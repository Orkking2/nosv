//! Completion-native file I/O with owned buffers.

use crate::io::{self, BufResult, IoBuf, IoBufMut, IoHandle, OpWork, OwnedOp};
use std::{
    ffi::CString,
    fmt,
    io::{Error, ErrorKind},
    os::{
        fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd},
        unix::{ffi::OsStrExt, fs::OpenOptionsExt},
    },
    path::Path,
    sync::Arc,
};
use uring::{opcode, squeue, types};

/// An asynchronously opened file bound to one runtime's I/O driver.
pub struct File {
    /// Descriptor retained by every in-flight operation.
    fd: Arc<OwnedFd>,

    /// Runtime driver used for subsequent operations.
    io: IoHandle,
    
    /// Whether the descriptor was opened with `O_APPEND`.
    append: bool,
}

impl File {
    /// Opens an existing file for reading on the current runtime.
    pub async fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let io = IoHandle::try_current().map_err(|_| io::runtime_closed_error())?;
        Self::open_on(&io, path).await
    }

    /// Opens an existing file for reading on an explicit runtime.
    pub async fn open_on(io: &IoHandle, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true);
        options.open_on(io, path).await
    }

    /// Creates or truncates a file for writing on the current runtime.
    pub async fn create(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let io = IoHandle::try_current().map_err(|_| io::runtime_closed_error())?;
        Self::create_on(&io, path).await
    }

    /// Creates or truncates a file for writing on an explicit runtime.
    pub async fn create_on(io: &IoHandle, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut options = OpenOptions::new();
        options.write(true).create(true).truncate(true);
        options.open_on(io, path).await
    }

    /// Creates a new file for writing on the current runtime.
    pub async fn create_new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let io = IoHandle::try_current().map_err(|_| io::runtime_closed_error())?;
        Self::create_new_on(&io, path).await
    }

    /// Creates a new file for writing on an explicit runtime.
    pub async fn create_new_on(io: &IoHandle, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        options.open_on(io, path).await
    }

    /// Reads at `offset`, returning both the result and buffer.
    pub async fn read_at<B: IoBufMut>(&self, buf: B, offset: u64) -> BufResult<usize, B> {
        self.read_at_offset(buf, offset, 0).await
    }

    /// Reads until the whole writable buffer is filled or EOF is reached.
    pub async fn read_exact_at<B: IoBufMut>(&self, mut buf: B, offset: u64) -> BufResult<(), B> {
        let total = buf.bytes_total();
        let mut filled = 0usize;
        while filled < total {
            let Some(file_offset) = offset.checked_add(filled as u64) else {
                return (
                    Err(Error::new(ErrorKind::InvalidInput, "file offset overflow")),
                    buf,
                );
            };
            let (result, returned) = self.read_at_offset(buf, file_offset, filled).await;
            buf = returned;
            match result {
                Ok(0) => {
                    return (
                        Err(Error::new(
                            ErrorKind::UnexpectedEof,
                            "failed to fill whole buffer",
                        )),
                        buf,
                    );
                }
                Ok(read) => filled += read,
                Err(error) => return (Err(error), buf),
            }
        }
        (Ok(()), buf)
    }

    /// Writes initialized bytes at `offset`, returning both result and buffer.
    pub async fn write_at<B: IoBuf>(&self, buf: B, offset: u64) -> BufResult<usize, B> {
        if self.append {
            return (
                Err(Error::new(
                    ErrorKind::InvalidInput,
                    "positioned writes are not allowed on append-mode files",
                )),
                buf,
            );
        }
        self.write_at_offset(buf, offset, 0).await
    }

    /// Writes all initialized bytes at `offset`.
    pub async fn write_all_at<B: IoBuf>(&self, mut buf: B, offset: u64) -> BufResult<(), B> {
        if self.append {
            return (
                Err(Error::new(
                    ErrorKind::InvalidInput,
                    "positioned writes are not allowed on append-mode files",
                )),
                buf,
            );
        }
        let total = buf.bytes_init();
        let mut written = 0usize;
        while written < total {
            let Some(file_offset) = offset.checked_add(written as u64) else {
                return (
                    Err(Error::new(ErrorKind::InvalidInput, "file offset overflow")),
                    buf,
                );
            };
            let (result, returned) = self.write_at_offset(buf, file_offset, written).await;
            buf = returned;
            match result {
                Ok(0) => {
                    return (
                        Err(Error::new(ErrorKind::WriteZero, "failed to write buffer")),
                        buf,
                    );
                }
                Ok(count) => written += count,
                Err(error) => return (Err(error), buf),
            }
        }
        (Ok(()), buf)
    }

    /// Appends initialized bytes using this append-mode descriptor.
    pub async fn append<B: IoBuf>(&self, buf: B) -> BufResult<usize, B> {
        if !self.append {
            return (
                Err(Error::new(
                    ErrorKind::InvalidInput,
                    "append requires a file opened in append mode",
                )),
                buf,
            );
        }
        self.write_at_offset(buf, u64::MAX, 0).await
    }

    /// Flushes file data and metadata to durable storage.
    pub async fn sync_all(&self) -> std::io::Result<()> {
        OwnedOp::new(
            &self.io,
            SyncWork {
                fd: self.fd.clone(),
                data_only: false,
            },
        )
        .await
    }

    /// Flushes file data, without requiring unrelated metadata, to storage.
    pub async fn sync_data(&self) -> std::io::Result<()> {
        OwnedOp::new(
            &self.io,
            SyncWork {
                fd: self.fd.clone(),
                data_only: true,
            },
        )
        .await
    }

    /// Issues one read into the buffer at an allocation-relative offset.
    async fn read_at_offset<B: IoBufMut>(
        &self,
        buf: B,
        file_offset: u64,
        buffer_offset: usize,
    ) -> BufResult<usize, B> {
        let remaining = buf.bytes_total().saturating_sub(buffer_offset);
        if remaining > u32::MAX as usize {
            return (
                Err(Error::new(
                    ErrorKind::InvalidInput,
                    "buffer is too large for io_uring",
                )),
                buf,
            );
        }
        OwnedOp::new(
            &self.io,
            ReadWork {
                fd: self.fd.clone(),
                buf,
                file_offset,
                buffer_offset,
                remaining,
            },
        )
        .await
    }

    /// Issues one write from the buffer at an allocation-relative offset.
    async fn write_at_offset<B: IoBuf>(
        &self,
        buf: B,
        file_offset: u64,
        buffer_offset: usize,
    ) -> BufResult<usize, B> {
        let remaining = buf.bytes_init().saturating_sub(buffer_offset);
        if remaining > u32::MAX as usize {
            return (
                Err(Error::new(
                    ErrorKind::InvalidInput,
                    "buffer is too large for io_uring",
                )),
                buf,
            );
        }
        OwnedOp::new(
            &self.io,
            WriteWork {
                fd: self.fd.clone(),
                buf,
                file_offset,
                buffer_offset,
                remaining,
            },
        )
        .await
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("File")
            .field("fd", &self.as_raw_fd())
            .field("append", &self.append)
            .finish_non_exhaustive()
    }
}

impl AsFd for File {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

impl AsRawFd for File {
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

/// Builder for standard file access, creation, and Unix flags.
#[derive(Clone, Debug)]
pub struct OpenOptions {
    /// Whether reads are permitted.
    read: bool,
    /// Whether ordinary writes are permitted.
    write: bool,
    /// Whether writes use append semantics.
    append: bool,
    /// Whether an existing file is truncated.
    truncate: bool,
    /// Whether a missing file is created.
    create: bool,
    /// Whether creation must be exclusive.
    create_new: bool,
    /// Unix permission bits used for creation.
    mode: u32,
    /// Additional caller-provided Unix open flags.
    custom_flags: i32,
}

impl OpenOptions {
    /// Creates an empty option set.
    pub fn new() -> Self {
        Self {
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
            mode: 0o666,
            custom_flags: 0,
        }
    }

    /// Enables or disables read access.
    pub fn read(&mut self, read: bool) -> &mut Self {
        self.read = read;
        self
    }

    /// Enables or disables write access.
    pub fn write(&mut self, write: bool) -> &mut Self {
        self.write = write;
        self
    }

    /// Enables or disables append access.
    pub fn append(&mut self, append: bool) -> &mut Self {
        self.append = append;
        self
    }

    /// Enables or disables truncation of an existing file.
    pub fn truncate(&mut self, truncate: bool) -> &mut Self {
        self.truncate = truncate;
        self
    }

    /// Enables or disables creation when the path is absent.
    pub fn create(&mut self, create: bool) -> &mut Self {
        self.create = create;
        self
    }

    /// Enables exclusive creation and failure when the path exists.
    pub fn create_new(&mut self, create_new: bool) -> &mut Self {
        self.create_new = create_new;
        self
    }

    /// Opens using the current runtime.
    pub async fn open(&self, path: impl AsRef<Path>) -> std::io::Result<File> {
        let io = IoHandle::try_current().map_err(|_| io::runtime_closed_error())?;
        self.open_on(&io, path).await
    }

    /// Opens using an explicit runtime I/O capability.
    pub async fn open_on(&self, io: &IoHandle, path: impl AsRef<Path>) -> std::io::Result<File> {
        let (flags, mode) = self.flags()?;
        let path = CString::new(path.as_ref().as_os_str().as_bytes()).map_err(|_| {
            Error::new(
                ErrorKind::InvalidInput,
                "file path contains an interior NUL byte",
            )
        })?;
        let fd = OwnedOp::new(io, OpenWork { path, flags, mode }).await?;
        Ok(File {
            fd: Arc::new(fd),
            io: io.clone(),
            append: self.append,
        })
    }

    /// Validates options and constructs Linux open flags.
    fn flags(&self) -> std::io::Result<(i32, libc::mode_t)> {
        if !self.read && !self.write && !self.append {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "an access mode is required",
            ));
        }
        if (self.truncate || self.create || self.create_new) && !self.write && !self.append {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "truncate and create options require write or append access",
            ));
        }

        let access = match (self.read, self.write || self.append) {
            (true, true) => libc::O_RDWR,
            (true, false) => libc::O_RDONLY,
            (false, true) => libc::O_WRONLY,
            (false, false) => unreachable!("validated access mode"),
        };
        let mut flags = access | libc::O_CLOEXEC;
        if self.append {
            flags |= libc::O_APPEND;
        }
        if self.truncate && !self.create_new {
            flags |= libc::O_TRUNC;
        }
        if self.create || self.create_new {
            flags |= libc::O_CREAT;
        }
        if self.create_new {
            flags |= libc::O_EXCL;
        }
        flags |= self.custom_flags & !libc::O_ACCMODE;
        Ok((flags, self.mode as libc::mode_t))
    }
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenOptionsExt for OpenOptions {
    fn mode(&mut self, mode: u32) -> &mut Self {
        self.mode = mode;
        self
    }

    fn custom_flags(&mut self, flags: i32) -> &mut Self {
        self.custom_flags = flags;
        self
    }
}

/// Owns an open path until `IORING_OP_OPENAT` completes.
struct OpenWork {
    /// Stable NUL-terminated path.
    path: CString,
    /// Validated open flags.
    flags: i32,
    /// Creation permission bits.
    mode: libc::mode_t,
}

impl OpWork<std::io::Result<OwnedFd>> for OpenWork {
    fn opcode(&self) -> u8 {
        opcode::OpenAt::CODE
    }

    fn entry(&mut self) -> squeue::Entry {
        opcode::OpenAt::new(types::Fd(libc::AT_FDCWD), self.path.as_ptr())
            .flags(self.flags)
            .mode(self.mode)
            .build()
    }

    fn complete(self: Box<Self>, result: i32) -> std::io::Result<OwnedFd> {
        if result < 0 {
            Err(Error::from_raw_os_error(-result))
        } else {
            // SAFETY: a successful OPENAT CQE transfers ownership of a new fd.
            Ok(unsafe { OwnedFd::from_raw_fd(result) })
        }
    }

    fn fail(self: Box<Self>, error: Error) -> std::io::Result<OwnedFd> {
        Err(error)
    }
}

/// Owns a mutable buffer and descriptor through one positioned read.
struct ReadWork<B> {
    /// Source descriptor retained through completion.
    fd: Arc<OwnedFd>,
    /// Destination buffer.
    buf: B,
    /// Positioned file offset.
    file_offset: u64,
    /// Allocation-relative destination offset.
    buffer_offset: usize,
    /// Maximum completion length.
    remaining: usize,
}

impl<B: IoBufMut> OpWork<BufResult<usize, B>> for ReadWork<B> {
    fn opcode(&self) -> u8 {
        opcode::Read::CODE
    }

    fn entry(&mut self) -> squeue::Entry {
        // SAFETY: buffer_offset is bounded by bytes_total in the constructor.
        let pointer = unsafe { self.buf.stable_mut_ptr().add(self.buffer_offset) };
        opcode::Read::new(
            types::Fd(self.fd.as_raw_fd()),
            pointer,
            self.remaining as u32,
        )
        .offset(self.file_offset)
        .build()
    }

    fn complete(mut self: Box<Self>, result: i32) -> BufResult<usize, B> {
        match io::result(result) {
            Ok(read) if read <= self.remaining => {
                let initialized = self.buf.bytes_init().max(self.buffer_offset + read);
                // SAFETY: the successful CQE reports that this prefix was initialized.
                unsafe { self.buf.set_init(initialized) };
                (Ok(read), self.buf)
            }
            Ok(_) => (
                Err(Error::new(
                    ErrorKind::InvalidData,
                    "kernel returned an oversized read",
                )),
                self.buf,
            ),
            Err(error) => (Err(error), self.buf),
        }
    }

    fn fail(self: Box<Self>, error: Error) -> BufResult<usize, B> {
        (Err(error), self.buf)
    }
}

/// Owns an initialized buffer and descriptor through one positioned write.
struct WriteWork<B> {
    /// Destination descriptor retained through completion.
    fd: Arc<OwnedFd>,
    /// Source buffer.
    buf: B,
    /// Positioned file offset, or `u64::MAX` for append.
    file_offset: u64,
    /// Allocation-relative source offset.
    buffer_offset: usize,
    /// Maximum completion length.
    remaining: usize,
}

impl<B: IoBuf> OpWork<BufResult<usize, B>> for WriteWork<B> {
    fn opcode(&self) -> u8 {
        opcode::Write::CODE
    }

    fn entry(&mut self) -> squeue::Entry {
        // SAFETY: buffer_offset is bounded by bytes_init in the constructor.
        let pointer = unsafe { self.buf.stable_ptr().add(self.buffer_offset) };
        opcode::Write::new(
            types::Fd(self.fd.as_raw_fd()),
            pointer,
            self.remaining as u32,
        )
        .offset(self.file_offset)
        .build()
    }

    fn complete(self: Box<Self>, result: i32) -> BufResult<usize, B> {
        match io::result(result) {
            Ok(written) if written <= self.remaining => (Ok(written), self.buf),
            Ok(_) => (
                Err(Error::new(
                    ErrorKind::InvalidData,
                    "kernel returned an oversized write",
                )),
                self.buf,
            ),
            Err(error) => (Err(error), self.buf),
        }
    }

    fn fail(self: Box<Self>, error: Error) -> BufResult<usize, B> {
        (Err(error), self.buf)
    }
}

/// Retains a file descriptor through a sync operation.
struct SyncWork {
    /// Descriptor retained through completion.
    fd: Arc<OwnedFd>,
    /// Whether to request `IORING_FSYNC_DATASYNC`.
    data_only: bool,
}

impl OpWork<std::io::Result<()>> for SyncWork {
    fn opcode(&self) -> u8 {
        opcode::Fsync::CODE
    }

    fn entry(&mut self) -> squeue::Entry {
        let operation = opcode::Fsync::new(types::Fd(self.fd.as_raw_fd()));
        if self.data_only {
            operation.flags(types::FsyncFlags::DATASYNC).build()
        } else {
            operation.build()
        }
    }

    fn complete(self: Box<Self>, result: i32) -> std::io::Result<()> {
        io::result(result).map(|_| ())
    }

    fn fail(self: Box<Self>, error: Error) -> std::io::Result<()> {
        Err(error)
    }
}
