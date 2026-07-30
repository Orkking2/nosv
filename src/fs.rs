//! Completion-native positional file I/O with owned buffers.
//!
//! [`File`] values are bound to the [`IoHandle`] used to open them. Read and write
//! methods take ownership of their buffers and return those buffers on every
//! operational outcome. Reads write from the start of an [`IoBufMut`] allocation
//! up to [`IoBufMut::bytes_total`]; writes consume the initialized prefix reported
//! by [`IoBuf::bytes_init`]. All ordinary file I/O is positioned and does not use
//! or advance a shared file cursor.
//!
//! Constructors without an `_on` suffix use [`IoHandle::try_current`]. Their `_on`
//! counterparts accept an explicit handle and may be constructed outside a runtime
//! polling scope. Dropping an in-flight future requests cancellation, while its
//! buffer, path, and descriptor remain retained through the terminal kernel
//! completion. A kernel side effect may still win cancellation.

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

/// An owned file descriptor bound to one runtime's I/O driver.
///
/// `File` is deliberately not [`Clone`]. It is `Send + Sync`, and each in-flight
/// operation internally retains the descriptor, so dropping the `File` does not
/// invalidate kernel work already submitted. Positioned methods do not modify a
/// shared cursor; [`File::append`] is available only when opened in append mode.
pub struct File {
    /// Descriptor retained by every in-flight operation.
    fd: Arc<OwnedFd>,

    /// Runtime driver used for subsequent operations.
    io: IoHandle,

    /// Whether the descriptor was opened with `O_APPEND`.
    append: bool,
}

impl File {
    /// Opens an existing file for read-only access on the current runtime.
    ///
    /// This is equivalent to enabling [`OpenOptions::read`] and calling
    /// [`OpenOptions::open`].
    ///
    /// # Errors
    ///
    /// Returns an error when no current I/O runtime exists, the path contains an
    /// interior NUL byte, `IORING_OP_OPENAT` is unsupported, the runtime closes,
    /// or the operating system rejects the open.
    pub async fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let io = IoHandle::try_current().map_err(|_| io::runtime_closed_error())?;
        Self::open_on(&io, path).await
    }

    /// Opens an existing file for read-only access on `io`.
    ///
    /// This form does not require a current-runtime polling scope.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid path, an unsupported open opcode, a closed
    /// or fork-inherited runtime, or an operating-system open failure.
    pub async fn open_on(io: &IoHandle, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true);
        options.open_on(io, path).await
    }

    /// Opens a file for write-only access, creating or truncating it.
    ///
    /// Creation uses mode `0o666` subject to the process umask.
    ///
    /// # Errors
    ///
    /// Returns an error when no current I/O runtime exists, the path is invalid,
    /// the open opcode is unsupported, the runtime closes, or the operating system
    /// rejects creation or truncation.
    pub async fn create(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let io = IoHandle::try_current().map_err(|_| io::runtime_closed_error())?;
        Self::create_on(&io, path).await
    }

    /// Opens a file on `io` for write-only access, creating or truncating it.
    ///
    /// Creation uses mode `0o666` subject to the process umask. This form does not
    /// require a current-runtime polling scope.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid path, an unsupported open opcode, a closed
    /// or fork-inherited runtime, or an operating-system open failure.
    pub async fn create_on(io: &IoHandle, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut options = OpenOptions::new();
        options.write(true).create(true).truncate(true);
        options.open_on(io, path).await
    }

    /// Creates a new write-only file and fails if the path already exists.
    ///
    /// Creation uses mode `0o666` subject to the process umask.
    ///
    /// # Errors
    ///
    /// In addition to runtime, path, capability, and operating-system failures,
    /// returns [`ErrorKind::AlreadyExists`] when the target already exists.
    pub async fn create_new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let io = IoHandle::try_current().map_err(|_| io::runtime_closed_error())?;
        Self::create_new_on(&io, path).await
    }

    /// Creates a new write-only file on `io` and fails if it already exists.
    ///
    /// Creation uses mode `0o666` subject to the process umask. This form does not
    /// require a current-runtime polling scope.
    ///
    /// # Errors
    ///
    /// In addition to closed-runtime, path, opcode, and operating-system failures,
    /// returns [`ErrorKind::AlreadyExists`] when the target already exists.
    pub async fn create_new_on(io: &IoHandle, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        options.open_on(io, path).await
    }

    /// Reads from the absolute file `offset` into the start of `buf`.
    ///
    /// At most [`IoBufMut::bytes_total`] bytes are read. Existing initialized bytes
    /// may be overwritten. On success, the returned count may be smaller than the
    /// writable size, including zero at end of file, and the returned buffer marks
    /// at least that prefix initialized. The operation does not change a file
    /// cursor.
    ///
    /// # Errors
    ///
    /// The result half reports validation, unsupported-opcode, runtime-closure, and
    /// kernel read errors. The buffer is returned unchanged on an error reported by
    /// this single operation.
    pub async fn read_at<B: IoBufMut>(&self, buf: B, offset: u64) -> BufResult<usize, B> {
        self.read_at_offset(buf, offset, 0).await
    }

    /// Reads from `offset` until the entire writable allocation is filled.
    ///
    /// The target length is [`IoBufMut::bytes_total`], not the buffer's initialized
    /// length. Multiple positioned reads may be issued. If a later read fails, the
    /// returned buffer retains bytes initialized by earlier successful reads.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnexpectedEof`] if EOF occurs before the allocation is
    /// full and [`ErrorKind::InvalidInput`] if the offset range overflows or the
    /// buffer exceeds one SQE's length limit. Runtime, opcode, and kernel errors are
    /// forwarded in the result half; the buffer is always returned.
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

    /// Writes initialized bytes from `buf` at the absolute file `offset`.
    ///
    /// One operation writes at most [`IoBuf::bytes_init`] bytes and may complete
    /// with a short count. It does not change a file cursor.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] without submission for an append-mode
    /// file or an oversized buffer. Other validation, unsupported-opcode, runtime,
    /// and kernel errors are returned alongside the original buffer.
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

    /// Writes the entire initialized prefix of `buf` starting at `offset`.
    ///
    /// Short writes are retried with increasing absolute offsets. The method does
    /// not change a file cursor and always returns the original buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] for append-mode files, offset overflow,
    /// or an oversized operation, and [`ErrorKind::WriteZero`] if a write makes no
    /// progress. Runtime, unsupported-opcode, and kernel errors are forwarded.
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

    /// Performs one append-mode write from the initialized prefix of `buf`.
    ///
    /// The descriptor must have been opened with [`OpenOptions::append`]. The
    /// kernel chooses the end-of-file position atomically according to `O_APPEND`;
    /// this method may return a short count and does not retry it.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if the file is not in append mode or the
    /// buffer is too large for one SQE. Runtime, unsupported-opcode, and kernel
    /// write errors are returned together with the original buffer.
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

    /// Requests synchronization of file data and associated metadata.
    ///
    /// This is the completion-native equivalent of `fsync(2)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `IORING_OP_FSYNC` is unsupported, the runtime is closed,
    /// or the operating system cannot synchronize the file.
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

    /// Requests synchronization of file data and metadata needed to retrieve it.
    ///
    /// This uses the `IORING_FSYNC_DATASYNC` flag and may omit metadata unrelated
    /// to subsequent data retrieval.
    ///
    /// # Errors
    ///
    /// Returns an error if `IORING_OP_FSYNC` is unsupported, the runtime is closed,
    /// or the operating system cannot synchronize the file.
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

/// Configures file access, creation behavior, Unix mode, and custom flags.
///
/// The builder follows the standard-library [`std::fs::OpenOptions`] model and
/// implements [`OpenOptionsExt`]. At least one of read, write, or append must be
/// enabled before opening. `O_CLOEXEC` is always applied and cannot be disabled by
/// custom flags.
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
    /// Creates an option set with no access mode enabled.
    ///
    /// Creation mode defaults to `0o666`; all access and creation flags default to
    /// false. Calling [`OpenOptions::open`] before enabling read, write, or append
    /// returns [`ErrorKind::InvalidInput`].
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
    ///
    /// Read combined with write or append produces read-write access.
    pub fn read(&mut self, read: bool) -> &mut Self {
        self.read = read;
        self
    }

    /// Enables or disables ordinary write access.
    ///
    /// This does not enable append semantics; use [`OpenOptions::append`] for that.
    pub fn write(&mut self, write: bool) -> &mut Self {
        self.write = write;
        self
    }

    /// Enables or disables append access (`O_APPEND`).
    ///
    /// Append counts as write access. Positioned [`File::write_at`] and
    /// [`File::write_all_at`] reject a file opened with this option; use
    /// [`File::append`] instead.
    pub fn append(&mut self, append: bool) -> &mut Self {
        self.append = append;
        self
    }

    /// Enables or disables truncation of an existing file (`O_TRUNC`).
    ///
    /// Truncation requires write or append access. It is ignored when
    /// [`OpenOptions::create_new`] is enabled.
    pub fn truncate(&mut self, truncate: bool) -> &mut Self {
        self.truncate = truncate;
        self
    }

    /// Enables or disables creation when the path is absent (`O_CREAT`).
    ///
    /// Creation requires write or append access and uses the configured Unix mode
    /// subject to the process umask.
    pub fn create(&mut self, create: bool) -> &mut Self {
        self.create = create;
        self
    }

    /// Enables or disables exclusive creation (`O_CREAT | O_EXCL`).
    ///
    /// This option requires write or append access, fails if the target exists,
    /// and suppresses truncation.
    pub fn create_new(&mut self, create_new: bool) -> &mut Self {
        self.create_new = create_new;
        self
    }

    /// Opens `path` with these options on the current runtime.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] for incompatible options or an interior
    /// NUL byte. It also reports absence of a current runtime, runtime closure, an
    /// unsupported open opcode, and operating-system open errors.
    pub async fn open(&self, path: impl AsRef<Path>) -> std::io::Result<File> {
        let io = IoHandle::try_current().map_err(|_| io::runtime_closed_error())?;
        self.open_on(&io, path).await
    }

    /// Opens `path` with these options on the explicit `io` capability.
    ///
    /// This form does not require a current-runtime polling scope.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] for incompatible options or an interior
    /// NUL byte. It also reports a closed or fork-inherited runtime, an unsupported
    /// open opcode, and operating-system open errors.
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

    /// Validates access/creation combinations and constructs Linux open flags.
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
