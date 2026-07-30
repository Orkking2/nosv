//! Numeric-address TCP sockets backed by completion-native `io_uring`.
//!
//! [`TcpStream::connect`] uses `IORING_OP_CONNECT`; owned-buffer send and receive
//! methods use `IORING_OP_SEND` and `IORING_OP_RECV`. [`TcpListener::accept`] waits
//! with `IORING_OP_POLL_ADD` and performs the final `accept4(2)` as a nonblocking
//! syscall with `CLOEXEC` and nonblocking flags set atomically. Socket creation,
//! binding, listening, options, and address queries are short synchronous syscalls.
//!
//! Only numeric address forms implementing this module's sealed [`ToSocketAddrs`]
//! trait are accepted; no hostname resolution runs on nOS-V workers. Buffer-taking
//! methods return their owned buffer on every outcome and retain it through terminal
//! kernel completion after caller cancellation. With `tokio-compat`, [`TcpStream`]
//! also implements Tokio's borrowed [`crate::io::AsyncRead`] and
//! [`crate::io::AsyncWrite`] using independent readiness lanes. A borrowed operation
//! that returns `Pending` has not modified or consumed its caller's buffer.

use crate::io::{self, BufResult, IoBuf, IoBufMut, IoHandle, OpWork, OwnedOp};
#[cfg(feature = "tokio-compat")]
use crate::util::lock;
use socket2::{Domain, SockAddr, Socket, Type};
use std::{
    fmt,
    io::{Error, ErrorKind},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr},
    os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, RawFd},
    sync::Arc,
};
use uring::{opcode, squeue, types};

/// Converts a supported numeric address form into ordered connection candidates.
///
/// Implementations are provided for [`SocketAddr`], `(IpAddr, u16)`,
/// `(Ipv4Addr, u16)`, `(Ipv6Addr, u16)`, socket-address arrays and slices, and
/// references to those forms. The trait is sealed so hostname-bearing forms cannot
/// introduce blocking DNS resolution on runtime workers. Constructors try returned
/// addresses sequentially and report the last failure. An empty array or slice is
/// rejected with [`ErrorKind::InvalidInput`].
pub trait ToSocketAddrs: sealed::Sealed {
    /// Copies numeric candidates in the order they should be attempted.
    fn to_socket_addrs(&self) -> Vec<SocketAddr>;
}

/// Seals numeric address conversions against hostname-bearing implementations.
mod sealed {
    /// Private supertrait for numeric address forms.
    pub trait Sealed {}

    impl Sealed for std::net::SocketAddr {}
    impl Sealed for (std::net::IpAddr, u16) {}
    impl Sealed for (std::net::Ipv4Addr, u16) {}
    impl Sealed for (std::net::Ipv6Addr, u16) {}
    impl Sealed for [std::net::SocketAddr] {}
    impl<const N: usize> Sealed for [std::net::SocketAddr; N] {}
    impl<T: Sealed + ?Sized> Sealed for &T {}
}

impl ToSocketAddrs for SocketAddr {
    fn to_socket_addrs(&self) -> Vec<SocketAddr> {
        vec![*self]
    }
}

impl ToSocketAddrs for (IpAddr, u16) {
    fn to_socket_addrs(&self) -> Vec<SocketAddr> {
        vec![SocketAddr::new(self.0, self.1)]
    }
}

impl ToSocketAddrs for (Ipv4Addr, u16) {
    fn to_socket_addrs(&self) -> Vec<SocketAddr> {
        vec![SocketAddr::new(self.0.into(), self.1)]
    }
}

impl ToSocketAddrs for (Ipv6Addr, u16) {
    fn to_socket_addrs(&self) -> Vec<SocketAddr> {
        vec![SocketAddr::new(self.0.into(), self.1)]
    }
}

impl ToSocketAddrs for [SocketAddr] {
    fn to_socket_addrs(&self) -> Vec<SocketAddr> {
        self.to_vec()
    }
}

impl<const N: usize> ToSocketAddrs for [SocketAddr; N] {
    fn to_socket_addrs(&self) -> Vec<SocketAddr> {
        self.to_vec()
    }
}

impl<T: ToSocketAddrs + ?Sized> ToSocketAddrs for &T {
    fn to_socket_addrs(&self) -> Vec<SocketAddr> {
        (**self).to_socket_addrs()
    }
}

/// A connected, nonblocking TCP stream bound to one I/O runtime.
///
/// The type is `Send + Sync` but deliberately not [`Clone`]. In-flight operations
/// retain the socket independently, so dropping the stream cannot invalidate a
/// submitted kernel request. With `tokio-compat`, independent read and write
/// readiness lanes permit full-duplex use by Tokio utilities such as `split` and
/// `copy_bidirectional`.
pub struct TcpStream {
    /// Socket retained by every in-flight operation.
    socket: Arc<Socket>,
    /// Runtime driver used for completion-native and readiness operations.
    io: IoHandle,
    #[cfg(feature = "tokio-compat")]
    /// Independent borrowed-read readiness lane.
    read_ready: std::sync::Mutex<ReadyLane>,
    #[cfg(feature = "tokio-compat")]
    /// Independent borrowed-write readiness lane.
    write_ready: std::sync::Mutex<ReadyLane>,
}

impl TcpStream {
    /// Connects to the first successful numeric address on the current runtime.
    ///
    /// Candidates are attempted sequentially in [`ToSocketAddrs`] order. Each
    /// attempt uses a new atomic `CLOEXEC`/nonblocking socket.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] for no candidates, or an error when no
    /// current runtime exists, `IORING_OP_CONNECT` is unsupported, the runtime
    /// closes, socket creation fails, or every candidate fails. For multiple
    /// candidates, the final attempt's error is returned.
    pub async fn connect(addresses: impl ToSocketAddrs) -> std::io::Result<Self> {
        let io = IoHandle::try_current().map_err(|_| io::runtime_closed_error())?;
        Self::connect_on(&io, addresses).await
    }

    /// Connects to the first successful numeric address using `io`.
    ///
    /// This form does not require a current-runtime polling scope. Candidates are
    /// attempted sequentially, each with a new atomic `CLOEXEC`/nonblocking socket.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] for no candidates. It also reports a
    /// closed or fork-inherited runtime, unsupported connect opcode, socket creation
    /// failure, and connection failures; the last candidate's error is returned.
    pub async fn connect_on(io: &IoHandle, addresses: impl ToSocketAddrs) -> std::io::Result<Self> {
        let mut last = None;
        let addresses = addresses.to_socket_addrs();

        if addresses.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "no socket addresses supplied",
            ));
        }

        for address in addresses {
            let socket = match tcp_socket(address) {
                Ok(socket) => socket,
                Err(error) => {
                    last = Some(error);
                    continue;
                }
            };
            let address = SockAddr::from(address);

            match OwnedOp::new(
                io,
                ConnectWork {
                    socket: Some(socket),
                    address,
                },
            )
            .await
            {
                Ok(socket) => return Ok(Self::from_socket(io, socket)),
                Err(error) => last = Some(error),
            }
        }

        Err(last.expect("nonempty address attempt has a result"))
    }

    /// Receives once into the start of the owned buffer.
    ///
    /// At most [`IoBufMut::bytes_total`] bytes are written. Existing initialized
    /// bytes may be overwritten. A successful count of zero means orderly peer
    /// shutdown, and the returned buffer marks at least the reported prefix
    /// initialized.
    ///
    /// # Errors
    ///
    /// Validation, unsupported-opcode, runtime-closure, and socket receive errors
    /// are returned alongside the original buffer.
    pub async fn recv<B: IoBufMut>(&self, buf: B) -> BufResult<usize, B> {
        self.recv_offset(buf, 0).await
    }

    /// Receives until the entire writable allocation is filled.
    ///
    /// The target length is [`IoBufMut::bytes_total`], and multiple receives may be
    /// issued. If a later receive fails, the returned buffer retains the prefix
    /// initialized by earlier completions.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnexpectedEof`] if the peer closes before the allocation
    /// is full and [`ErrorKind::InvalidInput`] for an oversized operation. Runtime,
    /// unsupported-opcode, and socket errors are returned with the buffer.
    pub async fn recv_exact<B: IoBufMut>(&self, mut buf: B) -> BufResult<(), B> {
        let total = buf.bytes_total();
        let mut filled = 0usize;
        while filled < total {
            let (result, returned) = self.recv_offset(buf, filled).await;
            buf = returned;
            match result {
                Ok(0) => {
                    return (
                        Err(Error::new(
                            ErrorKind::UnexpectedEof,
                            "connection closed early",
                        )),
                        buf,
                    );
                }
                Ok(count) => filled += count,
                Err(error) => return (Err(error), buf),
            }
        }
        (Ok(()), buf)
    }

    /// Sends once from the initialized prefix of an owned buffer.
    ///
    /// At most [`IoBuf::bytes_init`] bytes are sent, `MSG_NOSIGNAL` suppresses
    /// `SIGPIPE`, and a successful operation may report a short count.
    ///
    /// # Errors
    ///
    /// Validation, unsupported-opcode, runtime-closure, and socket send errors are
    /// returned alongside the original buffer.
    pub async fn send<B: IoBuf>(&self, buf: B) -> BufResult<usize, B> {
        self.send_offset(buf, 0).await
    }

    /// Sends the entire initialized prefix of an owned buffer.
    ///
    /// Short sends are retried and every attempt uses `MSG_NOSIGNAL`. The original
    /// buffer is returned after success or failure.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::WriteZero`] if an attempt makes no progress and
    /// [`ErrorKind::InvalidInput`] for an oversized operation. Runtime,
    /// unsupported-opcode, and socket errors are forwarded with the buffer.
    pub async fn send_all<B: IoBuf>(&self, mut buf: B) -> BufResult<(), B> {
        let total = buf.bytes_init();
        let mut sent = 0usize;
        while sent < total {
            let (result, returned) = self.send_offset(buf, sent).await;
            buf = returned;
            match result {
                Ok(0) => {
                    return (
                        Err(Error::new(ErrorKind::WriteZero, "failed to send buffer")),
                        buf,
                    );
                }
                Ok(count) => sent += count,
                Err(error) => return (Err(error), buf),
            }
        }
        (Ok(()), buf)
    }

    /// Returns the local IP address and port.
    ///
    /// # Errors
    ///
    /// Returns an operating-system address-query error or [`ErrorKind::InvalidData`]
    /// if the descriptor unexpectedly reports a non-IP address.
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        socket_addr(self.socket.local_addr()?)
    }

    /// Returns the connected peer's IP address and port.
    ///
    /// # Errors
    ///
    /// Returns an operating-system address-query error or [`ErrorKind::InvalidData`]
    /// if the descriptor unexpectedly reports a non-IP address.
    pub fn peer_addr(&self) -> std::io::Result<SocketAddr> {
        socket_addr(self.socket.peer_addr()?)
    }

    /// Returns and clears the pending `SO_ERROR`, if any.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket option cannot be queried.
    pub fn take_error(&self) -> std::io::Result<Option<Error>> {
        self.socket.take_error()
    }

    /// Returns whether `TCP_NODELAY` disables Nagle's algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket option cannot be queried.
    pub fn nodelay(&self) -> std::io::Result<bool> {
        self.socket.tcp_nodelay()
    }

    /// Enables or disables `TCP_NODELAY`.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket option cannot be changed.
    pub fn set_nodelay(&self, nodelay: bool) -> std::io::Result<()> {
        self.socket.set_tcp_nodelay(nodelay)
    }

    /// Disables further receives, sends, or both according to `how`.
    ///
    /// This calls the nonblocking `shutdown(2)` syscall immediately.
    ///
    /// # Errors
    ///
    /// Returns an operating-system shutdown error.
    pub fn shutdown(&self, how: Shutdown) -> std::io::Result<()> {
        self.socket.shutdown(how)
    }

    /// Constructs a public stream and its optional independent readiness lanes.
    fn from_socket(io: &IoHandle, socket: Socket) -> Self {
        Self {
            socket: Arc::new(socket),
            io: io.clone(),
            #[cfg(feature = "tokio-compat")]
            read_ready: std::sync::Mutex::new(ReadyLane::new(libc::POLLIN as u32)),
            #[cfg(feature = "tokio-compat")]
            write_ready: std::sync::Mutex::new(ReadyLane::new(libc::POLLOUT as u32)),
        }
    }

    /// Issues one receive at an allocation-relative buffer offset.
    async fn recv_offset<B: IoBufMut>(&self, buf: B, buffer_offset: usize) -> BufResult<usize, B> {
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
            RecvWork {
                socket: self.socket.clone(),
                buf,
                buffer_offset,
                remaining,
            },
        )
        .await
    }

    /// Issues one send at an allocation-relative buffer offset.
    async fn send_offset<B: IoBuf>(&self, buf: B, buffer_offset: usize) -> BufResult<usize, B> {
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
            SendWork {
                socket: self.socket.clone(),
                buf,
                buffer_offset,
                remaining,
            },
        )
        .await
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TcpStream")
            .field("fd", &self.as_raw_fd())
            .field("local_addr", &self.local_addr())
            .field("peer_addr", &self.peer_addr())
            .finish_non_exhaustive()
    }
}

impl AsFd for TcpStream {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.socket.as_fd()
    }
}

impl AsRawFd for TcpStream {
    fn as_raw_fd(&self) -> RawFd {
        self.socket.as_raw_fd()
    }
}

/// A nonblocking TCP listener bound to one I/O runtime.
///
/// Listener creation performs only numeric-address socket setup; it never resolves
/// hostnames. The type is `Send + Sync` and retains its descriptor for pending
/// readiness operations.
pub struct TcpListener {
    /// Listening descriptor retained by accept polls.
    socket: Arc<Socket>,
    /// Runtime driver used for readiness operations and accepted streams.
    io: IoHandle,
}

impl TcpListener {
    /// Binds and listens on the first successful address on the current runtime.
    ///
    /// Socket creation, `bind(2)`, and `listen(2)` are short synchronous setup calls
    /// performed when the future is first polled. Candidates are tried sequentially
    /// with a backlog of 1024.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] for no candidates, or an error when no
    /// current runtime exists, socket setup fails for every candidate, or the runtime
    /// is already closed. The final candidate's error is returned.
    pub async fn bind(addresses: impl ToSocketAddrs) -> std::io::Result<Self> {
        let io = IoHandle::try_current().map_err(|_| io::runtime_closed_error())?;
        Self::bind_on(&io, addresses).await
    }

    /// Binds and listens on the first successful address using `io`.
    ///
    /// This form does not require a current-runtime polling scope. Socket creation,
    /// `bind(2)`, and `listen(2)` are short synchronous setup calls; candidates are
    /// tried sequentially with a backlog of 1024.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] for no candidates. It also reports a
    /// closed or fork-inherited runtime and socket setup failures, returning the
    /// final candidate's error when all attempts fail.
    pub async fn bind_on(io: &IoHandle, addresses: impl ToSocketAddrs) -> std::io::Result<Self> {
        io.ensure_running()
            .map_err(|_| io::runtime_closed_error())?;
        let addresses = addresses.to_socket_addrs();
        if addresses.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "no socket addresses supplied",
            ));
        }
        let mut last = None;
        for address in addresses {
            match bind_socket(address) {
                Ok(socket) => {
                    return Ok(Self {
                        socket: Arc::new(socket),
                        io: io.clone(),
                    });
                }
                Err(error) => last = Some(error),
            }
        }
        Err(last.expect("nonempty address attempt has a result"))
    }

    /// Accepts one connection and returns its numeric peer address.
    ///
    /// The method first attempts nonblocking `accept4(2)`. If no connection is
    /// ready, it waits with a one-shot `IORING_OP_POLL_ADD` and retries. Accepted
    /// descriptors receive `CLOEXEC` and nonblocking status atomically. Dropping a
    /// pending future cancels only its readiness request; no borrowed state is
    /// retained and no later final accept syscall is performed by that future.
    ///
    /// # Errors
    ///
    /// Returns an error when poll-add is unsupported, the runtime closes, readiness
    /// polling fails, `accept4(2)` fails, or the peer address is not an IP address.
    pub async fn accept(&self) -> std::io::Result<(TcpStream, SocketAddr)> {
        loop {
            match accept_socket(&self.socket) {
                Ok((socket, address)) => {
                    return Ok((
                        TcpStream::from_socket(&self.io, socket),
                        socket_addr(address)?,
                    ));
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => {}
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
            OwnedOp::new(
                &self.io,
                PollWork {
                    socket: self.socket.clone(),
                    events: libc::POLLIN as u32,
                },
            )
            .await?;
        }
    }

    /// Returns the listener's bound IP address and port.
    ///
    /// # Errors
    ///
    /// Returns an operating-system address-query error or [`ErrorKind::InvalidData`]
    /// for an unexpected non-IP address.
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        socket_addr(self.socket.local_addr()?)
    }

    /// Returns and clears the listener's pending asynchronous socket error.
    ///
    /// A return value of `Ok(None)` means that no error is pending.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket option cannot be queried.
    pub fn take_error(&self) -> std::io::Result<Option<Error>> {
        self.socket.take_error()
    }
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TcpListener")
            .field("fd", &self.as_raw_fd())
            .field("local_addr", &self.local_addr())
            .finish_non_exhaustive()
    }
}

impl AsFd for TcpListener {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.socket.as_fd()
    }
}

impl AsRawFd for TcpListener {
    fn as_raw_fd(&self) -> RawFd {
        self.socket.as_raw_fd()
    }
}

/// Creates an atomic CLOEXEC/nonblocking stream socket for one address family.
fn tcp_socket(address: SocketAddr) -> std::io::Result<Socket> {
    let domain = if address.is_ipv4() {
        Domain::IPV4
    } else {
        Domain::IPV6
    };
    Socket::new(domain, Type::STREAM.nonblocking().cloexec(), None)
}

/// Creates, binds, and listens without running a blocking operation.
fn bind_socket(address: SocketAddr) -> std::io::Result<Socket> {
    let socket = tcp_socket(address)?;
    socket.bind(&SockAddr::from(address))?;
    socket.listen(1024)?;
    Ok(socket)
}

/// Accepts with CLOEXEC and nonblocking status set atomically.
fn accept_socket(listener: &Socket) -> std::io::Result<(Socket, SockAddr)> {
    // SAFETY: accept4 initializes an internet sockaddr and its length; a successful
    // descriptor is immediately adopted into exactly one owning Socket.
    unsafe {
        SockAddr::try_init(|storage, length| {
            let fd = libc::accept4(
                listener.as_raw_fd(),
                storage.cast(),
                length,
                libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
            );
            if fd < 0 {
                Err(Error::last_os_error())
            } else {
                // SAFETY: accept4 returned a new uniquely owned descriptor.
                Ok(Socket::from_raw_fd(fd))
            }
        })
    }
}

/// Converts only internet socket address families.
fn socket_addr(address: SockAddr) -> std::io::Result<SocketAddr> {
    address
        .as_socket()
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "socket returned a non-IP address"))
}

/// Owns a socket and address through asynchronous connection establishment.
struct ConnectWork {
    /// Socket transferred to the result after successful connection.
    socket: Option<Socket>,
    /// Stable peer address retained through completion.
    address: SockAddr,
}

impl OpWork<std::io::Result<Socket>> for ConnectWork {
    fn opcode(&self) -> u8 {
        opcode::Connect::CODE
    }

    fn entry(&mut self) -> squeue::Entry {
        opcode::Connect::new(
            types::Fd(self.socket.as_ref().expect("connect socket").as_raw_fd()),
            self.address.as_ptr().cast(),
            self.address.len(),
        )
        .build()
    }

    fn complete(mut self: Box<Self>, result: i32) -> std::io::Result<Socket> {
        io::result(result).map(|_| self.socket.take().expect("completed connect socket"))
    }

    fn fail(self: Box<Self>, error: Error) -> std::io::Result<Socket> {
        Err(error)
    }
}

/// Owns a socket and mutable buffer through one receive.
struct RecvWork<B> {
    /// Socket retained through completion.
    socket: Arc<Socket>,
    /// Destination buffer.
    buf: B,
    /// Allocation-relative destination offset.
    buffer_offset: usize,
    /// Maximum completion length.
    remaining: usize,
}

impl<B: IoBufMut> OpWork<BufResult<usize, B>> for RecvWork<B> {
    fn opcode(&self) -> u8 {
        opcode::Recv::CODE
    }

    fn entry(&mut self) -> squeue::Entry {
        // SAFETY: buffer_offset is bounded by bytes_total in the constructor.
        let pointer = unsafe { self.buf.stable_mut_ptr().add(self.buffer_offset) };
        opcode::Recv::new(
            types::Fd(self.socket.as_raw_fd()),
            pointer,
            self.remaining as u32,
        )
        .build()
    }

    fn complete(mut self: Box<Self>, result: i32) -> BufResult<usize, B> {
        match io::result(result) {
            Ok(read) if read <= self.remaining => {
                let initialized = self.buf.bytes_init().max(self.buffer_offset + read);
                // SAFETY: the successful CQE initialized the reported prefix.
                unsafe { self.buf.set_init(initialized) };
                (Ok(read), self.buf)
            }
            Ok(_) => (
                Err(Error::new(
                    ErrorKind::InvalidData,
                    "kernel returned an oversized receive",
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

/// Owns a socket and initialized buffer through one send.
struct SendWork<B> {
    /// Socket retained through completion.
    socket: Arc<Socket>,
    /// Source buffer.
    buf: B,
    /// Allocation-relative source offset.
    buffer_offset: usize,
    /// Maximum completion length.
    remaining: usize,
}

impl<B: IoBuf> OpWork<BufResult<usize, B>> for SendWork<B> {
    fn opcode(&self) -> u8 {
        opcode::Send::CODE
    }

    fn entry(&mut self) -> squeue::Entry {
        // SAFETY: buffer_offset is bounded by bytes_init in the constructor.
        let pointer = unsafe { self.buf.stable_ptr().add(self.buffer_offset) };
        opcode::Send::new(
            types::Fd(self.socket.as_raw_fd()),
            pointer,
            self.remaining as u32,
        )
        .flags(libc::MSG_NOSIGNAL)
        .build()
    }

    fn complete(self: Box<Self>, result: i32) -> BufResult<usize, B> {
        match io::result(result) {
            Ok(sent) if sent <= self.remaining => (Ok(sent), self.buf),
            Ok(_) => (
                Err(Error::new(
                    ErrorKind::InvalidData,
                    "kernel returned an oversized send",
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

/// Owns a socket through a one-shot readiness poll.
struct PollWork {
    /// Descriptor retained while the readiness request is live.
    socket: Arc<Socket>,
    /// Linux poll event mask.
    events: u32,
}

impl OpWork<std::io::Result<()>> for PollWork {
    fn opcode(&self) -> u8 {
        opcode::PollAdd::CODE
    }

    fn entry(&mut self) -> squeue::Entry {
        opcode::PollAdd::new(types::Fd(self.socket.as_raw_fd()), self.events).build()
    }

    fn complete(self: Box<Self>, result: i32) -> std::io::Result<()> {
        io::result(result).map(|_| ())
    }

    fn fail(self: Box<Self>, error: Error) -> std::io::Result<()> {
        Err(error)
    }
}

#[cfg(feature = "tokio-compat")]
/// One persistent readiness request for a borrowed-I/O direction.
struct ReadyLane {
    /// Read or write poll event mask.
    events: u32,
    /// Persistent readiness operation, if currently pending.
    operation: Option<OwnedOp<std::io::Result<()>>>,
}

#[cfg(feature = "tokio-compat")]
impl ReadyLane {
    /// Creates an empty one-direction readiness lane.
    fn new(events: u32) -> Self {
        Self {
            events,
            operation: None,
        }
    }

    /// Polls or lazily creates the lane's one-shot readiness operation.
    fn poll(
        &mut self,
        io: &IoHandle,
        socket: &Arc<Socket>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        use std::future::Future;
        if self.operation.is_none() {
            self.operation = Some(OwnedOp::new(
                io,
                PollWork {
                    socket: socket.clone(),
                    events: self.events,
                },
            ));
        }
        let result =
            std::pin::Pin::new(self.operation.as_mut().expect("readiness operation")).poll(cx);
        if result.is_ready() {
            self.operation = None;
        }
        result
    }
}

#[cfg(feature = "tokio-compat")]
/// Performs borrowed reads only after an immediate nonblocking receive succeeds or
/// the independent read readiness lane completes. `Pending` leaves `ReadBuf`
/// untouched.
impl tokio::io::AsyncRead for TcpStream {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let this = self.get_mut();
        loop {
            if buf.remaining() == 0 {
                return std::task::Poll::Ready(Ok(()));
            }
            // SAFETY: recv never deinitializes bytes and assume_init publishes only success.
            let unfilled = unsafe { buf.unfilled_mut() };
            // SAFETY: the socket is nonblocking and the slice is writable for its full length.
            let result = unsafe {
                libc::recv(
                    this.socket.as_raw_fd(),
                    unfilled.as_mut_ptr().cast(),
                    unfilled.len(),
                    0,
                )
            };
            if result >= 0 {
                let read = result as usize;
                // SAFETY: recv initialized exactly the returned prefix.
                unsafe { buf.assume_init(read) };
                buf.advance(read);
                return std::task::Poll::Ready(Ok(()));
            }
            let error = Error::last_os_error();
            if error.kind() == ErrorKind::Interrupted {
                continue;
            }
            if error.kind() != ErrorKind::WouldBlock {
                return std::task::Poll::Ready(Err(error));
            }
            match lock(&this.read_ready).poll(&this.io, &this.socket, cx) {
                std::task::Poll::Ready(Ok(())) => continue,
                other => return other,
            }
        }
    }
}

#[cfg(feature = "tokio-compat")]
/// Performs borrowed writes with `MSG_NOSIGNAL` and an independent write readiness
/// lane. Flush is immediate because the stream has no userspace output buffer;
/// shutdown closes the write half.
impl tokio::io::AsyncWrite for TcpStream {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let this = self.get_mut();
        loop {
            // SAFETY: the socket is nonblocking and buf remains borrowed for this call only.
            let result = unsafe {
                libc::send(
                    this.socket.as_raw_fd(),
                    buf.as_ptr().cast(),
                    buf.len(),
                    libc::MSG_NOSIGNAL,
                )
            };
            if result >= 0 {
                return std::task::Poll::Ready(Ok(result as usize));
            }
            let error = Error::last_os_error();
            if error.kind() == ErrorKind::Interrupted {
                continue;
            }
            if error.kind() != ErrorKind::WouldBlock {
                return std::task::Poll::Ready(Err(error));
            }
            match lock(&this.write_ready).poll(&this.io, &this.socket, cx) {
                std::task::Poll::Ready(Ok(())) => continue,
                std::task::Poll::Ready(Err(error)) => return std::task::Poll::Ready(Err(error)),
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(self.socket.shutdown(Shutdown::Write))
    }
}
