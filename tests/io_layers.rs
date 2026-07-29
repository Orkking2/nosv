#![cfg(feature = "io-uring")]

use nosv::{
    Runtime, RuntimeBuilder,
    fs::OpenOptions,
    io_uring::raw,
    net::{TcpListener, TcpStream},
};
use std::{
    future::{Future, poll_fn},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::Mutex,
    task::Poll,
};

static NATIVE_IO_LOCK: Mutex<()> = Mutex::new(());

fn temporary_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("nosv-{name}-{}", std::process::id()))
}

#[test]
fn file_owned_buffers_append_and_large_entry_erasure() {
    let _serial = NATIVE_IO_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let path = temporary_path("owned-file");
    let _ = std::fs::remove_file(&path);
    let runtime = RuntimeBuilder::<raw::squeue::Entry128, raw::cqueue::Entry32>::default()
        .build()
        .expect("large-entry runtime");
    let io = runtime.io_handle();

    runtime.block_on(async {
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(true);
        let file = options.open_on(&io, &path).await.expect("open file");

        let (result, buffer) = file.write_all_at(b"hello".to_vec(), 0).await;
        result.expect("positioned write");
        assert_eq!(buffer, b"hello");
        file.sync_data().await.expect("sync file data");

        let (result, buffer) = file.read_exact_at(vec![0; 5], 0).await;
        result.expect("positioned exact read");
        assert_eq!(buffer, b"hello");
        drop(file);

        let mut append = OpenOptions::new();
        append.read(true).append(true);
        let file = append.open_on(&io, &path).await.expect("append open");
        let (result, buffer) = file.write_at(b"bad".to_vec(), 0).await;
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(buffer, b"bad");
        let (result, buffer) = file.append(b"!".to_vec()).await;
        assert_eq!(result.expect("append"), 1);
        assert_eq!(buffer, b"!");
    });

    runtime.shutdown().expect("file runtime shutdown");
    assert_eq!(std::fs::read(&path).expect("read result"), b"hello!");
    std::fs::remove_file(path).expect("remove temporary file");
}

#[test]
fn tcp_owned_round_trip_and_numeric_fallback() {
    let _serial = NATIVE_IO_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = Runtime::new().expect("runtime");
    let io = runtime.io_handle();
    let tasks = runtime.handle();

    runtime.block_on(async {
        let listener = TcpListener::bind_on(&io, (IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("listener address");
        let candidates = [
            SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), address.port()),
            address,
        ];
        let client_io = io.clone();
        let client = tasks
            .spawn(async move {
                let stream = TcpStream::connect_on(&client_io, candidates)
                    .await
                    .expect("fallback connect");
                let (result, sent) = stream.send_all(b"ping".to_vec()).await;
                result.expect("send request");
                assert_eq!(sent, b"ping");
                let (result, response) = stream.recv_exact(vec![0; 4]).await;
                result.expect("receive response");
                response
            })
            .expect("spawn client");

        let (server, peer) = listener.accept().await.expect("accept client");
        assert!(peer.ip().is_loopback());
        let (result, request) = server.recv_exact(Vec::with_capacity(4)).await;
        result.expect("receive request");
        assert_eq!(request, b"ping");
        let (result, response) = server.send_all(b"pong".to_vec()).await;
        result.expect("send response");
        assert_eq!(response, b"pong");
        assert_eq!(client.await.expect("join client"), b"pong");
    });

    runtime.shutdown().expect("TCP runtime shutdown");
}

#[test]
fn dropping_pending_accept_is_drained_by_shutdown() {
    let _serial = NATIVE_IO_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = Runtime::new().expect("runtime");
    let io = runtime.io_handle();

    runtime.block_on(async {
        let listener = TcpListener::bind_on(&io, (Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind listener");
        let mut accept = Box::pin(listener.accept());
        poll_fn(|cx| {
            assert!(matches!(accept.as_mut().poll(cx), Poll::Pending));
            Poll::Ready(())
        })
        .await;
        drop(accept);
        drop(listener);
    });

    runtime.shutdown().expect("cancelled accept drained");
}

#[cfg(feature = "tokio-compat")]
#[test]
fn tokio_extensions_run_on_nosv_runtime() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let _serial = NATIVE_IO_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = Runtime::new().expect("runtime");
    let io = runtime.io_handle();
    let tasks = runtime.handle();

    runtime.block_on(async {
        let listener = TcpListener::bind_on(&io, (Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind listener");
        let address = listener.local_addr().unwrap();
        let client_io = io.clone();
        let client = tasks
            .spawn(async move {
                let stream = TcpStream::connect_on(&client_io, address).await.unwrap();
                let (mut reader, mut writer) = tokio::io::split(stream);
                writer.write_all(b"tokio").await.unwrap();
                writer.shutdown().await.unwrap();
                let mut response = Vec::new();
                reader.read_to_end(&mut response).await.unwrap();
                response
            })
            .unwrap();

        let (server, _) = listener.accept().await.unwrap();
        let (mut reader, mut writer) = tokio::io::split(server);
        assert_eq!(tokio::io::copy(&mut reader, &mut writer).await.unwrap(), 5);
        writer.shutdown().await.unwrap();
        assert_eq!(&client.await.unwrap(), b"tokio");
    });

    runtime
        .shutdown()
        .expect("Tokio-compatible runtime shutdown");
}

#[cfg(feature = "tokio-compat")]
#[test]
fn tokio_copy_bidirectional_bridges_two_nosv_streams() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let _serial = NATIVE_IO_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = Runtime::new().expect("runtime");
    let io = runtime.io_handle();
    let tasks = runtime.handle();

    runtime.block_on(async {
        let left_listener = TcpListener::bind_on(&io, (Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let right_listener = TcpListener::bind_on(&io, (Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let left_address = left_listener.local_addr().unwrap();
        let right_address = right_listener.local_addr().unwrap();

        let left_io = io.clone();
        let left = tasks
            .spawn(async move {
                let mut stream = TcpStream::connect_on(&left_io, left_address).await.unwrap();
                stream.write_all(b"left").await.unwrap();
                AsyncWriteExt::shutdown(&mut stream).await.unwrap();
                let mut received = Vec::new();
                stream.read_to_end(&mut received).await.unwrap();
                received
            })
            .unwrap();
        let right_io = io.clone();
        let right = tasks
            .spawn(async move {
                let mut stream = TcpStream::connect_on(&right_io, right_address)
                    .await
                    .unwrap();
                stream.write_all(b"right").await.unwrap();
                AsyncWriteExt::shutdown(&mut stream).await.unwrap();
                let mut received = Vec::new();
                stream.read_to_end(&mut received).await.unwrap();
                received
            })
            .unwrap();

        let (mut left_server, _) = left_listener.accept().await.unwrap();
        let (mut right_server, _) = right_listener.accept().await.unwrap();
        assert_eq!(
            tokio::io::copy_bidirectional(&mut left_server, &mut right_server)
                .await
                .unwrap(),
            (4, 5),
        );
        assert_eq!(&left.await.unwrap(), b"right");
        assert_eq!(&right.await.unwrap(), b"left");
    });

    runtime.shutdown().expect("bidirectional bridge shutdown");
}

#[cfg(feature = "tokio-compat")]
#[test]
fn tokio_pending_read_leaves_borrowed_buffer_untouched() {
    let _serial = NATIVE_IO_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = Runtime::new().expect("runtime");
    let io = runtime.io_handle();
    let tasks = runtime.handle();

    runtime.block_on(async {
        let listener = TcpListener::bind_on(&io, (Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let client_io = io.clone();
        let client = tasks
            .spawn(async move { TcpStream::connect_on(&client_io, address).await.unwrap() })
            .unwrap();
        let (mut server, _) = listener.accept().await.unwrap();
        let client = client.await.unwrap();

        let mut storage = [0x5a; 8];
        {
            let mut read_buf = tokio::io::ReadBuf::new(&mut storage);
            poll_fn(|cx| {
                assert!(matches!(
                    tokio::io::AsyncRead::poll_read(
                        std::pin::Pin::new(&mut server),
                        cx,
                        &mut read_buf,
                    ),
                    Poll::Pending,
                ));
                Poll::Ready(())
            })
            .await;
            assert!(read_buf.filled().is_empty());
        }
        assert_eq!(storage, [0x5a; 8]);
        drop(client);
        drop(server);
    });

    runtime
        .shutdown()
        .expect("pending readiness cancellation drained");
}
