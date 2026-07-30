use nosv::{
    Handle, JoinHandle,
    io::IoHandle,
    net::{TcpListener, TcpStream},
    time,
};
use std::{
    collections::VecDeque,
    io::{self, ErrorKind},
    net::{Ipv4Addr, SocketAddr},
    time::Duration,
};

const CLIENT_MESSAGES: [&str; 4] = [
    "hello nosv",
    "owned buffers",
    "completion-native io",
    "numeric tcp",
];
const FRAME_HEADER_LEN: usize = size_of::<u32>();
const MAX_FRAME_SIZE: usize = 64 * 1024;
const OPERATION_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug)]
struct ClientResult {
    id: usize,
    request: String,
    response: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = nosv::Runtime::new()?;
    let tasks = runtime.handle();
    let io = runtime.io_handle();

    let demo_result = runtime.block_on(run_demo(tasks, io));
    let shutdown_result = runtime.shutdown();

    demo_result?;
    shutdown_result?;
    Ok(())
}

async fn run_demo(tasks: Handle, io: IoHandle) -> io::Result<()> {
    let listener = TcpListener::bind_on(&io, (Ipv4Addr::LOCALHOST, 0)).await?;
    let address = listener.local_addr()?;
    println!("listening on {address}");

    let server_tasks = tasks.clone();
    let server = tasks
        .spawn(serve(listener, server_tasks, CLIENT_MESSAGES.len()))
        .map_err(spawn_error)?;

    let mut clients = VecDeque::with_capacity(CLIENT_MESSAGES.len());
    for (id, request) in CLIENT_MESSAGES.into_iter().enumerate() {
        let client_io = io.clone();
        match tasks.spawn(run_client(id, client_io, address, request)) {
            Ok(client) => clients.push_back(client),
            Err(error) => {
                abort_and_drain(clients).await;
                server.abort();
                let _ = server.await;
                return Err(spawn_error(error));
            }
        }
    }

    let client_results = match join_io_tasks(clients).await {
        Ok(results) => results,
        Err(error) => {
            server.abort();
            let _ = server.await;
            return Err(error);
        }
    };
    let processed = join_io_task(server).await?;

    for result in &client_results {
        println!(
            "client {}: {:?} -> {:?}",
            result.id, result.request, result.response
        );
    }
    println!(
        "verified {} clients and {processed} bytes",
        client_results.len()
    );
    Ok(())
}

async fn serve(listener: TcpListener, tasks: Handle, clients: usize) -> io::Result<usize> {
    let mut connections = VecDeque::with_capacity(clients);

    for _ in 0..clients {
        let (stream, peer) = match timed_accept(&listener).await {
            Ok(connection) => connection,
            Err(error) => {
                abort_and_drain(connections).await;
                return Err(error);
            }
        };

        match tasks.spawn(handle_connection(stream, peer)) {
            Ok(connection) => connections.push_back(connection),
            Err(error) => {
                abort_and_drain(connections).await;
                return Err(spawn_error(error));
            }
        }
    }

    Ok(join_io_tasks(connections).await?.into_iter().sum())
}

async fn handle_connection(stream: TcpStream, peer: SocketAddr) -> io::Result<usize> {
    let mut payload = recv_frame(&stream).await?;
    let bytes = payload.len();
    payload.make_ascii_uppercase();
    send_frame(&stream, payload).await?;
    println!("server: processed {bytes} bytes from {peer}");
    Ok(bytes)
}

async fn run_client(
    id: usize,
    io: IoHandle,
    address: SocketAddr,
    request: &'static str,
) -> io::Result<ClientResult> {
    let stream = timed_connect(&io, address).await?;
    stream.set_nodelay(true)?;
    send_frame(&stream, request.as_bytes().to_vec()).await?;

    let response = recv_frame(&stream).await?;
    let expected = request.to_ascii_uppercase().into_bytes();
    if response != expected {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "client {id} expected {:?}, received {:?}",
                String::from_utf8_lossy(&expected),
                String::from_utf8_lossy(&response),
            ),
        ));
    }

    Ok(ClientResult {
        id,
        request: request.to_owned(),
        response: String::from_utf8(response)
            .map_err(|error| io::Error::new(ErrorKind::InvalidData, error))?,
    })
}

async fn send_frame(stream: &TcpStream, payload: Vec<u8>) -> io::Result<()> {
    let header = encode_frame_len(payload.len())?.to_vec();
    timed_send_all(stream, header).await?;
    timed_send_all(stream, payload).await
}

async fn recv_frame(stream: &TcpStream) -> io::Result<Vec<u8>> {
    let header = timed_recv_exact(stream, Vec::with_capacity(FRAME_HEADER_LEN)).await?;
    let header: [u8; FRAME_HEADER_LEN] = header.try_into().map_err(|header: Vec<u8>| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!("received {} header bytes", header.len()),
        )
    })?;
    let payload_len = decode_frame_len(header)?;
    timed_recv_exact(stream, Vec::with_capacity(payload_len)).await
}

fn encode_frame_len(payload_len: usize) -> io::Result<[u8; FRAME_HEADER_LEN]> {
    if payload_len > MAX_FRAME_SIZE {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!("frame length {payload_len} exceeds the {MAX_FRAME_SIZE}-byte limit"),
        ));
    }
    let payload_len = u32::try_from(payload_len)
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "frame length does not fit u32"))?;
    Ok(payload_len.to_be_bytes())
}

fn decode_frame_len(header: [u8; FRAME_HEADER_LEN]) -> io::Result<usize> {
    let payload_len = u32::from_be_bytes(header) as usize;
    if payload_len > MAX_FRAME_SIZE {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("peer frame length {payload_len} exceeds the {MAX_FRAME_SIZE}-byte limit"),
        ));
    }
    Ok(payload_len)
}

async fn timed_connect(io: &IoHandle, address: SocketAddr) -> io::Result<TcpStream> {
    time::timeout(OPERATION_TIMEOUT, TcpStream::connect_on(io, address))
        .await
        .map_err(|_| timed_out("connect"))?
}

async fn timed_accept(listener: &TcpListener) -> io::Result<(TcpStream, SocketAddr)> {
    time::timeout(OPERATION_TIMEOUT, listener.accept())
        .await
        .map_err(|_| timed_out("accept"))?
}

async fn timed_send_all(stream: &TcpStream, buffer: Vec<u8>) -> io::Result<()> {
    let (result, _buffer) = time::timeout(OPERATION_TIMEOUT, stream.send_all(buffer))
        .await
        .map_err(|_| timed_out("send"))?;
    result
}

async fn timed_recv_exact(stream: &TcpStream, buffer: Vec<u8>) -> io::Result<Vec<u8>> {
    let (result, buffer) = time::timeout(OPERATION_TIMEOUT, stream.recv_exact(buffer))
        .await
        .map_err(|_| timed_out("receive"))?;
    result?;
    Ok(buffer)
}

fn timed_out(operation: &str) -> io::Error {
    io::Error::new(
        ErrorKind::TimedOut,
        format!("{operation} exceeded {OPERATION_TIMEOUT:?}"),
    )
}

fn spawn_error(error: nosv::SpawnError) -> io::Error {
    io::Error::other(error.to_string())
}

fn join_error(error: nosv::JoinError) -> io::Error {
    io::Error::other(error.to_string())
}

async fn join_io_task<T>(task: JoinHandle<io::Result<T>>) -> io::Result<T> {
    task.await.map_err(join_error)?
}

async fn join_io_tasks<T>(mut tasks: VecDeque<JoinHandle<io::Result<T>>>) -> io::Result<Vec<T>> {
    let mut outputs = Vec::with_capacity(tasks.len());
    while let Some(task) = tasks.pop_front() {
        match join_io_task(task).await {
            Ok(output) => outputs.push(output),
            Err(error) => {
                abort_and_drain(tasks).await;
                return Err(error);
            }
        }
    }
    Ok(outputs)
}

async fn abort_and_drain<T>(tasks: VecDeque<JoinHandle<T>>) {
    for task in &tasks {
        task.abort();
    }
    for task in tasks {
        let _ = task.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_lengths_round_trip() {
        for payload_len in [0, 1, 255, MAX_FRAME_SIZE] {
            let header = encode_frame_len(payload_len).expect("valid frame length");
            assert_eq!(
                decode_frame_len(header).expect("valid frame header"),
                payload_len
            );
        }
    }

    #[test]
    fn local_oversized_frame_is_rejected() {
        let error = encode_frame_len(MAX_FRAME_SIZE + 1).expect_err("oversized local frame");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn peer_oversized_frame_is_rejected() {
        let header = u32::try_from(MAX_FRAME_SIZE + 1)
            .expect("test length fits u32")
            .to_be_bytes();
        let error = decode_frame_len(header).expect_err("oversized peer frame");
        assert_eq!(error.kind(), ErrorKind::InvalidData);
    }
}
