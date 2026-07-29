#![cfg(feature = "io-uring")]

use nosv::{Runtime, RuntimeBuilder, io_uring::raw};
use std::sync::Mutex;

static NATIVE_IO_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn raw_nop_restores_user_data_from_block_on() {
    let _serial = NATIVE_IO_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = Runtime::new().expect("runtime and kernel ring");
    let io = runtime.io_uring_handle();
    runtime.block_on(async move {
        let entry = raw::opcode::Nop::new().build().user_data(0xfeed_beef);
        // SAFETY: NOP references no external kernel-visible resources.
        let mut completions = unsafe { io.submit_entries([entry]) }
            .await
            .expect("NOP admitted");
        let completion = completions
            .next()
            .await
            .expect("one NOP CQE")
            .expect("completion buffer did not overflow");
        assert_eq!(completion.index, 0);
        assert_eq!(completion.cqe.user_data(), 0xfeed_beef);
        assert_eq!(completion.cqe.result(), 0);
        assert!(completions.next().await.is_none());
    });
    runtime.shutdown().expect("runtime shutdown");
}

#[test]
fn concurrent_spawned_bursts_cross_ring_depth_and_reap_size() {
    let _serial = NATIVE_IO_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = Runtime::builder()
        .io_uring_config(nosv::io_uring::IoUringConfig {
            entries: 8,
            reap_size: 2,
            max_buffered_completions: 128,
            ..nosv::io_uring::IoUringConfig::default()
        })
        .build()
        .expect("small ring runtime");
    let tasks = runtime.handle();
    let first_io = runtime.io_uring_handle();
    let second_io = first_io.clone();
    let first = tasks
        .spawn(async move {
            let entries = (0..40).map(|value| raw::opcode::Nop::new().build().user_data(value));
            // SAFETY: NOP entries reference no external resources.
            let mut stream = unsafe { first_io.submit_entries(entries) }.await.unwrap();
            let mut seen = Vec::new();
            while let Some(completion) = stream.next().await {
                seen.push(completion.unwrap().index);
            }
            seen
        })
        .expect("first spawned submitter");
    let second = tasks
        .spawn(async move {
            let entries =
                (0..40).map(|value| raw::opcode::Nop::new().build().user_data(1000 + value));
            // SAFETY: NOP entries reference no external resources.
            let mut stream = unsafe { second_io.submit_entries(entries) }.await.unwrap();
            let mut seen = Vec::new();
            while let Some(completion) = stream.next().await {
                seen.push(completion.unwrap().index);
            }
            seen
        })
        .expect("second spawned submitter");
    let (first, second) =
        runtime.block_on(async move { (first.await.unwrap(), second.await.unwrap()) });
    assert_eq!(first.len(), 40);
    assert_eq!(second.len(), 40);
    assert_eq!(first.iter().copied().max(), Some(39));
    assert_eq!(second.iter().copied().max(), Some(39));
    runtime.shutdown().expect("burst runtime shutdown");
}

#[test]
fn explicit_async_cancel_reports_and_drains_original() {
    let _serial = NATIVE_IO_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = Runtime::new().expect("runtime and kernel ring");
    let io = runtime.io_uring_handle();
    runtime.block_on(async move {
        let timeout = Box::new(raw::types::Timespec::new().sec(60));
        let entry = raw::opcode::Timeout::new(timeout.as_ref())
            .build()
            .user_data(0xcafe);
        // SAFETY: the timeout allocation remains live through wait_drained.
        let completions = unsafe { io.submit_entries([entry]) }
            .await
            .expect("timeout admitted");
        let mut cancellation = completions.cancel();
        let cancel = cancellation
            .next()
            .await
            .expect("one AsyncCancel CQE")
            .expect("cancellation buffer did not overflow");
        assert_eq!(cancel.index, 0);
        assert_eq!(cancel.cqe.user_data(), 0xcafe);
        let cancel_result = cancel.cqe.result();
        assert!(
            cancel_result == 0
                || cancel_result == -libc::ENOENT
                || cancel_result == -libc::EALREADY
        );
        assert!(cancellation.next().await.is_none());
        cancellation.wait_drained().await;
        drop(timeout);
    });
    runtime
        .shutdown()
        .expect("runtime shutdown after cancellation");
}

#[test]
fn dropped_stream_is_cancelled_and_drained_by_shutdown() {
    let _serial = NATIVE_IO_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = Runtime::new().expect("runtime and kernel ring");
    let io = runtime.io_uring_handle();
    let timeout = Box::new(raw::types::Timespec::new().sec(60));
    let entry = raw::opcode::Timeout::new(timeout.as_ref())
        .build()
        .user_data(0xdead);
    runtime.block_on(async move {
        // SAFETY: the outer timeout allocation remains live until shutdown completes.
        let stream = unsafe { io.submit_entries([entry]) }
            .await
            .expect("timeout admitted");
        drop(stream);
    });
    runtime.shutdown().expect("shutdown drains dropped stream");
    drop(timeout);
}

#[test]
fn large_entries_use_the_same_raw_submission_api() {
    let _serial = NATIVE_IO_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = RuntimeBuilder::<raw::squeue::Entry128, raw::cqueue::Entry32>::default()
        .build()
        .expect("large-entry ring");
    let io = runtime.io_uring_handle();
    runtime.block_on(async move {
        let entry =
            raw::squeue::Entry128::from(raw::opcode::Nop::new().build().user_data(0x1234_5678));
        // SAFETY: NOP references no external kernel-visible resources.
        let mut completions = unsafe { io.submit_entries([entry]) }
            .await
            .expect("large NOP admitted");
        let completion = completions
            .next()
            .await
            .expect("one large CQE")
            .expect("completion buffer did not overflow");
        assert_eq!(completion.index, 0);
        assert_eq!(completion.cqe.user_data(), 0x1234_5678);
        assert_eq!(raw::cqueue::Entry::from(completion.cqe).result(), 0);
        assert!(completions.next().await.is_none());
    });
    runtime.shutdown().expect("large runtime shutdown");
}
