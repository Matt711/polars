use std::cmp::Reverse;
use std::path::PathBuf;

use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_io::json::BatchedWriter;
use polars_utils::priority::Priority;

use super::{SinkNode, SinkRecvPort};
use crate::async_executor::spawn;
use crate::nodes::{JoinHandle, MorselSeq, TaskPriority};

type Linearized = Priority<Reverse<MorselSeq>, Vec<u8>>;
pub struct NDJsonSinkNode {
    path: PathBuf,
}
impl NDJsonSinkNode {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl SinkNode for NDJsonSinkNode {
    fn name(&self) -> &str {
        "ndjson_sink"
    }

    fn is_sink_input_parallel(&self) -> bool {
        true
    }

    fn spawn_sink(
        &mut self,
        _num_pipelines: usize,
        recv_ports_recv: SinkRecvPort,
        _state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let (handle, rx_receivers, mut rx_linearizer) =
            recv_ports_recv.parallel_into_linearize::<Linearized>();
        join_handles.push(handle);

        // 16MB
        const DEFAULT_ALLOCATION_SIZE: usize = 1 << 24;

        // Encode task.
        //
        // Task encodes the columns into their corresponding JSON encoding.
        join_handles.extend(rx_receivers.into_iter().map(|mut rx_receiver| {
            spawn(TaskPriority::High, async move {
                // Amortize the allocations over time. If we see that we need to do way larger
                // allocations, we adjust to that over time.
                let mut allocation_size = DEFAULT_ALLOCATION_SIZE;

                while let Ok((outcome, mut rx, mut sender)) = rx_receiver.recv().await {
                    while let Ok(morsel) = rx.recv().await {
                        let (df, seq, _, consume_token) = morsel.into_inner();

                        let mut buffer = Vec::with_capacity(allocation_size);
                        let mut writer = BatchedWriter::new(&mut buffer);

                        writer.write_batch(&df)?;

                        allocation_size = allocation_size.max(buffer.len());
                        sender.insert(Priority(Reverse(seq), buffer)).await.unwrap();
                        drop(consume_token); // Keep the consume_token until here to increase the
                                             // backpressure.
                    }

                    outcome.stopped();
                }

                PolarsResult::Ok(())
            })
        }));

        let path = self.path.clone();

        // IO task.
        //
        // Task that will actually do write to the target file.
        let io_task = polars_io::pl_async::get_runtime().spawn(async move {
            use tokio::fs::OpenOptions;
            use tokio::io::AsyncWriteExt;

            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path.as_path())
                .await
                .map_err(|err| polars_utils::_limit_path_len_io_err(path.as_path(), err))?;

            while let Ok((outcome, mut linearizer)) = rx_linearizer.recv().await {
                while let Some(Priority(_, buffer)) = linearizer.get().await {
                    file.write_all(&buffer).await?;
                }
                outcome.stopped();
            }

            PolarsResult::Ok(())
        });
        join_handles.push(spawn(TaskPriority::Low, async move {
            io_task
                .await
                .unwrap_or_else(|e| Err(std::io::Error::from(e).into()))
        }));
    }
}
