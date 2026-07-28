use nosv::{RuntimeBuilder, task::yield_now};
use std::error::Error;

async fn fut(input: u64) -> Result<u64, Box<dyn Error>> {
    yield_now().await;

    Ok(input * 2)
}

fn main() -> Result<(), Box<dyn Error>> {
    let runtime = RuntimeBuilder::new().build()?;

    let input = 5;
    let output = runtime.block_on(fut(input));

    match output {
        Ok(val) => println!("input ({input}) produced value ({val})"),
        Err(err) => println!("input ({input}) produced error ({err})"),
    }

    Ok(())
}
