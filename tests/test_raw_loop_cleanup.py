import asyncio

from litdata.raw.dataset import _get_loop_runner


def test_raw_loop_and_executor_are_cleaned_up_after_use():
    runner = _get_loop_runner()
    runner.run(asyncio.to_thread(lambda: None))
    assert runner.is_alive()
