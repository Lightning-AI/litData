import asyncio

from litdata.raw.dataset import _get_loop_runner


def test_raw_loop_is_cleaned_up_after_use():
    runner = _get_loop_runner()
    runner.run(asyncio.sleep(0))
    assert runner.is_alive()
