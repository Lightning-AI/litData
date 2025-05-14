import pytest

from litdata.processing.item_provider import WorkerItemProvider  # replace with actual module name


def test_set_and_get_items_int_index():
    provider = WorkerItemProvider([[1, 2], [3, 4]], num_downloaders=2)
    provider.set_items(0, [10, 20])
    assert provider.get_items(0) == [10, 20]


def test_set_and_get_items_tuple_index():
    provider = WorkerItemProvider([[1, 2], [3, 4]], num_downloaders=2)
    provider.set_items((1, 0), [30])
    assert provider.get_items((1, 0)) == [30]
    provider.set_items(1, [10, 20])
    assert provider.get_items(1) == [10, 20]


def test_invalid_index_type():
    provider = WorkerItemProvider([[1, 2], [3, 4]], num_downloaders=1)
    with pytest.raises(ValueError, match="Invalid index type"):
        provider.set_items("invalid", [0])
    with pytest.raises(ValueError, match="Invalid index type"):
        provider.get_items("invalid")


@pytest.mark.parametrize("use_shared", [True, False])
def test_prepare_ready_to_use_queue(use_shared):
    provider = WorkerItemProvider([[1, 2, 3], [4, 5, 6]], num_downloaders=2)
    provider.prepare_ready_to_use_queue(use_shared, worker_index=0)
    provider.prepare_ready_to_use_queue(use_shared, worker_index=1)

    for worker_index in range(2):
        queue = provider.ready_to_process_shared_queue if use_shared else provider.ready_to_process_item[worker_index]
        contents = []
        none_count = 0
        while none_count < 2:
            item = queue.get_nowait()

            if item is None:
                none_count += 1
            else:
                contents.append(item)

        # should have 3 items + 2 sentinels (None)
        assert len(contents) == 3
        assert all(isinstance(i, tuple) for i in contents)
