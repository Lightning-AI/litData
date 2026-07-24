import itertools

from litdata.streaming.item_loader import Interval
from litdata.utilities.env import _DistributedEnv
from litdata.utilities.shuffle import (
    _associate_chunks_and_intervals_to_workers,
    _get_shared_chunks,
    _group_chunks_by_nodes,
    _intra_node_chunk_shuffle,
)


def test_intra_node_chunk_shuffle():
    chunks_per_workers = [
        [0, 1],  # rank 0, node 0, worker 0
        [2, 3],  # rank 0, node 0, worker 1
        [4, 5],  # rank 1, node 0, worker 0
        [6, 7],  # rank 1, node 0, worker 1
        [8, 9],  # rank 2, node 1, worker 0
        [10, 11],  # rank 2, node 1, worker 1
        [12, 13],  # rank 3, node 1, worker 0
        [14, 15],  # rank 3, node 1, worker 1
    ]

    # Each rank shuffles the chunks the same way
    shuffled_per_rank = [
        _intra_node_chunk_shuffle(
            distributed_env=_DistributedEnv(4, rank, 2),
            num_workers=2,
            chunks_per_workers=chunks_per_workers,
            seed=42,
            current_epoch=0,
        )
        for rank in range(4)
    ]
    expected = [4, 3, 7, 6, 0, 1, 5, 2, 12, 11, 15, 14, 8, 9, 13, 10]
    assert shuffled_per_rank[0] == shuffled_per_rank[1] == shuffled_per_rank[2] == shuffled_per_rank[3] == expected

    # shuffles are different each epoch
    shuffled_per_rank = [
        _intra_node_chunk_shuffle(
            distributed_env=_DistributedEnv(4, 0, 2),
            num_workers=2,
            chunks_per_workers=chunks_per_workers,
            seed=42,
            current_epoch=epoch,
        )
        for epoch in range(4)
    ]
    for i, j in itertools.product(range(4), range(4)):
        # check that the shuffles are different (pairwise comparison)
        if i <= j:
            continue
        assert shuffled_per_rank[i] != shuffled_per_rank[j]


def test_group_chunks_by_nodes():
    # 1 node x 1 processes x 2 workers
    chunks_per_workers = [[0, 1], [2, 3]]
    result = _group_chunks_by_nodes(chunks_per_workers, world_size=1, num_nodes=1, num_workers_per_process=2)
    expected = [[0, 1, 2, 3]]
    assert result == expected

    # 1 node x 2 processes x 2 workers
    chunks_per_workers = [
        [0, 1],  # rank 0, node 0, worker 0
        [2, 3],  # rank 0, node 0, worker 1
        [4, 5],  # rank 1, node 0, worker 0
        [6, 7],  # rank 1, node 0, worker 1
    ]
    result = _group_chunks_by_nodes(chunks_per_workers, world_size=2, num_nodes=1, num_workers_per_process=2)
    expected = [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert result == expected

    # 2 nodes x 2 processes x 2 workers
    chunks_per_workers = [
        [0, 1],  # rank 0, node 0, worker 0
        [2, 3],  # rank 0, node 0, worker 1
        [4, 5],  # rank 1, node 0, worker 0
        [6, 7],  # rank 1, node 0, worker 1
        [8, 9],  # rank 2, node 1, worker 0
        [10, 11],  # rank 2, node 1, worker 1
        [12, 13],  # rank 3, node 1, worker 0
        [14, 15],  # rank 3, node 1, worker 1
    ]
    result = _group_chunks_by_nodes(chunks_per_workers, world_size=4, num_nodes=2, num_workers_per_process=2)
    expected = [
        [0, 1, 2, 3, 4, 5, 6, 7],  # chunks in node 0
        [8, 9, 10, 11, 12, 13, 14, 15],  # chunks in node 1
    ]
    assert result == expected


def test_associate_chunks_and_intervals_to_workers():
    indexes = [0, 1, 2, 3, 4, 5, 6, 7]
    chunk_intervals = [
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
    ]

    workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert workers_chunks == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert workers_intervals == [
        [[0, 0, 50, 50], [0, 0, 50, 50]],
        [[0, 0, 50, 50], [0, 0, 50, 50]],
        [[0, 0, 50, 50], [0, 0, 50, 50]],
        [[0, 0, 50, 50], [0, 0, 50, 50]],
    ]

    chunk_intervals = [
        Interval(0, 0, 50, 50),
        Interval(0, 0, 150, 150),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 12, 12),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 27, 27),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 33, 33),
    ]

    workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert workers_chunks == [[0, 1], [1, 2], [2, 3, 4, 5], [5, 6, 7]]
    assert sum([interval[2] - interval[1] for interval in workers_intervals[0]]) == 105
    assert sum([interval[2] - interval[1] for interval in workers_intervals[1]]) == 105
    assert sum([interval[2] - interval[1] for interval in workers_intervals[2]]) == 105
    assert sum([interval[2] - interval[1] for interval in workers_intervals[3]]) == 105

    assert workers_intervals == [
        [[0, 0, 50, 50], [0, 0, 55, 150]],
        [[0, 55, 150, 150], [0, 0, 10, 50]],
        [[0, 10, 50, 50], [0, 0, 12, 12], [0, 0, 50, 50], [0, 0, 3, 27]],
        [[0, 3, 27, 27], [0, 0, 50, 50], [0, 0, 31, 33]],
    ]

    chunk_intervals = [
        Interval(0, 0, 5, 5),
        Interval(0, 0, 150, 150),
        Interval(0, 0, 7, 7),
        Interval(0, 0, 12, 12),
        Interval(0, 0, 4, 4),
        Interval(0, 0, 27, 27),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 1, 1),
    ]

    workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert workers_chunks == [[0, 1], [1], [1, 2, 3, 4, 5], [5, 6, 7]]
    assert sum([interval[2] - interval[1] for interval in workers_intervals[0]]) == 64
    assert sum([interval[2] - interval[1] for interval in workers_intervals[1]]) == 64
    assert sum([interval[2] - interval[1] for interval in workers_intervals[2]]) == 64
    assert sum([interval[2] - interval[1] for interval in workers_intervals[3]]) == 64
    assert workers_intervals == [
        [[0, 0, 5, 5], [0, 0, 59, 150]],
        [[0, 59, 123, 150]],
        [[0, 123, 150, 150], [0, 0, 7, 7], [0, 0, 12, 12], [0, 0, 4, 4], [0, 0, 14, 27]],
        [[0, 14, 27, 27], [0, 0, 50, 50], [0, 0, 1, 1]],
    ]

    chunk_intervals = [
        Interval(0, 0, 6, 6),
        Interval(0, 0, 6, 6),
        Interval(0, 0, 6, 6),
        Interval(0, 0, 6, 6),
    ]

    workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
        _DistributedEnv(1, 0, 1), range(0, 4), chunk_intervals, False, 8, 6
    )

    assert workers_intervals == [[[0, 0, 6, 6]], [[0, 0, 6, 6]], [[0, 0, 6, 6]], [[0, 0, 6, 6]], [], [], [], []]
    assert workers_chunks == [[0], [1], [2], [3], [], [], [], []]

    workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
        _DistributedEnv(2, 0, 1), range(0, 4), chunk_intervals, False, 8, 6
    )

    assert workers_chunks == [[0], [1], [], [], [], [], [], [], [2], [3], [], [], [], [], [], []]
    assert workers_intervals == [
        [[0, 0, 6, 6]],
        [[0, 0, 6, 6]],
        [],
        [],
        [],
        [],
        [],
        [],
        [[0, 0, 6, 6]],
        [[0, 0, 6, 6]],
        [],
        [],
        [],
        [],
        [],
        [],
    ]

    workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
        _DistributedEnv(1, 0, 1), range(0, 4), chunk_intervals, False, 2, 8
    )
    assert workers_chunks == [[0, 1, 2], [2, 3]]
    assert workers_intervals == [[[0, 0, 6, 6], [0, 0, 6, 6], [0, 0, 4, 6]], [[0, 4, 6, 6], [0, 0, 6, 6]]]

    chunk_intervals = [
        Interval(0, 0, 6, 6),
        Interval(0, 0, 7, 7),
        Interval(0, 0, 6, 6),
        Interval(0, 0, 7, 8),
    ]

    workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
        _DistributedEnv(2, 0, 1), range(0, 4), chunk_intervals, False, 8, 6
    )

    assert sum([y[2] - y[1] for x in workers_intervals for y in x]) == 26
    assert workers_chunks == [[0], [1], [], [], [], [], [], [], [1, 2], [2, 3], [3], [], [], [], [], []]
    assert workers_intervals == [
        [[0, 0, 6, 6]],
        [[0, 0, 6, 7]],
        [],
        [],
        [],
        [],
        [],
        [],
        [[0, 6, 7, 7], [0, 0, 5, 6]],
        [[0, 5, 6, 6], [0, 0, 5, 8]],
        [[0, 5, 7, 8]],
        [],
        [],
        [],
        [],
        [],
    ]

    workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
        _DistributedEnv(2, 0, 1), range(0, 4), chunk_intervals, True, 8, 6
    )

    assert sum([y[2] - y[1] for x in workers_intervals for y in x]) == 24
    assert workers_chunks == [[0], [1], [], [], [], [], [], [], [1, 2], [2, 3], [], [], [], [], [], []]
    assert workers_intervals == [
        [[0, 0, 6, 6]],
        [[0, 0, 6, 7]],
        [],
        [],
        [],
        [],
        [],
        [],
        [[0, 6, 7, 7], [0, 0, 5, 6]],
        [[0, 5, 6, 6], [0, 0, 5, 8]],
        [],
        [],
        [],
        [],
        [],
        [],
    ]


def test_get_shared_chunks():
    assert _get_shared_chunks([]) == {}
    assert _get_shared_chunks([[0]]) == {}
    assert _get_shared_chunks([[0], [1]]) == {}
    assert _get_shared_chunks([[0], [0, 1]]) == {0: [0, 1]}  # chunk 0 is shared by worker 0 and 1
    assert _get_shared_chunks([[0, 1], [1]]) == {1: [0, 1]}  # chunk 1 is shared by worker 0 and 1
    assert _get_shared_chunks([[2], [0, 1], [2, 3]]) == {2: [0, 2]}
    assert _get_shared_chunks([[2], [0, 1], [2, 3], [1, 4], [1]]) == {1: [1, 3, 4], 2: [0, 2]}
