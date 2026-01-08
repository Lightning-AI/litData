from litdata.utilities.env import _DistributedEnv


def test_distributed_env_from_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("GLOBAL_RANK", "1")
    monkeypatch.setenv("NNODES", "2")

    dist_env = _DistributedEnv.detect()
    assert dist_env.world_size == 2
    assert dist_env.global_rank == 1
    assert dist_env.num_nodes == 2
