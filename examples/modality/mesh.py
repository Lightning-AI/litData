"""Mesh: encode a trimesh object, stream a Trimesh.

Needs trimesh. Path and raw bytes are stored as-is; mesh= encodes.
"""

import trimesh

from litdata import Mesh, StreamingDataset, optimize


def make_sample(index: int) -> dict:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return {
        "index": index,
        # Mesh(path="m.glb")
        "mesh": Mesh(mesh=mesh, file_type="glb"),
    }


if __name__ == "__main__":
    optimize(
        fn=make_sample,
        inputs=list(range(4)),
        output_dir="example_optimize_dataset/mesh",
        num_workers=2,
        chunk_bytes="64MB",
    )

    sample = StreamingDataset("example_optimize_dataset/mesh")[0]
    mesh = sample["mesh"]  # Trimesh
    print(mesh.vertices.shape, mesh.faces.shape)
