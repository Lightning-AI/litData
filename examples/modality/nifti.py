"""Nifti: encode a volume, stream a nibabel image.

Needs nibabel.
"""

import numpy as np

from litdata import Nifti, StreamingDataset, optimize


def make_sample(index: int) -> dict:
    vol = np.random.randn(16, 16, 16).astype(np.float32)
    return {
        "index": index,
        # Nifti(path="v.nii.gz")
        "volume": Nifti(array=vol, affine=np.eye(4)),
    }


if __name__ == "__main__":
    optimize(
        fn=make_sample,
        inputs=list(range(4)),
        output_dir="example_optimize_dataset/nifti",
        num_workers=2,
        chunk_bytes="64MB",
    )

    sample = StreamingDataset("example_optimize_dataset/nifti")[0]
    nifti = sample["volume"]  # Nibabel
    volume = nifti.get_fdata()
    print(volume.shape, nifti.affine.shape)
