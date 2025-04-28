import numpy as np

from litdata import optimize


# Store random images into the data chunks
def random_images(index):
    # Seed uniquely for each index to ensure reproducibility
    rng = np.random.default_rng(seed=index)
    return {
        "index": index,  # int data type
        "image": rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),  # numpy array
        "class": rng.integers(10), # numpy array
    }


if __name__ == "__main__":
    optimize(
        fn=random_images,  # The function applied over each input.
        inputs=list(range(1000)),  # Provide any inputs. The fn is applied on each item.
        output_dir="my_optimized_dataset",  # The directory where the optimized data are stored.
        num_workers=4,  # The number of workers. The inputs are distributed among them.
        chunk_bytes="64MB",  # The maximum number of bytes to write into a data chunk.
    )
