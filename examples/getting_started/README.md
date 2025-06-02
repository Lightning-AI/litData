## Getting Started

### 1. Prepare Your Data

Convert your raw dataset into **LitData Optimized Streaming Format** using the `optimize` operator.

Here is an example with some random images.

```python
import numpy as np
from litdata import optimize
from PIL import Image


# Store random images into the data chunks
def random_images(index):
    data = {
        "index": index, # int data type
        "image": Image.fromarray(np.random.randint(0, 256, (32, 32, 3), np.uint8)), # PIL image data type
        "class": np.random.randint(10), # numpy array data type
    }
    # The data is serialized into bytes and stored into data chunks by the optimize operator.
    return data # The data is serialized into bytes and stored into data chunks by the optimize operator.

if __name__ == "__main__":
    optimize(
        fn=random_images,  # The function applied over each input.
        inputs=list(range(1000)),  # Provide any inputs. The fn is applied on each item.
        output_dir="my_optimized_dataset",  # The directory where the optimized data are stored.
        num_workers=4,  # The number of workers. The inputs are distributed among them.
        chunk_bytes="64MB"  # The maximum number of bytes to write into a data chunk.
    )

```

The `optimize` operator supports any data structures and types. Serialize whatever you want. The optimized data is stored under the output directory `my_optimized_dataset`.

### 2. Upload your Data to Cloud Storage

Cloud providers such as [AWS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), [Google Cloud](https://cloud.google.com/storage/docs/uploading-objects?hl=en#upload-object-cli), [Azure](https://learn.microsoft.com/en-us/azure/import-export/storage-import-export-data-to-files?tabs=azure-portal-preview) provide command line clients to upload your data to their storage solutions.

Here is how to upload the optimized dataset using the [AWS CLI](https://aws.amazon.com/s3) to [AWS S3](https://aws.amazon.com/s3/).

```bash
⚡ aws s3 cp --recursive my_optimized_dataset s3://my-bucket/my_optimized_dataset
```

### 3. Use StreamingDataset

Then, the Streaming Dataset can read the data directly from [AWS S3](https://aws.amazon.com/s3/).

```python
from litdata import StreamingDataset, StreamingDataLoader

# Remote path where full dataset is stored
input_dir = 's3://my-bucket/my_optimized_dataset'

# Create the Streaming Dataset
dataset = StreamingDataset(input_dir, shuffle=True)

# Access any elements of the dataset
sample = dataset[50]
img = sample['image']
cls = sample['class']

# Create dataLoader and iterate over it to train your AI models.

# Custom collate function to handle the batch (Optional)
def collate_fn(batch):
    return {
        "image": [sample["image"] for sample in batch],
        "class": [sample["class"] for sample in batch],
    }


dataloader = StreamingDataLoader(dataset, collate_fn=collate_fn)
for sample in dataloader:
    img, cls = sample["image"], sample["class"]
```
