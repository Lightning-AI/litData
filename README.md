<div align="center">
<img src="https://pl-flash-data.s3.amazonaws.com/lit_data_logo.webp" alt="LitData" width="800px"/>

&nbsp;
&nbsp;

**Transform datasets at scale.    
Optimize data for fast AI model training.**


<pre>
Transform                              Optimize
  
✅ Parallelize data processing       ✅ Stream large cloud datasets          
✅ Create vector embeddings          ✅ Accelerate training by 20x           
✅ Run distributed inference         ✅ Pause and resume data streaming      
✅ Scrape websites at scale          ✅ Use remote data without local loading
</pre>

---

![PyPI](https://img.shields.io/pypi/v/litdata)
![Downloads](https://img.shields.io/pypi/dm/litdata)
![License](https://img.shields.io/github/license/Lightning-AI/litdata)
[![Discord](https://img.shields.io/discord/1077906959069626439?label=Get%20Help%20on%20Discord)](https://discord.gg/VptPCZkGNa)

<p align="center">
  <a href="https://lightning.ai/">Lightning AI</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#speed-up-model-training">Optimize data</a> •
  <a href="#transform-datasets">Transform data</a> •
  <a href="#key-features">Features</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#start-from-a-template">Templates</a> •
  <a href="#community">Community</a>
</p>

&nbsp;

<a target="_blank" href="https://lightning.ai/docs/overview/optimize-data/optimize-datasets">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>

</div>

&nbsp;

# Transform data at scale. Optimize for fast model training.
LitData scales [data processing tasks](#transform-datasets) (data scraping, image resizing, distributed inference, embedding creation) on local or cloud machines. It also enables [optimizing datasets](#speed-up-model-training) to accelerate AI model training and work with large remote datasets without local loading.

&nbsp;

# Quick start
First, install LitData:

```bash
pip install litdata
```

Choose your workflow:

🚀 [Speed up model training](#speed-up-model-training)    
🚀 [Transform datasets](#transform-datasets)

&nbsp;

<details>
  <summary>Advanced install</summary>

Install all the extras
```bash
pip install 'litdata[extras]'
```

</details>

&nbsp;

----

# Speed up model training
Stream datasets directly from cloud storage without local downloads. Choose the approach that fits your workflow:

## Option 1: Start immediately with existing data ⚡⚡
Stream raw files directly from cloud storage - no pre-optimization needed.

```python
from litdata import StreamingRawDataset
from torch.utils.data import DataLoader

# Point to your existing cloud data
dataset = StreamingRawDataset("s3://my-bucket/raw-data/")
dataloader = DataLoader(dataset, batch_size=32)

for batch in dataloader:
    # Process raw bytes on-the-fly
    pass
```

**Key benefits:**

✅ **Instant access:**         Start streaming immediately without preprocessing.    
✅ **Zero setup time:**        No data conversion or optimization required.    
✅ **Native format:**          Work with original file formats (images, text, etc.).    
✅ **Flexible processing:**    Apply transformations on-the-fly during streaming.    
✅ **Cloud-native:**           Stream directly from S3, GCS, or Azure storage.    

## Option 2: Optimize for maximum performance ⚡⚡⚡  
Accelerate model training (20x faster) by optimizing datasets for streaming directly from cloud storage. Work with remote data without local downloads with features like loading data subsets, accessing individual samples, and resumable streaming.

**Step 1: Optimize your data (one-time setup)**

Transform raw data into optimized chunks for maximum streaming speed.
This step formats the dataset for fast loading by writing data in an efficient chunked binary format.

```python
import numpy as np
from PIL import Image
import litdata as ld

def random_images(index):
    # Replace with your actual image loading here (e.g., .jpg, .png, etc.)
    # Recommended: use compressed formats like JPEG for better storage and optimized streaming speed
    # You can also apply resizing or reduce image quality to further increase streaming speed and save space
    fake_images = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    fake_labels = np.random.randint(10)

    # You can use any key:value pairs. Note that their types must not change between samples, and Python lists must
    # always contain the same number of elements with the same types
    data = {"index": index, "image": fake_images, "class": fake_labels}

    return data

if __name__ == "__main__":
    # The optimize function writes data in an optimized format
    ld.optimize(
        fn=random_images,                   # the function applied to each input
        inputs=list(range(1000)),           # the inputs to the function (here it's a list of numbers)
        output_dir="fast_data",             # optimized data is stored here
        num_workers=4,                      # the number of workers on the same machine
        chunk_bytes="64MB"                  # size of each chunk
    )
```

**Step 2: Put the data on the cloud**

Upload the data to a [Lightning Studio](https://lightning.ai) (backed by S3) or your own S3 bucket:
```bash
aws s3 cp --recursive fast_data s3://my-bucket/fast_data
```

**Step 3: Stream the data during training**

Load the data by replacing the PyTorch Dataset and DataLoader with the StreamingDataset and StreamingDataLoader.

```python
import litdata as ld

dataset = ld.StreamingDataset('s3://my-bucket/fast_data', shuffle=True, drop_last=True)

# Custom collate function to handle the batch (optional)
def collate_fn(batch):
    return {
        "image": [sample["image"] for sample in batch],
        "class": [sample["class"] for sample in batch],
    }


dataloader = ld.StreamingDataLoader(dataset, collate_fn=collate_fn)
for sample in dataloader:
    img, cls = sample["image"], sample["class"]
```

**Key benefits:**

✅ **Accelerate training:**       Optimized datasets load 20x faster.      
✅ **Stream cloud datasets:**     Work with cloud data without downloading it.    
✅ **PyTorch-first:**             Works with PyTorch libraries like PyTorch Lightning, Lightning Fabric, Hugging Face.    
✅ **Easy collaboration:**        Share and access datasets in the cloud, streamlining team projects.     
✅ **Scale across GPUs:**         Streamed data automatically scales to all GPUs.      
✅ **Flexible storage:**          Use S3, GCS, Azure, or your own cloud account for data storage.    
✅ **Compression:**               Reduce your data footprint by using advanced compression algorithms.  
✅ **Run local or cloud:**        Run on your own machines or auto-scale to 1000s of cloud GPUs with Lightning Studios.         
✅ **Enterprise security:**       Self host or process data on your cloud account with Lightning Studios.  

&nbsp;

----

# Transform datasets
Accelerate data processing tasks (data scraping, image resizing, embedding creation, distributed inference) by parallelizing (map) the work across many machines at once.

Here's an example that resizes and crops a large image dataset:

```python
from PIL import Image
import litdata as ld

# use a local or S3 folder
input_dir = "my_large_images"     # or "s3://my-bucket/my_large_images"
output_dir = "my_resized_images"  # or "s3://my-bucket/my_resized_images"

inputs = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

# resize the input image
def resize_image(image_path, output_dir):
  output_image_path = os.path.join(output_dir, os.path.basename(image_path))
  Image.open(image_path).resize((224, 224)).save(output_image_path)

ld.map(
    fn=resize_image,
    inputs=inputs,
    output_dir="output_dir",
)
```

**Key benefits:**

✅ Parallelize processing:    Reduce processing time by transforming data across multiple machines simultaneously.    
✅ Scale to large data:       Increase the size of datasets you can efficiently handle.    
✅ Flexible usecases:         Resize images, create embeddings, scrape the internet, etc...    
✅ Run local or cloud:        Run on your own machines or auto-scale to 1000s of cloud GPUs with Lightning Studios.         
✅ Enterprise security:       Self host or process data on your cloud account with Lightning Studios.  

&nbsp;

----

# Key Features

## Features for optimizing and streaming datasets for model training

<details>
  <summary>✅ Stream raw datasets from cloud storage (beta)</summary>
  &nbsp;

Effortlessly stream raw files (images, text, etc.) directly from S3, GCS, and Azure cloud storage without any optimization or conversion. Ideal for workflows requiring instant access to original data in its native format.

**Prerequisites:**

Install the required dependencies to stream raw datasets from cloud storage like **Amazon S3** or **Google Cloud Storage**:

```bash
# for aws s3
pip install "litdata[extra]" s3fs

# for gcloud storage
pip install "litdata[extra]" gcsfs
```

**Usage Example:**
```python
from torch.utils.data import DataLoader
from litdata import StreamingRawDataset

dataset = StreamingRawDataset("s3://bucket/files/")

# Use with PyTorch DataLoader
loader = DataLoader(dataset, batch_size=32)
for batch in loader:
    # Each item is raw bytes
    pass
```

> Use `StreamingRawDataset` to stream your data as-is. Use `StreamingDataset` for fastest streaming after optimizing your data.


You can also customize how files are grouped by subclassing `StreamingRawDataset` and overriding the `setup` method. This is useful for pairing related files (e.g., image and mask, audio and transcript) or any custom grouping logic.

```python
from typing import Union
from torch.utils.data import DataLoader
from litdata import StreamingRawDataset
from litdata.raw.indexer import FileMetadata

class SegmentationRawDataset(StreamingRawDataset):
    def setup(self, files: list[FileMetadata]) -> Union[list[FileMetadata], list[list[FileMetadata]]]:
        # TODO: Implement your custom grouping logic here.
        # For example, group files by prefix, extension, or any rule you need.
        # Return a list of groups, where each group is a list of FileMetadata.
        # Example:
        #   return [[image, mask], ...]
        pass

# Initialize the custom dataset
dataset = SegmentationRawDataset("s3://bucket/files/")
loader = DataLoader(dataset, batch_size=32)
for item in loader:
    # Each item in the batch is a pair: [image_bytes, mask_bytes]
    pass
```

**Smart Index Caching**

`StreamingRawDataset` automatically caches the file index for instant startup. Initial scan, builds and caches the index, then subsequent runs load instantly.

**Two-Level Cache:**
- **Local:** Stored in your cache directory for instant access
- **Remote:** Automatically saved to cloud storage (e.g., `s3://bucket/files/index.json.zstd`) for reuse

**Force Rebuild:**
```python
# When dataset files have changed
dataset = StreamingRawDataset("s3://bucket/files/", recompute_index=True)
```

</details>

<details>
  <summary> ✅ Stream large cloud datasets</summary>
&nbsp;

Use data stored on the cloud without needing to download it all to your computer, saving time and space.

Imagine you're working on a project with a huge amount of data stored online. Instead of waiting hours to download it all, you can start working with the data almost immediately by streaming it.

Once you've optimized the dataset with LitData, stream it as follows:
```python
from litdata import StreamingDataset, StreamingDataLoader

dataset = StreamingDataset('s3://my-bucket/my-data', shuffle=True)
dataloader = StreamingDataLoader(dataset, batch_size=64)

for batch in dataloader:
    process(batch)  # Replace with your data processing logic

```


Additionally, you can inject client connection settings for [S3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html#boto3.session.Session.client) or GCP when initializing your dataset. This is useful for specifying custom endpoints and credentials per dataset.

```python
from litdata import StreamingDataset

# boto3 compatible storage options for a custom S3-compatible endpoint
storage_options = {
    "endpoint_url": "your_endpoint_url",
    "aws_access_key_id": "your_access_key_id",
    "aws_secret_access_key": "your_secret_access_key",
}

dataset = StreamingDataset('s3://my-bucket/my-data', storage_options=storage_options)

# s5cmd compatible storage options for a custom S3-compatible endpoint
# Note: If s5cmd is installed, it will be used by default for S3 operations. If you prefer not to use s5cmd, you can disable it by setting the environment variable: `DISABLE_S5CMD=1`
storage_options = {
    "AWS_ACCESS_KEY_ID": "your_access_key_id",
    "AWS_SECRET_ACCESS_KEY": "your_secret_access_key",
    "S3_ENDPOINT_URL": "your_endpoint_url",  # Required only for custom endpoints
}


dataset = StreamingDataset('s3://my-bucket/my-data', storage_options=storage_options)
```

Alternative: Using `s5cmd` for S3 Operations


Also, you can specify a custom cache directory when initializing your dataset. This is useful when you want to store the cache in a specific location.
```python
from litdata import StreamingDataset

# Initialize the StreamingDataset with the custom cache directory
dataset = StreamingDataset('s3://my-bucket/my-data', cache_dir="/path/to/cache")
```

</details>

<details>
  <summary> ✅ Stream Hugging Face 🤗 datasets</summary>

&nbsp;

To use your favorite  Hugging Face dataset with LitData, simply pass its URL to `StreamingDataset`.

<details>
  <summary>How to get HF dataset URI?</summary>

https://github.com/user-attachments/assets/3ba9e2ef-bf6b-41fc-a578-e4b4113a0e72

</details>

**Prerequisites:**

Install the required dependencies to stream Hugging Face datasets:
```sh
pip install "litdata[extra]" huggingface_hub

# Optional: To speed up downloads on high-bandwidth networks
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**Stream Hugging Face dataset:**

```python
import litdata as ld

# Define the Hugging Face dataset URI
hf_dataset_uri = "hf://datasets/leonardPKU/clevr_cogen_a_train/data"

# Create a streaming dataset
dataset = ld.StreamingDataset(hf_dataset_uri)

# Print the first sample
print("Sample", dataset[0])

# Stream the dataset using StreamingDataLoader
dataloader = ld.StreamingDataLoader(dataset, batch_size=4)
for sample in dataloader:
    pass 
```

You don’t need to worry about indexing the dataset or any other setup. **LitData** will **handle all the necessary steps automatically** and `cache` the `index.json` file, so you won't have to index it again.

This ensures that the next time you stream the dataset, the indexing step is skipped..

&nbsp;

### Indexing the HF dataset (Optional)

If the Hugging Face dataset hasn't been indexed yet, you can index it first using the `index_hf_dataset` method, and then stream it using the code above.

```python
import litdata as ld

hf_dataset_uri = "hf://datasets/leonardPKU/clevr_cogen_a_train/data"

ld.index_hf_dataset(hf_dataset_uri)
```

- Indexing the Hugging Face dataset ahead of time will make streaming abit faster, as it avoids the need for real-time indexing during streaming.

- To use `HF gated dataset`, ensure the `HF_TOKEN` environment variable is set.

**Note**: For HuggingFace datasets, `indexing` & `streaming` is supported only for datasets in **`Parquet format`**.

&nbsp;

### Full Workflow for Hugging Face Datasets

For full control over the cache path(`where index.json file will be stored`) and other configurations, follow these steps:

1. Index the Hugging Face dataset first:

```python
import litdata as ld

hf_dataset_uri = "hf://datasets/open-thoughts/OpenThoughts-114k/data"

ld.index_parquet_dataset(hf_dataset_uri, "hf-index-dir")
```

2. To stream HF datasets now, pass the `HF dataset URI`, the path where the `index.json` file is stored, and `ParquetLoader` as the `item_loader` to the **`StreamingDataset`**:

```python
import litdata as ld
from litdata.streaming.item_loader import ParquetLoader

hf_dataset_uri = "hf://datasets/open-thoughts/OpenThoughts-114k/data"

dataset = ld.StreamingDataset(hf_dataset_uri, item_loader=ParquetLoader(), index_path="hf-index-dir")

for batch in ld.StreamingDataLoader(dataset, batch_size=4):
  pass
```

&nbsp;

### LitData `Optimize` v/s `Parquet`
<!-- TODO: Update benchmark -->
Below is the benchmark for the `Imagenet dataset (155 GB)`, demonstrating that **`optimizing the dataset using LitData is faster and results in smaller output size compared to raw Parquet files`**.

| **Operation**                    | **Size (GB)** | **Time (seconds)** | **Throughput (images/sec)** |
|-----------------------------------|---------------|---------------------|-----------------------------|
| LitData Optimize Dataset          | 45            | 283.17             | 4000-4700                  |
| Parquet Optimize Dataset          | 51            | 465.96             | 3600-3900                  |
| Index Parquet Dataset (overhead)  | N/A           | 6                  | N/A                         |

</details>

<details>
  <summary> ✅ Streams on multi-GPU, multi-node</summary>

&nbsp;

Data optimized and loaded with Lightning automatically streams efficiently in distributed training across GPUs or multi-node.

The `StreamingDataset` and `StreamingDataLoader` automatically make sure each rank receives the same quantity of varied batches of data, so it works out of the box with your favorite frameworks ([PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), [Lightning Fabric](https://lightning.ai/docs/fabric/stable/), or [PyTorch](https://pytorch.org/docs/stable/index.html)) to do distributed training.

Here you can see an illustration showing how the Streaming Dataset works with multi node / multi gpu under the hood.

```python
from litdata import StreamingDataset, StreamingDataLoader

# For the training dataset, don't forget to enable shuffle and drop_last !!! 
train_dataset = StreamingDataset('s3://my-bucket/my-train-data', shuffle=True, drop_last=True)
train_dataloader = StreamingDataLoader(train_dataset, batch_size=64)

for batch in train_dataloader:
    process(batch)  # Replace with your data processing logic

val_dataset = StreamingDataset('s3://my-bucket/my-val-data', shuffle=False, drop_last=False)
val_dataloader = StreamingDataLoader(val_dataset, batch_size=64)

for batch in val_dataloader:
    process(batch)  # Replace with your data processing logic
```

![An illustration showing how the Streaming Dataset works with multi node.](https://pl-flash-data.s3.amazonaws.com/streaming_dataset.gif)

</details>

<details>
  <summary> ✅ Stream from multiple cloud providers</summary>

&nbsp;

The `StreamingDataset` provides support for reading optimized datasets from common cloud storage providers like AWS S3, Google Cloud Storage (GCS), and Azure Blob Storage. Below are examples of how to use StreamingDataset with each cloud provider.

```python
import os
import litdata as ld

# Read data from AWS S3 using boto3
aws_storage_options={
    "aws_access_key_id": os.environ['AWS_ACCESS_KEY_ID'],
    "aws_secret_access_key": os.environ['AWS_SECRET_ACCESS_KEY'],
}
# You can also pass the session options. (for boto3 only)
aws_session_options = {
  "profile_name": os.environ['AWS_PROFILE_NAME'],  # Required only for custom profiles
  "region_name": os.environ['AWS_REGION_NAME'],    # Required only for custom regions
}
dataset = ld.StreamingDataset("s3://my-bucket/my-data", storage_options=aws_storage_options, session_options=aws_session_options)

# Read Data from AWS S3 with Unsigned Request using boto3
aws_storage_options={
  "config": botocore.config.Config(
        retries={"max_attempts": 1000, "mode": "adaptive"}, # Configure retries for S3 operations
        signature_version=botocore.UNSIGNED, # Use unsigned requests
  )
}
dataset = ld.StreamingDataset("s3://my-bucket/my-data", storage_options=aws_storage_options)


# Read data from AWS S3 using s5cmd
# Note: If s5cmd is installed, it will be used by default for S3 operations. If you prefer not to use s5cmd, you can disable it by setting the environment variable: `DISABLE_S5CMD=1`
aws_storage_options={
    "AWS_ACCESS_KEY_ID": os.environ['AWS_ACCESS_KEY_ID'],
    "AWS_SECRET_ACCESS_KEY": os.environ['AWS_SECRET_ACCESS_KEY'],
    "S3_ENDPOINT_URL": os.environ['AWS_ENDPOINT_URL'],  # Required only for custom endpoints
}
dataset = ld.StreamingDataset("s3://my-bucket/my-data", storage_options=aws_storage_options)

# Read Data from AWS S3 with Unsigned Request using s5cmd
aws_storage_options={
  "AWS_NO_SIGN_REQUEST": "Yes" # Required for unsigned requests
  "S3_ENDPOINT_URL": os.environ['AWS_ENDPOINT_URL'],  # Required only for custom endpoints
}
dataset = ld.StreamingDataset("s3://my-bucket/my-data", storage_options=aws_storage_options)


# Read data from GCS
gcp_storage_options={
    "project": os.environ['PROJECT_ID'],
}
dataset = ld.StreamingDataset("gs://my-bucket/my-data", storage_options=gcp_storage_options)

# Read data from Azure
azure_storage_options={
    "account_url": f"https://{os.environ['AZURE_ACCOUNT_NAME']}.blob.core.windows.net",
    "credential": os.environ['AZURE_ACCOUNT_ACCESS_KEY']
}
dataset = ld.StreamingDataset("azure://my-bucket/my-data", storage_options=azure_storage_options)
```

</details>  

<details>
  <summary> ✅ Pause, resume data streaming</summary>
&nbsp;

Stream data during long training, if interrupted, pick up right where you left off without any issues.

LitData provides a stateful `Streaming DataLoader` e.g. you can `pause` and `resume` your training whenever you want.

Info: The `Streaming DataLoader` was used by [Lit-GPT](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/pretrain_tinyllama.md) to pretrain LLMs. Restarting from an older checkpoint was critical to get to pretrain the full model due to several failures (network, CUDA Errors, etc..).

```python
import os
import torch
from litdata import StreamingDataset, StreamingDataLoader

dataset = StreamingDataset("s3://my-bucket/my-data", shuffle=True)
dataloader = StreamingDataLoader(dataset, num_workers=os.cpu_count(), batch_size=64)

# Restore the dataLoader state if it exists
if os.path.isfile("dataloader_state.pt"):
    state_dict = torch.load("dataloader_state.pt")
    dataloader.load_state_dict(state_dict)

# Iterate over the data
for batch_idx, batch in enumerate(dataloader):

    # Store the state every 1000 batches
    if batch_idx % 1000 == 0:
        torch.save(dataloader.state_dict(), "dataloader_state.pt")
```

</details>


<details>
  <summary> ✅ Use shared queue for Optimizing</summary>
&nbsp;

If you are using multiple workers to optimize your dataset, you can use a shared queue to speed up the process.

This is especially useful when optimizing large datasets in parallel, where some workers may be slower than others.

It can also improve fault tolerance when workers fail due to out-of-memory (OOM) errors.

```python
import numpy as np
from PIL import Image
import litdata as ld

def random_images(index):
    fake_images = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    fake_labels = np.random.randint(10)

    data = {"index": index, "image": fake_images, "class": fake_labels}

    return data

if __name__ == "__main__":
    # The optimize function writes data in an optimized format.
    ld.optimize(
        fn=random_images,                   # the function applied to each input
        inputs=list(range(1000)),           # the inputs to the function (here it's a list of numbers)
        output_dir="fast_data",             # optimized data is stored here
        num_workers=4,                      # The number of workers on the same machine
        chunk_bytes="64MB" ,                 # size of each chunk
        keep_data_ordered=False,             # Use a shared queue to speed up the process
    )
```


### Performance Difference between using a shared queue and not using it:

**Note**: The following benchmarks were collected using the ImageNet dataset on an A10G machine with 16 workers.

| Configuration    | Optimize Time (sec) | Stream 1 (img/sec) | Stream 2 (img/sec) |
|------------------|---------------------|---------------------|---------------------|
| shared_queue (`keep_data_ordered=False`)     | 1281                | 5392                | 5732                |
| no shared_queue (`keep_data_ordered=True (default)`)  | 1187                | 5257                | 5746                |

📌 Note: The **shared_queue** option impacts optimization time, not streaming speed.
> While the streaming numbers may appear slightly different, this variation is incidental and not caused by shared_queue.
>
> Streaming happens after optimization and does not involve inter-process communication where shared_queue plays a role.

- 📄 Using a shared queue helps balance the load across workers, though it may slightly increase optimization time due to the overhead of pickling items sent between processes.

- ⚡ However, it can significantly improve optimizing performance — especially when some workers are slower than others.

</details>


<details>
  <summary> ✅ Use a <code>Queue</code> as input for optimizing data</summary>
&nbsp;

Sometimes you don’t have a static list of inputs to optimize — instead, you have a stream of data coming in over time. In such cases, you can use a multiprocessing.Queue to feed data into the optimize() function.

- This is especially useful when you're collecting data from a remote source like a web scraper, socket, or API.

- You can also use this setup to store `replay buffer` data during reinforcement learning and later stream it back for training.

```python
from multiprocessing import Process, Queue
from litdata.processing.data_processor import ALL_DONE
import litdata as ld
import time

def yield_numbers():
    for i in range(1000):
        time.sleep(0.01)
        yield (i, i**2)

def data_producer(q: Queue):
    for item in yield_numbers():
        q.put(item)

    q.put(ALL_DONE)  # Sentinel value to signal completion

def fn(index):
    return index  # Identity function for demo

if __name__ == "__main__":
    q = Queue(maxsize=100)

    producer = Process(target=data_producer, args=(q,))
    producer.start()

    ld.optimize(
        fn=fn,                   # Function to process each item
        queue=q,                 # 👈 Stream data from this queue
        output_dir="fast_data",  # Where to store optimized data
        num_workers=2,
        chunk_size=100,
        mode="overwrite",
    )

    producer.join()
```

📌 Note: Using queues to optimize your dataset impacts optimization time, not streaming speed.

> Irrespective of number of workers, you only need to put one sentinel value to signal completion.
>
> It'll be handled internally by LitData.

</details>


<details>
  <summary> ✅ LLM Pre-training </summary>
&nbsp;

LitData is highly optimized for LLM pre-training. First, we need to tokenize the entire dataset and then we can consume it.

```python
import json
from pathlib import Path
import zstandard as zstd
from litdata import optimize, TokensLoader
from tokenizer import Tokenizer
from functools import partial

# 1. Define a function to convert the text within the jsonl files into tokens
def tokenize_fn(filepath, tokenizer=None):
    with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
        for row in f:
            text = json.loads(row)["text"]
            if json.loads(row)["meta"]["redpajama_set_name"] == "RedPajamaGithub":
                continue  # exclude the GitHub data since it overlaps with starcoder
            text_ids = tokenizer.encode(text, bos=False, eos=True)
            yield text_ids

if __name__ == "__main__":
    # 2. Generate the inputs (we are going to optimize all the compressed json files from SlimPajama dataset )
    input_dir = "./slimpajama-raw"
    inputs = [str(file) for file in Path(f"{input_dir}/SlimPajama-627B/train").rglob("*.zst")]

    # 3. Store the optimized data wherever you want under "/teamspace/datasets" or "/teamspace/s3_connections"
    outputs = optimize(
        fn=partial(tokenize_fn, tokenizer=Tokenizer(f"{input_dir}/checkpoints/Llama-2-7b-hf")), # Note: You can use HF tokenizer or any others
        inputs=inputs,
        output_dir="./slimpajama-optimized",
        chunk_size=(2049 * 8012),
        # This is important to inform LitData that we are encoding contiguous 1D array (tokens). 
        # LitData skips storing metadata for each sample e.g all the tokens are concatenated to form one large tensor.
        item_loader=TokensLoader(),
    )
```

```python
import os
from litdata import StreamingDataset, StreamingDataLoader, TokensLoader
from tqdm import tqdm

# Increase by one because we need the next word as well
dataset = StreamingDataset(
  input_dir=f"./slimpajama-optimized/train",
  item_loader=TokensLoader(block_size=2048 + 1),
  shuffle=True,
  drop_last=True,
)

train_dataloader = StreamingDataLoader(dataset, batch_size=8, pin_memory=True, num_workers=os.cpu_count())

# Iterate over the SlimPajama dataset
for batch in tqdm(train_dataloader):
    pass
```

</details>

<details>
  <summary> ✅ Filter illegal data </summary>
&nbsp;

Sometimes, you have bad data that you don't want to include in the optimized dataset. With LitData, yield only the good data sample to include. 


```python
from litdata import optimize, StreamingDataset

def should_keep(index) -> bool:
  # Replace with your own logic
  return index % 2 == 0


def fn(data):
    if should_keep(data):
        yield data

if __name__ == "__main__":
    optimize(
        fn=fn,
        inputs=list(range(1000)),
        output_dir="only_even_index_optimized",
        chunk_bytes="64MB",
        num_workers=1
    )

    dataset = StreamingDataset("only_even_index_optimized")
    data = list(dataset)
    print(data)
    # [0, 2, 4, 6, 8, 10, ..., 992, 994, 996, 998]
```

You can even use try/expect.  

```python
from litdata import optimize, StreamingDataset

def fn(data):
    try:
        yield 1 / data 
    except:
        pass

if __name__ == "__main__":
    optimize(
        fn=fn,
        inputs=[0, 0, 0, 1, 2, 4, 0],
        output_dir="only_defined_ratio_optimized",
        chunk_bytes="64MB",
        num_workers=1
    )

    dataset = StreamingDataset("only_defined_ratio_optimized")
    data = list(dataset)
    # The 0 are filtered out as they raise a division by zero 
    print(data)
    # [1.0, 0.5, 0.25] 
```
</details>

<details>
  <summary> ✅ Combine datasets</summary>
&nbsp;

Mix and match different sets of data to experiment and create better models.

Combine datasets with `CombinedStreamingDataset`.  As an example, this mixture of [Slimpajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) & [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) was used in the [TinyLLAMA](https://github.com/jzhang38/TinyLlama) project to pretrain a 1.1B Llama model on 3 trillion tokens.

```python
from litdata import StreamingDataset, CombinedStreamingDataset, StreamingDataLoader, TokensLoader
from tqdm import tqdm
import os

train_datasets = [
    StreamingDataset(
        input_dir="s3://tinyllama-template/slimpajama/train/",
        item_loader=TokensLoader(block_size=2048 + 1), # Optimized loader for tokens used by LLMs
        shuffle=True,
        drop_last=True,
    ),
    StreamingDataset(
        input_dir="s3://tinyllama-template/starcoder/",
        item_loader=TokensLoader(block_size=2048 + 1), # Optimized loader for tokens used by LLMs
        shuffle=True,
        drop_last=True,
    ),
]

# Mix SlimPajama data and Starcoder data with these proportions:
weights = (0.693584, 0.306416)
combined_dataset = CombinedStreamingDataset(datasets=train_datasets, seed=42, weights=weights, iterate_over_all=False)

train_dataloader = StreamingDataLoader(combined_dataset, batch_size=8, pin_memory=True, num_workers=os.cpu_count())

# Iterate over the combined datasets
for batch in tqdm(train_dataloader):
    pass
```

**Batching Methods**

The `CombinedStreamingDataset` supports two different batching methods through the `batching_method` parameter:

**Stratified Batching (Default)**:
With `batching_method="stratified"` (the default), each batch contains samples from multiple datasets according to the specified weights:

```python
# Default stratified batching - batches mix samples from all datasets
combined_dataset = CombinedStreamingDataset(
    datasets=[dataset1, dataset2], 
    batching_method="stratified"  # This is the default
)
```

**Per-Stream Batching**:
With `batching_method="per_stream"`, each batch contains samples exclusively from a single dataset. This is useful when datasets have different shapes or structures:

```python
# Per-stream batching - each batch contains samples from only one dataset
combined_dataset = CombinedStreamingDataset(
    datasets=[dataset1, dataset2], 
    batching_method="per_stream"
)

# This ensures each batch has consistent structure, helpful for datasets with varying:
# - Image sizes
# - Sequence lengths  
# - Data types
# - Feature dimensions
```
</details>

<details>
  <summary> ✅ Parallel streaming</summary>
&nbsp;

While `CombinedDataset` allows to fetch a sample from one of the datasets it wraps at each iteration, `ParallelStreamingDataset` can be used to fetch a sample from all the wrapped datasets at each iteration:

```python
from litdata import StreamingDataset, ParallelStreamingDataset, StreamingDataLoader
from tqdm import tqdm

parallel_dataset = ParallelStreamingDataset(
    [
        StreamingDataset(input_dir="input_dir_1"),
        StreamingDataset(input_dir="input_dir_2"),
    ],
)

dataloader = StreamingDataLoader(parallel_dataset)

for batch_1, batch_2 in tqdm(dataloader):
    pass
```

This is useful to generate new data on-the-fly using a sample from each dataset. To do so, provide a ``transform`` function to `ParallelStreamingDataset`:

```python
def transform(samples: Tuple[Any]):
    sample_1, sample_2 = samples  # as many samples as wrapped datasets
    return sample_1 + sample_2  # example transformation

parallel_dataset = ParallelStreamingDataset([dset_1, dset_2], transform=transform)

dataloader = StreamingDataLoader(parallel_dataset)

for transformed_batch in tqdm(dataloader):
    pass
```

If the transformation requires random number generation, internal random number generators provided by `ParallelStreamingDataset` can be used. These are seeded using the current dataset state at the beginning of each epoch, which allows for reproducible and resumable data transformation. To use them, define a ``transform`` which takes a dictionary of random number generators as its second argument:

```python
def transform(samples: Tuple[Any], rngs: Dict[str, Any]):
    sample_1, sample_2 = samples  # as many samples as wrapped datasets
    rng = rngs["random"]  # "random", "numpy" and "torch" keys available
    return rng.random() * sample_1 + rng.random() * sample_2  # example transformation

parallel_dataset = ParallelStreamingDataset([dset_1, dset_2], transform=transform)
```
</details>

<details>
  <summary> ✅ Cycle datasets</summary>
&nbsp;

`ParallelStreamingDataset` can also be used to cycle a `StreamingDataset`. This allows to dissociate the epoch length from the number of samples in the dataset.

To do so, set the `length` option to the desired number of samples to yield per epoch. If ``length`` is greater than the number of samples in the dataset, the dataset is cycled. At the beginning of a new epoch, the dataset resumes from where it left off at the end of the previous epoch.

```python
from litdata import StreamingDataset, ParallelStreamingDataset, StreamingDataLoader
from tqdm import tqdm

dataset = StreamingDataset(input_dir="input_dir")

cycled_dataset = ParallelStreamingDataset([dataset], length=100)

print(len(cycled_dataset)))  # 100

dataloader = StreamingDataLoader(cycled_dataset)

for batch, in tqdm(dataloader):
    pass
```

You can even set `length` to `float("inf")` for an infinite dataset!
</details>

<details>
  <summary> ✅ Merge datasets</summary>
&nbsp;

Merge multiple optimized datasets into one.

```python
import numpy as np
from PIL import Image

from litdata import StreamingDataset, merge_datasets, optimize


def random_images(index):
    return {
        "index": index,
        "image": Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)),
        "class": np.random.randint(10),
    }


if __name__ == "__main__":
    out_dirs = ["fast_data_1", "fast_data_2", "fast_data_3", "fast_data_4"]  # or ["s3://my-bucket/fast_data_1", etc.]"
    for out_dir in out_dirs:
        optimize(fn=random_images, inputs=list(range(250)), output_dir=out_dir, num_workers=4, chunk_bytes="64MB")

    merged_out_dir = "merged_fast_data" # or "s3://my-bucket/merged_fast_data"
    merge_datasets(input_dirs=out_dirs, output_dir=merged_out_dir)

    dataset = StreamingDataset(merged_out_dir)
    print(len(dataset))
    # out: 1000
```
</details>

<details>
  <summary> ✅ Transform datasets while Streaming</summary>
&nbsp;

Transform datasets on-the-fly while streaming them, allowing for efficient data processing without the need to store intermediate results.

- You can use the `transform` argument in `StreamingDataset` to apply a `transformation function` or `a list of transformation functions` to each sample as it is streamed.

```python
# Define a simple transform function
torch_transform = transforms.Compose([
  transforms.Resize((256, 256)),       # Resize to 256x256
  transforms.ToTensor(),               # Convert to PyTorch tensor (C x H x W)
  transforms.Normalize(                # Normalize using ImageNet stats
      mean=[0.485, 0.456, 0.406], 
      std=[0.229, 0.224, 0.225]
  )
])

def transform_fn(x, *args, **kwargs):
    """Define your transform function."""
    return torch_transform(x)  # Apply the transform to the input image

# Create dataset with appropriate configuration
dataset = StreamingDataset(data_dir, cache_dir=str(cache_dir), shuffle=shuffle, transform=[transform_fn])
```

Or, you can create a subclass of `StreamingDataset` and override its `transform` method to apply custom transformations to each sample.

```python
class StreamingDatasetWithTransform(StreamingDataset):
        """A custom dataset class that inherits from StreamingDataset and applies a transform."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.torch_transform = transforms.Compose([
                transforms.Resize((256, 256)),       # Resize to 256x256
                transforms.ToTensor(),               # Convert to PyTorch tensor (C x H x W)
                transforms.Normalize(                # Normalize using ImageNet stats
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])

        # Define your transform method
        def transform(self, x, *args, **kwargs):
            """A simple transform function."""
            return self.torch_transform(x)


dataset = StreamingDatasetWithTransform(data_dir, cache_dir=str(cache_dir), shuffle=shuffle)
```

</details>

<details>
  <summary> ✅ Split datasets for train, val, test</summary>

&nbsp;

Split a dataset into train, val, test splits with `train_test_split`.

```python
from litdata import StreamingDataset, train_test_split

dataset = StreamingDataset("s3://my-bucket/my-data") # data are stored in the cloud

print(len(dataset)) # display the length of your data
# out: 100,000

train_dataset, val_dataset, test_dataset = train_test_split(dataset, splits=[0.3, 0.2, 0.5])

print(train_dataset)
# out: 30,000

print(val_dataset)
# out: 20,000

print(test_dataset)
# out: 50,000
```

</details>

<details>
  <summary> ✅ Load a subset of the remote dataset</summary>

&nbsp;
Work on a smaller, manageable portion of your data to save time and resources.


```python
from litdata import StreamingDataset, train_test_split

dataset = StreamingDataset("s3://my-bucket/my-data", subsample=0.01) # data are stored in the cloud

print(len(dataset)) # display the length of your data
# out: 1000
```

</details>

<details>
  <summary> ✅ Upsample from your source datasets </summary>

&nbsp;
Use to control the size of one iteration of a StreamingDataset using repeats. Contains `floor(N)` possibly shuffled copies of the source data, then a subsampling of the remainder.


```python
from litdata import StreamingDataset

dataset = StreamingDataset("s3://my-bucket/my-data", subsample=2.5, shuffle=True)

print(len(dataset)) # display the length of your data
# out: 250000
```

</details>

<details>
  <summary> ✅ Easily modify optimized cloud datasets</summary>
&nbsp;

Add new data to an existing dataset or start fresh if needed, providing flexibility in data management.

LitData optimized datasets are assumed to be immutable. However, you can make the decision to modify them by changing the mode to either `append` or `overwrite`.

```python
from litdata import optimize, StreamingDataset

def compress(index):
    return index, index**2

if __name__ == "__main__":
    # Add some data
    optimize(
        fn=compress,
        inputs=list(range(100)),
        output_dir="./my_optimized_dataset",
        chunk_bytes="64MB",
    )

    # Later on, you add more data
    optimize(
        fn=compress,
        inputs=list(range(100, 200)),
        output_dir="./my_optimized_dataset",
        chunk_bytes="64MB",
        mode="append",
    )

    ds = StreamingDataset("./my_optimized_dataset")
    assert len(ds) == 200
    assert ds[:] == [(i, i**2) for i in range(200)]
```

The `overwrite` mode will delete the existing data and start from fresh.

</details>

<details>
  <summary> ✅ Stream parquet datasets</summary>
&nbsp;

Stream Parquet datasets directly with LitData—no need to convert them into LitData’s optimized binary format! If your dataset is already in Parquet format, you can efficiently index and stream it using `StreamingDataset` and `StreamingDataLoader`.

**Assumption:**

Your dataset directory contains one or more Parquet files.

**Prerequisites:**

Install the required dependencies to stream Parquet datasets from cloud storage like **Amazon S3** or **Google Cloud Storage**:

```bash
# For Amazon S3
pip install "litdata[extra]" s3fs

# For Google Cloud Storage
pip install "litdata[extra]" gcsfs
```

**Index Your Dataset**: 

Index your Parquet dataset to create an index file that LitData can use to stream the dataset.

```python
import litdata as ld

# Point to your data stored in the cloud
pq_dataset_uri = "s3://my-bucket/my-parquet-data"  # or "gs://my-bucket/my-parquet-data"

ld.index_parquet_dataset(pq_dataset_uri)
```

**Stream the Dataset**

Use `StreamingDataset` with `ParquetLoader` to load and stream the dataset efficiently:


```python
import litdata as ld
from litdata.streaming.item_loader import ParquetLoader

# Specify your dataset location in the cloud
pq_dataset_uri = "s3://my-bucket/my-parquet-data"  # or "gs://my-bucket/my-parquet-data"

# Set up the streaming dataset
dataset = ld.StreamingDataset(pq_dataset_uri, item_loader=ParquetLoader())

print("Sample", dataset[0])

dataloader = ld.StreamingDataLoader(dataset, batch_size=4)
for sample in dataloader:
    pass
```

</details>

<details>
  <summary> ✅ Use compression</summary>
&nbsp;

Reduce your data footprint by using advanced compression algorithms.

```python
import litdata as ld

def compress(index):
    return index, index**2

if __name__ == "__main__":
    # Add some data
    ld.optimize(
        fn=compress,
        inputs=list(range(100)),
        output_dir="./my_optimized_dataset",
        chunk_bytes="64MB",
        num_workers=1,
        compression="zstd"
    )
```

Using [zstd](https://github.com/facebook/zstd), you can achieve high compression ratio like 4.34x for this simple example.

| Without | With |
| -------- | -------- | 
| 2.8kb | 646b |


</details>

<details>
  <summary> ✅ Access samples without full data download</summary>
&nbsp;

Look at specific parts of a large dataset without downloading the whole thing or loading it on a local machine.

```python
from litdata import StreamingDataset

dataset = StreamingDataset("s3://my-bucket/my-data") # data are stored in the cloud

print(len(dataset)) # display the length of your data

print(dataset[42]) # show the 42th element of the dataset
```

</details>

<details>
  <summary> ✅ Use any data transforms</summary>
&nbsp;

Customize how your data is processed to better fit your needs.

Subclass the `StreamingDataset` and override its `__getitem__` method to add any extra data transformations.

```python
from litdata import StreamingDataset, StreamingDataLoader
import torchvision.transforms.v2.functional as F

class ImagenetStreamingDataset(StreamingDataset):

    def __getitem__(self, index):
        image = super().__getitem__(index)
        return F.resize(image, (224, 224))

dataset = ImagenetStreamingDataset(...)
dataloader = StreamingDataLoader(dataset, batch_size=4)

for batch in dataloader:
    print(batch.shape)
    # Out: (4, 3, 224, 224)
```

</details>

<details>
  <summary> ✅ Profile data loading speed</summary>
&nbsp;

Measure and optimize how fast your data is being loaded, improving efficiency.

The `StreamingDataLoader` supports profiling of your data loading process. Simply use the `profile_batches` argument to specify the number of batches you want to profile:

```python
from litdata import StreamingDataset, StreamingDataLoader

StreamingDataLoader(..., profile_batches=5)
```

This generates a Chrome trace called `result.json`. Then, visualize this trace by opening Chrome browser at the `chrome://tracing` URL and load the trace inside.

</details>

<details>
  <summary> ✅ Reduce memory use for large files</summary>
&nbsp;

Handle large data files efficiently without using too much of your computer's memory.

When processing large files like compressed [parquet files](https://en.wikipedia.org/wiki/Apache_Parquet), use the Python yield keyword to process and store one item at the time, reducing the memory footprint of the entire program.

```python
from pathlib import Path
import pyarrow.parquet as pq
from litdata import optimize
from tokenizer import Tokenizer
from functools import partial

# 1. Define a function to convert the text within the parquet files into tokens
def tokenize_fn(filepath, tokenizer=None):
    parquet_file = pq.ParquetFile(filepath)
    # Process per batch to reduce RAM usage
    for batch in parquet_file.iter_batches(batch_size=8192, columns=["content"]):
        for text in batch.to_pandas()["content"]:
            yield tokenizer.encode(text, bos=False, eos=True)

# 2. Generate the inputs
input_dir = "/teamspace/s3_connections/tinyllama-template"
inputs = [str(file) for file in Path(f"{input_dir}/starcoderdata").rglob("*.parquet")]

# 3. Store the optimized data wherever you want under "/teamspace/datasets" or "/teamspace/s3_connections"
outputs = optimize(
    fn=partial(tokenize_fn, tokenizer=Tokenizer(f"{input_dir}/checkpoints/Llama-2-7b-hf")), # Note: Use HF tokenizer or any others
    inputs=inputs,
    output_dir="/teamspace/datasets/starcoderdata",
    chunk_size=(2049 * 8012), # Number of tokens to store by chunks. This is roughly 64MB of tokens per chunk.
)
```

</details>

<details>
  <summary> ✅ Limit local cache space</summary>
&nbsp;

Limit the amount of disk space used by temporary files, preventing storage issues.

Adapt the local caching limit of the `StreamingDataset`. This is useful to make sure the downloaded data chunks are deleted when used and the disk usage stays low.

```python
from litdata import StreamingDataset

dataset = StreamingDataset(..., max_cache_size="10GB")
```

</details>

<details>
  <summary> ✅ Change cache directory path</summary>
&nbsp;

Specify the directory where cached files should be stored, ensuring efficient data retrieval and management. This is particularly useful for organizing your data storage and improving access times.

```python
from litdata import StreamingDataset
from litdata.streaming.cache import Dir

cache_dir = "/path/to/your/cache"
data_dir = "s3://my-bucket/my_optimized_dataset"

dataset = StreamingDataset(input_dir=Dir(path=cache_dir, url=data_dir))
```

</details>

<details>
  <summary> ✅ Optimize loading on networked drives</summary>
&nbsp;

Optimize data handling for computers on a local network to improve performance for on-site setups.

On-prem compute nodes can mount and use a network drive. A network drive is a shared storage device on a local area network. In order to reduce their network overload, the `StreamingDataset` supports `caching` the data chunks.

```python
from litdata import StreamingDataset

dataset = StreamingDataset(input_dir="local:/data/shared-drive/some-data")
```

</details>

<details>
  <summary> ✅ Optimize dataset in distributed environment</summary>
&nbsp;

Lightning can distribute large workloads across hundreds of machines in parallel. This can reduce the time to complete a data processing task from weeks to minutes by scaling to enough machines.

To apply the optimize operator across multiple machines, simply provide the num_nodes and machine arguments to it as follows:

```python
import os
from litdata import optimize, Machine

def compress(index):
    return (index, index ** 2)

optimize(
    fn=compress,
    inputs=list(range(100)),
    num_workers=2,
    output_dir="my_output",
    chunk_bytes="64MB",
    num_nodes=2,
    machine=Machine.DATA_PREP, # You can select between dozens of optimized machines
)
```

If the `output_dir` is a local path, the optimized dataset will be present in: `/teamspace/jobs/{job_name}/nodes-0/my_output`. Otherwise, it will be stored in the specified `output_dir`.

Read the optimized dataset:

```python
from litdata import StreamingDataset

output_dir = "/teamspace/jobs/litdata-optimize-2024-07-08/nodes.0/my_output"

dataset = StreamingDataset(output_dir)

print(dataset[:])
```

</details>

<details>
  <summary> ✅ Encrypt, decrypt data at chunk/sample level</summary>
&nbsp;

Secure data by applying encryption to individual samples or chunks, ensuring sensitive information is protected during storage.

This example shows how to use the `FernetEncryption` class for sample-level encryption with a data optimization function.

```python
from litdata import optimize
from litdata.utilities.encryption import FernetEncryption
import numpy as np
from PIL import Image

# Initialize FernetEncryption with a password for sample-level encryption
fernet = FernetEncryption(password="your_secure_password", level="sample")
data_dir = "s3://my-bucket/optimized_data"

def random_image(index):
    """Generate a random image for demonstration purposes."""
    fake_img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    return {"image": fake_img, "class": index}

# Optimize data while applying encryption
optimize(
    fn=random_image,
    inputs=list(range(5)),  # Example inputs: [0, 1, 2, 3, 4]
    num_workers=1,
    output_dir=data_dir,
    chunk_bytes="64MB",
    encryption=fernet,
)

# Save the encryption key to a file for later use
fernet.save("fernet.pem")
```

Load the encrypted data using the `StreamingDataset` class as follows:

```python
from litdata import StreamingDataset
from litdata.utilities.encryption import FernetEncryption

# Load the encryption key
fernet = FernetEncryption(password="your_secure_password", level="sample")
fernet.load("fernet.pem")

# Create a streaming dataset for reading the encrypted samples
ds = StreamingDataset(input_dir=data_dir, encryption=fernet)
```

Implement your own encryption method: Subclass the `Encryption` class and define the necessary methods:

```python
from litdata.utilities.encryption import Encryption

class CustomEncryption(Encryption):
    def encrypt(self, data):
        # Implement your custom encryption logic here
        return data

    def decrypt(self, data):
        # Implement your custom decryption logic here
        return data
```

This allows the data to remain secure while maintaining flexibility in the encryption method.
</details>

<details>
  <summary> ✅ Debug & Profile LitData with logs & Litracer</summary>

&nbsp;

LitData comes with built-in logging and profiling capabilities to help you debug and profile your data streaming workloads.

<img width="1439" alt="431247797-0e955e71-2f9a-4aad-b7c1-a8218fed2e2e" src="https://github.com/user-attachments/assets/4e40676c-ba0b-49af-acac-975977173669" />

- e.g., with LitData Streaming

```python
import litdata as ld
from litdata.debugger import enable_tracer

# WARNING: Remove existing trace `litdata_debug.log` file if it exists before re-tracing
enable_tracer()

if __name__ == "__main__":
    dataset = ld.StreamingDataset("s3://my-bucket/my-data", shuffle=True)
    dataloader = ld.StreamingDataLoader(dataset, batch_size=64)

    for batch in dataloader:
        print(batch)  # Replace with your data processing logic
```

1. Generate Debug Log:

    - Run your Python program and it'll create a log file containing detailed debug information.

    ```bash
      python main.py
    ```

2. Install [Litracer](https://github.com/deependujha/litracer/):

    - Option 1: Using Go (recommended)
        - Install Go on your system.
        - Run the following command to install Litracer:

        ```bash
          go install github.com/deependujha/litracer@latest
        ```

    - Option 2: Download Binary
        - Visit the [LitRacer GitHub Releases](https://github.com/deependujha/litracer/releases) page.
        - Download the appropriate binary for your operating system and follow the installation instructions.

3. Convert Debug Log to trace JSON:

    - Use litracer to convert the generated log file into a trace JSON file. This command uses 100 workers for conversion:

    ```bash
      litracer litdata_debug.log -o litdata_trace.json -w 100
    ```

4. Visualize the trace:

    - Use either `chrome://tracing` in the Chrome browser or `ui.perfetto.dev` to view the `litdata_trace.json` file for in-depth performance insights. You can also use `SQL queries` to analyze the logs.
    - `Perfetto` is recommended over `chrome://tracing` for visualization & analyzing.

- Key Points:

    - For very large trace.json files (`> 2GB`), refer to the [Perfetto documentation](https://perfetto.dev/docs/visualization/large-traces) for using native accelerators.
    - If you are trying to connect Perfetto to the RPC server, it is recommended to use Chrome over Brave, as it has been observed that Perfetto in Brave does not autodetect the RPC server.

</details>

<details>
  <summary> ✅ Lightning AI Data Connections - Direct download and upload </summary>

&nbsp;

[Lightning Studios](https://lightning.ai/) have special directories for data connections that are available to an entire teamspace. LitData functions that reference those directories will experience a significant performance increase as uploads and downloads will happen directly from the bucket that backs the folder.

For example, output artifacts from this code will be directly uploaded to the `my-data-1` s3 bucket.

```python
from litdata import optimize

def should_keep(data):
    if data % 2 == 0:
        yield data

if __name__ == "__main__":
    optimize(
        fn=should_keep,
        inputs=list(range(1000)),
        output_dir="/teamspace/s3_connections/my-data-1/output",
        chunk_bytes="64MB",
        num_workers=1
    )
```


Similarly, data will be downloaded directly from the `my-data-1` s3 bucket in this example code.

```python
from litdata import StreamingRawDataset

if __name__ == "__main__":
    data_dir = "/teamspace/s3_connections/my-bucket-1/data"

    raw_dataset = StreamingRawDataset(data_dir)

    data = list(raw_dataset)
    print(data)
```

References to any of the following directories will work similarly:
1. `/teamspace/lightning_storage/...`
2. `/teamspace/s3_connections/...`
3. `/teamspace/gcs_connections/...`
4. `/teamspace/s3_folders/...`
5. `/teamspace/gcs_folders/...`
</details>

&nbsp;


## Features for transforming datasets

<details>
  <summary> ✅ Parallelize data transformations (map)</summary>
&nbsp;

Apply the same change to different parts of the dataset at once to save time and effort.

The `map` operator can be used to apply a function over a list of inputs.

Here is an example where the `map` operator is used to apply a `resize_image` function over a folder of large images.

```python
from litdata import map
from PIL import Image

# Note: Inputs could also refer to files on s3 directly.
input_dir = "my_large_images"
inputs = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

# The resize image takes one of the input (image_path) and the output directory.
# Files written to output_dir are persisted.
def resize_image(image_path, output_dir):
  output_image_path = os.path.join(output_dir, os.path.basename(image_path))
  Image.open(image_path).resize((224, 224)).save(output_image_path)

map(
    fn=resize_image,
    inputs=inputs,
    output_dir="s3://my-bucket/my_resized_images",
)
```

</details>

&nbsp;

----

# Benchmarks
In this section we show benchmarks for speed to optimize a dataset and the resulting streaming speed ([Reproduce the benchmark](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries)).

## Streaming speed 
### LitData Chunks
Data optimized and streamed with LitData achieves a 20x speed up over non optimized data and 2x speed up over other streaming solutions.

Speed to stream Imagenet 1.2M from AWS S3:

| Framework | Images / sec  1st Epoch (float32)  | Images / sec   2nd Epoch (float32) | Images / sec 1st Epoch (torch16) | Images / sec 2nd Epoch (torch16) |
|---|---|---|---|---|
| LitData | **5839** | **6692**  | **6282**  | **7221**  |
| Web Dataset  | 3134 | 3924 | 3343 | 4424 |
| Mosaic ML  | 2898 | 5099 | 2809 | 5158 |

<details>
  <summary> Benchmark details</summary>
&nbsp;

- [Imagenet-1.2M dataset](https://www.image-net.org/) contains `1,281,167 images`.
- To align with other benchmarks, we measured the streaming speed (`images per second`) loaded from [AWS S3](https://aws.amazon.com/s3/) for several frameworks.

</details>
&nbsp;

Speed to stream Imagenet 1.2M from other cloud storage providers:

| Storage Provider | Framework | Images / sec 1st Epoch (float32) | Images / sec 2nd Epoch (float32) |
|---|---|---|---|
| Cloudflare R2 | LitData | **5335** | **5630** |

Speed to stream Imagenet 1.2M from local disk with ffcv vs LitData:
| Framework | Dataset Mode | Dataset Size @ 256px | Images / sec 1st Epoch (float32) | Images / sec 2nd Epoch (float32) |
|---|---|---|---|---|
| LitData | PIL RAW | 168 GB | 6647 | 6398 | 
| LitData | JPEG 90% | 12 GB | 6553 | 6537 |
| ffcv (os_cache=True) | RAW | 170 GB | 7263 | 6698 |
| ffcv (os_cache=False) | RAW | 170 GB | 7556 | 8169 |
| ffcv(os_cache=True) | JPEG 90% | 20 GB | 7653 | 8051 |
| ffcv(os_cache=False) | JPEG 90% | 20 GB | 8149 | 8607 |

### Raw Dataset

Speed to stream raw Imagenet 1.2M from different cloud storage providers:


| Storage | Images / s (without transform) | Images / s (with transform) |
|---------|-------------------|----------------|
| AWS S3  | ~6400 +/- 100     | ~3200 +/- 100  |
| Google Cloud Storage | ~5650 +/- 100     | ~3100 +/- 100  |

> **Note:**
> Use `StreamingRawDataset` if you want to stream your data as-is. Use `StreamingDataset` if you want the fastest streaming and are okay with optimizing your data first.

&nbsp;

## Time to optimize data
LitData optimizes the Imagenet dataset for fast training 3-5x faster than other frameworks:

Time to optimize 1.2 million ImageNet images (Faster is better):
| Framework |Train Conversion Time | Val Conversion Time | Dataset Size | # Files |
|---|---|---|---|---|
| LitData  |  **10:05 min** | **00:30 min** | **143.1 GB**  | 2.339  |
| Web Dataset  | 32:36 min | 01:22 min | 147.8 GB | 1.144 |
| Mosaic ML  | 49:49 min | 01:04 min | **143.1 GB** | 2.298 |

&nbsp;

----

# Parallelize transforms and data optimization on cloud machines
<div align="center">
<img alt="Lightning" src="https://pl-flash-data.s3.amazonaws.com/data-prep.jpg" width="700px">
</div>

## Parallelize data transforms

Transformations with LitData are linearly parallelizable across machines.

For example, let's say that it takes 56 hours to embed a dataset on a single A10G machine. With LitData,
this can be speed up by adding more machines in parallel

| Number of machines | Hours |
|-----------------|--------------|
| 1               | 56           |
| 2               | 28           |
| 4               | 14           |
| ...               | ...            |
| 64              | 0.875        |

To scale the number of machines, run the processing script on [Lightning Studios](https://lightning.ai/):

```python
from litdata import map, Machine

map(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP, # Select between dozens of optimized machines
)
```

## Parallelize data optimization
To scale the number of machines for data optimization, use [Lightning Studios](https://lightning.ai/):

```python
from litdata import optimize, Machine

optimize(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP, # Select between dozens of optimized machines
)
```

&nbsp;

Example: [Process the LAION 400 million image dataset in 2 hours on 32 machines, each with 32 CPUs](https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset).

&nbsp;

----

# Start from a template
Below are templates for real-world applications of LitData at scale.

## Templates: Transform datasets

| Studio | Data type | Time (minutes) | Machines | Dataset |
| ------------------------------------ | ----------------- | ----------------- | -------------- | -------------- |
| [Download LAION-400MILLION dataset](https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset) | Image & Text | 120 | 32 |[LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) |
| [Tokenize 2M Swedish Wikipedia Articles](https://lightning.ai/lightning-ai/studios/tokenize-2m-swedish-wikipedia-articles) | Text | 7 | 4 | [Swedish Wikipedia](https://huggingface.co/datasets/wikipedia) |
| [Embed English Wikipedia under 5 dollars](https://lightning.ai/lightning-ai/studios/embed-english-wikipedia-under-5-dollars) | Text | 15 | 3 | [English Wikipedia](https://huggingface.co/datasets/wikipedia) |

## Templates: Optimize + stream data

| Studio | Data type | Time (minutes) | Machines | Dataset |
| -------------------------------- | ----------------- | ----------------- | -------------- | -------------- |
| [Benchmark cloud data-loading libraries](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries) | Image & Label | 10 | 1 | [Imagenet 1M](https://paperswithcode.com/sota/image-classification-on-imagenet?tag_filter=171) |
| [Optimize GeoSpatial data for model training](https://lightning.ai/lightning-ai/studios/convert-spatial-data-to-lightning-streaming) | Image & Mask | 120 | 32 | [Chesapeake Roads Spatial Context](https://github.com/isaaccorley/chesapeakersc) |
| [Optimize TinyLlama 1T dataset for training](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset) | Text | 240 | 32 | [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) & [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) |
| [Optimize parquet files for model training](https://lightning.ai/lightning-ai/studios/convert-parquets-to-lightning-streaming) | Parquet Files | 12 | 16 | Randomly Generated data |

&nbsp;

----

# Community
LitData is a community project accepting contributions -  Let's make the world's most advanced AI data processing framework.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)    
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litdata/blob/main/LICENSE)


----

## Citation

```
@misc{litdata2023,
  author       = {Thomas Chaton and Lightning AI},
  title        = {LitData: Transform datasets at scale. Optimize datasets for fast AI model training.},
  year         = {2023},
  howpublished = {\url{https://github.com/Lightning-AI/litdata}},
  note         = {Accessed: 2025-04-09}
}
```

----

## Papers with LitData

* [Towards Interpretable Protein Structure
Prediction with Sparse Autoencoders](https://arxiv.org/pdf/2503.08764) | [Github](https://github.com/johnyang101/reticular-sae) | (Nithin Parsan, David J. Yang and John J. Yang)

----

# Governance

## Maintainers

* Thomas Chaton ([tchaton](https://github.com/tchaton))
* Bhimraj Yadav ([bhimrazy](https://github.com/bhimrazy))
* Deependu ([deependujha](https://github.com/deependujha))


## Emeritus Maintainers

* Luca Antiga ([lantiga](https://github.com/lantiga))
* Justus Schock ([justusschock](https://github.com/justusschock))
* Jirka Borda ([Borda](https://github.com/Borda))

<details>
  <summary>Alumni</summary>

* Adrian Wälchli ([awaelchli](https://github.com/awaelchli))

</details>
