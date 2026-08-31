<div align="center">
<h1>
  Speed up model training by fixing data loading
</h1>  
<img src="https://pl-flash-data.s3.amazonaws.com/lit_data_logo.webp" alt="LitData" width="800px"/>

&nbsp;
&nbsp;

<table>
<tr>
<td valign="top" align="left">

**Transform**

✅ Parallelize data processing  
✅ Create vector embeddings  
✅ Run distributed inference  
✅ Scrape websites at scale

</td>
<td valign="top" align="left">

**Optimize / Stream**

✅ Stream raw files with no prep  
✅ Stream large cloud datasets  
✅ Accelerate training by 20x  
✅ Pause and resume data streaming  
✅ Use remote data without local loading

</td>
</tr>
</table>

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
  <a href="#modality">Modality</a> •
  <a href="#huggingface-datasets">Hugging Face Datasets</a> •
  <a href="#key-features">Features</a> •
  <a href="#stream-raw">Stream raw files</a> •
  <a href="#resolve-paths">Paths & cloud URLs</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#start-from-a-template">Templates</a> •
  <a href="#used-by">Used by</a> •
  <a href="#skills">Skills</a> •
  <a href="#community">Community</a>
</p>

&nbsp;

<a target="_blank" href="https://lightning.ai/docs/overview/optimize-data/optimize-datasets">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>

</div>

&nbsp;

# Why LitData?
Speeding up model training involves more than kernel tuning. Data loading frequently slows down training, because datasets are too large to fit on disk, consist of millions of small files, or stream slowly from the cloud. 

LitData provides tools to preprocess and optimize datasets into a format that streams efficiently from any cloud or local source. It also includes a map operator for distributed data processing before optimization. This makes data pipelines faster, cloud-agnostic, and can improve training throughput by up to 20×.

&nbsp;

# Looking for GPUs?
Over 340,000 developers use [Lightning Cloud](https://lightning.ai/?utm_source=litdata&utm_medium=referral&utm_campaign=litdata) - purpose-built for PyTorch and PyTorch Lightning. 
- [GPUs](https://lightning.ai/pricing?utm_source=litdata&utm_medium=referral&utm_campaign=litdata) from $0.19.   
- [Clusters](https://lightning.ai/clusters?utm_source=litdata&utm_medium=referral&utm_campaign=litdata): frontier-grade training/inference clusters.   
- [AI Studio (vibe train)](https://lightning.ai/studios?utm_source=litdata&utm_medium=referral&utm_campaign=litdata): workspaces where AI helps you debug, tune and vibe train.
- [AI Studio (vibe deploy)](https://lightning.ai/studios?utm_source=litdata&utm_medium=referral&utm_campaign=litdata): workspaces where AI helps you optimize, and deploy models.     
- [Notebooks](https://lightning.ai/notebooks?utm_source=litdata&utm_medium=referral&utm_campaign=litdata): Persistent GPU workspaces where AI helps you code and analyze.
- [Inference](https://lightning.ai/deploy?utm_source=litdata&utm_medium=referral&utm_campaign=litdata): Deploy models as inference APIs.

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

On Linux/macOS, `[extras]` includes optional `uvloop` for a faster asyncio event loop used by `StreamingRawDataset` (stdlib asyncio is the fallback when it is not installed).

</details>

<details>
  <summary>AI agent skill (Cursor, Claude Code, …)</summary>

Install the LitData expert skill so coding agents know the full API, path resolver, optimize/stream recipes, and internals. Full file map → [Skills](#skills).

```bash
npx skills add Lightning-AI/litData
```

Source: [`.claude/skills/litdata/`](.claude/skills/litdata/) in this repository ([skills CLI](https://github.com/vercel-labs/skills)).

</details>

&nbsp;

----

# Speed up model training
Stream datasets directly from cloud storage without local downloads. Choose the approach that fits your workflow:

## Option 1: Stream existing files as-is ⚡⚡ — `StreamingRawDataset`

**No optimize step.** Point LitData at a folder of images, audio, text, or any files (local or cloud) and train with a normal PyTorch `DataLoader`. Downloads are **fully asynchronous** and **batched**; cloud clients include **built-in retries**. You receive **raw `bytes`** — decode, parse, or transform however you want.

Details → [Stream raw files](#stream-raw).

```python
from litdata import StreamingRawDataset
from torch.utils.data import DataLoader
from PIL import Image
import io

dataset = StreamingRawDataset(
    "s3://my-bucket/raw-images/",          # or gs://, azure://, /teamspace/s3_connections/..., local path
    transform=lambda b: Image.open(io.BytesIO(b)).convert("RGB"),  # optional — default is raw bytes
)
loader = DataLoader(dataset, batch_size=32, num_workers=8)

for batch in loader:
    train_step(batch)
```

**Key benefits:**

✅ **Zero preprocess:**     No chunking job — use the files you already have.    
✅ **Raw bytes, your rules:** Each sample is file `bytes`; decode with PIL, torchaudio, json, or any custom logic (`transform=` optional).    
✅ **Fully async + batched:** Concurrent downloads via `asyncio` / `__getitems__` (not one-file-at-a-time).    
✅ **Built-in retries:**     Cloud downloads retry transient failures (adaptive client retries).    
✅ **Cloud-native:**        S3 / GCS / Azure / Studio connections; same path resolver as optimized streaming.    
✅ **Grouped samples:**     Override `setup()` to yield image+mask, audio+transcript, etc.    
✅ **Indexed once:**        `index.json.zstd` cached locally and on the bucket for fast restarts.    
✅ **Upgrade path:**        When I/O becomes the bottleneck, `optimize` → `StreamingDataset` for max throughput.    

## Option 2: Optimize for maximum performance ⚡⚡⚡  
Accelerate model training (20x faster) by optimizing datasets for streaming directly from cloud storage. Work with remote data without local downloads with features like loading data subsets, accessing individual samples, and resumable streaming.

**Step 1: Optimize your data (one-time setup)**

Transform raw data into optimized chunks for maximum streaming speed.
This step formats the dataset for fast loading by writing data in an efficient chunked binary format.

```python
import numpy as np
import litdata as ld

def random_images(index):
    # Replace with your files: Image(path="photo.jpg") or Image(bytes=...).
    # Wrappers pick the serializer (a caption string is not an image).
    # quality/format encode JPEG — not uncompressed PIL RAW.
    array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    return {
        "index": index,
        "image": ld.Image(array=array, quality=95, format="jpeg"),
        "class": np.random.randint(10),
    }

if __name__ == "__main__":
    # Exactly one of chunk_bytes or chunk_size
    ld.optimize(
        fn=random_images,                   # the function applied to each input
        inputs=list(range(1000)),           # the inputs to the function (here it's a list of numbers)
        output_dir="fast_data",             # optimized data is stored here
        num_workers=4,                      # the number of workers on the same machine
        chunk_bytes="64MB"                  # default; see FAQ for larger samples
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

dataset = ld.StreamingDataset(
    's3://my-bucket/fast_data',
    shuffle=True,
    drop_last=True,  # important for multi-GPU so every rank sees the same length
    seed=42,
)

# Custom collate function to handle the batch (optional)
def collate_fn(batch):
    return {
        "image": [sample["image"] for sample in batch],
        "class": [sample["class"] for sample in batch],
    }


dataloader = ld.StreamingDataLoader(dataset, batch_size=64, collate_fn=collate_fn)
for sample in dataloader:
    img, cls = sample["image"], sample["class"]
```

**Keyed lookup and in-place patches** (needs `polars` and `optimize(..., key_fn=...)` or `build_keys_index`):

```python
ld.optimize(fn=fn, inputs=inputs, output_dir="fast_data", chunk_bytes="64MB", key_fn=lambda s: s["id"])

ds = ld.StreamingDataset("fast_data")
sample = ds["entity-id"]          # str keys
sample = ds.get_by_key(42)        # int entity keys; ds[42] is still positional

with ld.dataset_update("fast_data") as update:  # local directory only
    update["entity-id"] = {"id": "entity-id", "x": 1}
    update.commit()
```

`mode="append"` continues chunk numbering. `use_checkpoint=True` tries to resume an interrupted optimize. They are not the same.

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

# Modality <a id="media-types"></a>

Wrap each file so a caption is not treated as a path: Text(path=...), Image(path=...), Audio(path=...). Path and raw bytes are stored as-is; array / image / mesh encode.

<table width="100%">
<tr>
<th align="left">Type</th>
<th align="left">Write</th>
<th align="left">Stream</th>
</tr>
<tr>
<td colspan="3"><strong>Text</strong></td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/text.py">Text</a></td>
<td>Text(path="a.txt")<br>Text(bytes=utf8)<br>Text(text="a caption")</td>
<td>text  # str</td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/text.py">Tokens</a></td>
<td>Tensor(array=token_ids)<br>optimize(..., item_loader=TokensLoader())</td>
<td>tokens  # Tensor, length block_size — <a href="#llm-training">LLM training</a></td>
</tr>
<tr>
<td colspan="3"><strong>Image</strong></td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/image.py">Image</a></td>
<td>Image(path="a.jpg")<br>Image(bytes=jpeg)<br>Image(array=hwc, quality=95, format="jpeg")</td>
<td>image.shape  # Tensor CHW</td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/jpeg.py">Jpeg</a></td>
<td>Jpeg(path="a.jpg")<br>Jpeg(array=hwc, quality=95)</td>
<td>image.shape  # Tensor CHW</td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/jpeg_array.py">JpegArray</a></td>
<td>JpegArray(images=[Jpeg(path=p) for p in frames])</td>
<td>images[0].shape  # Tensor CHW</td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/pil.py">Pil</a></td>
<td>Pil(path="a.png")<br>Pil(image=pil_img, mode="RGB")</td>
<td>pil_img.size  # PIL.Image</td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/tiff.py">Tiff</a></td>
<td>Tiff(path="a.tif")<br>Tiff(array=hw)</td>
<td>array.shape  # NumPy</td>
</tr>
<tr>
<td colspan="3"><strong>Audio and Video</strong></td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/audio.py">Audio</a></td>
<td>Audio(path="a.wav")<br>Audio(bytes=wav)<br>Audio(array=wave, sampling_rate=16000)</td>
<td>audio["array"]<br>audio["sampling_rate"]</td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/video.py">Video</a></td>
<td>Video(path="c.mp4")<br>Video(bytes=mp4)<br>Video(array=frames, fps=25)</td>
<td>video.get_frames_at(0)<br>video.get_frames_in_range(0, 8)</td>
</tr>
<tr>
<td colspan="3"><strong>File</strong></td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/file.py">File</a></td>
<td>File(path="doc.bin")<br>File(bytes=blob)</td>
<td>sidecar  # raw bytes</td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/pdf.py">Pdf</a></td>
<td>Pdf(path="p.pdf")<br>Pdf(pdf=pdfplumber_doc)</td>
<td>pdf.pages[0]  # Pdfplumber</td>
</tr>
<tr>
<td colspan="3"><strong>3D and volume</strong></td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/mesh.py">Mesh</a></td>
<td>Mesh(path="m.glb")<br>Mesh(mesh=trimesh_obj, file_type="glb")</td>
<td>mesh.vertices  # Trimesh</td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/nifti.py">Nifti</a></td>
<td>Nifti(path="v.nii.gz")<br>Nifti(array=vol, affine=np.eye(4))</td>
<td>nifti.get_fdata()  # Nibabel</td>
</tr>
<tr>
<td colspan="3"><strong>Array and Graph</strong></td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/numpy_array.py">Numpy</a></td>
<td>np.load("a.npy")<br>np.zeros((3, 4, 4))</td>
<td>array  # NumPy</td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/tensor.py">Tensor</a></td>
<td>Tensor(array=torch.randn(3, 4, 4))</td>
<td>feat  # Tensor — 1-D token ids use TokensLoader under Text</td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/graph.py">Graph</a></td>
<td>Data(x=…, edge_index=…, y=…)<br>Graph(x=…, edge_index=…, y=…)<br>Graph(data=pyg_data)</td>
<td>graph.x, graph.edge_index  # PyG Data or Graph — <a href="#pyg-graphs">PyG graphs</a></td>
</tr>
<tr>
<td colspan="3"><strong>Parquet</strong></td>
</tr>
<tr>
<td valign="top"><a href="examples/modality/parquet.py">Parquet</a></td>
<td>folder of .parquet files<br>StreamingDataset(..., item_loader=ParquetLoader())</td>
<td>row["col"]  # dict of columns — <a href="#stream-parquet">stream parquet</a></td>
</tr>
</table>

Examples (path on disk → optimize → batch): [examples/modality](examples/modality).

----

# Hugging Face Datasets 🤗 <a id="huggingface-datasets"></a><a id="stream-hf"></a>

Train on the Hub datasets you already use. Drop in a Hugging Face dataset with one call — then optimize once onto Lightning Cloud storage when you want faster training I/O than Hub parquet or Hugging Face `datasets` streaming.

<details>
  <summary>How to get the dataset URL</summary>

https://github.com/user-attachments/assets/3ba9e2ef-bf6b-41fc-a578-e4b4113a0e72

</details>

```bash
pip install 'litdata[extras]' huggingface_hub
```

Gated datasets: set `HF_TOKEN`.

**Load from the Hub**

```python
import litdata as ld

dataset = ld.StreamingDataset("hf://datasets/HuggingFaceH4/ultrachat_200k/data/train_sft-*.parquet")
print("Sample", dataset[0])

dataloader = ld.StreamingDataLoader(
    dataset, batch_size=4, num_workers=4, multiprocessing_context="spawn"
)
for sample in dataloader:
    pass
```

**Optimize once for faster training**

```python
import litdata as ld

ld.optimize_hf("stanfordnlp/imdb", output_dir="imdb-opt", split="train")
dataset = ld.StreamingDataset("imdb-opt", shuffle=True, drop_last=True)
```

`optimize_hf` converts a Hub dataset into LitData chunks. Later calls reuse `output_dir` if it is already optimized. Pass `revision=`, `config=`, `chunk_size=`, or `fn=` when you need them.

### Faster than Hub parquet and `datasets` streaming

`StreamingDataset("hf://...")` loads Hub parquet as-is. Run `optimize_hf` once when you want higher training throughput. Both beat Hugging Face `datasets` streaming (`load_dataset(..., streaming=True)`) on every split below. Optimized chunks are **not** always fastest: **12/15** prefer `optimize_hf`; parquet stays ahead on [OpenThoughts](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k), [food101](https://huggingface.co/datasets/ethz/food101), and [cifar100](https://huggingface.co/datasets/uoft-cs/cifar100). Minipile, food101, and superb ks were re-measured with current defaults; the other twelve rows are unchanged.

Sequential one-epoch read after a 200-row warmup. Reproduce: `scripts/bench/bench_hf_hub_suite.py`.

| Dataset | `optimize_hf` rows/s | parquet (`hf://`) rows/s | HF streaming rows/s |
| --- | ---: | ---: | ---: |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) / train_sft | **47,285** | 31,209 | 6,752 |
| [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) / train | 6,508 | **12,292** | 1,492 |
| [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) / train / 3.0.0 | **57,944** | 44,286 | 13,874 |
| [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) / train | **108,420** | 74,502 | 8,280 |
| [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) / train | **164,825** | 141,235 | 58,331 |
| [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) / train | **214,217** | 162,448 | 14,613 |
| [Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full) / train | **123,670** | 88,400 | 58,621 |
| [EdinburghNLP/xsum](https://huggingface.co/datasets/EdinburghNLP/xsum) / train | **55,941** | 37,684 | 15,377 |
| [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) / train | **92,741** | 46,573 | 18,263 |
| [JeanKaddour/minipile](https://huggingface.co/datasets/JeanKaddour/minipile) / train | **81,871** | 50,160 | 3,928 |
| [ethz/food101](https://huggingface.co/datasets/ethz/food101) / train | 1,989 | **4,782** | 384 |
| [zh-plus/tiny-imagenet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) / train | **57,855** | 43,535 | 6,491 |
| [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10) / train | **31,671** | 20,657 | 8,843 |
| [uoft-cs/cifar100](https://huggingface.co/datasets/uoft-cs/cifar100) / train | 22,448 | **24,332** | 7,553 |
| [s3prl/superb](https://huggingface.co/datasets/s3prl/superb) / train / ks | **7,101** | 6,775 | 314 |

Your own parquet on disk or S3 → [Stream parquet datasets](#stream-parquet).

# Key Features

## Features for optimizing and streaming datasets for model training

<details>
  <summary> ✅ Stream raw files as-is (no optimize) — StreamingRawDataset <a id="stream-raw" href="#stream-raw">🔗</a> </summary>
  &nbsp;

`StreamingRawDataset` streams **your existing files** from local disk or cloud storage with **no conversion step**. It is a map-style `torch.utils.data.Dataset`: use a standard PyTorch `DataLoader` (not `StreamingDataLoader`).

**You get raw `bytes`.** LitData does not impose a sample schema — open images with PIL, parse JSONL, decode audio, run your own tokenizer, or pass a `transform=` if you prefer. Grouped items yield `list[bytes]` (e.g. image + mask).

Downloads are **fully asynchronous** and **batched**: when the DataLoader requests a batch, `__getitems__` fetches those files concurrently with `asyncio.gather`. Cloud clients include **built-in retries** for transient network errors.

Use it when you want to train or prototype on JPEGs, masks, audio, JSONL, etc. **immediately**. Switch to [`optimize` → `StreamingDataset`](#speed-up-model-training) later if you need maximum cloud training throughput.

| | `StreamingRawDataset` | `StreamingDataset` (optimized) |
|--|----------------------|--------------------------------|
| Prep | None — point at a folder | One-time `optimize` → `chunk-*.bin` + `index.json` |
| Item | **Raw file `bytes`** (you decide how to decode) | Deserialized samples (dict/tensor/…) |
| I/O | Fully async, batched downloads + retries | Chunk prefetch / cache pipeline |
| Loader | `torch.utils.data.DataLoader` | Prefer `StreamingDataLoader` (shuffle, resume) |
| Best for | Instant start, full control over bytes | Highest sustained training I/O |

### Install (cloud)

```bash
pip install "litdata[extra]" s3fs    # Amazon S3
pip install "litdata[extra]" gcsfs  # Google Cloud Storage
# Azure / Studio connections: see Paths & cloud URLs
```

### Quick start

```python
from torch.utils.data import DataLoader
from litdata import StreamingRawDataset
from PIL import Image
import io

def to_image(data: bytes):
    return Image.open(io.BytesIO(data)).convert("RGB")

dataset = StreamingRawDataset(
    "s3://my-bucket/images/",   # also: gs://, azure://, /teamspace/s3_connections/..., local path
    transform=to_image,         # optional; default yields raw bytes
    storage_options={},         # optional cloud credentials / endpoint
)
loader = DataLoader(dataset, batch_size=32, num_workers=8)

for batch in loader:
    train_step(batch)
```

### Constructor knobs

| Arg | Default | Purpose |
|-----|---------|---------|
| `input_dir` | required | Folder URL/path (same [resolver](#resolve-paths) as optimized streaming) |
| `cache_dir` | LitData default cache | Where the file index (and optional file cache) live |
| `cache_files` | `False` | If `True`, keep downloaded files on disk under `cache_dir` (mirror remote layout) |
| `recompute_index` | `False` | Force re-scan when remote files changed |
| `transform` | `None` | `fn(bytes) -> Any` or `fn(list[bytes]) -> Any` for grouped items |
| `storage_options` | `{}` | Cloud client options |
| `indexer` | `FileIndexer()` | Custom discovery (subclass `BaseIndexer`) |
| `max_concurrent_downloads` | `None` (adaptive) | Per-worker in-flight downloads. `None` = size-aware budget (bandwidth; Little’s-law only for medians &lt;~8 MiB) split across workers; single-process capped at 128. An explicit `int` is used exactly (no silent clamp) |
| `max_prefetch` | `16` | Per-worker sequential look-ahead after each batch (default on). When `num_workers > 1`, effective look-ahead is `min(max_prefetch, 64 // num_workers)` so aggregate stays ~64 items. Pass `0` to disable |
| `prefetch_cache_size` | auto | LRU cap for prefetched items (defaults from `max_prefetch`) |
| `hedge_delay` | `0` | Seconds before a hedged duplicate GET for a slow download (`0` = off, default; opt-in) |
| `range_parallel_threshold` | `0` | Objects ≥ this many bytes use parallel ranged GETs (`0` = whole-object only; opt-in) |
| `item_type` | `"bytes"` | `"bytes"` buffers in RAM; `"path"` returns local cache paths (`cache_files=True` required) |

### Group related files (`setup`)

Default: **one file = one sample**. Override `setup` to filter or group (image + mask, audio + transcript, …). Return either a list of `FileMetadata` or a list of groups (`list[list[FileMetadata]]`).

```python
from collections import defaultdict
from torch.utils.data import DataLoader
from litdata import StreamingRawDataset
from litdata.raw.indexer import FileMetadata

class SegmentationRawDataset(StreamingRawDataset):
    def setup(self, files: list[FileMetadata]) -> list[list[FileMetadata]]:
        # Pair img_001.jpg with img_001.png (mask) by stem
        by_stem: dict[str, dict[str, FileMetadata]] = defaultdict(dict)
        for f in files:
            name = f.path.rsplit("/", 1)[-1]
            stem, _, ext = name.rpartition(".")
            by_stem[stem][ext.lower()] = f
        items = []
        for stem, parts in sorted(by_stem.items()):
            if "jpg" in parts and "png" in parts:
                items.append([parts["jpg"], parts["png"]])
        return items

dataset = SegmentationRawDataset(
    "s3://bucket/seg/",
    transform=lambda pair: (pair[0], pair[1]),  # list[bytes]: [image, mask]
)
loader = DataLoader(dataset, batch_size=16, num_workers=4)
for images, masks in loader:
    ...
```

### Index caching (`index.json.zstd`)

First open scans the tree and writes a compressed file list:

- **Local cache** under your LitData cache dir (fast restart on the same machine)
- **Remote copy** next to the data when possible (e.g. `s3://bucket/files/index.json.zstd`) so every machine skips the scan

```python
# After adding/removing files on the bucket:
dataset = StreamingRawDataset("s3://bucket/files/", recompute_index=True)
```

Do **not** confuse this with optimized LitData’s `index.json` (chunk metadata). Raw indexing only lists files.

### How downloads work

1. DataLoader asks for a batch of indices → `__getitems__`.
2. LitData **asynchronously** downloads those files **in parallel** (`asyncio.gather` + `adownload_fileobj`).
3. Cloud SDKs apply **retries** on transient failures (e.g. S3 adaptive retries).
4. Each item is returned as **`bytes`** (or `list[bytes]` if `setup` grouped files), then optional `transform`.

Your training loop stays normal PyTorch — no async/`await` in user code.

```python
# Default: you own the bytes
dataset = StreamingRawDataset("s3://bucket/files/")
raw: bytes = dataset[0]
# e.g. Image.open(io.BytesIO(raw)), json.loads(raw), np.frombuffer(raw), ...
```

### Tips

- Prefer `num_workers > 0` so worker processes overlap async batch downloads with training. Scale workers toward host vCPUs for network-bound JPEG-sized objects — avoid saturating every vCPU.
- On Linux, after any parent-process dataset I/O, use `DataLoader(..., multiprocessing_context="spawn", persistent_workers=True)` — default `fork` can hang S3 clients in workers.
- Default `max_prefetch=16` enables sequential look-ahead **per DataLoader worker**; shuffled access disables it. Pass `0` to turn off. When `num_workers > 1`, look-ahead and download concurrency both scale down with worker count so aggregate in-flight work stays bounded.
- Prefer an `s3://` / `gs://` URL or `/teamspace/s3_connections/...` so LitData hits the bucket directly ([resolver](#resolve-paths)) — avoid reading through FUSE.
- Leave `range_parallel_threshold=0` (default) for typical JPEGs; raise it only for large objects where parallel ranged GETs help.
- Best for medium/large files. Tiny objects (≲100 KB) are request-overhead bound — pack with [`optimize`](#speed-up-model-training) → `StreamingDataset` when I/O plateaus.

### Throughput

On ImageNet val raw over S3 (50 k JPEGs, batch size 64, spawn workers), throughput gains are clearest at **low worker counts / notebooks** (**+20–80%** at ≤8 workers). At **high workers** (≥16), results are roughly **parity within run-to-run noise**.

| workers | before | after | Δ |
|--------:|-------:|------:|--:|
| 0 | 543 | 735 | **+35%** |
| 2 | 816 | 1475 | **+81%** |
| 8 | 4841 | 5718 | **+18%** |
| 16+ | ~6k | ~6k | ~parity |

Useful knobs: `num_workers`, `max_prefetch` (default 16; worker-aware), `download_timeout` (batch-level hang protection). Ranged parallel downloads stay opt-in (`range_parallel_threshold=0`).

</details>

<details>
  <summary> ✅ Stream large cloud datasets <a id="stream-large" href="#stream-large">🔗</a> </summary>
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
```

Also, you can specify a custom cache directory when initializing your dataset. This is useful when you want to store the cache in a specific location.
```python
from litdata import StreamingDataset

# Initialize the StreamingDataset with the custom cache directory
dataset = StreamingDataset('s3://my-bucket/my-data', cache_dir="/path/to/cache")
```

Any local path, `s3://` / `gs://` / `r2://` / `azure://` / `hf://`, `local:` network drive, or Lightning `/teamspace/...` connection works — see [Resolve any path or cloud URL](#resolve-paths).

</details>

<details>
  <summary> ✅ Optimize images as JPEG (not raw PIL) <a id="optimize-jpeg" href="#optimize-jpeg">🔗</a> </summary>
&nbsp;

How you return images from `optimize` controls storage size and streaming speed.

| What you return | Serializer | Result |
|-----------------|------------|--------|
| `litdata.Image(path=...)` / `Image(array=..., quality=95, format="jpeg")` | `image` | Compressed bytes — **preferred** |
| `litdata.Jpeg(path=...)` / `Jpeg(array=..., quality=95)` | `jpeg` | JPEG bytes |
| `PIL.JpegImageFile` (e.g. `PIL.Image.open("x.jpg")`) | `jpeg` | Compressed bytes |
| Plain `PIL.Image` / `Image.fromarray(...)` | `pil` | Uncompressed pixels — often **10×+ larger** |

**Best practice:** wrap with `Image` / `Jpeg` at **quality ≈ 95**, or keep existing `.jpg` files via `Image(path=...)`. Resize when helpful.

```python
import litdata as ld

def load_image(path):
    return {"image": ld.Image(path=path, quality=95, format="jpeg"), "id": path}

if __name__ == "__main__":
    ld.optimize(fn=load_image, inputs=list_of_paths, output_dir="fast_data", chunk_bytes="64MB", num_workers=8)
```

Ready-made ImageNet optimize/stream scripts: `benchmarks/litdata/` (`--write_mode jpeg --quality 90`).

</details>

<details>
  <summary> ✅ Custom serializers <a id="serializers" href="#serializers">🔗</a> </summary>
&nbsp;

LitData serializes each **pytree leaf** with a pluggable registry. Built-ins (tried in order) include: `str`, `bool`, `int`, `float`, `video`, `audio`, `image`, `nifti`, `mesh`, `pdf`, `tifffile`, `file`, `pil`, `jpeg`, `jpeg_array`, `bytes`, `numpy` / `tensor` (and no-header variants), `graph`, and `pickle` (fallback).

Prefer [typed media wrappers](#media-types) (`Audio`, `Video`, `Image`, `Graph`, …) so a filepath is not confused with a caption. For images, `Image(..., quality=95, format="jpeg")` or a `JpegImageFile` stores JPEG; a plain `PIL.Image` selects **`pil`** (raw pixels). See [Optimize images as JPEG](#optimize-jpeg).

Pass custom serializers when **streaming** (and when using the lower-level `Cache` writer):

```python
from litdata import StreamingDataset
from litdata.streaming.serializers import Serializer

class MyTypeSerializer(Serializer):
    def serialize(self, item):
        return item.to_bytes(), None  # (bytes, optional metadata string)

    def deserialize(self, data: bytes):
        return MyType.from_bytes(data)

    def can_serialize(self, item) -> bool:
        return isinstance(item, MyType)

dataset = StreamingDataset(
    "s3://bucket/data",
    serializers={"my_type": MyTypeSerializer()},  # merged on top of built-ins
)
```

Keys you pass are tried before the defaults (so they win over `pickle`). `optimize()` uses the built-in registry based on the Python types your `fn` returns — prefer typed wrappers / JPEG / numpy / tensor leaves for best results.

</details>

<details>
  <summary> ✅ Stream PyG graphs <a id="pyg-graphs" href="#pyg-graphs">🔗</a> </summary>
&nbsp;

Store [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) `Data` / `HeteroData` as packed tensors (`to_dict()`), not `torch.save` / pickle. On read, LitData reconstructs with `from_dict` when `torch-geometric` is installed. `optimize` needs a **top-level** function (spawn).

`StreamingDataLoader` uses `litdata_collate` by default: graph samples become a `DataBatch` (`Batch.from_data_list`); everything else uses PyTorch `default_collate`. For `follow_batch` / `exclude_keys`, use `torch_geometric.loader.DataLoader`. Without PyG, graph batches stay a list of `Graph`.

### Homogeneous `Data` + GCN

```python
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from litdata import StreamingDataLoader, StreamingDataset, optimize

def make_graph(i: int) -> Data:
    n = 8 + i % 5
    src = torch.randint(0, n, (12,), dtype=torch.long)
    dst = torch.randint(0, n, (12,), dtype=torch.long)
    return Data(
        x=torch.randn(n, 8),
        edge_index=torch.stack([src, dst], 0),
        y=torch.tensor(i % 3),
        train_mask=torch.ones(n, dtype=torch.bool),
        num_nodes=n,
    )

optimize(make_graph, inputs=list(range(1024)), output_dir="graphs", chunk_size=64)

dataset = StreamingDataset("graphs")
sample = dataset[0]  # Data when PyG is installed, else Graph
loader = StreamingDataLoader(dataset, batch_size=32, shuffle=True)
batch = next(iter(loader))  # DataBatch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GCNConv(8, 16)
        self.lin = torch.nn.Linear(16, 3)

    def forward(self, data):
        x = F.relu(self.conv(data.x, data.edge_index))
        return self.lin(global_mean_pool(x, data.batch))
```

### `Graph` wrapper (no PyG at write time)

```python
from litdata import Graph, optimize

def make_graph(i: int) -> Graph:
    n = 6
    return Graph(
        x=torch.randn(n, 4),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        y=torch.tensor(i % 2),
        data={"num_nodes": n},  # extra tensors/scalars; field kwargs override data=
    )

optimize(make_graph, inputs=list(range(256)), output_dir="graphs")
# later: sample.to_pyg()  if the stream returned Graph
```

`Graph(data=pyg_data)` uses `pyg_data.to_dict()`. Do not mix tensor fields with an opaque NetworkX `data=`.

### Heterogeneous `HeteroData`

```python
from torch_geometric.data import HeteroData

from litdata import StreamingDataLoader, StreamingDataset, optimize

def make_hetero(i: int) -> HeteroData:
    data = HeteroData()
    data["paper"].x = torch.randn(8, 16)
    data["author"].x = torch.randn(4, 8)
    data["author", "writes", "paper"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 2, 4, 6]], dtype=torch.long
    )
    data.y = torch.tensor(i % 3)
    return data

optimize(make_hetero, inputs=list(range(512)), output_dir="hetero", chunk_size=32)

dataset = StreamingDataset("hetero")
sample = dataset[0]  # HeteroData
print(sample["paper"].x.shape, sample["author", "writes", "paper"].edge_index.shape)

loader = StreamingDataLoader(dataset, batch_size=16)
batch = next(iter(loader))  # HeteroDataBatch
# batch["paper"].x, batch["paper"].batch, batch["author", "writes", "paper"].edge_index
```

### Graph plus metadata in one sample

```python
def make_row(i: int) -> dict:
    return {"id": i, "graph": make_graph(i)}

optimize(make_row, inputs=list(range(1024)), output_dir="rows")
loader = StreamingDataLoader(StreamingDataset("rows"), batch_size=8)
batch = next(iter(loader))
# batch["id"] is a tensor; batch["graph"] is a DataBatch
```

### Sample subgraphs first, then stream

`NeighborLoader` needs one in-memory graph. To stream, run the sampler in `optimize` and store each subgraph as a `Data`:

```python
# sampler = NeighborSampler(big_graph, num_neighbors=[10, 10])

def sample_seed(seed: int) -> Data:
    out = sampler.sample_from_nodes(torch.tensor([seed]))
    return Data(x=out.x, edge_index=out.edge_index, y=out.y)

optimize(sample_seed, inputs=train_seeds.tolist(), output_dir="subgraphs")
```

NetworkX (or any non-tensor object) uses `Graph(data=nx_graph)` → `graph:pickle`. Do not `torch.save` a graph into the sample.

</details>

<details>
  <summary> ✅ Stream MosaicML MDS datasets <a id="stream-mds" href="#stream-mds">🔗</a> </summary>
&nbsp;

If you already have datasets written in [MosaicML Streaming](https://github.com/mosaicml/streaming) MDS (Mosaic Data Shard) format, you can stream them directly with LitData—no re-optimization or conversion required!

LitData's default `PyTreeLoader` natively understands the MDS binary layout, so you can read existing MDS shards using the familiar `StreamingDataset` and `StreamingDataLoader` APIs.

**Assumption:**

Your dataset directory contains MDS shard files (e.g. `shard.00000.mds`, ...) along with an `index.json` describing the shards and their `column_sizes`/`column_names`.

**Stream the MDS dataset:**

```python
import litdata as ld

# point to your MDS dataset stored locally or in the cloud

mds_dataset_uri = "s3://my-bucket/my-mds-data" # or a local path

# LitData automatically detects and deserializes the MDS format

dataset = ld.StreamingDataset(mds_dataset_uri)

print("Sample", dataset[0])

dataloader = ld.StreamingDataLoader(dataset, batch_size=4)
for sample in dataloader:
  pass
```

**How it works:**

- LitData reads the `format` field from the dataset config. When it's set to `"mds"`, the item loader uses MDS-aware deserialization (`mds_deserialize`) that respects the per-column sizes stored in each shard.
- Fixed-size columns are read directly, while variable-size columns are prefixed with a `uint32` length header—exactly as in the MosaicML MDS spec.
- Each sample is reconstructed into its original Python structure via LitData's `data_spec`.

**Key benefits:**

✅ **Zero conversion:**       Reuse existing MDS shards as-is.    
✅ **Drop-in APIs:**          Use the same `StreamingDataset` / `StreamingDataLoader` you already know.    
✅ **Cloud-native:**          Stream MDS shards directly from S3, GCS, or Azure.    
✅ **Easy migration:**        Move from MosaicML Streaming to LitData without re-optimizing.    

> **Note:** Encrypted data loading is not currently supported for the MDS format.

</details>

<details>
  <summary> ✅ Streams on multi-GPU, multi-node <a id="multi-gpu" href="#multi-gpu">🔗</a> </summary>

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
  <summary> ✅ Shuffle, seed, and drop_last <a id="shuffle" href="#shuffle">🔗</a> </summary>
&nbsp;

Shuffling is **deterministic** and designed for distributed training:

1. Chunks are assigned (and possibly split) across ranks/workers.
2. Items inside each chunk are permuted.

The permutation depends on `seed`, the epoch, and chunk metadata — the same settings always yield the same order (required for resumable `state_dict`).

**Object storage (`s3://`, `gs://`, …)** globally permutes chunks (`FullShuffle`). Random chunk order is cheap once files are already copied into the local cache. Items *inside* a chunk are shuffled as aligned ``batch_decode`` windows. Pass ``item_shuffle_window=256`` (default / ``"auto"``) so blocks are shuffled, then the items inside each block — training still reuses the decode cache. ``item_shuffle_window=0`` (or ``"full"``) restores a full in-chunk permutation. ``LITDATA_ITEM_SHUFFLE_WINDOW`` applies only when the argument is omitted.

**POSIX-fast** (automatic for any local path) mmaps chunks in place. **Vast / NFS / Lustre / GPFS** (and `LITDATA_POSIX_FAST=1`) use `WindowShuffle`: each worker gets **whole chunks** in a sequential stripe, then shuffles chunk order inside a sliding window (default **16**, `LITDATA_POSIX_SHUFFLE_WINDOW`). In-chunk items still use ``item_shuffle_window``. Local disks (ext4/xfs) keep global `FullShuffle`. Object URLs stay on `FullShuffle`. `LITDATA_POSIX_FAST=0` disables in-place mmap.

```python
from litdata import StreamingDataset, StreamingDataLoader

train = StreamingDataset(
    "s3://my-bucket/train",
    shuffle=True,
    drop_last=True,  # keep every rank/worker at the same length (default True under DDP)
    seed=42,         # default is 42; keep stable when resuming
    item_shuffle_window=256,  # 0 / "full" = permute every item in the chunk
)
loader = StreamingDataLoader(train, batch_size=64, num_workers=8)

# shuffle=/drop_last= on the loader override the dataset
loader = StreamingDataLoader(train, batch_size=64, shuffle=True, drop_last=True)
```

**Notes**

- Val/test: usually `shuffle=False`, `drop_last=False`.
- If `drop_last=False` under multi-GPU, LitData warns — collectives can hang when ranks see different lengths.
- Resume with `loader.state_dict()` / `load_state_dict()`. To deliberately ignore checkpointed shuffle settings, set `force_override_state_dict=True` on the dataset.

</details>

<details>
  <summary> ✅ FAQ: chunk size &amp; shuffle before optimize <a id="faq-chunk-shuffle" href="#faq-chunk-shuffle">🔗</a> </summary>
&nbsp;

### What `chunk_bytes` should I use?

Default is **64MB** — a good starting point for typical small/medium samples.

When each datapoint is large (e.g. a few MB), prefer a **larger chunk** (practical range often **256–512MB**) so each chunk holds more samples and **intra-chunk batch randomization** has a bigger pool. Tradeoff: larger chunks take **longer to download** before they can be used.

This is expert guidance (recommended-range mindset), not a published chunk-size sweep.

### Is StreamingDataset shuffle enough if my source data is ordered?

**Not always.** LitData handles **distributed sampling** and **bucket sampling within chunks** automatically (`shuffle=True` randomizes chunk order and item order inside each chunk). That is **not** a substitute for a fully shuffled file-level DataLoader when the source has strong structure (same subject/set contiguous, class blocks, etc.).

If ordered data would make chunked sampling problematic and you cannot embed the grouping as the sample unit:

- Shuffle the list of samples **before** `optimize` so chunks mix well, **or**
- Use [`StreamingRawDataset`](#stream-raw) (per-file random access via a standard PyTorch `DataLoader` with `shuffle=True`) instead of optimize → `StreamingDataset`.

### FUSE vs LitData (Lightning Studios)

`/teamspace/s3_connections` (and related mounts) are **FUSE** — fine for browsing, not for training I/O. Under load they are very slow and can crash. Pass the same path into LitData (`StreamingRawDataset` / `StreamingDataset` / `optimize`): LitData resolves it and talks **directly** to the bucket ([Resolve any path](#resolve-paths)).

Rough ImageNet order-of-magnitude on a Studio (not hard guarantees; right tuning for raw): FUSE hand-read ~**600** images/s · [`StreamingRawDataset`](#stream-raw) ~**6–7k** · optimized [`StreamingDataset`](#speed-up-model-training) (64MB chunks) ~**11k**.

</details>

<details>
  <summary> ✅ StreamingDataset & StreamingDataLoader knobs <a id="streaming-kwargs" href="#streaming-kwargs">🔗</a> </summary>
&nbsp;

**`StreamingDataset`**

| Argument | Default | Description |
|----------|---------|-------------|
| `input_dir` | required | Local path, cloud URI, `Dir`, or parquet path (basename wildcards OK) |
| `cache_dir` | `LITDATA_CACHE_DIR` or `~/.lightning/chunks` | Where chunks are cached |
| `item_loader` | from index | `TokensLoader`, `ParquetLoader`, … |
| `shuffle` | `False` | Deterministic shuffle (see [Shuffle](#shuffle)) |
| `drop_last` | `True` if distributed else `False` | Equal length across ranks |
| `seed` | `42` | Shuffle / subsample RNG |
| `serializers` | built-ins | Custom serialize/deserialize map |
| `max_cache_size` | `None` | Evict consumed chunks beyond this size. Default: 75% of free disk, leaving ≥50GB when possible. Pin with `"100G"` / `"50GB"`, a fraction (`0.90`), or `MAX_CACHE_SIZE`. |
| `max_pre_download` | `2` | Chunks each worker may prefetch (raise for throughput; watch disk / RAM) |
| `subsample` | `1.0` | Fraction of data (`0.01`) or upsample (`2.5`) |
| `encryption` | `None` | `FernetEncryption` / `RSAEncryption` / custom |
| `storage_options` | `{}` | Cloud client options |
| `session_options` | `{}` | boto3 session options (S3) |
| `index_path` | `None` | Parquet/HF `index.json` file or directory |
| `force_override_state_dict` | `False` | Local ctor args override loaded checkpoint |
| `transform` | `None` | Callable or list of callables per sample |

Peak disk ≈ `num_workers × max_pre_download × mean_chunk_size`.

On **Vast / NFS / local disk**, POSIX-fast is on by default (`LITDATA_POSIX_FAST=0` to disable). `WILLNEED` prefetch and `num_workers` are capped when they would exceed about half of `MemAvailable`. Idle **hugepages** (common on GPU nodes) do not count as available RAM — drop unused `nr_hugepages` if `MemAvailable` looks tiny next to `MemTotal`.

**`StreamingDataLoader`**

| Argument | Description |
|----------|-------------|
| All usual `torch.utils.data.DataLoader` kwargs | `batch_size`, `num_workers`, `collate_fn`, `pin_memory`, … |
| `shuffle` / `drop_last` | Forwarded to the streaming dataset |
| `profile_batches` | `int` / `True` / `False` — viztracer worker trace (see [Profile data loading](#profile-loading)) |
| `profile_cprofile` | `True` — stdlib cProfile of the main process + worker 0 (see [Profile data loading](#profile-loading)) |
| `profile_skip_batches` / `profile_dir` | Warm-up skip count; output dir for viztracer / cProfile files |
| `multiprocessing_context` | Use **`"spawn"`** (or `"forkserver"`) with `ParquetLoader` + `num_workers>0` on Linux |

Prefer `StreamingDataLoader` over a plain PyTorch `DataLoader` for optimized / combined / parallel datasets (resume + correct batch metadata).

</details>

<details>
  <summary> ✅ Stream from multiple cloud providers <a id="cloud-providers" href="#cloud-providers">🔗</a> </summary>

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

aws_storage_options={
    "AWS_ACCESS_KEY_ID": os.environ['AWS_ACCESS_KEY_ID'],
    "AWS_SECRET_ACCESS_KEY": os.environ['AWS_SECRET_ACCESS_KEY'],
    "S3_ENDPOINT_URL": os.environ['AWS_ENDPOINT_URL'],  # Required only for custom endpoints
}
dataset = ld.StreamingDataset("s3://my-bucket/my-data", storage_options=aws_storage_options)

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
  <summary> ✅ Pause, resume data streaming <a id="pause-resume" href="#pause-resume">🔗</a> </summary>
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

Same `seed` and `shuffle` are required. For a **`StreamingDataset`**, **`num_workers` and `world_size` may change**: LitData drops a global `sample_in_epoch` prefix and restripes the rest (never duplicates remaining IDs). For a matching loss curve keep **global batch size** (`world_size * batch_size`) constant and DDP ranks in lockstep. `num_canonical_nodes` (default: first-run `world_size`) is frozen in the checkpoint. POSIX `WindowShuffle` resumes whole remaining chunks. **`CombinedStreamingDataset` and `ParallelStreamingDataset` resume only with the same `world_size`, `num_workers`, and `batch_size`.**

```python
dataset = StreamingDataset("s3://my-bucket/my-data", shuffle=True, num_canonical_nodes=8)
```

</details>


<details>
  <summary> ✅ Use shared queue for Optimizing <a id="shared-queue" href="#shared-queue">🔗</a> </summary>
&nbsp;

`optimize` / `map` default to a **shared per-node queue** (`keep_data_ordered=False`). Work is packed per node, then every worker on that node pulls the next item, so a slow worker does not leave others idle. Set `keep_data_ordered=True` to keep a static per-worker slice (required for `use_checkpoint` and `align_chunking`).

Local `output_dir` writes chunks in place. Remote inputs and outputs use the streaming downloader (`adownload_file` / `aupload_file`, obstore when available).

```python
import numpy as np
import litdata as ld

def random_images(index):
    array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    return {
        "index": index,
        "image": ld.Image(array=array, quality=95, format="jpeg"),
        "class": np.random.randint(10),
    }

if __name__ == "__main__":
    # The optimize function writes data in an optimized format.
    ld.optimize(
        fn=random_images,                   # the function applied to each input
        inputs=list(range(1000)),           # the inputs to the function (here it's a list of numbers)
        output_dir="fast_data",             # optimized data is stored here
        num_workers=4,                      # The number of workers on the same machine
        chunk_bytes="64MB" ,                 # size of each chunk
        keep_data_ordered=False,             # default: shared queue (set True to keep input order)
    )
```


### Shared queue vs ordered (skewed local files)

`scripts/bench/bench_node_queue.py --files 4000 --workers 8` (first 500 files are 1 MiB). On `main`, unordered optimize sat on a 200s empty-queue timeout after work finished.

| Tree | Mode | Time | Throughput |
|------|------|-----:|-----------:|
| `main` (old default) | `keep_data_ordered=True` | 23.7s | 169 files/s |
| `main` | `keep_data_ordered=False` | 223.6s | 18 files/s |
| this tree | `keep_data_ordered=True` | 22.9s | 175 files/s |
| this tree (**new default**) | `keep_data_ordered=False` | **18.8s** | 213 files/s |

Shared-queue **before → after: ~12×**. New default vs old ordered default: **1.22×**.

### Local / remote input × output

`python scripts/bench/bench_node_queue.py --files 200 --workers 4 --io-matrix` (first 50 files are 1 MiB).

| Topology | Ordered | Shared | Speedup |
|----------|--------:|-------:|--------:|
| local → local | 6.55s | **2.96s** | 2.21× |
| remote → local | 7.44s | **3.48s** | 2.14× |
| local → remote | 10.98s | **6.51s** | 1.69× |
| remote → remote | 11.48s | **8.55s** | 1.34× |

Shared queue balances uneven workers. It does not change later `StreamingDataset` throughput.

</details>


<details>
  <summary> ✅ Use a <code>Queue</code> as input for optimizing data <a id="queue-input" href="#queue-input">🔗</a> </summary>
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
  <summary> ✅ LLM Pre-training <a id="llm-training" href="#llm-training">🔗</a> </summary>
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
  <summary> ✅ Filter illegal data <a id="filter-data" href="#filter-data">🔗</a> </summary>
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
  <summary> ✅ Combine datasets <a id="combine-datasets" href="#combine-datasets">🔗</a> </summary>
&nbsp;

Mix and match different sets of data to experiment and create better models.

Combine datasets with `CombinedStreamingDataset`.  As an example, this mixture of [Slimpajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) & [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) was used in the [TinyLLAMA](https://github.com/jzhang38/TinyLlama) project to pretrain a 1.1B Llama model on 3 trillion tokens.

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
combined_dataset = CombinedStreamingDataset(
    datasets=train_datasets,
    seed=42,
    weights=weights,
    iterate_over_all=False,  # required when passing weights (see below)
)

train_dataloader = StreamingDataLoader(combined_dataset, batch_size=8, pin_memory=True, num_workers=os.cpu_count())

# Iterate over the combined datasets
for batch in tqdm(train_dataloader):
    pass
```

**`iterate_over_all` vs `weights` (important)**

| Mode | Behavior |
|------|----------|
| `iterate_over_all=True` (default) | Iterate until **all** datasets are exhausted. Do **not** pass `weights` — LitData derives them from dataset lengths (raises `ValueError` if you pass both). |
| `iterate_over_all=False` | Stop when **any** dataset is exhausted. Pass explicit `weights` for your mixture (e.g. TinyLlama). Length may be `None` (variable). |

**Batching Methods** (`batching_method`)

**Stratified** (default): each batch mixes samples from multiple datasets according to the weights.

```python
combined_dataset = CombinedStreamingDataset(
    datasets=[dataset1, dataset2],
    batching_method="stratified",  # default
)
```

**Per-stream**: each batch comes from only one randomly selected dataset (useful when shapes/dtypes differ).

```python
combined_dataset = CombinedStreamingDataset(
    datasets=[dataset1, dataset2],
    batching_method="per_stream",
)
```

Other knobs: `seed` (default `42`), `force_override_state_dict=True` to let local ctor args override a loaded checkpoint.
</details>

<details>
  <summary> ✅ Parallel streaming <a id="parallel-streaming" href="#parallel-streaming">🔗</a> </summary>
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
  <summary> ✅ Cycle datasets <a id="cycle-datasets" href="#cycle-datasets">🔗</a> </summary>
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
  <summary> ✅ Merge datasets <a id="merge-datasets" href="#merge-datasets">🔗</a> </summary>
&nbsp;

Merge multiple optimized datasets into one.

```python
import numpy as np

from litdata import Image, StreamingDataset, merge_datasets, optimize


def random_images(index):
    array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    return {
        "index": index,
        "image": Image(array=array, quality=95, format="jpeg"),
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

If you wrote chunks yourself (`Cache` / `BinaryWriter`) and only have `{rank}.index.json` shards, finish the dataset:

```python
from litdata import complete_dataset, StreamingDataset

complete_dataset("my_chunks")  # no-op if index.json already exists
StreamingDataset("my_chunks")  # also tries this automatically
```

</details>

<details>
  <summary> ✅ Transform datasets while Streaming <a id="transform-streaming" href="#transform-streaming">🔗</a> </summary>
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
  <summary> ✅ Split datasets for train, val, test <a id="split-datasets" href="#split-datasets">🔗</a> </summary>

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

Or pick exact indices with `StreamingDataset.subset`:

```python
train = dataset.subset(range(0, 30_000))
# dataset.subset(slice(0, 1000))
```

</details>

<details>
  <summary> ✅ Load a subset of the remote dataset <a id="load-subset" href="#load-subset">🔗</a> </summary>

&nbsp;
Work on a smaller, manageable portion of your data to save time and resources.


```python
from litdata import StreamingDataset, train_test_split

dataset = StreamingDataset("s3://my-bucket/my-data", subsample=0.01) # data are stored in the cloud

print(len(dataset)) # display the length of your data
# out: 1000

# or a list / slice of global indices
small = dataset.subset([0, 10, 20])
```

</details>

<details>
  <summary> ✅ Upsample from your source datasets <a id="upsample-datasets" href="#upsample-datasets">🔗</a> </summary>

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
  <summary> ✅ Easily modify optimized cloud datasets <a id="modify-datasets" href="#modify-datasets">🔗</a> </summary>
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
  <summary> ✅ Stream parquet datasets <a id="stream-parquet" href="#stream-parquet">🔗</a> </summary>
&nbsp;

Stream existing Parquet files with LitData **without** converting them to LitData chunks — or convert them when you need LitData’s optimized binary format. Hugging Face Hub datasets are covered in [Hugging Face Datasets](#huggingface-datasets).

### Stream vs optimize vs map

| Goal | Use |
|------|-----|
| Train on parquet as-is (no conversion) | `index_parquet_dataset` → `StreamingDataset` + `ParquetLoader` |
| Faster I/O / tokenize / custom sample shape | `optimize(fn)` that `yield`s rows from parquet ([reduce memory](#reduce-memory)) |
| Reshard huge parquet files while mapping | `map(..., reader=ParquetReader(cache_folder, num_rows=...))` |

Each sample from `ParquetLoader` is a **`dict`** (column name → value).

### Prerequisites

```bash
pip install 'litdata[extras]'   # includes polars + pyarrow
# Cloud listing/index extras as needed:
pip install s3fs    # s3://
pip install gcsfs   # gs://
```

### Index a parquet directory

```python
import litdata as ld

ld.index_parquet_dataset(
    "s3://my-bucket/my-parquet-data",  # local path, s3://, gs://, or hf://
    cache_dir=None,                   # see table below
    storage_options={},               # cloud credentials / endpoints
    num_workers=4,                    # parallel metadata reads
)
```

| Scheme | Where `index.json` is written |
|--------|-------------------------------|
| Local directory | Next to the files, or under `cache_dir` if set |
| `s3://` / `gs://` | **Uploaded to the bucket** at `{url}/index.json` (needs write access) |
| `hf://` | **Local** `cache_dir` (required for HF indexing via this helper) |

**Indexing notes**

- Lists **top-level** `.parquet` files only (not recursive subfolders).
- All files must share the same schema.
- Supported for indexing today: local, `s3://`, `gs://`, `hf://` (not `r2://` / `azure://` yet).
- For Hub datasets, use `StreamingDataset("hf://...")` — see [Hugging Face Datasets](#huggingface-datasets).

### Stream with `ParquetLoader`

If the folder looks like parquet and has no `index.json`, `StreamingDataset` now **builds the index automatically**. You still need `ParquetLoader` for local/S3/GCS (it must match `index.json`). `hf://` already auto-indexes and selects the loader.

```python
import litdata as ld
from litdata.streaming.item_loader import ParquetLoader

uri = "s3://my-bucket/my-parquet-data"
dataset = ld.StreamingDataset(
    uri,
    item_loader=ParquetLoader(low_memory=True),  # default: row-group streaming
    # index_path="/path/to/index.json",         # optional if index lives elsewhere
)

# Basename wildcards when the path ends with .parquet:
# dataset = ld.StreamingDataset("s3://bucket/data/train-*.parquet", item_loader=ParquetLoader())

print(dataset[0])  # dict of columns

# Linux + num_workers>0: use spawn (Polars + fork deadlocks)
dataloader = ld.StreamingDataLoader(
    dataset,
    batch_size=4,
    num_workers=4,
    multiprocessing_context="spawn",
)
for batch in dataloader:
    pass
```

### `ParquetLoader` knobs

| Arg | Default | Meaning |
|-----|---------|---------|
| `low_memory` | `True` | Stream by row group (lower RAM). `False` loads each whole file into memory (warns). |
| `pre_load_chunk` | `False` | Prefetch full DataFrame — **only effective when `low_memory=False`**. |

Import: `from litdata.streaming.item_loader import ParquetLoader` (not re-exported at `litdata` top level).

### Reshard parquet for `map` / `optimize`

```python
from litdata import map
from litdata.processing.readers import ParquetReader

def process(pq_file, output_dir):
    # pq_file is a pyarrow.parquet.ParquetFile
    ...

map(
    fn=process,
    inputs=list_of_parquet_paths,
    output_dir="s3://bucket/out",
    reader=ParquetReader(cache_folder="/tmp/pq-shards", num_rows=65536),
)
```

`ParquetReader` splits inputs that exceed `num_rows` into smaller cached files before your `fn` runs.

</details>

<details>
  <summary> ✅ Use compression <a id="compression" href="#compression">🔗</a> </summary>
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

`compression="zstd"` (default `compression_level="batch"`, `compression_batch_size=256`) keeps a `.bin` with an uncompressed directory plus zstd frames of 256 items — the same window as Arrow IPC record batches and aligned decode. `compression_level="chunk"` wraps each pytree chunk as one whole-file `.zstd.bin`. `compression_level="sample"` zstd-compresses each item between the pytree offsets. Numeric zstd levels are `compression="zstd:4"`, not `compression_level`. Nested JSON / Hub Arrow IPC still uses Arrow’s own per-batch zstd when you pass `"zstd"`.


</details>

<details>
  <summary> ✅ Access samples without full data download <a id="access-samples" href="#access-samples">🔗</a> </summary>
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
  <summary> ✅ Use any data transforms <a id="data-transforms" href="#data-transforms">🔗</a> </summary>
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
  <summary> ✅ Profile data loading speed <a id="profile-loading" href="#profile-loading">🔗</a> </summary>
&nbsp;

`StreamingDataLoader` can record a **viztracer** Chrome trace of the DataLoader worker loop so you can see where time goes (fetch, deserialize, collate, IPC).

### Prerequisites

```bash
pip install viztracer
```

Profiling requires **`num_workers >= 1`** (raises otherwise). On multi-GPU, only **global rank 0** installs the worker profiler.

### Usage

```python
from litdata import StreamingDataset, StreamingDataLoader

dataset = StreamingDataset("s3://my-bucket/my-data", shuffle=True, drop_last=True)

loader = StreamingDataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    profile_batches=20,          # record this many batches (int), or True for the whole run
    profile_skip_batches=5,      # warm up / skip cold-start batches before recording
    profile_dir="./profiles",    # where to write result.json (default: cwd)
)

for batch in loader:
    train_step(batch)
    # after profile_batches (+ skip) complete, worker 0 saves the trace and prints the path
```

| Arg | Default | Meaning |
|-----|---------|---------|
| `profile_batches` | `False` | `int` → stop after that many **recorded** batches; `True` → profile until the iterator ends; `False` → off |
| `profile_skip_batches` | `0` | Batches to skip before the tracer starts (useful to skip cache cold-start) |
| `profile_dir` | current working directory | Directory for `result.json` (overwrites an existing file) |

Only **worker 0** is instrumented. When an `int` is used, the tracer wraps `fetcher.fetch` and stops after `profile_skip_batches + profile_batches` fetch calls. When `True`, tracing runs for the lifetime of that worker loop.

### View the trace

```bash
# Option A — Chrome
# open chrome://tracing and load profiles/result.json

# Option B — Perfetto (often better for large traces)
# open https://ui.perfetto.dev and load the same file
```

### Tips

- Delete or change `profile_dir` between runs — LitData removes an existing `result.json` before starting.
- Pair with a wiped chunk cache if you care about **cold** epoch behavior (`litdata cache clear`).
### cProfile (main + worker 0)

Stdlib statistical profile of **the parent process** (queue wait, unpickle, collate handoff) and **worker 0** (fetch, decode, transforms). No extra package.

```python
loader = StreamingDataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    profile_cprofile=True,
    profile_dir="./profiles",
)

for batch in loader:
    train_step(batch)
# writes profiles/cprofile_main.prof + cprofile_worker0.prof (and .txt summaries)
```

```bash
python -m pstats profiles/cprofile_worker0.prof
# then: sort tottime   stats 30
```

Do not set `profile_cprofile` and `profile_batches` together — both install a `sys.setprofile` hook. With `num_workers=0` only the main file is written. The parent profiler starts **after** workers spawn so fork does not inherit an active cProfile.

- For deeper LitData internals (download / read / delete timeline), use `enable_tracer()` + [Litracer](https://github.com/Lightning-AI/litracer) instead — see [Debug & Profile LitData](#debug-profile). That path is complementary: viztracer = DataLoader worker CPU timeline; cProfile = function self/cum time; Litracer = LitData pipeline events.

</details>

<details>
  <summary> ✅ Reduce memory use for large files <a id="reduce-memory" href="#reduce-memory">🔗</a> </summary>
&nbsp;

Handle large data files efficiently without using too much of your computer's memory.

**Optimize from parquet** (convert into LitData chunks) when you need tokenization or LitData’s binary format. To **stream parquet without converting**, see [Stream parquet datasets](#stream-parquet).

When processing large parquet files, `yield` one item at a time to keep memory low:

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
  <summary> ✅ Limit local cache space <a id="limit-cache" href="#limit-cache">🔗</a> </summary>
&nbsp;

Control how much disk the local chunk cache may use. Downloaded chunks are deleted after use once the cache exceeds the limit.

Default `max_cache_size` is **`None`**: LitData uses **75% of currently free disk** and still leaves **≥50GB** free when the volume is large enough for checkpoints. Pass `"100G"` / `"50GB"` for a fixed budget, or a float (`0.90`) for that fraction of currently free space. `MAX_CACHE_SIZE` overrides the constructor (size or fraction).

Peak disk in flight is roughly:

```
num_workers × max_pre_download × mean_chunk_size
```

Keep `max_cache_size` comfortably above that peak. For remote datasets, async chunk prefetch often raises `max_pre_download` to **≥4** automatically — see [async prefetch & environment variables](#async-prefetch-env).

```python
from litdata import StreamingDataset

dataset = StreamingDataset(
    "s3://my-bucket/my-data",
    max_cache_size="10GB",
    max_pre_download=4,  # chunks each worker may prefetch (default 2; async may floor to 4)
)
```

</details>

<details>
  <summary> ✅ Async chunk prefetch & environment variables <a id="async-prefetch-env" href="#async-prefetch-env">🔗</a> </summary>
&nbsp;

### Async chunk prefetch

LitData can overlap **remote chunk downloads** with training using `asyncio` inside each DataLoader worker’s prepare thread. This is **not** an async DataLoader — your loop stays:

```python
for batch in StreamingDataLoader(dataset, batch_size=64, num_workers=8):
    train_step(batch)
```

| Situation | Async prefetch |
|-----------|----------------|
| Remote dataset (`s3://`, `gs://`, …) | **On** by default |
| Local-only dataset | **Off** by default |
| `LITDATA_ASYNC_CHUNK_PREFETCH=1` | Force on |
| `LITDATA_ASYNC_CHUNK_PREFETCH=0` | Force off |

When async is on, LitData raises `max_pre_download` to at least **4** so `asyncio.gather` has enough in-flight downloads (override with `LITDATA_ASYNC_MIN_PRE_DOWNLOAD`; set `0` to disable the floor). Concurrent GETs are capped by `LITDATA_ASYNC_DOWNLOAD_CONCURRENCY` (default **8**). The prepare thread drains the prefetch queue up to that gather width whenever a cache slot is free. Peak disk ≈ `num_workers × max_pre_download × chunk_size` — size `max_cache_size` accordingly.

The reader does **not** poll the cache directory for each chunk. After a download finishes, the downloader atomically `os.replace`s the temp file and sets an in-process `Event`. Compressed chunks set that Event only after decompress publishes the readable `.bin`. Other DataLoader workers still fall back to a short filesystem check (Events are per process).

```bash
# Debugging download/delete races — force synchronous downloads
export LITDATA_ASYNC_CHUNK_PREFETCH=0

# Keep max_pre_download=2 even with async enabled
export LITDATA_ASYNC_MIN_PRE_DOWNLOAD=0

# Cap overlapping remote GETs (default 8)
export LITDATA_ASYNC_DOWNLOAD_CONCURRENCY=4
```

### Common environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LITDATA_CACHE_DIR` | `~/.lightning/chunks` | Default chunk cache directory |
| `LITDATA_ITEM_SHUFFLE_WINDOW` | `256` | In-chunk shuffle block size when `item_shuffle_window` is omitted (`0` = full permute) |
| `LITDATA_ASYNC_CHUNK_PREFETCH` | on for remote | `0`/`1` force async chunk download overlap |
| `LITDATA_ASYNC_MIN_PRE_DOWNLOAD` | `4` | Floor for `max_pre_download` when async is on (`0` = no floor) |
| `LITDATA_ASYNC_DOWNLOAD_CONCURRENCY` | `8` | Max in-flight chunk GETs per gather |
| `LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB` | `8` | S3 obstore stream chunk size (MiB) |
| `MAX_WAIT_TIME` | `120` | Seconds to wait for a chunk before error |
| `FORCE_DOWNLOAD_TIME` | `30` | Seconds before force re-download of a missing chunk |
| `LITDATA_CHECK_UPDATES` | unset | `1` enables the PyPI upgrade tip (off by default) |
| `LITDATA_DISABLE_VERSION_CHECK` | on unless updates enabled | `1` skips the upgrade tip |
| `HF_TOKEN` | — | Gated Hugging Face datasets |
| `DEBUG_LITDATA` / `PRINT_DEBUG_LOGS` | `0` | Internal debug / stdout logs |
| `LITDATA_LOG_FILE` | `litdata_debug.log` | `enable_tracer()` output path |
| `LITDATA_TRACE_LEVEL` | unset | `batch` / `chunk` / `sample` / `debug` / `off` (see [Debug & Profile](#debug-profile)) |
| `LITDATA_TRACE_CATEGORIES` | from level | Comma-separated cats, e.g. `download,read,delete` |

Multi-node `optimize`/`map` on Studios also uses `DATA_OPTIMIZER_*` (set by the platform). Full catalog (debug logs, Studio injects, torchrun): see the LitData skill `reference/env-vars.md` when using agent skills, or the source modules `constants.py` / `async_prefetch.py`.

</details>

<details>
  <summary> ✅ Change cache directory path <a id="cache-directory" href="#cache-directory">🔗</a> </summary>
&nbsp;

Specify where cached chunk files are stored.

```python
from litdata import StreamingDataset
from litdata.streaming.cache import Dir

# Simple: dedicated cache directory
dataset = StreamingDataset("s3://my-bucket/my_optimized_dataset", cache_dir="/path/to/your/cache")

# Or when cache path and remote URL should differ:
dataset = StreamingDataset(input_dir=Dir(path="/path/to/your/cache", url="s3://my-bucket/my_optimized_dataset"))
```

Global default without passing `cache_dir` every time:

```bash
export LITDATA_CACHE_DIR=/path/to/your/cache
```

CLI:

```bash
litdata cache path    # print the active cache directory
litdata cache clear   # delete cached chunks
```

</details>

<details>
  <summary> ✅ Optimize loading on networked drives <a id="networked-drives" href="#networked-drives">🔗</a> </summary>
&nbsp;

Optimize data handling for computers on a local network to improve performance for on-site setups.

On-prem compute nodes can mount and use a network drive. A network drive is a shared storage device on a local area network. In order to reduce their network overload, the `StreamingDataset` supports `caching` the data chunks.

```python
from litdata import StreamingDataset

dataset = StreamingDataset(input_dir="local:/data/shared-drive/some-data")
```

</details>

<details>
  <summary> ✅ Optimize / map across multiple machines (Lightning Studios) <a id="distributed-optimization" href="#distributed-optimization">🔗</a> </summary>
&nbsp;

On [Lightning Studios](https://lightning.ai/), `num_nodes` and `machine` scale `optimize` / `map` across many machines. This is **not** the same as `num_workers` (processes on one machine).

**How it works**

1. You call `optimize(..., num_nodes=N, machine=...)` (or `map`) inside a Studio.
2. LitData starts a **data-prep job** that re-runs your script on **N** machines.
3. Each machine processes a shard of the inputs (`num_nodes × num_workers` total workers). The last node merges chunk indexes into a single `index.json`.
4. Your local call blocks until the job finishes; open the printed Runs URL to monitor.

Outside Studio, passing `num_nodes` / `machine` raises an error (create a Studio account to use multi-node).

```python
from litdata import optimize, Machine

def compress(index):
    return (index, index ** 2)

if __name__ == "__main__":
    optimize(
        fn=compress,
        inputs=list(range(100)),
        num_workers=8,              # processes per machine
        output_dir="/teamspace/s3_connections/my-data/optimized-v1",  # durable bucket (recommended)
        chunk_bytes="64MB",
        num_nodes=32,               # machines in the job
        machine=Machine.DATA_PREP,  # or omit to use the current Studio machine type
    )
```

**Where outputs land**

| `output_dir` | Result |
|--------------|--------|
| `/teamspace/s3_connections/...`, `/teamspace/datasets/...`, `s3://...`, `gs://...` | Written directly to that store (**recommended**) |
| Local or `/teamspace/studios/this_studio/...` | Remapped to the job’s **artifacts** storage; the Studio UI may also expose it under `/teamspace/jobs/<job>/...` |

```python
from litdata import StreamingDataset

# Prefer the same connection / cloud URL you wrote to:
dataset = StreamingDataset("/teamspace/s3_connections/my-data/optimized-v1")
```

The same `num_nodes` / `machine` pattern works with `map`. See also [Parallelize transforms and data optimization](#parallelize-transforms-and-data-optimization-on-cloud-machines).

</details>

<details>
  <summary> ✅ Encrypt, decrypt data at chunk/sample level <a id="encrypt-decrypt" href="#encrypt-decrypt">🔗</a> </summary>
&nbsp;

Encrypt optimized data at **sample** or **chunk** level. Built-ins: `FernetEncryption` and `RSAEncryption` (`litdata.utilities.encryption`). Requires the `cryptography` package. **Not supported for Mosaic MDS.**

| `level` | Meaning |
|---------|---------|
| `"sample"` (default) | Encrypt each sample independently |
| `"chunk"` | Encrypt whole chunks |

**Fernet (symmetric)**

```python
from litdata import optimize, StreamingDataset
from litdata.utilities.encryption import FernetEncryption

fernet = FernetEncryption(password="your_secure_password", level="sample")  # or level="chunk"
data_dir = "s3://my-bucket/optimized_data"

def fn(index):
    return {"index": index, "value": index**2}

if __name__ == "__main__":
    optimize(
        fn=fn,
        inputs=list(range(5)),
        num_workers=1,
        output_dir=data_dir,
        chunk_bytes="64MB",
        encryption=fernet,
    )
    fernet.save("fernet.pem")  # persist salt/level; keep the password safe

# Later — load key material with the same password
fernet = FernetEncryption.load("fernet.pem", password="your_secure_password")
ds = StreamingDataset(input_dir=data_dir, encryption=fernet)
```

**RSA (asymmetric)**

```python
from litdata.utilities.encryption import RSAEncryption

rsa = RSAEncryption(password="your_secure_password", level="sample")  # or "chunk"
optimize(fn=fn, inputs=list(range(5)), output_dir=data_dir, chunk_bytes="64MB", encryption=rsa)
rsa.save("rsa.pem")

rsa = RSAEncryption.load("rsa.pem", password="your_secure_password")
ds = StreamingDataset(input_dir=data_dir, encryption=rsa)
```

**Custom algorithm** — subclass `Encryption` and implement `encrypt` / `decrypt` / `save` / `load` / `state_dict` / `algorithm`.

</details>

<details>
  <summary> ✅ Debug & Profile LitData with logs & Litracer <a id="debug-profile" href="#debug-profile">🔗</a> </summary>

&nbsp;

`enable_tracer()` records the streaming pipeline (download vs read vs delete vs batch) as one-line events. [Litracer](https://github.com/Lightning-AI/litracer) converts that log into a Chrome / [Perfetto](https://ui.perfetto.dev) trace.

This is complementary to [`profile_batches`](#profile-loading) (viztracer = DataLoader worker **CPU**; Litracer = LitData **pipeline** events).

<img width="1439" alt="431247797-0e955e71-2f9a-4aad-b7c1-a8218fed2e2e" src="https://github.com/user-attachments/assets/4e40676c-ba0b-49af-acac-975977173669" />

```python
import litdata as ld
from litdata.debugger import enable_tracer

# Call once per process, before the DataLoader. Delete an existing log before re-tracing (append).
enable_tracer(level="chunk", log_file="litdata_debug.log")
# level="batch" | "chunk" (default) | "sample" | "debug" | "off"
# enable_tracer(categories=["download", "read", "delete"])

if __name__ == "__main__":
    dataset = ld.StreamingDataset("s3://my-bucket/my-data", shuffle=True)
    for batch in ld.StreamingDataLoader(dataset, batch_size=64, num_workers=8):
        ...
```

| Level | Events |
| ----- | ------ |
| `batch` | Epoch + per-batch spans, plus crashes |
| `chunk` (default) | + `download`, `read`, `delete`, `decompress`, `prefetch` |
| `sample` | + per-item `__getitem__` (high volume) |
| `debug` | + `.cnt` lock refcount spans |
| `off` | Disable |

Event **names** are stable (`download`, `read`, `delete`, `batch`, `sample`, `crash`). Chunk / sample indexes live in args so Perfetto groups all downloads together. Each line is `key: value;` pairs with Chrome **microsecond** timestamps. Crashes are a one-line instant (`ph: I`, `name: crash`); the Python traceback is printed to **stderr**, not the log file (a multi-line `logger.exception` would break Litracer).

Env overrides: `LITDATA_LOG_FILE`, `LITDATA_TRACE_LEVEL`, `LITDATA_TRACE_CATEGORIES` (comma-separated). Tracer calls are no-ops when tracing is off.

1. Generate the log:

    ```bash
    python train.py   # writes litdata_debug.log
    ```

2. Install [Litracer](https://github.com/Lightning-AI/litracer) (Go 1.23+):

    ```bash
    git clone https://github.com/Lightning-AI/litracer.git
    cd litracer && go build -o litracer .
    ```

    Or `go install github.com/deependujha/litracer@latest` (published Go module path). Until `go.mod` is renamed, `go install github.com/Lightning-AI/litracer@latest` does not work. Release binaries: [GitHub Releases](https://github.com/Lightning-AI/litracer/releases).

3. Convert and open in Perfetto:

    ```bash
    litracer --quiet --validate -o litdata_trace.json.gz litdata_debug.log
    litracer --quiet --cat download,read,delete -o io.json.gz litdata_debug.log
    # open the .json.gz at https://ui.perfetto.dev (preferred) or chrome://tracing
    ```

    `--quiet` prints a one-line summary (per-category durations, unmatched B/E, crashes). `--cat` keeps only those categories. Matched B/E pairs become complete (`ph: X`) spans unless `--no-complete`. Default output is gzip Chrome JSON (`.json.gz`) — both Perfetto and `chrome://tracing` open it; pass `-o file.json` for uncompressed.

- For trace files `> 2GB`, see [Perfetto large traces](https://perfetto.dev/docs/visualization/large-traces).
- If you connect Perfetto to the RPC server, prefer Chrome over Brave (Brave often does not autodetect the RPC server).

**Multi-worker `s3://` `FileNotFoundError` after ~120s:** `num_workers=0` working while `num_workers>0` fails usually means the DataLoader parent started obstore (tokio) before fork and worker GETs hung. Current LitData fetches `index.json` with boto3 so workers can lazy-init obstore; they fall back to boto3 if the parent already started the runtime. On Studio R2 / `lightning_storage`, the same symptom can be a prefetch-thread crash (`data_connection_id` / `endpoint_url` into `boto3.Session`) — look for `[litdata] PrepareChunksThread CRASHED` on stderr and a `crash` instant in the trace.

</details>

<details>
  <summary> ✅ Resolve any path or cloud URL (local, S3, GCS, R2, Azure, HF, Studio) <a id="resolve-paths" href="#resolve-paths">🔗</a> </summary>

&nbsp;

LitData **resolves** every dataset path you pass to `StreamingDataset`, `StreamingRawDataset`, `optimize`, `map`, and related APIs. You write one path string; LitData figures out whether to read locally, download from object storage, or (inside [Lightning Studios](https://lightning.ai/)) talk **directly to the bucket** behind a `/teamspace/...` mount instead of going through slow FUSE I/O.

### Supported URI schemes

| Scheme | Example | Use when |
|--------|---------|----------|
| Local path | `./data` or `/data/imagenet` | Files on disk |
| `s3://` | `s3://my-bucket/optimized` | AWS S3 |
| `gs://` | `gs://my-bucket/optimized` | Google Cloud Storage |
| `r2://` | `r2://my-bucket/optimized` | Cloudflare R2 |
| `azure://` | `azure://container/optimized` | Azure Blob Storage |
| `hf://` | `hf://datasets/org/name/data` | Hugging Face datasets (parquet) |
| `local:` | `local:/mnt/nfs/dataset` | Network / shared drive (LitData still caches chunks locally to reduce NAS load) |

```python
from litdata import StreamingDataset, optimize

# Same APIs — only the path changes
StreamingDataset("s3://my-bucket/fast_data", shuffle=True, drop_last=True)
StreamingDataset("gs://my-bucket/fast_data")
StreamingDataset("r2://my-bucket/fast_data", storage_options={...})
StreamingDataset("azure://my-container/fast_data", storage_options={...})
StreamingDataset("hf://datasets/org/name/data")
StreamingDataset("local:/data/shared-drive/some-data")
StreamingDataset("/var/data/fast_data")  # plain local directory
```

Pass cloud credentials with `storage_options` (and optional `session_options` for boto3 profiles/regions). See [Stream from multiple cloud providers](#cloud-providers).

### Cache directory vs remote URL

By default LitData caches downloaded chunks under `~/.lightning/chunks` (override with `cache_dir=` or `LITDATA_CACHE_DIR`). When the cache location and the dataset URL must differ, use `Dir`:

```python
from litdata import StreamingDataset
from litdata.streaming.resolver import Dir

dataset = StreamingDataset(
    Dir(path="/fast-ssd/cache/run-1", url="s3://my-bucket/fast_data")
)
# Equivalent:
dataset = StreamingDataset("s3://my-bucket/fast_data", cache_dir="/fast-ssd/cache/run-1")
```

```bash
export LITDATA_CACHE_DIR=/fast-ssd/cache
litdata cache path    # show active cache directory
litdata cache clear   # wipe cached chunks
```

### Date/time path templates

Embed a `strftime` pattern in `{...}` and LitData expands it to the current time (useful for versioned `output_dir`s):

```python
# e.g. on 2025-05-05 → ".../run_2025-05-05"
optimize(
    fn=fn,
    inputs=inputs,
    output_dir="s3://my-bucket/datasets/run_{%Y-%m-%d}",
    chunk_bytes="64MB",
)
```

### Lightning Studio `/teamspace/...` paths (direct bucket I/O)

In Lightning Studios, data connections appear under `/teamspace/...`. **Prefer these paths in LitData** — optimize/map uploads and StreamingDataset downloads use the **backing object store URL** (and temporary credentials when needed), which is much faster than reading every file through the FUSE mount.

| Path prefix | What LitData does |
|-------------|-------------------|
| `/teamspace/studios/this_studio/...` | Local Studio workspace disk (not a cloud URL) |
| `/teamspace/studios/<other_studio>/...` | Resolves to that Studio’s content bucket (`s3://` or `gs://`) |
| `/teamspace/s3_connections/<name>/...` | Direct S3 to the connection’s bucket |
| `/teamspace/gcs_connections/<name>/...` | Direct GCS |
| `/teamspace/s3_folders/<name>/...` | S3 folder connection |
| `/teamspace/gcs_folders/<name>/...` | GCS folder connection |
| `/teamspace/lightning_storage/<name>/...` | Lightning-managed object storage (R2-style) |
| `/teamspace/datasets/...` | Teamspace datasets mount → project datasets bucket |

```python
from litdata import StreamingDataset, StreamingRawDataset, optimize

# Stream optimized data from an attached S3 connection (direct bucket download)
dataset = StreamingDataset("/teamspace/s3_connections/my-data-1/fast_data", shuffle=True, drop_last=True)

# Stream raw files from a connection
raw = StreamingRawDataset("/teamspace/s3_connections/my-bucket-1/raw")

# Optimize *into* a connection — chunks upload straight to the bucket
def should_keep(data):
    if data % 2 == 0:
        yield data

if __name__ == "__main__":
    optimize(
        fn=should_keep,
        inputs=list(range(1000)),
        output_dir="/teamspace/s3_connections/my-data-1/output",
        chunk_bytes="64MB",
        num_workers=1,
    )
```

**Tips**

- Version remote outputs (`.../v2`, `.../run_{%Y-%m-%d}`). Optimized datasets are immutable unless you pass `mode="append"` or `mode="overwrite"`.
- Outside Studio, use `s3://` / `gs://` / … with your own credentials — `/teamspace/...` resolution needs Lightning Studio environment variables.
- `optimize` / `map` with `num_nodes` launch a Studio **job** (not local multi-process). Prefer a connection / cloud `output_dir`; local / `this_studio` optimize outputs go to job artifacts (UI may show `/teamspace/jobs/...`). Details: [distributed optimization](#distributed-optimization).

</details>

&nbsp;


## Features for transforming datasets

<details>
  <summary> ✅ Parallelize data transformations (map) <a id="map" href="#map">🔗</a> </summary>
&nbsp;

Apply the same change to different parts of the dataset at once to save time and effort.

The `map` operator applies a function over a list of inputs. **`fn` must write into `output_dir` and return `None`.** Guard with `if __name__ == "__main__"` when using multiple workers.

```python
import os
from litdata import map
from PIL import Image

input_dir = "my_large_images"  # or s3://...
inputs = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

def resize_image(image_path, output_dir):
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    Image.open(image_path).resize((224, 224)).save(output_image_path)

if __name__ == "__main__":
    map(
        fn=resize_image,
        inputs=inputs,
        output_dir="s3://my-bucket/my_resized_images",
        num_workers=8,
    )
```

**`map` arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `fn` | required | `fn(input, output_dir) -> None` |
| `inputs` | required | Sequence or `StreamingDataLoader` |
| `output_dir` | required | Local or cloud path ([resolver](#resolve-paths)) |
| `input_dir` | `None` | Root for remote inputs (background download while processing) |
| `weights` | `None` | Per-input weights to balance workers |
| `num_workers` | CPU count | Local process workers |
| `fast_dev_run` | `False` | Process only a few items (`True` → small default, or an int) |
| `num_nodes` / `machine` | `None` | Scale out on [Lightning Studios](https://lightning.ai/) |
| `num_downloaders` / `num_uploaders` | auto | I/O concurrency per worker |
| `reorder_files` | `True` | Pack by file size for balance; `False` preserves order |
| `error_when_not_empty` | `False` | Error if `output_dir` already has files |
| `reader` | default | Custom reader for inputs |
| `batch_size` | `None` | Group inputs into batches for `fn` |
| `start_method` | spawn† | Multiprocessing start method (†spawn unless IPython) |
| `optimize_dns` | `None` | Optimized DNS (Studio / cloud) |
| `storage_options` | `{}` | Cloud credentials / endpoints |
| `keep_data_ordered` | `False` | Shared work queue (faster for uneven workers). `True` keeps a static per-worker slice. Forced `True` with `use_checkpoint` / `align_chunking`. |

</details>

<details>
  <summary> ✅ <code>optimize</code> arguments reference <a id="optimize-kwargs" href="#optimize-kwargs">🔗</a> </summary>
&nbsp;

Full knob list for `litdata.optimize` (see Quick start for the minimal recipe). **Exactly one of `chunk_bytes` or `chunk_size`.** Use `if __name__ == "__main__"`.

| Argument | Default | Description |
|----------|---------|-------------|
| `fn` | required | Maps each input → sample (or `yield` samples / skip bad ones) |
| `inputs` | `None` | Sequence or `StreamingDataLoader` (ignored if `queue` is set) |
| `queue` | `None` | `multiprocessing.Queue` of live inputs; send **one** `ALL_DONE` when finished |
| `output_dir` | `"optimized_data"` | Local or cloud ([resolver](#resolve-paths)); version remote prefixes |
| `input_dir` | `None` | Remote input root for background download |
| `weights` | `None` | Per-input weights to balance workers |
| `chunk_bytes` | `None` | Max bytes per chunk (e.g. `"64MB"`; see [FAQ](#faq-chunk-shuffle) for larger samples) |
| `chunk_size` | `None` | Max items (or tokens with `TokensLoader`) per chunk |
| `align_chunking` | `False` | Match single-worker chunk boundaries (needs `chunk_size`; uneven load) |
| `compression` | `None` | `"zstd"` or `"zstd:N"` (numeric level). Pair with `compression_level` for pytree wrap |
| `compression_level` | `"batch"` (zstd) | `"batch"` framed `.bin`; `"chunk"` whole-file `.zstd.bin`; `"sample"` per-item zstd |
| `compression_batch_size` | `256` | Items per frame when `compression_level="batch"` (matches Arrow IPC / decode windows) |
| `encryption` | `None` | `FernetEncryption` / `RSAEncryption` / custom ([encrypt](#encrypt-decrypt)) |
| `num_workers` | CPU count | Local workers |
| `fast_dev_run` | `False` | Smoke a subset of inputs |
| `num_nodes` / `machine` | `None` | Multi-node on Lightning Studios |
| `num_downloaders` / `num_uploaders` | auto | I/O concurrency per worker |
| `reorder_files` | `True` | Size-based packing; `False` preserves order |
| `reader` | default | Custom input reader |
| `batch_size` | `None` | Group inputs for `fn` |
| `mode` | `None` | `"append"` or `"overwrite"` existing dataset; default treats data as immutable |
| `use_checkpoint` | `False` | Resume an interrupted optimize from `.checkpoints` |
| `item_loader` | `None` | e.g. `TokensLoader()` for contiguous tokens |
| `start_method` | spawn† | Multiprocessing start method |
| `optimize_dns` | `None` | Optimized DNS |
| `storage_options` | `{}` | Cloud credentials / endpoints |
| `keep_data_ordered` | `False` | Shared queue among workers. `True` keeps input order. Forced `True` with `use_checkpoint` / `align_chunking`. |
| `verbose` | `True` | Progress logging |

Related features: [shared queue](#shared-queue), [queue input](#queue-input), [append/overwrite](#modify-datasets), [compression](#compression), [TokensLoader / LLM](#llm-training), [filter](#filter-data).

</details>

<details>
  <summary> ✅ Cloud-optimized <code>walk</code> (list files at scale) <a id="walk" href="#walk">🔗</a> </summary>
&nbsp;

`litdata.walk` is a threaded, cloud-friendly alternative to `os.walk` for building large `inputs=` lists (especially on Lightning Studios). Yields `(dirpath, dirnames, filenames)` like `os.walk`, but **order is not depth-first**.

```python
from litdata import walk, optimize

inputs = []
for root, dirs, files in walk("/teamspace/s3_connections/my-data/raw", max_workers=32):
    for name in files:
        if name.endswith(".jpg"):
            inputs.append(f"{root}/{name}")

if __name__ == "__main__":
    optimize(fn=load_image, inputs=inputs, output_dir="...", chunk_bytes="64MB")
```

Prints a warning outside Lightning Studio — it is optimized for that environment; elsewhere prefer `os.walk` or your cloud SDK’s listing API.

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

Speed to stream a synthetic ImageNet-scale set from **Vast NFS** with POSIX-fast (mmap in place, decode only, no transforms):

| Workers | Images / sec |
|---|---|
| 48 | **18.2k** |
| 208 | **35.8k** |

### Raw Dataset

Speed to stream raw Imagenet 1.2M from different cloud storage providers:


| Storage | Images / s (without transform) | Images / s (with transform) |
|---------|-------------------|----------------|
| AWS S3  | ~6400 +/- 100     | ~3200 +/- 100  |
| Google Cloud Storage | ~5650 +/- 100     | ~3100 +/- 100  |

> **Also see:** [`StreamingRawDataset`](#stream-raw) streams existing files with **no optimize step** (great default to start). Use `StreamingDataset` after `optimize` when you need the highest sustained training throughput.

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

Transformations with LitData are linearly parallelizable across machines on [Lightning Studios](https://lightning.ai/) (see [distributed optimization](#distributed-optimization) for how the job launch works).

For example, let's say that it takes 56 hours to embed a dataset on a single A10G machine. With LitData,
this can be speed up by adding more machines in parallel

| Number of machines | Hours |
|-----------------|--------------|
| 1               | 56           |
| 2               | 28           |
| 4               | 14           |
| ...               | ...            |
| 64              | 0.875        |

```python
from litdata import map, Machine

map(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP,  # or omit to inherit the Studio machine
  # Prefer output_dir on /teamspace/s3_connections/... or s3://...
)
```

## Parallelize data optimization

Same Studio job launch as `map` — `num_nodes` machines × `num_workers` processes; last node merges the index.

```python
from litdata import optimize, Machine

optimize(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP,
  output_dir="/teamspace/s3_connections/my-data/optimized-v1",
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
| [Optimize TinyLlama 1T dataset for training](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset) | Text | 240 | 32 | [SlimPajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) & [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) |
| [Optimize parquet files for model training](https://lightning.ai/lightning-ai/studios/convert-parquets-to-lightning-streaming) | Parquet Files | 12 | 16 | Randomly Generated data |

&nbsp;

----

# Used by

<table width="100%">
<tr>
<th align="left" width="18%">Project</th>
<th align="left">Description</th>
</tr>
<tr>
<td valign="top"><a href="https://github.com/sunlabuiuc/PyHealth">PyHealth</a></td>
<td>Deep-learning toolkit for clinical prediction (MIMIC, eICU, OMOP, sleep, CXR). <code>set_task()</code> writes processed samples with LitData; <code>SampleDataset</code> subclasses <code>StreamingDataset</code> so training streams chunked EHR tensors instead of holding the cohort in RAM.</td>
</tr>
<tr>
<td valign="top"><a href="https://github.com/prescient-design/lobster">LBSTER</a></td>
<td>Protein and biological-sequence language models from Prescient Design (Genentech). Pre-training and concept-bottleneck models (fitness, embeddings, guided generation) stream large sequence corpora through LitData.</td>
</tr>
<tr>
<td valign="top"><a href="https://github.com/OpenSynth-energy/OpenSynth">OpenSynth</a></td>
<td>Open toolkit for synthetic smart-meter / energy time series. Generated or historical meter traces are optimized and streamed for model training.</td>
</tr>
<tr>
<td valign="top"><a href="https://github.com/BiomedSciAI/biomed-multi-view">biomed-multi-view</a></td>
<td>IBM BiomedSciAI multi-view biomedical models. LitData is used to cache and stream paired modalities during training.</td>
</tr>
<tr>
<td valign="top"><a href="https://github.com/cma2015/DEM">DEM</a></td>
<td>Phenotype and gene-mining pipeline (<code>biodem</code>). Large genomic / trait tables are packed into LitData chunks for repeated training passes.</td>
</tr>
<tr>
<td valign="top"><a href="https://pypi.org/project/deeptan/">deeptan</a></td>
<td>Graph multi-task models for multi-omics trait-associated networks. Guide graphs and expression tables are converted to LitData chunks before GNN training.</td>
</tr>
<tr>
<td valign="top"><a href="https://github.com/avitai/datarax">datarax</a></td>
<td>Data tooling with an optional cloud-streaming extra that uses LitData to read remote datasets without a full local copy.</td>
</tr>
<tr>
<td valign="top"><a href="https://pypi.org/project/fasr/">fasr</a></td>
<td>Speech ASR framework. The LitData extra streams audio and transcripts for training instead of random-access file lists.</td>
</tr>
</table>

# Skills <a id="skills"></a>

Coding agents (Cursor, Claude Code, and others) should load the LitData skill instead of guessing the API.

```bash
npx skills add Lightning-AI/litData
```

Useful options: `-g` (user-global), `-a cursor` (Cursor only), `-y` (non-interactive). In this repo the skill already lives at [`.claude/skills/litdata/`](.claude/skills/litdata/). Installer: [skills CLI](https://github.com/vercel-labs/skills).

Start at [`SKILL.md`](.claude/skills/litdata/SKILL.md), then load [`reference/using-litdata.md`](.claude/skills/litdata/reference/using-litdata.md) before writing examples.

| File | When to load |
| --- | --- |
| [SKILL.md](.claude/skills/litdata/SKILL.md) | Triggers, public API, traps |
| [using-litdata.md](.claude/skills/litdata/reference/using-litdata.md) | Optimize / stream / raw / modality cookbook |
| [streaming.md](.claude/skills/litdata/reference/streaming.md) | Read path, shuffle, resume, serializers |
| [processing.md](.claude/skills/litdata/reference/processing.md) | optimize / map orchestration |
| [data-movement.md](.claude/skills/litdata/reference/data-movement.md) | Download / upload / FUSE vs direct I/O |
| [multi-node.md](.claude/skills/litdata/reference/multi-node.md) | Studio num_nodes jobs |
| [resolver.md](.claude/skills/litdata/reference/resolver.md) | Paths, URLs, Studio mounts |
| [storage-format.md](.claude/skills/litdata/reference/storage-format.md) | Chunks, `index.json`, writer / reader |
| [cache-and-chunk-lifecycle.md](.claude/skills/litdata/reference/cache-and-chunk-lifecycle.md) | Prefetch and eviction |
| [env-vars.md](.claude/skills/litdata/reference/env-vars.md) | LITDATA_* and DATA_OPTIMIZER_* |
| [keyed-lookup.md](.claude/skills/litdata/reference/keyed-lookup.md) | key_fn, dataset_update |
| [debugging.md](.claude/skills/litdata/reference/debugging.md) | enable_tracer, Litracer |
| [benchmarking.md](.claude/skills/litdata/reference/benchmarking.md) | Fair benches |
| [lightning-studio.md](.claude/skills/litdata/reference/lightning-studio.md) | Studio env and credentials |
| [testing.md](.claude/skills/litdata/reference/testing.md) | Pytest / CI |
| [contributing.md](.claude/skills/litdata/reference/contributing.md) | PR / lint path |

Offline streaming what-if (not the Python package): [simulator/](simulator/) (litsim).

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
  howpublished = {\url{https://github.com/Lightning-AI/litdata}}
}
```

----

## Papers with LitData

Papers that train or stream with LitData (`optimize` / `StreamingDataset`). Scholar hits for “litdata streaming” are often weather *lightning* data, or mention LitData only as example source code.

| Paper | Venue | How LitData is used |
|---|---|---|
| [Towards Interpretable Protein Structure Prediction with Sparse Autoencoders](https://arxiv.org/abs/2503.08764) ([code](https://github.com/johnyang101/reticular-sae)) | ICLR 2025 GEM | `optimize` shards ESM-2 embeddings; `StreamingDataset` streams from S3 for multi-GPU SAE training |
| [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385) ([code](https://github.com/jzhang38/TinyLlama)) | arXiv 2024 | 1.1B pretrain on SlimPajama + StarCoder via Lit-GPT’s `lightning.data` stack (now LitData): `CombinedStreamingDataset` + `TokensLoader` |

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
