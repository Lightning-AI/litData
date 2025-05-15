First step is to prepare the dataset.
And for that we need to have the dataset in the right format. 
So the idea here is to first have the dataset in the this machine and make it comaptibe to be direclty read via ImageFolderDataset. and int heb format of image and it's label index. 

Since the dataset is already present in the lightning platform so I woudl skip the download process and direclty copy from there.

Copy the imagenet raw dataset to the data/imagenet-1m-raw 
```sh
s5cmd cp "s3://imagenet-1m-template/raw/train/*" data/imagenet-1m-raw/train
```

Convert the imaagenet raw dataset to be used as the imagefolder by converting the original subfolders present in imaganet raw train as class index subfolders
```sh
 python convert_imagenet_to_pytorch_style.py --data_dir data/imagenet-1m-raw/train
    ```
    
now install the ffcv library
```sh
sh install_ffcv.sh
```
## Prepare the dataset
Now we prepare the ffcv dataset for the imagenet dataset.
(max 256px, 0% JPEG, quality 100)
```sh
    python write_imagenet.py \
        --cfg.dataset=imagenet \
        --cfg.split=train \
        --cfg.data_dir=/path/to/imagenet/train \
        --cfg.write_path=/your/output/path/train_256_0.0_100.ffcv \
        --cfg.max_resolution=256 \
        --cfg.write_mode=proportion \
        --cfg.compress_probability=0.0 \
        --cfg.jpeg_quality=100
```
(max 256px, 100% JPEG, quality 90)
```sh
    python write_imagenet.py \
        --cfg.dataset=imagenet \
        --cfg.split=train \
        --cfg.data_dir=/path/to/imagenet/train \
        --cfg.write_path=/your/output/path/train_256_100.0_90.ffcv \
        --cfg.max_resolution=256 \
        --cfg.write_mode=proportion \
        --cfg.compress_probability=100.0 \
        --cfg.jpeg_quality=90
```

## Stream the dataset
Now we can stream the dataset using the ffcv library.
```sh
python stream_imagenet.py \
   --cfg.data_path=/path/to/train_256_0.0_100.ffcv \
   --cfg.batch_size=256 \
   --cfg.num_workers=32 \
   --cfg.epochs=2 \
   --cfg.os_cache=False 
```


# Benchmark litdata vs ffcv
```
 ~  aws s3 ls s3://xxxxxxx/datasets/imagenet-1m-ffcv/
2025-04-27 19:38:18 181937712488 train_256_0.0_100.ffcv
2025-04-27 19:38:18 21295869288 train_256_100.0_90.ffcv
⚡ ~  aws s3 ls s3://xxxxxxx/datasets/imagenet-1m-litdata/
                           PRE train_256_jpg_90/
                           PRE train_256_raw_pil/

```
To benchmark them data is already present in the litdata teamspace so we can directly copy from the s3 to the local machine or you can also prepare by yourself as shown above for ffcv and litdata.