# Modality

Text, image, audio, and video start from files on disk: `list_media_folder` + `Text(path=...)` / `Image(path=...)` / `Audio(path=...)` / `Video(path=...)`. The caption is a string. For LLM pre-training, tokenize to a 1-D `Tensor` and pass `item_loader=TokensLoader()` — see `text.py`.

```bash
python examples/modality/image.py
python examples/modality/audio.py
python examples/modality/video.py
```

Audio and video need torchcodec (`pip install "litdata[extra]"`). Mesh, Pdf, Nifti, and Tiff need trimesh, pdfplumber, nibabel, and tifffile.

`optimize.py` / `stream.py` write several types in one sample. PyG details: [README — PyG graphs](../../README.md#pyg-graphs).
