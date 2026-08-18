# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import random
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest
import tifffile
import torch
from lightning_utilities.core.imports import RequirementCache

from litdata.streaming.serializers import (
    _NUMPY_DTYPES_MAPPING,
    _SERIALIZERS,
    _TORCH_DTYPES_MAPPING,
    AudioSerializer,
    BooleanSerializer,
    FileSerializer,
    ImageSerializer,
    IntegerSerializer,
    JPEGArraySerializer,
    JPEGSerializer,
    MeshSerializer,
    NiftiSerializer,
    NoHeaderNumpySerializer,
    NoHeaderTensorSerializer,
    NumpySerializer,
    PDFSerializer,
    PILSerializer,
    TensorSerializer,
    TIFFSerializer,
    VideoSerializer,
    _get_serializers,
    _LitAudioDecoder,
    _torchcodec_usable,
    _torchvision_read_video_available,
)
from litdata.types import Audio, File, Image, Jpeg, JpegArray, Mesh, Nifti, Pdf, Pil, Tiff, Video


def seed_everything(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


_PIL_AVAILABLE = RequirementCache("PIL")
_TIFFFILE_AVAILABLE = RequirementCache("tifffile")


def test_serializers():
    keys = list(_SERIALIZERS.keys())
    assert keys == [
        "str",
        "text",
        "bool",
        "int",
        "float",
        "video",
        "audio",
        "image",
        "nifti",
        "mesh",
        "pdf",
        "graph",
        "tifffile",
        "file",
        "pil",
        "jpeg",
        "jpeg_array",
        "bytes",
        "no_header_numpy",
        "numpy",
        "no_header_tensor",
        "tensor",
        "pickle",
    ]


def test_int_serializer():
    serializer = IntegerSerializer()

    for i in range(100):
        data, _ = serializer.serialize(i)
        assert isinstance(data, bytes)
        assert i == serializer.deserialize(data)


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
@pytest.mark.parametrize("mode", ["I", "L", "RGB"])
def test_pil_serializer(mode):
    serializer = PILSerializer()

    from PIL import Image

    np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
    img = Image.fromarray(np_data).convert(mode)

    data, _ = serializer.serialize(img)
    assert isinstance(data, bytes)

    deserialized_img = serializer.deserialize(data)
    deserialized_img = deserialized_img.convert("I")
    np_dec_data = np.asarray(deserialized_img, dtype=np.uint32)
    assert isinstance(deserialized_img, Image.Image)

    # Validate data content
    assert np.array_equal(np_data, np_dec_data)


def test_pil_serializer_available():
    serializer = PILSerializer()
    with mock.patch("litdata.streaming.serializers._PIL_AVAILABLE", False):
        assert not serializer.can_serialize(None)


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
def test_jpeg_serializer():
    serializer = JPEGSerializer()

    from PIL import Image

    array = np.random.randint(255, size=(28, 28, 3), dtype=np.uint8)
    img = Image.fromarray(array)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    img = Image.open(io.BytesIO(img_bytes))

    data, _ = serializer.serialize(img)
    assert isinstance(data, bytes)

    deserialized_img = serializer.deserialize(data)
    assert deserialized_img.shape == torch.Size([3, 28, 28])


def test_jpeg_serializer_available():
    serializer = JPEGSerializer()
    with mock.patch("litdata.streaming.serializers._PIL_AVAILABLE", False):
        assert not serializer.can_serialize(None)


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
def test_jpeg_array_serializer():
    """Test the JPEGArraySerializer with various inputs and edge cases."""
    from PIL import Image

    serializer = JPEGArraySerializer()

    # Helper function to create a list of jpeg images
    def create_test_jpeg_image(width, height):
        array = np.random.randint(255, size=(height, width, 3), dtype=np.uint8)
        img = Image.fromarray(array)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        return Image.open(io.BytesIO(img_bytes.getvalue()))

    # Test 1: Basic functionality - List of JPEG images
    images = [
        create_test_jpeg_image(100, 100),
        create_test_jpeg_image(200, 150),
        create_test_jpeg_image(300, 200),
    ]

    # Verify can_serialize
    assert serializer.can_serialize(images)
    assert serializer.can_serialize(tuple(images))
    assert not serializer.can_serialize([b"not a image"])

    # Test serialization and deserialization
    data, name = serializer.serialize(images)
    assert isinstance(data, bytes)

    # Deserialize and verify
    deserialized_images = serializer.deserialize(data)
    assert len(deserialized_images) == 3
    assert all(isinstance(img, torch.Tensor) for img in deserialized_images)
    # Verify image dimensions
    assert deserialized_images[0].shape == torch.Size([3, 100, 100])  # CHW
    assert deserialized_images[1].shape == torch.Size([3, 150, 200])
    assert deserialized_images[2].shape == torch.Size([3, 200, 300])

    # Test 2: Single image
    single_image_list = [create_test_jpeg_image(50, 50)]
    data, _ = serializer.serialize(single_image_list)
    deserialized_single = serializer.deserialize(data)
    assert len(deserialized_single) == 1
    assert isinstance(deserialized_single[0], torch.Tensor)
    assert deserialized_single[0].shape == torch.Size([3, 50, 50])

    # Test 3: Large batch of images (using list comprehension)
    large_batch = [create_test_jpeg_image(10, 10) for _ in range(10)]
    data, _ = serializer.serialize(large_batch)
    deserialized_batch = serializer.deserialize(data)
    assert all(isinstance(img, torch.Tensor) for img in deserialized_batch)
    # Verify image dimensions
    assert all(img.shape == torch.Size([3, 10, 10]) for img in deserialized_batch)

    # Test 4: Error handling with corrupted data
    with pytest.raises(ValueError, match="Input data is too short"):
        serializer.deserialize(b"abc")  # Too short data


@pytest.mark.flaky(reruns=3)
@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_tensor_serializer():
    seed_everything(42)

    serializer_tensor = TensorSerializer()

    shapes = [(10,), (10, 10), (10, 10, 10), (10, 10, 10, 5), (10, 10, 10, 5, 4)]
    for dtype in _TORCH_DTYPES_MAPPING.values():
        for shape in shapes:
            # Not serializable for some reasons
            if dtype in [torch.bfloat16]:
                continue
            tensor = torch.ones(shape, dtype=dtype)

            data, _ = serializer_tensor.serialize(tensor)
            deserialized_tensor = serializer_tensor.deserialize(data)

            assert deserialized_tensor.dtype == dtype
            assert torch.equal(tensor, deserialized_tensor)


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_numpy_serializer():
    seed_everything(42)

    serializer_tensor = NumpySerializer()

    shapes = [(10,), (10, 10), (10, 10, 10), (10, 10, 10, 5), (10, 10, 10, 5, 4)]
    for dtype in _NUMPY_DTYPES_MAPPING.values():
        # Those types aren't supported
        if dtype.name in ["object", "bytes", "str", "void"]:
            continue
        for shape in shapes:
            tensor = np.ones(shape, dtype=dtype)
            data, _ = serializer_tensor.serialize(tensor)
            deserialized_tensor = serializer_tensor.deserialize(data)
            assert deserialized_tensor.dtype == dtype
            assert deserialized_tensor.flags.writeable
            np.testing.assert_equal(tensor, deserialized_tensor)


def test_assert_bfloat16_tensor_serializer():
    serializer = TensorSerializer()
    tensor = torch.ones((10,), dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="Got unsupported ScalarType BFloat16"):
        serializer.serialize(tensor)


def test_assert_no_header_tensor_serializer():
    serializer = NoHeaderTensorSerializer()
    t = torch.ones((10,))
    data, name = serializer.serialize(t)
    assert name == "no_header_tensor:1"
    assert serializer._dtype is None
    serializer.setup(name)
    assert serializer._dtype == torch.float32
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)


def test_assert_no_header_numpy_serializer():
    serializer = NoHeaderNumpySerializer()
    t = np.ones((10,), dtype=np.float64)
    assert serializer.can_serialize(t)
    data, name = serializer.serialize(t)
    try:
        assert name == "no_header_numpy:10"
    except AssertionError as e:  # debug what np.core.sctypes looks like on Windows
        raise ValueError(np.core.sctypes) from e
    assert serializer._dtype is None
    serializer.setup(name)
    assert serializer._dtype == np.dtype("float64")
    new_t = serializer.deserialize(data)
    assert new_t.flags.writeable
    np.testing.assert_equal(t, new_t)


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
@pytest.mark.skipif(
    condition=not _torchcodec_usable() and not _torchvision_read_video_available(),
    reason="Requires torchcodec or torchvision.io.read_video",
)
def test_wav_deserialization(tmpdir):
    from torch.hub import download_url_to_file

    video_file = os.path.join(tmpdir, "video.mp4")
    key = "tutorial-assets/mptestsrc.mp4"  # E501
    try:
        download_url_to_file(f"https://download.pytorch.org/torchaudio/{key}", video_file)
    except Exception as exc:
        pytest.skip(f"Could not download the torchaudio tutorial clip: {exc}")

    serializer = VideoSerializer(decode="all")
    assert serializer.can_serialize(video_file)
    data, name = serializer.serialize(video_file)
    assert len(data) > 1000
    assert name == "video:mp4"
    vframes, aframes, info = serializer.deserialize(data)
    assert vframes.ndim == 4
    assert vframes.shape[-1] == 3
    assert vframes.shape[0] > 0
    assert aframes.ndim == 2
    # The metadata keys for video serialization may vary by serializer.
    # For example, `torchvision` typically uses `video_fps`, while `torchcodec` uses `average_fps`.
    # Despite these naming differences, both keys represent the same fps value,
    # ensuring consistency in video frame rate representation across serialization methods.
    assert "video_fps" in info or "average_fps" in info
    fps = info.get("video_fps", info.get("average_fps"))
    assert fps is not None
    assert abs(float(fps) - 25.0) < 0.1


_MIN_PDF = b"%PDF-1.1\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
_MIN_STL = b"""solid simple
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid simple
"""


def test_video_serializer_accepts_path_and_dict(tmpdir):
    video_file = os.path.join(tmpdir, "clip.mp4")
    with open(video_file, "wb") as handle:
        handle.write(b"ftypfake")

    serializer = VideoSerializer(decode="bytes")
    assert serializer.can_serialize(video_file)
    data, name = serializer.serialize(video_file)
    assert name == "video:mp4"
    assert serializer.deserialize(data) == b"ftypfake"

    assert serializer.can_serialize({"path": video_file, "bytes": None})
    data, name = serializer.serialize({"path": video_file, "bytes": b"from-dict"})
    assert name == "video:mp4"
    assert data == b"from-dict"


@pytest.mark.skipif(not _torchcodec_usable(), reason="Requires a working torchcodec install")
def test_video_serializer_default_is_torchcodec_decoder(tmpdir):
    from torch.hub import download_url_to_file
    from torchcodec.decoders import VideoDecoder

    video_file = os.path.join(tmpdir, "video.mp4")
    download_url_to_file("https://download.pytorch.org/torchaudio/tutorial-assets/mptestsrc.mp4", video_file)
    serializer = VideoSerializer()
    data, _ = serializer.serialize(video_file)
    decoded = serializer.deserialize(data)
    assert isinstance(decoded, VideoDecoder)
    clip = decoded.get_frames_in_range(0, 4)
    assert clip.data.shape[0] == 4


def test_mesh_serializer_roundtrip_bytes(tmpdir):
    mesh_file = os.path.join(tmpdir, "mesh.stl")
    with open(mesh_file, "wb") as handle:
        handle.write(_MIN_STL)

    serializer = MeshSerializer(decode=False)
    assert serializer.can_serialize(mesh_file)
    data, name = serializer.serialize(mesh_file)
    assert name == "mesh:stl"
    assert serializer.deserialize(data) == _MIN_STL

    serializer.setup("mesh:stl")
    if RequirementCache("trimesh"):
        decoded = MeshSerializer(decode=True)
        decoded.setup("mesh:stl")
        mesh = decoded.deserialize(data)
        assert mesh is not None


def test_pdf_serializer_roundtrip_bytes(tmpdir):
    pdf_file = os.path.join(tmpdir, "doc.pdf")
    with open(pdf_file, "wb") as handle:
        handle.write(_MIN_PDF)

    serializer = PDFSerializer(decode=False)
    assert serializer.can_serialize(pdf_file)
    assert serializer.can_serialize(_MIN_PDF)
    data, name = serializer.serialize(pdf_file)
    assert name == "pdf"
    assert serializer.deserialize(data).startswith(b"%PDF")


def test_media_types_are_leaves_and_not_strings():
    from litdata.utilities._pytree import tree_flatten

    caption = "a recording of a dog"
    audio = Audio(path="does-not-need-to-exist.wav")
    leaves, _ = tree_flatten({"caption": caption, "audio": audio})
    assert leaves == [caption, audio]
    assert AudioSerializer().can_serialize(audio)
    assert not AudioSerializer().can_serialize(caption)
    assert not VideoSerializer().can_serialize(audio)
    assert VideoSerializer().can_serialize(Video(bytes=b"ftyp"))
    assert ImageSerializer().can_serialize(Image(bytes=b"\xff\xd8"))
    assert MeshSerializer(decode=False).can_serialize(Mesh(bytes=_MIN_STL))
    assert PDFSerializer(decode=False).can_serialize(Pdf(bytes=_MIN_PDF))
    assert NiftiSerializer(decode=False).can_serialize(Nifti(bytes=b"nii"))
    assert JPEGSerializer().can_serialize(Jpeg(bytes=b"\xff\xd8"))
    assert JPEGArraySerializer().can_serialize(JpegArray(images=[Jpeg(bytes=b"\xff\xd8")]))
    assert FileSerializer().can_serialize(File(bytes=b"raw"))
    assert TIFFSerializer().can_serialize(Tiff(bytes=b"II*\x00"))
    assert PILSerializer().can_serialize(Pil(bytes=b"\x89PNG"))


def test_audio_type_path_and_pcm(tmpdir):
    wav = os.path.join(tmpdir, "tone.wav")
    array = np.zeros(8000, dtype=np.float32)
    data, name = AudioSerializer(decode="bytes").serialize({"array": array, "sampling_rate": 8000})
    with open(wav, "wb") as handle:
        handle.write(data)

    serializer = AudioSerializer(decode="bytes")
    assert serializer.can_serialize(Audio(path=wav))
    again, name = serializer.serialize(Audio(path=wav))
    assert name == "audio:wav"
    assert again[:4] == b"RIFF"

    pcm_path = os.path.join(tmpdir, "raw.pcm")
    pcm = (np.zeros(16, dtype=np.int16)).tobytes()
    with open(pcm_path, "wb") as handle:
        handle.write(pcm)
    encoded, name = serializer.serialize(Audio(path=pcm_path, sampling_rate=8000))
    assert name == "audio:wav"
    assert encoded[:4] == b"RIFF"


def test_audio_decoder_is_subscriptable():
    class _FakeDecoder:
        def get_all_samples(self):
            class _Samples:
                data = torch.zeros(1, 8)

            return _Samples()

        def get_samples_played_in_range(self, start: float, end: float):
            class _Range:
                sample_rate = 8000

            return _Range()

    decoder = _LitAudioDecoder(_FakeDecoder())
    assert decoder["sampling_rate"] == 8000
    assert decoder["array"].shape == (8,)


def test_as_bytes_accepts_memoryview():
    from litdata.streaming.serializers import _as_bytes

    assert _as_bytes(b"abc") == b"abc"
    assert _as_bytes(memoryview(b"abc")) == b"abc"


@pytest.mark.skipif(not _torchcodec_usable(), reason="Requires a working torchcodec install")
def test_audio_decoder_hf_getitem():
    array = np.zeros(1600, dtype=np.float32)
    data, _ = AudioSerializer(decode="bytes").serialize({"array": array, "sampling_rate": 8000})
    decoder = AudioSerializer().deserialize(data)
    assert decoder["sampling_rate"] == 8000
    waveform = decoder["array"]
    assert waveform.ndim == 1
    assert waveform.shape[0] > 0


def test_image_hf_encode_tricks():
    from litdata.streaming.serializers import _image_array_for_pil, _native_pil_format

    down = _image_array_for_pil(np.zeros((4, 4, 3), dtype=np.float32))
    assert down.dtype == np.dtype("|u1")

    from PIL import Image as PILImage

    rgb = PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    assert _native_pil_format(rgb) == "PNG"
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG")
    buf.seek(0)
    jpeg = PILImage.open(buf)
    assert _native_pil_format(jpeg) == "JPEG"
    data, name = ImageSerializer().serialize(Image(image=jpeg))
    assert name == "image:jpg"
    assert data[:2] == b"\xff\xd8"


def test_image_array_quality():
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    data, name = ImageSerializer().serialize(Image(array=array, quality=95, format="jpeg"))
    assert name == "image:jpg"
    assert data[:2] == b"\xff\xd8"

    jpeg, _ = JPEGSerializer().serialize(Jpeg(array=array, quality=80))
    assert jpeg[:2] == b"\xff\xd8"

    tiff, name = TIFFSerializer().serialize(Tiff(array=np.zeros((4, 4), dtype=np.uint8)))
    assert name == "tiff"
    assert tifffile.imread(io.BytesIO(tiff)).shape == (4, 4)


def test_jpeg_tiff_file_types(tmpdir):
    jpeg_path = os.path.join(tmpdir, "a.jpg")
    with open(jpeg_path, "wb") as handle:
        handle.write(b"\xff\xd8fake")
    data, _ = JPEGSerializer().serialize(Jpeg(path=jpeg_path))
    assert data == b"\xff\xd8fake"

    packed, _ = JPEGArraySerializer().serialize(JpegArray(images=[Jpeg(bytes=b"\xff\xd8a"), Jpeg(bytes=b"\xff\xd8b")]))
    assert packed[:4] == np.uint32(2).tobytes()

    tiff_path = os.path.join(tmpdir, "a.tif")
    with open(tiff_path, "wb") as handle:
        handle.write(b"II*\x00fake")
    data, name = TIFFSerializer().serialize(Tiff(path=tiff_path))
    assert name == "tiff"
    assert data.startswith(b"II")

    data, name = FileSerializer().serialize(File(path=jpeg_path))
    assert name == "file:jpg"
    assert data == b"\xff\xd8fake"


def test_video_type_bytes():
    serializer = VideoSerializer(decode="bytes")
    data, name = serializer.serialize(Video(bytes=b"ftypfake", path="clip.mp4"))
    assert name == "video:mp4"
    assert data == b"ftypfake"


def test_audio_serializer_from_array(tmpdir):
    rate = 8000
    array = np.zeros(rate, dtype=np.float32)
    serializer = AudioSerializer(decode="bytes")
    assert serializer.can_serialize({"array": array, "sampling_rate": rate})
    data, name = serializer.serialize({"array": array, "sampling_rate": rate})
    assert name == "audio:wav"
    assert data[:4] == b"RIFF"

    wav_path = os.path.join(tmpdir, "tone.wav")
    with open(wav_path, "wb") as handle:
        handle.write(data)
    assert serializer.can_serialize(wav_path)
    again, name = serializer.serialize(wav_path)
    assert name == "audio:wav"
    assert again[:4] == b"RIFF"


def test_nifti_serializer_roundtrip_bytes(tmpdir):
    path = os.path.join(tmpdir, "vol.nii")
    with open(path, "wb") as handle:
        handle.write(b"niftifake")
    serializer = NiftiSerializer(decode=False)
    assert serializer.can_serialize(path)
    data, name = serializer.serialize(path)
    assert name == "nifti:nii"
    assert serializer.deserialize(data) == b"niftifake"

    gz_path = os.path.join(tmpdir, "vol.nii.gz")
    with open(gz_path, "wb") as handle:
        handle.write(b"\x1f\x8bfake")
    data, name = serializer.serialize(gz_path)
    assert name == "nifti:nii.gz"


def test_get_serializers():
    class CustomSerializer(NoHeaderTensorSerializer):
        pass

    serializers = _get_serializers({"no_header_tensor": CustomSerializer(), "custom": CustomSerializer()})

    assert isinstance(serializers["no_header_tensor"], CustomSerializer)
    assert isinstance(serializers["custom"], CustomSerializer)


def test_deserialize_empty_tensor():
    serializer = TensorSerializer()
    t = torch.ones((0, 3)).int()
    data, _ = serializer.serialize(t)
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)

    t = torch.ones((0, 3)).float()
    data, _ = serializer.serialize(t)
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)


def test_deserialize_scalar_tensor():
    serializer = TensorSerializer()
    t = torch.tensor(0)
    data, _ = serializer.serialize(t)
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)


def test_deserialize_empty_no_header_tensor():
    serializer = NoHeaderTensorSerializer()
    t = torch.ones((0,)).int()
    data, name = serializer.serialize(t)
    serializer.setup(name)
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)

    t = torch.ones((0,)).float()
    data, name = serializer.serialize(t)
    serializer.setup(name)
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)


def test_can_serialize_tensor():
    serializer = TensorSerializer()
    # Check that the TensorSerializer can serialize scalar valued tensors as well as higher order (>1) Tensors
    assert serializer.can_serialize(torch.tensor(0))
    assert serializer.can_serialize(torch.tensor([[0, 0]]))
    # Check that it does not serialize Tensors of order 1, those are treated by the dedicated NoHeaderTensorSerializer
    assert not serializer.can_serialize(torch.tensor([0, 0]))


@pytest.mark.skipif(not _TIFFFILE_AVAILABLE, reason="Requires: ['tifffile']")
def test_tiff_serializer():
    serializer = TIFFSerializer()

    # Create a synthetic multispectral image
    height, width, bands = 28, 28, 12
    np_data = np.random.randint(0, 65535, size=(height, width, bands), dtype=np.uint16)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
        tifffile.imwrite(tmp_file.name, np_data)
        file_path = tmp_file.name

    # Test can_serialize
    assert serializer.can_serialize(file_path)

    # Serialize
    data, _ = serializer.serialize(file_path)
    assert isinstance(data, bytes)

    # Deserialize
    deserialized_data = serializer.deserialize(data)
    assert isinstance(deserialized_data, np.ndarray)
    assert deserialized_data.shape == (height, width, bands)
    assert deserialized_data.dtype == np.uint16

    # Validate data content
    assert np.array_equal(np_data, deserialized_data)

    # Clean up
    os.remove(file_path)


def test_boolean_serializer():
    serializer = BooleanSerializer()

    # Test serialization and deserialization of True
    data, _ = serializer.serialize(True)
    assert isinstance(data, bytes)
    assert serializer.deserialize(data) is True

    # Test serialization and deserialization of False
    data, _ = serializer.serialize(False)
    assert isinstance(data, bytes)
    assert serializer.deserialize(data) is False

    assert serializer.size == 1
    assert len(data) == 1

    # Test can_serialize method
    assert serializer.can_serialize(True)
    assert serializer.can_serialize(False)
    assert not serializer.can_serialize(1)
    assert not serializer.can_serialize("True")
    assert not serializer.can_serialize(None)
