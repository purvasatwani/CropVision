import numpy as np
from PIL import Image

import disease_info
from imaging import InvalidImage, open_image, resize_for_model, to_model_array
from conftest import png_bytes
import pytest


def test_preprocessing_shape_and_range():
    arr = to_model_array(Image.new("RGB", (640, 480), (255, 0, 0)))
    assert arr.shape == (1, 224, 224, 3) and arr.dtype == np.float32
    assert arr.max() == 255.0 and arr.min() == 0.0  # raw 0-255, scaling happens inside the model


def test_resize_is_square_centre_crop():
    im = Image.new("RGB", (400, 200), (0, 0, 0))
    im.paste((255, 255, 255), (150, 0, 250, 200))
    out = resize_for_model(im)
    assert out.size == (224, 224) and out.getpixel((112, 112)) == (255, 255, 255)


def test_open_image_rejects_tiny_and_junk():
    with pytest.raises(InvalidImage):
        open_image(png_bytes((10, 10)))
    import io
    with pytest.raises(InvalidImage):
        open_image(io.BytesIO(b"GIF89a-not-really"))


def test_disease_lookup():
    assert disease_info.lookup("Tomato___Late_blight")["condition"] == "Late blight"
    assert disease_info.lookup("Apple___healthy")["healthy"] is True
    assert disease_info.lookup("Corn_(maize)___Common_rust_")["condition"] == "Common rust"
    assert disease_info.lookup("Mystery___Thing")["found"] is False
    assert disease_info.display_name("Pepper,_bell___Bacterial_spot") == "Pepper, bell: Bacterial spot"
