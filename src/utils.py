# Standard libraries
import numpy as np

# Third party libraries
from PIL import Image
import tensorflow as tf


def load_image(file_path: str) -> Image:
    parts = tf.strings.split(file_path, '\\')
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def parse_np_array_image(image: np.array) -> Image:
    image = (image * 255 / np.max(image)).astype('uint8')
    image = Image.fromarray(image, 'RGB')

    return image
