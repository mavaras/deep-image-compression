# Standard libraries
from argparse import ArgumentParser
import numpy as np

# Third party libraries
from PIL import Image
import tensorflow as tf
from tensorflow.train import (
    Checkpoint,
    latest_checkpoint
)

# Local libraries
from model.cae import CAE
from utils import (
    load_image,
    parse_np_array_image
)


def predict(model: CAE, image: Image) -> Image:
    image = tf.expand_dims(image, 0)
    pred_image = model(image)
    pred_image = pred_image.numpy()
    pred_image = pred_image[0, :, :, :]

    return parse_np_array_image(pred_image)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument(
        '--image-output-path',
        required=False,
        default='pred_image.png',
        type=str
    )
    args = parser.parse_args()

    model = CAE()
    ckpt = Checkpoint(transformer=model)
    ckpt.restore(latest_checkpoint(args.model_path)).expect_partial()
    pred_image = predict(model, load_image(args.image_path))
    pred_image.save(args.image_output_path)
