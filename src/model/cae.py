# Standard libraries
from typing import Tuple
from functools import partial

# Third party modules
from tensorflow import Tensor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Layer,
    ZeroPadding2D
)
from tensorflow.keras.models import Model
from tensorflow.nn import leaky_relu

# Local libraries
from model.reflection_pad_2d import ReflectionPad2D


def get_conv2d(
    output_dim: int,
    kernel_size: Tuple[int] = (3, 3),
    strides: Tuple[int] = (1, 1),
    activation = partial(leaky_relu, alpha=0.01)
) -> Layer:
    return Conv2D(
        output_dim,
        kernel_size=kernel_size,
        activation=activation,
        padding='valid',
        strides=strides
    )


class CAE(Model):

  def __init__(self) -> None:
    super(CAE, self).__init__()
    self.setup_layers()

  def setup_layers(self) -> None:
    # padding: (top, bottom), (left, right)
    self.e_conv_1 = Sequential([
      ZeroPadding2D(padding=((1, 2), (1, 2))),
      get_conv2d(64, kernel_size=(5, 5), strides=(2, 2))
    ])
    self.e_conv_2 = Sequential([
      ZeroPadding2D(padding=((1, 2), (1, 2))),
      get_conv2d(128, kernel_size=(5, 5), strides=(2, 2))
    ])
    self.e_block_1 = Sequential([
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128),
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128, activation=None)
    ])
    self.e_block_2 = Sequential([
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128),
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128, activation=None)
    ])
    self.e_block_3 = Sequential([
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128),
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128, activation=None)
    ])
    self.e_conv_3 = Sequential([
      ZeroPadding2D(padding=((2, 2), (2, 2))),
      get_conv2d(32, activation='tanh', kernel_size=(5, 5))
    ])

    self.d_conv_1 = Sequential([
      get_conv2d(64),
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      Conv2DTranspose(128, kernel_size=(2, 2), padding='valid', strides=(2, 2))
    ])
    self.d_block_1 = Sequential([
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128),
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128, activation=None)
    ])
    self.d_block_2 = Sequential([
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128),
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128, activation=None)
    ])
    self.d_block_3 = Sequential([
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128),
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      get_conv2d(128, activation=None)
    ])
    self.d_conv_2 = Sequential([
      get_conv2d(32),
      ZeroPadding2D(padding=((1, 1), (1, 1))),
      Conv2DTranspose(256, kernel_size=(2, 2), padding='valid', strides=(2, 2))
    ])
    self.d_conv_3 = Sequential([
      get_conv2d(16),
      ReflectionPad2D((2, 2, 2, 2)),
      get_conv2d(3, activation='tanh')
    ])

  def call(self, input_value: Tensor) -> Tensor:
    ec1 = self.e_conv_1(input_value)
    ec2 = self.e_conv_2(ec1)
    eblock1 = self.e_block_1(ec2) + ec2
    eblock2 = self.e_block_2(eblock1) + eblock1
    eblock3 = self.e_block_3(eblock2) + eblock2
    ec3 = self.e_conv_3(eblock3)

    return self.decode(ec3)
  
  def decode(self, encoded_value: Tensor) -> Tensor:
    y = encoded_value# * 2.0 - 1
    uc1 = self.d_conv_1(encoded_value)
    dblock1 = self.d_block_1(uc1) + uc1
    dblock2 = self.d_block_2(dblock1) + dblock1
    dblock3 = self.d_block_3(dblock2) + dblock2
    uc2 = self.d_conv_2(dblock3)
    dec = self.d_conv_3(uc2)

    return dec
