# Standard libraries
from typing import Tuple

# Third party libraries
from tensorflow import (
  Tensor,
  pad
)
from tensorflow.keras.layers import Layer


class ReflectionPad2D(Layer):

  def __init__(self, paddings: Tuple[int] = (1, 1, 1, 1)) -> None:
    super(ReflectionPad2D, self).__init__()
    self.paddings = paddings

  def call(self, input_value: Tensor) -> Tensor:
    left, right, top, bottom = self.paddings

    return pad(
      input_value,
      paddings=[[0, 0], [top, bottom], [left, right], [0, 0]],
      mode='REFLECT'
    )
