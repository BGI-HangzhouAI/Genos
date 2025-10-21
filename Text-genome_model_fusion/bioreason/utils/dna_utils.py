from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

from transformers.utils import is_torch_available

if is_torch_available():
    import torch

DNAInput = Union[
    str, list[int], np.ndarray, "torch.Tensor", list[str], list[list[int]], list[np.ndarray], list["torch.Tensor"]
]  # noqa