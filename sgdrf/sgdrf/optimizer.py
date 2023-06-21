# Copyright 2023 John San Soucie
#
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

from enum import Enum, auto
import pyro.optim as optim
from pyro.optim import PyroOptim


class OptimizerType(Enum):
    """A wrapper for ::py:class:`pyro.optim.PyroOptim` optimizers."""

    Adadelta = auto()
    """::py:class:`pyro.optim.Adadelta`"""
    Adagrad = auto()
    """::py:class:`pyro.optim.Adagrad`"""
    Adam = auto()
    """::py:class:`pyro.optim.Adam`"""
    AdamW = auto()
    """::py:class:`pyro.optim.AdamW`"""
    Adamax = auto()
    """::py:class:`pyro.optim.Adamax`"""
    SGD = auto()
    """::py:class:`pyro.optim.SGD`"""

    def instantiate(self, lr: float = 0.001, clip_norm: float = 10.0) -> PyroOptim:
        """
        Instantiate the optimizer.

        Parameters
        ----------
        lr : float, optional
            Optimizer learning rate, by default 0.001
        clip_norm : float, optional
            Optimizer gradient norm maximum, by default 10.0

        Returns
        -------
        PyroOptim
            The instantiated optimizer

        """
        return getattr(optim, self._name_)(
            {"lr": lr}, clip_args={"clip_norm": clip_norm}
        )
