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
from typing import Optional

import pyro.contrib.gp.kernels as kernels
from torch import Tensor


class KernelType(Enum):
    """A wrapper for ::py:class:`pyro.contrib.gp.kernels.Isotropy` kernels."""

    RBF = auto()
    """::py:class:`pyro.contrib.gp.kernels.RBF`"""
    RationalQuadratic = auto()
    """::py:class:`pyro.contrib.gp.kernels.RationalQuadratic`"""
    Exponential = auto()
    """::py:class:`pyro.contrib.gp.kernels.Exponential`"""
    Matern32 = auto()
    """::py:class:`pyro.contrib.gp.kernels.Matern32`"""
    Matern52 = auto()
    """::py:class:`pyro.contrib.gp.kernels.Matern52`"""

    def instantiate(
        self,
        input_dim: int,
        lengthscale: Tensor,
        variance: Tensor,
        active_dims: Optional[list[int]] = None,
    ) -> kernels.Isotropy:
        """Instantiate the kernel.

        Parameters
        ----------
        input_dim : int
            Number of feature dimensions of inputs
        lengthscale : Tensor
            Length-scale parameter of this kernel
        variance : Tensor
            Variance parameter of this kernel
        active_dims : Optional[list[int]], optional
             List of feature dimensions of the input which the kernel acts on, by default None

        Returns
        -------
        kernels.Isotropy
            The instantiated kernel
        """
        return getattr(kernels, self._name_)(
            input_dim=input_dim,
            lengthscale=lengthscale,
            variance=variance,
            active_dims=active_dims,
        )
