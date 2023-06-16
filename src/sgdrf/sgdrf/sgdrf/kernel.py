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
import pyro.contrib.gp.kernels as kernels
from torch import Tensor
from typing import Optional


class KernelType(Enum):
    RBF = auto()
    RationalQuadratic = auto()
    Exponential = auto()
    Matern32 = auto()
    Matern52 = auto()

    def instantiate(
        self,
        input_dim: int,
        lengthscale: Tensor,
        variance: Tensor,
        active_dims: Optional[list[int]] = None,
    ) -> kernels.Isotropy:
        return getattr(kernels, self._name_)(
            input_dim=input_dim,
            lengthscale=lengthscale,
            variance=variance,
            active_dims=active_dims,
        )
