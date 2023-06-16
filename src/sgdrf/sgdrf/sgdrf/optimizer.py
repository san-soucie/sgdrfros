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
    Adadelta = auto()
    Adagrad = auto()
    Adam = auto()
    AdamW = auto()
    Adamax = auto()
    SGD = auto()

    def instantiate(self, lr: float = 0.001, clip_norm: float = 10.0) -> PyroOptim:
        return getattr(optim, self._name_)(
            {"lr": lr}, clip_args={"clip_norm": clip_norm}
        )
