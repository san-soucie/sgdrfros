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

from enum import Flag, auto


class SubsampleType(Flag):
    """A simple wrapper for the various subsampling strategies."""

    latest = auto()
    """Always sample the most recent `N` observations"""
    uniform = auto()
    """Sample uniformly at random from all past observations"""
    exponential = auto()
    """Sample with probability proportional to :math:`exp[{\\alpha}(t-{\\tau})]`"""
    exponential_plus_uniform = exponential | uniform
    """Sample with probabilities representing a weighted sum of exponential and uniform strategies"""
    exponential_plus_latest = exponential | latest
    """Sample with probabilities representing a weighted sum of exponential and latest strategies"""
    uniform_plus_latest = uniform | latest
    """Sample with probabilities representing a weighted sum of uniform and latest strategies"""
