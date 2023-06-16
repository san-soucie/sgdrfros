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

import rclpy
import rclpy.node
from rcl_interfaces.msg import (
    ParameterDescriptor,
    FloatingPointRange,
    IntegerRange,
    ParameterType,
)
from sgdrf_interfaces.msg import CategoricalObservation
from sgdrf_interfaces.srv import TopicProb, WordProb, WordTopicMatrix
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
import torch
import pyro
import pyro.util
from .sgdrf import SGDRF, KernelType, SubsampleType, OptimizerType
from typing import Any, Optional, Union
import sys

RANGETYPE = tuple[Union[int, float], Union[int, float], Union[int, float]]


class SGDRFNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("sgdrf_node")

        self.setup_parameters()

        self.sgdrf = self.initialize_sgdrf()
        self.obs_subscription = self.create_subscription(
            CategoricalObservation,
            f"categorical_observation__{self.sgdrf.V}__",
            self.new_obs_callback,
            10,
        )
        self.loss_publisher = self.create_publisher(Float64, "loss", 10)

        self.topic_prob_service = self.create_service(
            TopicProb, "topic_prob", self.topic_prob_service_callback
        )
        self.word_prob_service = self.create_service(
            WordProb, "word_prob", self.word_prob_service_callback
        )
        self.word_topic_matrix_service = self.create_service(
            WordTopicMatrix,
            "word_topic_matrix",
            self.word_topic_matrix_service_callback,
        )

        self.timer = self.create_timer(0.1, self.training_step_callback)

    @staticmethod
    def generate_parameter(
        name: str,
        value: Any,
        type: int,
        description: str,
        constraints: str = "",
        read_only: bool = False,
        dynamic_typing: bool = False,
        _range: Optional[RANGETYPE] = None,
    ):
        parameter_descriptor_kwargs = {
            "name": name,
            "type": type,
            "description": description,
            "additional_constraints": constraints,
            "read_only": read_only,
            "dynamic_typing": dynamic_typing,
        }
        if _range is not None:
            if type == ParameterType.PARAMETER_INTEGER:
                assert all(
                    isinstance(x, int) for x in _range
                ), f"error: _range must be 3 ints for integer parameter type (you gave {_range})"
                parameter_descriptor_kwargs["integer_range"] = [
                    IntegerRange(
                        from_value=_range[0], to_value=_range[1], step=_range[2]
                    )
                ]
            elif type == ParameterType.PARAMETER_DOUBLE:
                parameter_descriptor_kwargs["floating_point_range"] = [
                    FloatingPointRange(
                        from_value=_range[0], to_value=_range[1], step=_range[2]
                    )
                ]
            else:
                raise ValueError(
                    f"parameter type {type} not compatible with _range argument"
                )
        return (name, value, ParameterDescriptor(**parameter_descriptor_kwargs))

    def setup_parameters(self):
        parameters = []

        parameters.append(
            self.generate_parameter(
                name="dims",
                value=1,
                type=ParameterType.PARAMETER_INTEGER,
                description="number of spatial dimensions",
                _range=(1, 3, 1),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="xu_ns",
                value=[25],
                type=ParameterType.PARAMETER_INTEGER_ARRAY,
                description="number of inducing points per dimension",
                constraints="one positive integer per dimension",
            )
        )
        parameters.append(
            self.generate_parameter(
                name="d_mins",
                value=[0.0],
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="minimum value of each dimension",
                constraints="one float per dimension",
            )
        )
        parameters.append(
            self.generate_parameter(
                name="d_maxs",
                value=[1.0],
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="maximum value of each dimension",
                constraints="one float per dimension, greater than the minimum",
            )
        )
        parameters.append(
            self.generate_parameter(
                name="V",
                value=10,
                type=ParameterType.PARAMETER_INTEGER,
                description="number of observation categories",
                _range=(2, sys.maxsize, 1),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="K",
                value=2,
                type=ParameterType.PARAMETER_INTEGER,
                description="number of latent communities",
                _range=(2, sys.maxsize, 1),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="max_obs",
                value=1000,
                type=ParameterType.PARAMETER_INTEGER,
                description="maximum number of simultaneous categorical observations",
                _range=(2, sys.maxsize, 1),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="dir_p",
                value=1.0,
                type=ParameterType.PARAMETER_DOUBLE,
                description="uniform dirichlet hyperparameter",
                constraints="must be greater than zero",
                _range=(0, sys.float_info.max, 0),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="kernel_type",
                value="Matern32",
                type=ParameterType.PARAMETER_STRING,
                description="kernel type",
                constraints="must be one of: " + str(KernelType._member_names_),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="kernel_lengthscale",
                value=1.0,
                type=ParameterType.PARAMETER_DOUBLE,
                description="isotropic kernel lengthscale",
                constraints="must be greater than zero",
                _range=(0, sys.float_info.max, 0),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="kernel_variance",
                value=1.0,
                type=ParameterType.PARAMETER_DOUBLE,
                description="isotropic kernel variance",
                constraints="must be greater than zero",
                _range=(0, sys.float_info.max, 0),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="optimizer_type",
                value="Adam",
                type=ParameterType.PARAMETER_STRING,
                description="optimizer type",
                constraints="must be one of: " + str(OptimizerType._member_names_),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="optimizer_lr",
                value=0.001,
                type=ParameterType.PARAMETER_DOUBLE,
                description="optimizer learning rate",
                constraints="must be greater than zero",
                _range=(0, sys.float_info.max, 0),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="optimizer_clip_norm",
                value=10.0,
                type=ParameterType.PARAMETER_DOUBLE,
                description="optimizer norm maximum allowed value",
                constraints="must be greater than zero",
                _range=(0, sys.float_info.max, 0),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="device",
                value="cpu",
                type=ParameterType.PARAMETER_STRING,
                description="pytorch device to use",
                constraints='must be a valid torch device (e.g. "cpu", "cuda", "cuda:0", etc.)',
            )
        )
        parameters.append(
            self.generate_parameter(
                name="subsample_n",
                value=5,
                type=ParameterType.PARAMETER_INTEGER,
                description="number of past observations for each subsample",
                _range=(1, sys.maxsize, 1),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="subsample_type",
                value="uniform",
                type=ParameterType.PARAMETER_STRING,
                description="subsample type",
                constraints="must be one of: " + str(SubsampleType._member_names_),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="subsample_weight",
                value=0.5,
                type=ParameterType.PARAMETER_DOUBLE,
                description="weight to assign to first component of compound subsample strategy",
                _range=(0.0, 1.0, 0),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="subsample_exp",
                value=0.1,
                type=ParameterType.PARAMETER_DOUBLE,
                description="exponential parameter for subsample strategy, if applicable",
                constraints="must be greater than zero",
                _range=(0, sys.float_info.max, 0),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="whiten",
                value=False,
                type=ParameterType.PARAMETER_BOOL,
                description="whether or not the GP inputs are whitened",
            )
        )
        parameters.append(
            self.generate_parameter(
                name="fail_on_nan_loss",
                value=False,
                type=ParameterType.PARAMETER_BOOL,
                description="whether or not to fail if a NaN loss is encountered",
            )
        )
        parameters.append(
            self.generate_parameter(
                name="num_particles",
                value=1,
                type=ParameterType.PARAMETER_INTEGER,
                description="number of parallel samples from approximate posterior",
                _range=(1, sys.maxsize, 1),
            )
        )
        parameters.append(
            self.generate_parameter(
                name="jit",
                value=False,
                type=ParameterType.PARAMETER_BOOL,
                description="whether or not to JIT compile the prior and approximate posterior",
            )
        )
        parameters.append(
            self.generate_parameter(
                name="random_seed",
                value=777,
                type=ParameterType.PARAMETER_INTEGER,
                description="random seed",
                _range=(0, sys.maxsize, 1),
            )
        )
        self.declare_parameters(self.get_name(), parameters)

    def init_random_seed(self):
        random_seed = (
            self.get_parameter(f"{self.get_name()}.random_seed")
            .get_parameter_value()
            .integer_value
        )
        pyro.util.set_rng_seed(random_seed)

    def initialize_sgdrf(self):
        dims = (
            self.get_parameter(f"{self.get_name()}.dims")
            .get_parameter_value()
            .integer_value
        )
        xu_ns = (
            self.get_parameter(f"{self.get_name()}.xu_ns")
            .get_parameter_value()
            .integer_array_value
        )
        d_mins = (
            self.get_parameter(f"{self.get_name()}.d_mins")
            .get_parameter_value()
            .double_array_value
        )
        d_maxs = (
            self.get_parameter(f"{self.get_name()}.d_maxs")
            .get_parameter_value()
            .double_array_value
        )

        # check that the xu_ns, d_mins, d_maxs, and dims all agree
        assert len(d_mins) == len(d_maxs), "dimensions constraint lengths must match"
        assert (len(xu_ns) == 1) or (
            len(xu_ns) == len(d_mins)
        ), "inducing point numbers and dimensions constraint lengths must match"
        assert (dims == len(d_mins)) or (
            len(d_mins) == 1
        ), "dimension constraint lengths must match number of dimensions"
        if len(xu_ns) == 1:
            xu_ns = xu_ns * dims
        if len(d_mins) == 1:
            d_mins = d_mins * dims
        if len(d_maxs) == 1:
            d_maxs = d_maxs * dims

        assert all(
            x < y for x, y in zip(d_mins, d_maxs)
        ), "d_mins must be smaller than d_maxs in every dimension"

        V = (
            self.get_parameter(f"{self.get_name()}.V")
            .get_parameter_value()
            .integer_value
        )
        K = (
            self.get_parameter(f"{self.get_name()}.K")
            .get_parameter_value()
            .integer_value
        )
        max_obs = (
            self.get_parameter(f"{self.get_name()}.max_obs")
            .get_parameter_value()
            .integer_value
        )
        dir_p = (
            self.get_parameter(f"{self.get_name()}.dir_p")
            .get_parameter_value()
            .double_value
        )
        assert dir_p > 0, "dir_p must be greater than zero"
        kernel_type_string = (
            self.get_parameter(f"{self.get_name()}.kernel_type")
            .get_parameter_value()
            .string_value
        )
        assert (
            kernel_type_string in KernelType._member_names_
        ), "kernel_type must be one of " + str(KernelType._member_names_)
        kernel_type = KernelType[kernel_type_string]
        kernel_lengthscale = (
            self.get_parameter(f"{self.get_name()}.kernel_lengthscale")
            .get_parameter_value()
            .double_value
        )
        assert kernel_lengthscale > 0, "kernel_lengthscale must be greater than zero"
        kernel_variance = (
            self.get_parameter(f"{self.get_name()}.kernel_variance")
            .get_parameter_value()
            .double_value
        )
        assert kernel_variance > 0, "kernel_variance must be greater than zero"
        optimizer_type_string = (
            self.get_parameter(f"{self.get_name()}.optimizer_type")
            .get_parameter_value()
            .string_value
        )
        assert (
            optimizer_type_string in OptimizerType._member_names_
        ), "optimizer_type must be one of " + str(OptimizerType._member_names_)
        optimizer_type = OptimizerType[optimizer_type_string]
        optimizer_lr = (
            self.get_parameter(f"{self.get_name()}.optimizer_lr")
            .get_parameter_value()
            .double_value
        )
        assert optimizer_lr > 0, "optimizer_lr must be greater than zero"
        optimizer_clip_norm = (
            self.get_parameter(f"{self.get_name()}.optimizer_clip_norm")
            .get_parameter_value()
            .double_value
        )
        assert optimizer_clip_norm > 0, "optimizer_lr must be greater than zero"
        device_string = (
            self.get_parameter(f"{self.get_name()}.device")
            .get_parameter_value()
            .string_value
        )
        device = torch.device(device_string)
        subsample_n = (
            self.get_parameter(f"{self.get_name()}.subsample_n")
            .get_parameter_value()
            .integer_value
        )
        subsample_type_string = (
            self.get_parameter(f"{self.get_name()}.subsample_type")
            .get_parameter_value()
            .string_value
        )
        assert (
            subsample_type_string in SubsampleType._member_names_
        ), "subsample_type must be one of " + str(SubsampleType._member_names_)
        subsample_type = SubsampleType[subsample_type_string]
        subsample_weight = (
            self.get_parameter(f"{self.get_name()}.subsample_weight")
            .get_parameter_value()
            .double_value
        )
        subsample_exp = (
            self.get_parameter(f"{self.get_name()}.subsample_exp")
            .get_parameter_value()
            .double_value
        )
        assert subsample_exp > 0, "subsample_exp must be greater than zero"
        whiten = (
            self.get_parameter(f"{self.get_name()}.whiten")
            .get_parameter_value()
            .bool_value
        )
        fail_on_nan_loss = (
            self.get_parameter(f"{self.get_name()}.fail_on_nan_loss")
            .get_parameter_value()
            .bool_value
        )
        num_particles = (
            self.get_parameter(f"{self.get_name()}.num_particles")
            .get_parameter_value()
            .integer_value
        )
        jit = (
            self.get_parameter(f"{self.get_name()}.jit")
            .get_parameter_value()
            .bool_value
        )
        subsample_params = {
            "subsample_weight": subsample_weight,
            "subsample_exp": subsample_exp,
        }
        sgdrf = SGDRF(
            xu_ns=xu_ns,
            d_mins=d_mins,
            d_maxs=d_maxs,
            V=V,
            K=K,
            max_obs=max_obs,
            dir_p=dir_p,
            kernel_type=kernel_type,
            kernel_lengthscale=kernel_lengthscale,
            kernel_variance=kernel_variance,
            optimizer_type=optimizer_type,
            optimizer_lr=optimizer_lr,
            optimizer_clip_norm=optimizer_clip_norm,
            device=device,
            subsample_n=subsample_n,
            subsample_type=subsample_type,
            subsample_params=subsample_params,
            whiten=whiten,
            fail_on_nan_loss=fail_on_nan_loss,
            num_particles=num_particles,
            jit=jit,
        )
        return sgdrf

    def categorical_observation_to_tensors(self, msg: CategoricalObservation):
        pose_stamped = msg.pose_stamped
        obs = msg.obs
        assert (
            len(obs) == self.sgdrf.V
        ), "message observation length does not match SGDRF initialized vocabulary size"
        pose = pose_stamped.pose
        point = pose.position
        x_raw = [point.x]
        if self.sgdrf.dims >= 2:
            x_raw.append([point.y])
        if self.sgdrf.dims == 3:
            x_raw.append([point.z])
        xs = torch.tensor(x_raw, dtype=torch.float, device=self.sgdrf.device).unsqueeze(
            0
        )
        ws = torch.tensor(obs, dtype=torch.int, device=self.sgdrf.device).unsqueeze(0)
        return xs, ws

    def new_obs_callback(self, msg: CategoricalObservation):
        xs, ws = self.categorical_observation_to_tensors(msg)
        self.sgdrf.process_inputs(xs, ws)

    def training_step_callback(self):
        if self.sgdrf.n_xs > 0:
            loss = self.sgdrf.step()

            loss_msg = Float64()
            loss_msg.data = loss
            self.loss_publisher.publish(loss_msg)

    def point_array_to_tensor(self, point_array: list[Point]):
        coord_list = []
        if self.sgdrf.dims >= 1:
            coord_list += [[p.x for p in point_array]]
        if self.sgdrf.dims >= 2:
            coord_list += [[p.y for p in point_array]]
        if self.sgdrf.dims >= 3:
            coord_list += [[p.z for p in point_array]]
        return torch.tensor(coord_list, dtype=torch.float, device=self.sgdrf.device)

    def topic_prob_service_callback(self, request, response):
        xs = self.point_array_to_tensor(request.xs)
        response.probs = self.sgdrf.topic_prob(xs).detach().cpu().squeeze().tolist()
        return response

    def word_prob_service_callback(self, request, response):
        xs = self.point_array_to_tensor(request.xs)
        response.probs = self.sgdrf.word_prob(xs).detach().cpu().squeeze().tolist()
        return response

    def word_topic_matrix_service_callback(self, request, response):
        wt_prob = self.sgdrf.word_topic_prob()
        response.probs = torch.flatten(wt_prob.detach().cpu()).tolist()  # type: ignore
        return response


def main():
    rclpy.init()
    node = SGDRFNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
