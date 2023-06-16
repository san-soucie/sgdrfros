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

import pyro
import pyro.distributions as dist
import pyro.distributions.util
import pyro.infer
import pyro.contrib
import pyro.optim
import pyro.util
from pyro.contrib.gp.util import conditional
from pyro.nn.module import PyroParam, pyro_method

import torch
import torch.distributions
import torch.distributions.constraints
from torch.nn.functional import normalize
from typing import Union, Collection, Optional
from .optimizer import OptimizerType
from .kernel import KernelType
from .subsample import SubsampleType

EPSILON = 1e-2


class SGDRF(pyro.contrib.gp.Parameterized):
    def __init__(
        self,
        xu_ns: Union[int, Collection[int]],
        d_mins: Collection[float],
        d_maxs: Collection[float],
        V: int,
        K: int,
        max_obs: int,
        dir_p: float,
        kernel_type: KernelType = KernelType.Matern52,
        kernel_lengthscale: float = 1.0,
        kernel_variance: float = 1.0,
        optimizer_type: OptimizerType = OptimizerType.Adam,
        optimizer_lr: float = 0.01,
        optimizer_clip_norm: float = 10.0,
        device: torch.device = torch.device("cpu"),
        subsample_n: int = 1,
        subsample_type: SubsampleType = SubsampleType.uniform,
        subsample_params: Optional[dict[str, float]] = None,
        whiten: bool = False,
        fail_on_nan_loss: bool = True,
        num_particles: int = 1,
        jit: bool = False,
    ):
        super().__init__()

        xu_dims = []
        if isinstance(xu_ns, int):
            xu_ns = [xu_ns] * len(d_mins)
        for min_d, max_d, xu_n in zip(d_mins, d_maxs, xu_ns):
            delta_d = max_d - min_d
            dd = delta_d / (xu_n - 1)
            xu_dims.append(torch.arange(start=min_d, end=max_d + dd / 2, step=dd))
        xu = torch.cartesian_prod(*xu_dims).to(device).type(torch.float)
        if len(xu.shape) == 1:
            xu = xu.unsqueeze(-1)
        self.xu = xu
        self.dims = len(d_maxs)

        self.V = V
        self.K = K
        self.M = self.xu.size(0)
        self.latent_shape = (self.K,)
        self.max_obs = max_obs
        self.device = device
        self.dir_p = torch.tensor(
            [[dir_p] * V] * K, dtype=torch.float, device=device, requires_grad=True
        )
        self.jitter = 1e-5
        self.zero_loc = self.xu.new_zeros(self.latent_shape + (self.M,))
        self.uloc = PyroParam(self.zero_loc, dist.constraints.real, event_dim=None)
        self.uscaletril = PyroParam(
            pyro.distributions.util.eye_like(self.xu, self.M).repeat(
                self.latent_shape + (1, 1)
            ),
            constraint=dist.constraints.stack(
                [dist.constraints.lower_cholesky for _ in range(K)], dim=-3
            ),
            event_dim=None,
        )
        self.word_topic_probs = PyroParam(
            self.dir_p,
            constraint=dist.constraints.stack(
                [dist.constraints.simplex for _ in range(K)], dim=-2
            ),
            event_dim=None,
        )
        self.kernel = kernel_type.instantiate(
            input_dim=self.xu.size(1),
            variance=torch.tensor(kernel_variance, dtype=torch.float, device=device),
            lengthscale=torch.tensor(
                kernel_lengthscale, dtype=torch.float, device=device
            ),
        )

        self.whiten = whiten
        if subsample_params is None:
            subsample_params = {"exponential": 0.1, "weight": 0.5}
        self.subsample_type = subsample_type
        self.subsample_params = subsample_params
        self.subsample_n = subsample_n
        self.subsample_n_tensor = torch.tensor(
            self.subsample_n, dtype=torch.int, device=self.device
        )
        self.num_particles = num_particles
        self.objective_type = pyro.infer.JitTrace_ELBO if jit else pyro.infer.Trace_ELBO
        self.objective = self.objective_type(
            num_particles=self.num_particles,
            vectorize_particles=True,
            max_plate_nesting=1,
        )
        self.xs = torch.empty(0, *self.xu.shape[1:], device=device, dtype=torch.float)
        self.ws = torch.empty(0, self.V, device=device, dtype=torch.int)
        self.optimizer = optimizer_type.instantiate(
            lr=optimizer_lr, clip_norm=optimizer_clip_norm
        )
        self.svi = pyro.infer.SVI(
            self.model, self.guide, self.optimizer, self.objective
        )
        self.fail_on_nan_loss = fail_on_nan_loss

        self.n_xs = 0

    @staticmethod
    def entropy(x: torch.Tensor) -> torch.Tensor:
        p = torch.div(x, torch.linalg.vector_norm(x, ord=1, dim=-1, keepdim=True))
        sij = -p * torch.log(p)
        return torch.sum(sij, dim=-2)

    def subsample(self, t=None):
        n = self.subsample_n
        t = t if t is not None else self.xs.size(0) - 1
        if (self.subsample_type == "full") or (t <= n):
            return torch.tensor(list(range(t + 1)), dtype=torch.long)

        rtp1 = torch.arange(t + 1, device=self.device)
        latest = normalize(
            torch.tensor(
                [0 for _ in range(t)]
                + [
                    1,
                ],
                device=self.device,
            ),
            p=1.0,
        )
        exponential = normalize(
            torch.exp(-(t - rtp1) * self.subsample_params["exponential"]), p=1.0
        )
        uniform = normalize(0 * rtp1 + 1, p=1.0)
        if t < 1 or self.subsample_type == "latest":
            probs = latest
        elif self.subsample_type == "exponential":
            probs = exponential
        elif self.subsample_type == "uniform":
            probs = uniform
        elif self.subsample_type == "exponential+uniform":
            probs = normalize(
                exponential * self.subsample_params["weight"]
                + uniform * (1.0 - self.subsample_params["weight"]),
                p=1.0,
            )
        elif self.subsample_type == "exponential+latest":
            probs = normalize(
                exponential * self.subsample_params["weight"]
                + latest * (1.0 - self.subsample_params["weight"]),
                p=1.0,
            )
        elif self.subsample_type == "uniform+latest":
            probs = normalize(
                uniform * self.subsample_params["weight"]
                + latest * (1.0 - self.subsample_params["weight"]),
                p=1.0,
            )
        else:
            raise ValueError(f'invalid subsample_type "{self.subsample_type}"')
        return dist.Categorical(probs).sample([n])

    def topic_prob(self, xs=None):
        f_loc, _ = conditional(
            xs if xs is not None else self.xs,
            self.xu,
            self.kernel,
            self.uloc,
            self.uscaletril,
            full_cov=False,
            whiten=self.whiten,
            jitter=self.jitter,
        )
        topic_probs = torch.softmax(f_loc, -2).squeeze(-2)
        return topic_probs

    def word_topic_prob(self):
        return self.word_topic_probs

    def word_prob(self, xs=None):
        topic_probs = self.topic_prob(xs)
        word_topic_probs = self.word_topic_prob()
        return topic_probs.transpose(-2, -1) @ word_topic_probs

    @staticmethod
    def check_inputs(xs=None, ws=None):
        if xs is None or ws is None:
            assert (xs is None) and (ws is None), "inputs do not agree"
        else:
            assert not ((xs is None) and (ws is None)) and (
                xs.size(0) == ws.size(0)
            ), "inputs do not agree"

    def perplexity(self, xs=None, ws=None):
        self.check_inputs(xs, ws)
        xs = xs if xs is not None else self.xs
        ws = ws if ws is not None else self.ws
        return (
            ((ws * self.word_prob(xs).log()).sum() / -ws.sum())
            .exp()
            .detach()
            .cpu()
            .item()
        )

    def process_inputs(self, xs=None, ws=None):
        self.check_inputs(xs, ws)
        if xs is not None:
            self.xs = torch.cat((self.xs, xs), dim=0)
            self.ws = torch.cat((self.ws, ws), dim=0)
            self.n_xs += self.xs.shape[0]

    def get_step_args_kwargs(self):
        args = [
            self.xs,
            self.ws,
            self.xu,
            self.uloc,
            self.uscaletril,
            self.subsample(),
            self.dir_p,
            self.lengthscale,
            self.variance,
            self.jitter,
        ]
        kwargs = dict(K=self.K, whiten=self.whiten, max_obs=self.max_obs)
        return args, kwargs

    def step(self):
        loss = self.svi.step(self.xs, self.ws, self.subsample())
        if self.fail_on_nan_loss and pyro.util.torch_isnan(loss):
            raise ValueError("loss is NaN")
        return loss

    def dry_run(self, xs, ws):
        self.check_inputs(xs, ws)
        self.model(xs, ws, self.subsample())
        self.guide(xs, ws, self.subsample())

    @pyro_method
    def model(self, xs, ws, subsample):
        self.set_mode("model")
        N = xs.size(0)

        with pyro.plate("topics", self.K, device=xs.device):
            with pyro.util.ignore_jit_warnings():
                Kuu = self.kernel(self.xu).contiguous()
            Luu = torch.linalg.cholesky(Kuu)
            sc = (
                pyro.distributions.util.eye_like(self.xu, self.M)
                if self.whiten
                else Luu
            )
            pyro.sample(
                "log_topic_prob_u",
                dist.MultivariateNormal(self.zero_loc, scale_tril=sc),
            )
            word_topic_probs = pyro.sample(
                "word_topic_prob", dist.Dirichlet(self.word_topic_probs)
            )
            with pyro.util.ignore_jit_warnings():
                f_loc, _ = conditional(
                    xs,
                    self.xu,
                    self.kernel,
                    self.uloc,
                    self.uscaletril,
                    Lff=Luu,
                    full_cov=False,
                    whiten=self.whiten,
                    jitter=self.jitter,
                )
        topic_probs = torch.softmax(f_loc, -2).contiguous().squeeze(-2)
        word_probs = topic_probs.transpose(-2, -1) @ word_topic_probs

        with pyro.plate("words", size=N, device=xs.device, subsample=subsample) as i:
            obs = pyro.sample(
                "obs",
                dist.Multinomial(total_count=self.max_obs, probs=word_probs[..., i, :]),
                obs=ws[..., i, :],
            )
        return obs

    @pyro_method
    def guide(self, xs, ws, subsample):
        self.set_mode("guide")
        self._load_pyro_samples()

        with pyro.plate("topics", self.K, device=xs.device):
            pyro.sample(
                "log_topic_prob_u",
                dist.MultivariateNormal(self.uloc, scale_tril=self.uscaletril),
            )
            pyro.sample(
                "word_topic_prob",
                pyro.distributions.Delta(self.word_topic_probs).to_event(1),
            )

    def forward(self, xs):
        self.set_mode("guide")
        return self.word_prob(xs)
