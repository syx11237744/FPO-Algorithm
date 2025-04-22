# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
# ==============================================================================
"""Implementation of the FPONG algorithm."""

from __future__ import annotations

import torch
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter import FPOAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base import NaturalPG
from omnisafe.common.buffer import VectorFPOBuffer
from omnisafe.models.actor_critic.fpo_actor_critic import FPOActorCritic
from omnisafe.utils import distributed
from omnisafe.utils.CustomDataLoader import CustomDataLoader
from omnisafe.common.lagrange import Lagrange


@registry.register
class FPONG(NaturalPG):
    """The Feasible Policy Optimization with Natural Gradient (FPONG) algorithm.

    """

    def _init_env(self) -> None:
        """Initialize the environment.

        FPO uses :class:`omnisafe.adapter.FPOAdapter` to adapt the environment to the
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._env: FPOAdapter = FPOAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """
        self._actor_critic: FPOActorCritic = FPOActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self._buf: VectorFPOBuffer = VectorFPOBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            cost_gamma=self._cfgs.algo_cfgs.cost_gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )
        self._lagrange_in_region: Lagrange = Lagrange(**self._cfgs.lagrange_in_cfgs)
        self._lagrange_out_region: Lagrange = Lagrange(**self._cfgs.lagrange_out_cfgs)
        self._feasibility_threshold = self._cfgs.algo_cfgs.feasibility_threshold
        self._leaky_relu = torch.nn.LeakyReLU(negative_slope=self._cfgs.algo_cfgs.leaky_alpha)

    def _init_log(self) -> None:
        super()._init_log()

        # log information about actor
        self._logger.register_key('Value/Adv_r')
        self._logger.register_key('Value/Adv_c')
        self._logger.register_key('Value/Adv_rc')
        self._logger.register_key('Value/Adv_c_unstandardized')

        # log information about cost critic
        self._logger.register_key('Loss/Loss_recover_critic', delta=True)
        self._logger.register_key('Value/recover')

        # log information about lagrange multipliers
        self._logger.register_key('Train/penalty_term_in')
        self._logger.register_key('Train/penalty_term_out')
        self._logger.register_key('Train/in_region_ratio')
        self._logger.register_key('Metrics/InRegionLagrangeMultiplier')
        self._logger.register_key('Metrics/OutRegionLagrangeMultiplier')

    def _update(self) -> None:
        # update policy and value function
        data = self._buf.get()
        self._update_actor_critic(data)

        # update Lagrange multipliers
        penalty_term_in, penalty_term_out = self._calculate_penalty_term(data)
        self._lagrange_in_region.update_lagrange_multiplier(penalty_term_in)
        self._lagrange_out_region.update_lagrange_multiplier(penalty_term_out)

        self._logger.store({
            'Train/penalty_term_in': penalty_term_in,
            'Train/penalty_term_out': penalty_term_out,
            'Metrics/InRegionLagrangeMultiplier': self._lagrange_in_region.lagrangian_multiplier.item(),
            'Metrics/OutRegionLagrangeMultiplier': self._lagrange_out_region.lagrangian_multiplier.item(),
        })

    def _update_actor_critic(self, data: dict[str, torch.Tensor]) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. hint::

            +----------------+------------------------------------------------------------------+
            | obs            | ``observation`` sampled from buffer.                             |
            +================+==================================================================+
            | act            | ``action`` sampled from buffer.                                  |
            +----------------+------------------------------------------------------------------+
            | target_value_r | ``target reward value`` sampled from buffer.                     |
            +----------------+------------------------------------------------------------------+
            | target_value_c | ``target feasibility value`` sampled from buffer.                       |
            +----------------+------------------------------------------------------------------+
            | logp           | ``log probability`` sampled from buffer.                         |
            +----------------+------------------------------------------------------------------+
            | adv_r          | ``estimated advantage`` (e.g. **GAE**) sampled from buffer.      |
            +----------------+------------------------------------------------------------------+
            | adv_c          | ``estimated feasibility advantage`` (e.g. **GAE**) sampled from buffer. |
            +----------------+------------------------------------------------------------------+


        -  Update value net by :meth:`_update_reward_critic`.
        -  Update feasibility net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the data from buffer.
        #. Shuffle the data and split it into mini-batch data.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the number of mini-batch data is used up.
        #. Repeat steps 2, 3, 4 until the KL divergence violates the limit.
        """

        obs = data['obs']
        act = data['act']
        cost = data['cost']
        logp = data['logp']
        target_value_r = data['target_value_r']
        target_value_c = data['target_value_c']
        target_value_rc = data['target_value_rc']
        adv_r = data['adv_r']
        adv_c = data['adv_c']
        adv_rc = data['adv_rc']
        value_c = data['value_c']
        unstandardized_adv_c = data['unstandardized_adv_c']

        fea = value_c < self._feasibility_threshold
        vio = cost > 0

        in_region_multiplier = self._lagrange_in_region.lagrangian_multiplier.item()
        out_region_multiplier = self._lagrange_out_region.lagrangian_multiplier.item()

        weight = torch.clamp((1 - value_c / self._feasibility_threshold), 0, 1) ** in_region_multiplier
        weight = weight + (1 - weight) / (1 + out_region_multiplier)

        adv_in = weight * adv_r - (1 - weight) * adv_c
        adv_out = (adv_r - out_region_multiplier * adv_c) / (1 + out_region_multiplier)
        adv = torch.where(vio, adv_rc, torch.where(fea, adv_in, adv_out))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self._update_actor(obs, act, logp, adv, None)

        dataloader = CustomDataLoader(
            obs, target_value_r, target_value_c, target_value_rc,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True
        )

        update_counts = 0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for (obs, target_value_r, target_value_c, target_value_rc) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                self._update_cost_critic(obs, target_value_c)
                self._update_recover_critic(obs, target_value_rc)

                update_counts += 1

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv_r': adv_r.mean().item(),
                'Value/Adv_c': adv_c.mean().item(),
                'Value/Adv_rc': adv_rc.mean().item(),
                'Value/Adv_c_unstandardized': unstandardized_adv_c.mean().item(),
                'Train/in_region_ratio': fea.float().mean().item(),
            },
        )

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        return adv_r

    def _calculate_penalty_term(
        self,
        data: dict[str, torch.Tensor],
    ) -> tuple[float, float]:
        obs = data['obs']
        act = data['act']
        logp = data['logp']
        unstandardized_adv_c = data['unstandardized_adv_c']
        value_c = data['value_c']

        with torch.no_grad():
            _ = self._actor_critic.actor(obs)
            logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)

        term_out = unstandardized_adv_c * ratio
        term_in = term_out + value_c - self._feasibility_threshold

        fea = value_c < self._feasibility_threshold

        penalty_term_in = masked_mean(self._leaky_relu(term_in), fea).item()
        penalty_term_out = masked_mean(self._leaky_relu(term_out), ~fea).item()

        return penalty_term_in, penalty_term_out

    def _update_recover_critic(self, obs: torch.Tensor, target_value_rc: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_c (torch.Tensor): The ``target_value_c`` sampled from buffer.
        """
        self._actor_critic.recover_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.recover_critic(obs)[0], target_value_rc)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.recover_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.recover_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.recover_critic)
        self._actor_critic.recover_critic_optimizer.step()

        self._logger.store({'Loss/Loss_recover_critic': loss.mean().item()})


def masked_mean(x, mask):
    return (x * mask).sum() / max(mask.sum(), 1)
