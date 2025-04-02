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
"""Implementation of the Policy Gradient algorithm."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import FPOAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base import PolicyGradient
from omnisafe.common.buffer import VectorFPOBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.fpo_actor_critic import FPOActorCritic
from omnisafe.utils import distributed
from omnisafe.common.lagrange import Lagrange


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods,line-too-long
class FPO(PolicyGradient):
    """The FPO algorithm.

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
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
            cost_gamma=self._cfgs.algo_cfgs.cost_gamma,
            feasibility_threshold=self._cfgs.algo_cfgs.feasibility_threshold,
        )
        self._lagrange_out_region: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
        self._feasibility_threshold = self._cfgs.algo_cfgs.feasibility_threshold
        self._leaky_relu = torch.nn.LeakyReLU(negative_slope=self._cfgs.algo_cfgs.leaky_alpha)

    def _init_log(self) -> None:
        """Log info about epoch.

        +-----------------------+----------------------------------------------------------------------+
        | Things to log         | Description                                                          |
        +=======================+======================================================================+
        | Train/Epoch           | Current epoch.                                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpCost        | Average cost of the epoch.                                           |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpRet         | Average return of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpLen         | Average length of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Values/reward         | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-----------------------+----------------------------------------------------------------------+
        | Values/cost           | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-----------------------+----------------------------------------------------------------------+
        | Values/Adv            | Average reward advantage of the epoch.                               |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi          | Loss of the policy network.                                          |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic | Loss of the feasibility critic network.                                     |
        +-----------------------+----------------------------------------------------------------------+
        | Train/Entropy         | Entropy of the policy network.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Train/StopIters       | Number of iterations of the policy network.                          |
        +-----------------------+----------------------------------------------------------------------+
        | Train/PolicyRatio     | Ratio of the policy network.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Train/LR              | Learning rate of the policy network.                                 |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/Seed             | Seed of the experiment.                                              |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps    | Total steps of the experiment.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Time                  | Total time.                                                          |
        +-----------------------+----------------------------------------------------------------------+
        | FPS                   | Frames per second of the epoch.                                      |
        +-----------------------+----------------------------------------------------------------------+
        """
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self._actor_critic.actor
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key(
            'Metrics/EpRet',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpCost',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpLen',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Train/PolicyRatio', min_and_max=True)
        self._logger.register_key('Train/LR')
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self._logger.register_key('Train/PolicyStd')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Value/Adv_r')
        self._logger.register_key('Value/Adv_c')
        self._logger.register_key('Value/Adv_rc')
        self._logger.register_key('Value/Adv_c_unstandardized')

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')

        # log information about cost critic
        self._logger.register_key('Loss/Loss_cost_critic', delta=True)
        self._logger.register_key('Loss/Loss_recover_critic', delta=True)
        self._logger.register_key('Value/cost')
        self._logger.register_key('Value/recover')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

        self._logger.register_key('Train/penalty_term_in')
        self._logger.register_key('Train/penalty_term_out')
        self._logger.register_key('Train/in_region_ratio')
        self._logger.register_key('Train/neg_in_ratio')
        self._logger.register_key('Metrics/InRegionLagrangeMultiplier')
        self._logger.register_key('Metrics/InRegionLagrangeMultiplierStd')
        self._logger.register_key('Metrics/OutRegionLagrangeMultiplier')

        # register environment specific keys
        for env_spec_key in self._env.env_spec_keys:
            self.logger.register_key(env_spec_key)
    
    def _update(self) -> None:
        
        # then update the policy and value function
        data = self._buf.get()
        self._update_actor_critic(data)

        # compute penalty_term_in and out
        penalty_term_in, penalty_term_out = self._calculate_penalty_term(data)

        # update Lagrange multiplier parameter
        statewise_multiplier = self._actor_critic.multiplier(data['value_c'].unsqueeze(1))[0]
        statewise_multiplier_loss = masked_mean(-statewise_multiplier * penalty_term_in, data['mask_in_region'])
        self._actor_critic.multiplier_optimizer.zero_grad()
        statewise_multiplier_loss.backward()
        self._actor_critic.multiplier_optimizer.step()

        self._lagrange_out_region.update_lagrange_multiplier(penalty_term_out)

        statewise_multiplier = statewise_multiplier.detach()
        self._logger.store({'Metrics/InRegionLagrangeMultiplier': statewise_multiplier.mean().item()})
        self._logger.store({'Metrics/InRegionLagrangeMultiplierStd': statewise_multiplier.std().item()})
        self._logger.store({'Metrics/OutRegionLagrangeMultiplier': self._lagrange_out_region.lagrangian_multiplier.item()})

    def _update_actor_critic(
            self,
            data: dict[str, torch.Tensor],
        ) -> None:
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
        mask_in_region = data['mask_in_region']

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        with torch.no_grad():
            statewise_multiplier = self._actor_critic.multiplier(value_c.unsqueeze(1))[0]

        # dataloader = DataLoader(
        #     dataset=TensorDataset(
        #         obs, act, cost, logp, target_value_r, target_value_c, target_value_rc, adv_r, adv_c, adv_rc,
        #         mask_in_region, statewise_multiplier),
        #     batch_size=self._cfgs.algo_cfgs.batch_size,
        #     shuffle=False,
        # )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            # for (
            #     obs,
            #     act,
            #     cost,
            #     logp,
            #     target_value_r,
            #     target_value_c,
            #     target_value_rc,
            #     adv_r,
            #     adv_c,
            #     adv_rc,
            #     mask_in_region,
            #     statewise_multiplier,
            # ) in dataloader:
            self._update_reward_critic(obs, target_value_r)
            self._update_cost_critic(obs, target_value_c)
            self._update_recover_critic(obs, target_value_rc)
            self._update_actor(obs, act, cost, logp, adv_r, adv_c, adv_rc, mask_in_region, statewise_multiplier)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv_r': adv_r.mean().item(),
                'Value/Adv_c': adv_c.mean().item(),
                'Value/Adv_rc': adv_rc.mean().item(),
                'Train/KL': final_kl,
            },
        )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        adv_rc: torch.Tensor,
        mask_in_region: torch.Tensor,
        statewise_multiplier: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        #. Compute the loss function.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        .. warning::
            For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
            the ``KL divergence`` between the old policy and the new policy is calculated.
            And the ``KL divergence`` is used to determine whether the update is successful.
            If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log_p`` sampled from buffer.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``feasibility_advantage`` sampled from buffer.
        """
        loss = self._loss_pi(
            obs=obs,
            act=act,
            cost=cost,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            adv_rc=adv_rc,
            mask_in_region=mask_in_region,
            statewise_multiplier=statewise_multiplier,
        )
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        adv_rc: torch.Tensor,
        mask_in_region: torch.Tensor,
        statewise_multiplier: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        In Proximal Policy Optimization, the loss is defined as:

        .. math::

            L^{CLIP} = \underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                \min ( r_t A^{R}_{\pi_{\theta}} (s_t, a_t) , \text{clip} (r_t, 1 - \epsilon, 1 + \epsilon)
                A^{R}_{\pi_{\theta}} (s_t, a_t)
            \right]

        where :math:`r_t = \frac{\pi_{\theta}^{'} (a_t|s_t)}{\pi_{\theta} (a_t|s_t)}`,
        :math:`\epsilon` is the clip parameter, and :math:`A^{R}_{\pi_{\theta}} (s_t, a_t)` is the
        advantage.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` processed. ``reward_advantage`` here.

        Returns:
            The loss of pi/actor.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        out_region_multiplier = self._lagrange_out_region.lagrangian_multiplier.item()

        adv = torch.where(~mask_in_region, (adv_r - out_region_multiplier * adv_c) / (1 + out_region_multiplier),
                          (adv_r - statewise_multiplier * adv_c) / (1 + statewise_multiplier))
        adv = torch.where(cost > 0, adv_rc, adv)
        # standized adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ratio_cliped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )
        loss = -torch.min(ratio * adv, ratio_cliped * adv).mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()
        # useful extra info
        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss

    def _calculate_penalty_term(
        self,
        data: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, float]:
        obs = data['obs']
        act = data['act']
        logp = data['logp']
        unstandardized_adv_c = data['unstandardized_adv_c']
        value_c = data['value_c']
        mask_in_region = data['mask_in_region']

        with torch.no_grad():
            _ = self._actor_critic.actor(obs)
            logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)

        term_out = unstandardized_adv_c * ratio
        term_in = term_out + value_c - self._feasibility_threshold

        in_region_ratio = mask_in_region.float().mean().item()
        neg_in_ratio = masked_mean((term_in < 0).float(), mask_in_region).item()
        penalty_term_in = self._leaky_relu(term_in)
        penalty_term_out = masked_mean(self._leaky_relu(term_out), ~mask_in_region).item()

        self._logger.store({
            'Value/Adv_c_unstandardized': unstandardized_adv_c.mean().item(),
            'Train/in_region_ratio': in_region_ratio,
            'Train/neg_in_ratio': neg_in_ratio,
            'Train/penalty_term_in': masked_mean(penalty_term_in, mask_in_region).item(),
            'Train/penalty_term_out': penalty_term_out,
        })

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
