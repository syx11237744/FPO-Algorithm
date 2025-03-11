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

import time
from typing import Any
import numpy as np

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
from omnisafe.models.actor_critic import ConstraintActorCritic
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
        self._lagrange_in_region: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
        self._lagrange_out_region: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
        # self._penalty_term_out = self._cfgs.algo_cfgs.init_penalty_term_out
        # self._penalty_term_in = self._cfgs.algo_cfgs.init_penalty_term_in
        self._feasibility_threshold = self._cfgs.algo_cfgs.feasibility_threshold

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
        # self._logger.register_key('Loss/after_update_Loss_pi', delta=True)
        # self._logger.register_key('Loss/after_update_out_Loss_pi', delta=False, window_length=1)
        # self._logger.register_key('Loss/after_update_in_Loss_pi', delta=False, window_length=1)
        self._logger.register_key('Loss/out_Loss_pi', delta=False)
        self._logger.register_key('Loss/in_Loss_pi', delta=False)
        # self._logger.register_key('Train/before_clip_out_ratio*adv', delta=False, window_length=1)
        # self._logger.register_key('Train/after_update_before_clip_out_ratio*adv', delta=False, window_length=1)
        self._logger.register_key('Value/Adv_r')
        self._logger.register_key('Value/Adv_f')
        self._logger.register_key('Value/out_region_standardized_adv_r')
        self._logger.register_key('Value/in_region_standardized_adv_r')
        self._logger.register_key('Value/out_region_standardized_adv_f')
        self._logger.register_key('Value/in_region_standardized_adv_f')
        

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')

        # if self._cfgs.algo_cfgs.use_feasibility:
            # log information about feasibility critic
        self._logger.register_key('Loss/Loss_cost_critic', delta=True)
        self._logger.register_key('Value/feasibility')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

        self._logger.register_key('Train/penalty_term_in')
        self._logger.register_key('Train/penalty_term_out')
        # self._logger.register_key('Train/penalty_in')
        # self._logger.register_key('Train/penalty_out')
        self._logger.register_key('Metrics/InRegionLagrangeMultiplier')
        self._logger.register_key('Metrics/OutRegionLagrangeMultiplier')

        self._logger.register_key('Train/in_to_out_ratio')
        self._logger.register_key('Train/out_to_in_ratio')
        self._logger.register_key('Train/in_region_ratio')
        self._logger.register_key('Train/neg_out_ratio')
        self._logger.register_key('Train/neg_in_ratio')
        
        self._logger.register_key('Train/out_region_cadv_mean')
        self._logger.register_key('Train/in_region_cadv_mean')
        self._logger.register_key('Train/out_region_adv_mean')
        self._logger.register_key('Train/in_region_adv_mean')
        # self._logger.register_key('Train/out_IS_ratio')
        # self._logger.register_key('Train/out_adv_f')
        # self._logger.register_key('Train/out_IS_ratio*out_adv_f')
        # self._logger.register_key('Train/out_IS_ratio*out_stand_adv_f')

        self._logger.register_key('Train/p_divide_p_')


        # register environment specific keys
        for env_spec_key in self._env.env_spec_keys:
            self.logger.register_key(env_spec_key)
    
    def _update(self) -> None:
        # import pdb; pdb.set_trace()
        
        # then update the policy and value function
        data = self._buf.get()
        self._update_actor_critic(data)

        # 计算cg 和 penalty_term_in and out
        penalty_term_in, penalty_term_out = self._calculate_penalty_term_and_cg(data)

        # note that logger already uses MPI statistics across all processes..
        assert not np.isnan(penalty_term_in), 'penalty_term_in for updating lagrange multiplier is nan'
        assert not np.isnan(penalty_term_out), 'penalty_term_out for updating lagrange multiplier is nan'
        
        # first update Lagrange multiplier parameter
        self._lagrange_in_region.update_lagrange_multiplier(penalty_term_in)
        self._lagrange_out_region.update_lagrange_multiplier(penalty_term_out)
        
        self._logger.store({'Metrics/InRegionLagrangeMultiplier': self._lagrange_in_region.lagrangian_multiplier})
        self._logger.store({'Metrics/OutRegionLagrangeMultiplier': self._lagrange_out_region.lagrangian_multiplier})



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
            | target_value_f | ``target feasibility value`` sampled from buffer.                       |
            +----------------+------------------------------------------------------------------+
            | logp           | ``log probability`` sampled from buffer.                         |
            +----------------+------------------------------------------------------------------+
            | adv_r          | ``estimated advantage`` (e.g. **GAE**) sampled from buffer.      |
            +----------------+------------------------------------------------------------------+
            | adv_f          | ``estimated feasibility advantage`` (e.g. **GAE**) sampled from buffer. |
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
        # data = self._buf.get()
        obs, act, logp, target_value_r, target_value_f, adv_r, adv_f, value_feasibility,\
        in_region_standardized_adv_r, out_region_standardized_adv_r,\
        in_region_standardized_adv_f, out_region_standardized_adv_f ,\
        in_region_adv_mean, out_region_adv_mean, in_region_cadv_mean, out_region_cadv_mean,\
        mask_in_region  = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_f'],
            data['adv_r'],
            data['adv_f'],
            data['value_feasibility'],
            data['in_region_standardized_adv_r'],
            data['out_region_standardized_adv_r'],
            data['in_region_standardized_adv_f'],
            data['out_region_standardized_adv_f'],
            data['in_region_adv_mean'],
            data['out_region_adv_mean'],
            data['in_region_cadv_mean'],
            data['out_region_cadv_mean'],
            data['mask_in_region']
        )
        self._logger.store(
            {
                'Train/in_region_adv_mean': in_region_adv_mean,
                'Train/out_region_adv_mean': out_region_adv_mean,
                'Train/in_region_cadv_mean': in_region_cadv_mean,
                'Train/out_region_cadv_mean': out_region_cadv_mean,
                'Value/in_region_standardized_adv_r': masked_mean(in_region_standardized_adv_r, mask_in_region),
                'Value/out_region_standardized_adv_r': masked_mean(out_region_standardized_adv_r, ~mask_in_region),
                'Value/in_region_standardized_adv_f': masked_mean(in_region_standardized_adv_f, mask_in_region),
                'Value/out_region_standardized_adv_f': masked_mean(out_region_standardized_adv_f, ~mask_in_region),
            },
        )

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_f, adv_r, adv_f, value_feasibility,\
                                  in_region_standardized_adv_r, out_region_standardized_adv_r,\
                                  in_region_standardized_adv_f, out_region_standardized_adv_f,\
                                  mask_in_region),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0
        logp_ = self._actor_critic.actor.log_prob(act)
        self._logger.store({'Train/p_divide_p_':torch.exp(logp_ - logp)})

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_f,
                adv_r,
                adv_f,
                value_feasibility,
                in_region_standardized_adv_r,
                out_region_standardized_adv_r,
                in_region_standardized_adv_f,
                out_region_standardized_adv_f,
                mask_in_region,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                self._update_cost_critic(obs, target_value_f)
                self._update_actor(obs, act, logp, adv_r, value_feasibility, adv_f, target_value_f, \
                                   in_region_standardized_adv_r, out_region_standardized_adv_r, \
                                   in_region_standardized_adv_f, out_region_standardized_adv_f, \
                                   mask_in_region)

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
                'Value/Adv_r': adv_r.abs().mean().item(),
                'Value/Adv_f': adv_f.abs().mean().item(),
                'Train/KL': final_kl,
            },
        )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        value_feasibility: torch.Tensor,
        adv_f: torch.Tensor,
        target_value_f: torch.Tensor,
        in_region_standardized_adv_r: torch.Tensor,
        out_region_standardized_adv_r: torch.Tensor,
        in_region_standardized_adv_f: torch.Tensor,
        out_region_standardized_adv_f: torch.Tensor,
        mask_in_region: torch.Tensor
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
            adv_f (torch.Tensor): The ``feasibility_advantage`` sampled from buffer.
        """
        loss = self._loss_pi(
            obs = obs, 
            act=act, 
            logp=logp, 
            adv_r=adv_r, 
            value_feasibility=value_feasibility, 
            adv_f=adv_f, 
            target_value_f=target_value_f,
            in_region_standardized_adv_r=in_region_standardized_adv_r,
            out_region_standardized_adv_r=out_region_standardized_adv_r,
            in_region_standardized_adv_f=in_region_standardized_adv_f,
            out_region_standardized_adv_f=out_region_standardized_adv_f,
            mask_in_region=mask_in_region
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
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        value_feasibility: torch.Tensor,
        adv_f: torch.Tensor,
        target_value_f: torch.Tensor,
        in_region_standardized_adv_r: torch.Tensor,
        out_region_standardized_adv_r: torch.Tensor,
        in_region_standardized_adv_f: torch.Tensor,
        out_region_standardized_adv_f: torch.Tensor,
        mask_in_region: torch.Tensor
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
        lagrangian_multiplier_in_region = self._lagrange_in_region.lagrangian_multiplier.item()
        # lagrangian_multiplier_out_region = self._lagrange_out_region.lagrangian_multiplier.item()
        
        # For in-region samples
        term_in = adv_f * ratio + value_feasibility - self._feasibility_threshold
        mask_in_region_positive = (term_in > 0) & (value_feasibility < self._feasibility_threshold)

        # term_out = adv_f * ratio
        # mask_out_region_positive = (term_out > 0) & (value_feasibility >= self._feasibility_threshold)

        

        adv = torch.where(~mask_in_region  , -out_region_standardized_adv_f, (in_region_standardized_adv_r-lagrangian_multiplier_in_region)/(1+lagrangian_multiplier_in_region))
        adv = torch.where(mask_in_region_positive, (-in_region_standardized_adv_f), adv)
        # adv = torch.where(mask_in_region_positive, -in_region_standardized_adv_f, adv)
        # lagrangian_multiplier_in_region = self._lagrange_in_region.lagrangian_multiplier.item()
        # lagrangian_multiplier_out_region = self._lagrange_out_region.lagrangian_multiplier.item()
        
        # # For in-region samples
        # term_in = adv_f * ratio + value_feasibility - self._feasibility_threshold
        # mask_in_region_positive = (term_in > 0) & (value_feasibility < self._feasibility_threshold)
        # penalty_in = torch.where(mask_in_region_positive, adv_f * lagrangian_multiplier_in_region, torch.zeros_like(term_in))

        # # For out-region samples
        # term_out = adv_f * ratio
        # mask_out_region_positive = (term_out > 0) & (value_feasibility >= self._feasibility_threshold)
        # penalty_out = torch.where(mask_out_region_positive, adv_f * lagrangian_multiplier_out_region, torch.zeros_like(term_out))

        # lagrangian_multiplier_term = mask_in_region_positive * lagrangian_multiplier_in_region + mask_out_region_positive * lagrangian_multiplier_out_region + 1
        # adv = (adv_r - penalty_in - penalty_out) / lagrangian_multiplier_term
        
        ratio_cliped = torch.clamp( 
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )
        loss_term = -torch.min(ratio * adv, ratio_cliped * adv)

        loss = loss_term.mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()
        # useful extra info
        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                # 'Train/penalty_in': penalty_in.mean().item(),
                # 'Train/penalty_out': penalty_out.mean().item(),
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
                'Loss/out_Loss_pi': masked_mean(loss_term,~mask_in_region),
                'Loss/in_Loss_pi': masked_mean(loss_term,mask_in_region),
                # 'Train/before_clip_out_ratio*adv': masked_mean(ratio * adv,~mask_in_region)
            },
        )
        return loss
    
    def _calculate_penalty_term_and_cg(
        self,
        data: dict[str, torch.Tensor],
    ) -> tuple[float, float]:
        obs, logp, adv_f, act, value_feasibility, out_region_standardized_adv_f, \
        in_region_standardized_adv_r, in_region_standardized_adv_f = (
            data['obs'],
            data['logp'],
            data['adv_f'],
            data['act'],
            data['value_feasibility'],
            data['out_region_standardized_adv_f'],
            data['in_region_standardized_adv_r'],
            data['in_region_standardized_adv_f']
        )

        # _, _, value_feasibility_, logp_ = self._actor_critic.step(obs)
        # self._actor_critic.actor.log_prob(next_action)
        with torch.no_grad():
            value_feasibility_ = self._actor_critic.cost_critic(obs)[0]
            logp_ = self._actor_critic.actor.log_prob(act)

        relu = torch.nn.LeakyReLU()
        ratio = torch.exp(logp_ - logp)

        mask_in_region = value_feasibility < self._feasibility_threshold
        mask_out_region = ~mask_in_region

        term_in = adv_f * ratio + value_feasibility - self._feasibility_threshold
        count_neg_in = torch.sum(torch.logical_and(term_in < 0, mask_in_region)).item()
        in_region_count = torch.sum(mask_in_region).item()
        neg_in_ratio = count_neg_in / in_region_count

        term_out = adv_f * ratio
        count_neg_out = torch.sum(torch.logical_and(term_out < 0, mask_out_region)).item()
        out_region_count = torch.sum(mask_out_region).item()
        neg_out_ratio = count_neg_out / out_region_count

        penalty_term_in = torch.where(mask_in_region, relu(term_in), torch.zeros_like(term_in)).mean().item()
        penalty_term_out = torch.where(mask_out_region, relu(term_out), torch.zeros_like(term_out)).mean().item()

        in_region_ratio = in_region_count / mask_in_region.shape[0]

        mask_in_region_ = value_feasibility_ < self._feasibility_threshold
        mask_out_region_ = ~mask_in_region_

        # 计算过去在in region的但是之后在out region的样本的比例
        in_to_out_ratio = torch.sum(mask_in_region & mask_out_region_) / in_region_count
        # 计算过去在out region的但是之后在in region的样本的比例
        out_to_in_ratio = torch.sum(mask_out_region & mask_in_region_) / out_region_count

        # mask_in_region_positive = (term_in > 0) & (value_feasibility < self._feasibility_threshold)
        

        # adv = torch.where(~mask_in_region , -out_region_standardized_adv_f, in_region_standardized_adv_r)
        # adv = torch.where(mask_in_region_positive, (-in_region_standardized_adv_f), adv)
        
        # ratio_cliped = torch.clamp( 
        #     ratio,
        #     1 - self._cfgs.algo_cfgs.clip,
        #     1 + self._cfgs.algo_cfgs.clip,
        # )
        # loss_term = -torch.min(ratio * adv, ratio_cliped * adv)
        # loss = loss_term.mean()


        self._logger.store({
            'Train/in_region_ratio': in_region_ratio,
            'Train/neg_out_ratio': neg_out_ratio,
            'Train/neg_in_ratio': neg_in_ratio,
            'Train/in_to_out_ratio': in_to_out_ratio,
            'Train/out_to_in_ratio': out_to_in_ratio,
            'Train/penalty_term_in': penalty_term_in,
            'Train/penalty_term_out': penalty_term_out,
            # 'Train/out_IS_ratio': masked_mean(ratio, ~mask_in_region),
            # 'Train/out_adv_f': masked_mean(adv_f, ~mask_in_region),
            # 'Train/out_IS_ratio*out_adv_f': masked_mean(adv_f * ratio, ~mask_in_region),
            # 'Train/out_IS_ratio*out_stand_adv_f': masked_mean(out_region_standardized_adv_f * ratio, ~mask_in_region),
            # 'Loss/after_update_Loss_pi': loss.mean().item(),
            # 'Loss/after_update_out_Loss_pi': masked_mean(loss_term,~mask_in_region),
            # 'Loss/after_update_in_Loss_pi': masked_mean(loss_term,mask_in_region),
            # 'Train/after_update_before_clip_out_ratio*adv': masked_mean(ratio * adv,~mask_in_region)
        })


        return penalty_term_in, penalty_term_out


def masked_mean(x, mask):
    return (x * mask).sum() / torch.maximum(mask.sum(), torch.tensor(1.0).to(mask.device))