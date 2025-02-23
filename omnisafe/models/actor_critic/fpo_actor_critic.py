
from omnisafe.models.actor_critic import ActorCritic

from __future__ import annotations

import torch
from torch import optim

from omnisafe.models.actor_critic.actor_critic import ActorCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


class FPOActorCritic(ActorCritic):
    """FPOActorCritic is a wrapper around ActorCritic that adds a feasibility critic to the model.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+-------------------------------------------------------------+
    | Model                  | Description                                          |
    +=================+=============================================================+
    | Actor                  | Input is observation. Output is action.              |
    +-----------------+-------------------------------------------------------------+
    | Reward V Critic        | Input is observation. Output is reward value.        |
    +-----------------+-------------------------------------------------------------+
    | feasibility V Critic   | Input is observation. Output is feasibility value.   |
    +-----------------+-------------------------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        feasibility_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize an instance of :class:`FPOActorCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)
        self.feasibility_critic: Critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic('v')
        self.add_module('feasibility_critic', self.feasibility_critic)

        if model_cfgs.critic.lr is not None:
            self.feasibility_critic_optimizer: optim.Optimizer
            self.feasibility_critic_optimizer = optim.Adam(
                self.feasibility_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

    def step(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_feasibility: The feasibility value of the observation.
            log_prob: The log probability of the action.
        """
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            value_feasibility = self.feasibility_critic(obs)

            action = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(action)

        return action, value_r[0], value_feasibility[0], log_prob

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_feasibility: The feasibility value of the observation.
            log_prob: The log probability of the action.
        """
        return self.step(obs, deterministic=deterministic)
