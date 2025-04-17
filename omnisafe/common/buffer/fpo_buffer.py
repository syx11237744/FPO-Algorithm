from __future__ import annotations

import torch

from omnisafe.common.buffer import OnPolicyBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace
from omnisafe.utils.math import discount_cumsum


class FPOBuffer(OnPolicyBuffer):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        cost_gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`FPOBuffer`."""
        super().__init__(
            obs_space=obs_space,
            act_space=act_space,
            size=size,
            gamma=gamma,
            cost_gamma=cost_gamma,
            lam=lam,
            advantage_estimator=advantage_estimator,
            device=device,
            lam_c=lam_c,
            penalty_coefficient=penalty_coefficient,
            standardized_adv_r=standardized_adv_r,
            standardized_adv_c=standardized_adv_c,
        )
        self.cost_one_positions: list[int] = []  # 记录cost=1的轨迹位置
        self.cost_zero_positions: list[int] = []  # 记录cost=0的轨迹位置
        self.data['adv_rc'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['value_rc'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_rc'] = torch.zeros((size,), dtype=torch.float32, device=device)
        assert advantage_estimator == "gae", 'FPOBuffer only supports GAE advantage estimator.'

    def store(self, **data: torch.Tensor) -> None:
        """Store data into the buffer and record positions where cost equals one.

        #? 除了一般的数据或许还需要一个数据的存储是Feasible Function对于这个状态的输出, 那么我们的adapter就需要传进来这个数据，
        #? 我们是可以获取这个数据的吗？那么就需要看一看adapter了。是可以的，我们只需要在我们的actor_critic 提供一个可以接受状态进行预测的接口即可
        #! 实际是就是value_c，我们也可以进行重命名，这样更加直观

        Args:
            is_cost_one (bool): Whether the cost equals one at this position.
            data (torch.Tensor): The data to store.
        """
        assert self.ptr < self.max_size, 'No more space in the buffer!'
        for key, value in data.items():
            self.data[key][self.ptr] = value
        cost = data.get('cost', torch.tensor(0)).item()
        assert cost in (0, 1), f'Cost value must be 0 or 1, but got {cost}'
        if cost == 1:
            self.cost_one_positions.append(self.ptr - self.path_start_idx)
        else:
            self.cost_zero_positions.append(self.ptr - self.path_start_idx)

        self.ptr += 1

    def get(self) -> dict[str, torch.Tensor]:
        self.ptr, self.path_start_idx = 0, 0
        return self.data

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
        last_value_rc: torch.Tensor | None = None,
    ) -> None:
        """
        在原有的finish path的基础上，我们需要修改cost的advantage的计算方式
        """
        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_c is None:
            last_value_c = torch.zeros(1, device=self._device)
        if last_value_rc is None:
            last_value_rc = torch.ones(1, device=self._device)

        path_slice = slice(self.path_start_idx, self.ptr)
        path_length = self.ptr - self.path_start_idx
        last_value_r = last_value_r.to(self._device)
        last_value_c = last_value_c.to(self._device)
        last_value_rc = last_value_rc.to(self._device)
        
        rewards = torch.cat([self.data['reward'][path_slice], last_value_r])
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])
        values_rc = torch.cat([self.data['value_rc'][path_slice], last_value_rc])

        discountred_ret = discount_cumsum(rewards, self._gamma)[:-1]
        self.data['discounted_ret'][path_slice] = discountred_ret
        rewards -= self._penalty_coefficient * costs

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r,
            rewards,
            lam=self._lam,
        )

        adv_c, target_value_c = self._process_segments(
            path_length=path_length,
            costs=costs,
            values=values_c,
            segment_positions=self.cost_one_positions,
        )

        adv_rc, target_value_rc = self._process_segments(
            path_length=path_length,
            costs=1 - costs,
            values=values_rc,
            segment_positions=self.cost_zero_positions,
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c
        self.data['adv_rc'][path_slice] = adv_rc
        self.data['target_value_rc'][path_slice] = target_value_rc

        self.path_start_idx = self.ptr
        self.cost_one_positions = []
        self.cost_zero_positions = []

    def _process_segments(
        self,
        path_length: int,
        costs: torch.Tensor,
        values: torch.Tensor,
        segment_positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process path segments separated by cost=1 positions.
        
        Args:
            path_length: Length of the path
            costs: Cost values tensor
            value_feasibility: Feasibility values tensor
        
        Returns:
            Tuple of (advantages, target values, deltas)
        """
        # Initialize tensors
        adv_f = torch.zeros(path_length, device=self._device)
        target_value_f = torch.zeros(path_length, device=self._device)

        # Create segment boundaries
        if len(segment_positions) == 0 or segment_positions[-1] < path_length - 1:
            segment_positions.append(path_length - 1)

        # Process each segment
        start = 0
        for end in segment_positions:
            # Create masks for the current segment
            path_slice = slice(start, end + 1)
            value_slice = slice(start, end + 2)
            start = end + 1

            # Calculate advantages for the segment
            segment_adv, segment_target = self._calculate_feasibility_advantage(
                costs=costs[value_slice],
                values=values[value_slice],
                lam=self._lam_c,
            )

            # Update results
            adv_f[path_slice] = segment_adv
            target_value_f[path_slice] = segment_target

        return adv_f, target_value_f

    def _calculate_feasibility_advantage(
        self,
        costs: torch.Tensor,           # c(s)
        values: torch.Tensor,    # F^π(s)
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate feasibility advantage using GAE estimation.
        
        Args:
            costs: Cost values for each state
            feasibility_values: F^π(s) values
            next_feasibility_values: F^π(s') values 
            gamma: Discount factor
            lam: GAE lambda parameter
        """
        # 计算类似TD误差的值
        deltas = (
            costs[:-1] +  # c(s)
            (1 - costs[:-1]) * self._cost_gamma * values[1:] -  # (1-c(s))γF^π(s')
            values[:-1]  # -F^π(s)
        )
        
        # 使用GAE方式计算优势
        advantages = discount_cumsum(deltas, self._cost_gamma * lam)
        
        feasibility_targets = advantages + values[:-1]
        
        return advantages, feasibility_targets
