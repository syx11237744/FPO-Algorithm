

from __future__ import annotations

import torch

from omnisafe.common.buffer import OnPolicyBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed
from omnisafe.utils.math import discount_cumsum
class FPOBuffer(OnPolicyBuffer):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
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
            lam=lam,
            advantage_estimator=advantage_estimator,
            device=device,
            lam_c=lam_c,
            penalty_coefficient=penalty_coefficient,
            standardized_adv_r=standardized_adv_r,
            standardized_adv_c=False,
        )
        self._lam_c = lam_c
        self.cost_one_positions: list[int] = []  # 记录cost=1的轨迹位置
        self.data['value_feasibility'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['adv_f'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_f'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['deltas_f'] = torch.zeros((size,), dtype=torch.float32, device=device)
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
        # import pdb
        # pdb.set_trace()
        for key, value in data.items():
            self.data[key][self.ptr] = value
        cost = data.get('cost', torch.tensor(0)).item()
        assert cost in (0, 1), f'Cost value must be 0 or 1, but got {cost}'
        if cost == 1:
            self.data['value_feasibility'][self.ptr] = 1
            self.cost_one_positions.append(self.ptr - self.path_start_idx)
        self.ptr += 1

    def get(self) -> dict[str, torch.Tensor]:
        self.ptr, self.path_start_idx = 0, 0

        data = {
            'obs': self.data['obs'],
            'act': self.data['act'],
            'target_value_r': self.data['target_value_r'],
            'adv_r': self.data['adv_r'],
            'logp': self.data['logp'],
            'discounted_ret': self.data['discounted_ret'],
            'adv_f': self.data['adv_f'],
            'target_value_f': self.data['target_value_f'],
            'deltas_f': self.data['deltas_f'],
            'value_feasibility': self.data['value_feasibility']
        }

        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_f'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_f'] = data['adv_f'] - cadv_mean

        return data


    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_feasibility: torch.Tensor | None = None,
    ) -> None:
        """
        在原有的finish path的基础上，我们需要修改cost的advantage的计算方式
        """
        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_feasibility is None:
            last_value_feasibility = torch.zeros(1, device=self._device)

        path_slice = slice(self.path_start_idx, self.ptr)
        path_length = self.ptr - self.path_start_idx
        last_value_r = last_value_r.to(self._device)
        last_value_feasibility = last_value_feasibility.to(self._device)
        
        rewards = torch.cat([self.data['reward'][path_slice], last_value_r])
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_feasibility])
        value_feasibility = torch.cat([self.data['value_feasibility'][path_slice], last_value_feasibility])

        discountred_ret = discount_cumsum(rewards, self._gamma)[:-1]
        self.data['discounted_ret'][path_slice] = discountred_ret
        # penalties = torch.cat([costs,last_value_feasibility])
        rewards -= self._penalty_coefficient * costs

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r,
            rewards,
            lam=self._lam,
        )

        #! 想一下cost=1在最后一个位置会怎么样,应该不会怎么样，因为连乘的是乘到倒数第二个位置上的，所以不用担心？
        if len(self.cost_one_positions) == 0:
            adv_f, target_value_f, deltas_f = self._calculate_feasibility_advantage(
                costs=costs,
                value_feasibility=value_feasibility,
                lam=self._lam_c,
            )
        else:
            adv_f, target_value_f, deltas_f = self._process_segments(
                path_length=path_length,
                costs=costs,
                value_feasibility=value_feasibility,
            )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_f'][path_slice] = adv_f
        self.data['target_value_f'][path_slice] = target_value_f
        self.data['deltas_f'][path_slice] = deltas_f
        

        self.path_start_idx = self.ptr
        self.cost_one_positions = []
    
    def _process_segments(
        self,
        path_length: int,
        costs: torch.Tensor,
        value_feasibility: torch.Tensor,
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
        deltas_f = torch.zeros(path_length, device=self._device)
        
        # Create segment boundaries
        cost_one_tensor = torch.tensor(self.cost_one_positions, device=self._device)
        segments = torch.cat([
            torch.tensor([0], device=self._device),
            cost_one_tensor + 1,
            torch.tensor([path_length], device=self._device)
        ])
        
        # Process each segment
        for start, end in zip(segments[:-1], segments[1:]):
            #! 需要对最后一个cost可能=1的情况进行特殊判断，不然会出现数据丢失的问题
            # 对于这种情况的话，start == end == path_length，那么我们给start - 1
            if start >= end:
                if start == path_length and start > 0:
                    # 处理最后一个位置的cost=1情况
                    start = start - 1
                else:
                    continue
                
            # Create masks for the current segment
            path_slice = slice(start, end)
            value_slice = slice(start, end + 1)
            
            # Calculate advantages for the segment
            segment_adv, segment_target, segment_deltas = self._calculate_feasibility_advantage(
                costs=costs[value_slice],
                value_feasibility=value_feasibility[value_slice],
                lam=self._lam_c,
            )
            
            # Update results
            adv_f[path_slice] = segment_adv
            target_value_f[path_slice] = segment_target
            deltas_f[path_slice] = segment_deltas
        
        return adv_f, target_value_f, deltas_f


    def _calculate_feasibility_advantage(
        self,
        costs: torch.Tensor,           # c(s)
        value_feasibility: torch.Tensor,    # F^π(s)
        # next_value_feasibility: torch.Tensor,  # F^π(s') value_feasibility[1:]
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
            (1 - costs[:-1]) * self._gamma * value_feasibility[1:] -  # (1-c(s))γF^π(s')
            value_feasibility[:-1]  # -F^π(s)
        )
        
        # 使用GAE方式计算优势
        advantages = discount_cumsum(deltas, self._gamma * lam).to(torch.float32)
        
        # clip <= 1
        feasibility_targets = torch.min(advantages + value_feasibility[:-1], torch.ones_like(advantages))
        
        return advantages, feasibility_targets, deltas



        
        


        

    

    