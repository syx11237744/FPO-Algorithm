
import torch
from typing import Dict, List, Optional, Tuple, Union

from omnisafe.common.buffer import FPOBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed

class VectorFPOBuffer(FPOBuffer):
    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        cost_gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float,
        standardized_adv_r: bool,
        standardized_adv_c: bool,
        feasibility_threshold: float,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`VectorFPOBuffer`."""
        self._num_buffers: int = num_envs
        self._standardized_adv_r: bool = standardized_adv_r
        self._standardized_adv_c: bool = standardized_adv_c
        self._feasibility_threshold: float = feasibility_threshold

        if num_envs < 1:
            raise ValueError('num_envs must be greater than 0.')
        self.buffers: list[FPOBuffer] = [
            FPOBuffer(
                obs_space=obs_space,
                act_space=act_space,
                size=size,
                gamma=gamma,
                cost_gamma=cost_gamma,
                lam=lam,
                lam_c=lam_c,
                advantage_estimator=advantage_estimator,
                penalty_coefficient=penalty_coefficient,
                device=device,
            )
            for _ in range(num_envs)
        ]

    @property
    def num_buffers(self) -> int:
        """Number of buffers."""
        return self._num_buffers

    def store(self, **data: torch.Tensor) -> None:
        """Store vectorized data into vectorized buffer."""
        for i, buffer in enumerate(self.buffers):
            buffer.store(**{k: v[i] for k, v in data.items()})

    def finish_path(
        self,
        last_value_r: Optional[torch.Tensor] = None,
        last_value_c: Optional[torch.Tensor] = None,
        last_value_rc: Optional[torch.Tensor] = None,
        idx: int = 0,
    ) -> None:
        """Get the data in the buffer.

        In vector-FPO buffer, we get the data from each buffer and then concatenate them.
        """
        self.buffers[idx].finish_path(last_value_r, last_value_c, last_value_rc)

    def get(self) -> Dict[str, torch.Tensor]:
        """Get the data in the buffer.

        We provide a trick to standardize the advantages of state-action pairs. We calculate the
        mean and standard deviation of the advantages of state-action pairs and then standardize
        the advantages of state-action pairs. You can turn on this trick by setting the
        ``standardized_adv_r`` to ``True``. The same trick is applied to the advantages of the
        cost.

        Returns:
            The data stored and calculated in the buffer.
        """
        data_pre = {k: [v] for k, v in self.buffers[0].get().items()}
        for buffer in self.buffers[1:]:
            for k, v in buffer.get().items():
                data_pre[k].append(v)
        data = {k: torch.cat(v, dim=0) for k, v in data_pre.items()}

        mask_in_region = data['value_c'] < self._feasibility_threshold
        mask_out_region = ~mask_in_region

        in_region_adv_mean, in_region_adv_std = compute_region_statistics(data['adv_r'], mask_in_region)
        in_region_cadv_mean, in_region_cadv_std = compute_region_statistics(data['adv_c'], mask_in_region)
        out_region_adv_mean, out_region_adv_std = compute_region_statistics(data['adv_r'], mask_out_region)
        out_region_cadv_mean, out_region_cadv_std = compute_region_statistics(data['adv_c'], mask_out_region)
        rcadv_mean, rcadv_std, *_ = distributed.dist_statistics_scalar(data['adv_rc'])

        # adv_mean, adv_std = compute_region_statistics(data['adv_r'], torch.ones_like(mask_in_region))
        # cadv_mean, cadv_std = compute_region_statistics(data['adv_f'], torch.ones_like(mask_in_region))

        if self._standardized_adv_r:
            # data['standardized_adv_r'] = standardize_adv(
            #     data['adv_r'], adv_mean, adv_std, torch.ones_like(mask_in_region)
            # )
            # data['in_region_standardized_adv_r'] = data['standardized_adv_r'] * mask_in_region
            # data['out_region_standardized_adv_r'] = data['standardized_adv_r'] * mask_out_region
            data['in_region_standardized_adv_r'] = standardize_adv(
                data['adv_r'], in_region_adv_mean, in_region_adv_std, mask_in_region
            )
            data['out_region_standardized_adv_r'] = standardize_adv(
                data['adv_r'], out_region_adv_mean, out_region_adv_std, mask_out_region
            )

        if self._standardized_adv_c:
            # data['standardized_adv_f'] = standardize_adv(
            #     data['adv_f'], cadv_mean, cadv_std, torch.ones_like(mask_in_region)
            # )
            # data['in_region_standardized_adv_f'] = data['standardized_adv_f'] * mask_in_region
            # data['out_region_standardized_adv_f'] = data['standardized_adv_f'] * mask_out_region
            data['in_region_standardized_adv_c'] = standardize_adv(
                data['adv_c'], in_region_cadv_mean, in_region_cadv_std, mask_in_region
            )
            data['out_region_standardized_adv_c'] = standardize_adv(
                data['adv_c'], out_region_cadv_mean, out_region_cadv_std, mask_out_region
            )
            data['adv_rc'] = (data['adv_rc'] - rcadv_mean) / (rcadv_std + 1e-8)

        # if any(mask_in_region):
        #     # in_region_adv_mean, in_region_adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'][mask_in_region])
        #     in_region_adv_mean, in_region_adv_std = torch.mean(data['adv_r'][mask_in_region]), torch.std(data['adv_r'][mask_in_region])
        #     # in_region_cadv_mean, in_region_cadv_std, *_ = distributed.dist_statistics_scalar(data['adv_f'][mask_in_region])
        #     in_region_cadv_mean, in_region_cadv_std = torch.mean(data['adv_f'][mask_in_region]), torch.std(data['adv_f'][mask_in_region])
        # else:
        #     in_region_adv_mean = 0
        #     in_region_adv_std = 0
        #     in_region_cadv_mean = 0
        #     in_region_cadv_std = 0
        
        # if any(~mask_in_region):
        #     # out_region_adv_mean, out_region_adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'][~mask_in_region])
        #     out_region_adv_mean, out_region_adv_std = torch.mean(data['adv_r'][~mask_in_region]), torch.std(data['adv_r'][~mask_in_region])
        #     # out_region_cadv_mean, out_region_cadv_std, *_ = distributed.dist_statistics_scalar(data['adv_f'][~mask_in_region])
        #     out_region_cadv_mean, out_region_cadv_std = torch.mean(data['adv_f'][~mask_in_region]), torch.std(data['adv_f'][~mask_in_region])
        # else:
        #     out_region_adv_mean = 0
        #     out_region_adv_std = 0
        #     out_region_cadv_mean = 0
        #     out_region_cadv_std = 0

        # if self._standardized_adv_r:
        #     data['in_region_standardized_adv_r'] = ((data['adv_r'] - in_region_adv_mean) / (in_region_adv_std + 1e-8)) * mask_in_region
        #     data['out_region_standardized_adv_r'] = ((data['adv_r'] - out_region_adv_mean) / (out_region_adv_std + 1e-8)) * ~mask_in_region

        # if self._standardized_adv_c:
        #     data['in_region_standardized_adv_f'] = ((data['adv_f'] - in_region_cadv_mean) / (in_region_cadv_std + 1e-8)) * mask_in_region
        #     data['out_region_standardized_adv_f'] = ((data['adv_f'] - out_region_cadv_mean) / (out_region_cadv_std + 1e-8)) * ~mask_in_region
        
        # if any(~mask_in_region):
        #     data['out_region_standardized_adv_f'] = torch.zeros_like(data['out_region_standardized_adv_f'])
        #     data['out_region_standardized_adv_r'] = torch.zeros_like(data['out_region_standardized_adv_r'])
        
        data['in_region_cadv_mean'] = in_region_cadv_mean
        data['in_region_adv_mean'] = in_region_adv_mean
        data['out_region_cadv_mean'] = out_region_cadv_mean
        data['out_region_adv_mean'] = out_region_adv_mean
        data['mask_in_region'] = mask_in_region

        return data

def compute_region_statistics(data_tensor, mask):
    if torch.any(mask):
        selected_data = data_tensor[mask]
        mean = torch.mean(selected_data)
        if selected_data.numel() > 1:  # 只有1个元素时，std设为0
            std = torch.std(selected_data)
        else:
            std = torch.std(selected_data, unbiased=False)
        return mean, std
    return 0.0, 0.0

def standardize_adv(adv, mean, std, mask):
    return ((adv - mean) / (std + 1e-8)) * mask