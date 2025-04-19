from typing import Dict, Optional

import torch

from omnisafe.common.buffer import FPOBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace


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
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`VectorFPOBuffer`."""
        self._num_buffers: int = num_envs
        self._standardized_adv_r: bool = standardized_adv_r
        self._standardized_adv_c: bool = standardized_adv_c

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
        data = {k: torch.cat(v) for k, v in data_pre.items()}

        adv_mean, adv_std = compute_statistics(data['adv_r'])
        cadv_mean, cadv_std = compute_statistics(data['adv_c'])
        rcadv_mean, rcadv_std = compute_statistics(data['adv_rc'])

        if self._standardized_adv_r:
            data['adv_r'] = standardize_adv(data['adv_r'], adv_mean, adv_std)
        if self._standardized_adv_c:
            data['unstandardized_adv_c'] = data['adv_c']
            data['adv_c'] = standardize_adv(data['adv_c'], cadv_mean, cadv_std)
            data['adv_rc'] = standardize_adv(data['adv_rc'], rcadv_mean, rcadv_std)

        return data


def compute_statistics(data):
    return torch.mean(data), torch.std(data)


def standardize_adv(adv, mean, std):
    return (adv - mean) / (std + 1e-8)
