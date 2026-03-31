import torch


class StatewiseLagrangian:
    def __init__(
        self,
        pid_kp: float,
        pid_ki: float,
        pid_kd: float,
        ema_alpha: float,
    ) -> None:
        self._pid_kp: float = pid_kp
        self._pid_ki: float = pid_ki
        self._pid_kd: float = pid_kd
        self._ema_alpha: float = ema_alpha

    @property
    def multiplier(self) -> torch.Tensor:
        return self._multiplier

    def reset(self, value: torch.Tensor) -> None:
        self._error_ema = None
        self._error_int = torch.zeros_like(value)
        self._multiplier = value

    def update(self, error: torch.Tensor) -> None:
        if self._error_ema is None:
            self._error_ema = error
            new_error_ema = error
        else:
            new_error_ema = self._ema_alpha * self._error_ema + (1 - self._ema_alpha) * error
        self._error_int = self._error_int + error
        self._multiplier = torch.clamp_min(self._multiplier + 
                                           self._pid_kp * new_error_ema + 
                                           self._pid_ki * self._error_int + 
                                           self._pid_kd * (new_error_ema - self._error_ema), 0)
        self._error_ema = new_error_ema
