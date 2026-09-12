from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str = ''
    width: int = 832
    height: int = 480
    frames: int = 81
    fps: int = 16
    seed: int = 42
    steps: int = 50
    guidance: float = 5.0

    def validate(self):
        if not self.prompt.strip():
            raise ValueError('Prompt must not be empty.')
        for key in ('width','height','frames','fps','steps'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive.')
        if not 0 <= self.seed < 2**63:
            raise ValueError('seed must be between 0 and 2^63-1.')
        if not math.isfinite(self.guidance) or self.guidance < 0:
            raise ValueError('guidance must be finite and nonnegative.')
