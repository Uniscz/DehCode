from typing import Protocol
from dehcode.core.request import GenerationRequest


class Adapter(Protocol):
    def validate(self, request: GenerationRequest): ...
    def load(self, path: str): ...
    def arguments(self, request: GenerationRequest) -> dict: ...


def get_adapter(name):
    if name == 'wan':
        from .wan import WanAdapter
        return WanAdapter()
    raise ValueError(f'Adapter not implemented: {name}')
