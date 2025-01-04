from .base import DannsBase

class Codats(DannsBase):
    """
    CoDATS model https://arxiv.org/abs/2005.10996
    """

    def __init__(self, experiment: str) -> None:
        super().__init__(experiment)
