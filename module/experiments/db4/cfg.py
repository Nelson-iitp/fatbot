
import fatbot as fb

from ..config import PPOCONFIG
from .db import envF, isdF, isdL


class C(PPOCONFIG):
    def __init__(self, alias: str, common_initial_states: str, results_dir=None) -> None:
        super().__init__(alias, common_initial_states, results_dir, envF, isdF, isdL)