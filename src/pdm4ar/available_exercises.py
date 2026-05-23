from typing import Callable, Mapping

from frozendict import frozendict

from pdm4ar.exercises_def.ex14 import get_exercise14
from pdm4ar.exercises_def.structures import Exercise

available_exercises: Mapping[str, Callable[[], Exercise]] = frozendict(
    {
        "14": get_exercise14,
    }
)
