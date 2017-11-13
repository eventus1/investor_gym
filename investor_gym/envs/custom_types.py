# pylint: disable=C0103, C0111
from typing import NewType, Tuple, Dict, Any

Currency = NewType("Currency", float)
Return = NewType("Return", float)  # A float in (-inf, inf)
Proportion = NewType("Proportion", float)  # A float in [0.0, 1.0]
Days = NewType("Days", int)
Observation = NewType("Observation", Tuple[Return, ...])
Reward = NewType("Reward", Return)
Done = NewType("Done", bool)
Info = NewType("Info", Dict[str, Any])
Action = NewType("Action", Tuple[Proportion, ...])
Weights = NewType("Weights", Tuple[Proportion, ...])
# TODO can't import Asset bc cycle import
Holdings = NewType("Holdings",
                   Tuple[Tuple["Asset", Currency], ...])  # type: ignore
