from typing import Any
from inspect import isfunction

def exists(x: Any) -> bool:
    return x is not None

def default(val: Any, d: Any) -> Any:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num: int, divisor: int):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
