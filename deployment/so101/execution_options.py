"""Duration policy for bounded trials and continuous full-task execution."""

import math


def resolve_duration(full_task, requested=None, configured=60):
    value = float(configured if requested is None else requested)
    if not math.isfinite(value) or not 0 <= value <= 120:
        raise ValueError("Duration must be 0 (continuous) or greater than 0 and at most 120 seconds")
    if not full_task:
        return 3
    return None if value == 0 else value
