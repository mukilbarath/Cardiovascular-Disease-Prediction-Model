"""Synthetic fixtures exclusively for software tests, never evidence of research performance."""

import numpy as np
import pandas as pd
import pytest

from cardio_research.data import CATEGORICAL, NUMERIC


@pytest.fixture
def synthetic_frame():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(100, 5)), columns=NUMERIC)
    for column in CATEGORICAL:
        frame[column] = np.tile([0, 1], 50)
    frame["target"] = np.tile([0, 1, 1, 0], 25)
    return frame
