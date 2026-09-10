from __future__ import annotations

import pandas as pd


def add_refractivity_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """Reserved for refractivity features after dew-point and vapor-pressure units are audited."""
    raise NotImplementedError(
        "Refractivity features require a unit audit for dew_0_21 and upper-air dew_* columns before use."
    )
