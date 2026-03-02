"""Additional feature-engineering helpers for the Terry Stops dataset.

Each function accepts a pandas DataFrame and returns it with new
columns appended.  Functions are designed to be called independently
or chained together.
"""

import logging

logger = logging.getLogger(__name__)


def create_officer_experience_proxy(df):
    """Create a proxy for officer experience based on stop frequency.

    Counts the number of stops associated with each ``Officer ID`` and
    maps that count back as a new column ``officer_stop_count``.

    Args:
        df: DataFrame containing an ``Officer ID`` column.

    Returns:
        The same DataFrame with an added ``officer_stop_count`` column.
    """
    if "Officer ID" not in df.columns:
        logger.warning("Column 'Officer ID' not found; skipping experience proxy.")
        return df

    counts = df["Officer ID"].value_counts()
    df["officer_stop_count"] = df["Officer ID"].map(counts).fillna(0).astype(int)
    logger.info("Created officer_stop_count (max=%d).", df["officer_stop_count"].max())
    return df


def create_neighborhood_risk_score(df):
    """Compute a historical arrest rate per precinct.

    For each ``Precinct``, the risk score is the proportion of records
    with ``Arrest Flag == 1`` (or ``'Y'``).  The score is mapped back
    to every row as ``precinct_arrest_rate``.

    Args:
        df: DataFrame containing ``Precinct`` and ``Arrest Flag``
            columns.

    Returns:
        The same DataFrame with an added ``precinct_arrest_rate``
        column.
    """
    if "Precinct" not in df.columns or "Arrest Flag" not in df.columns:
        logger.warning("Required columns missing; skipping neighborhood risk score.")
        return df

    arrest = df["Arrest Flag"]
    if arrest.dtype == object:
        arrest = arrest.map({"Y": 1, "N": 0}).fillna(0)

    rate = arrest.groupby(df["Precinct"]).mean()
    df["precinct_arrest_rate"] = (
        df["Precinct"].map(rate).fillna(0.0).round(4)
    )
    logger.info("Created precinct_arrest_rate for %d precincts.", rate.shape[0])
    return df


def create_repeat_subject_flag(df):
    """Flag subjects who appear more than once in the dataset.

    Uses ``Subject ID`` to determine repeats.  A new boolean column
    ``is_repeat_subject`` is added (1 if the subject has multiple
    records, 0 otherwise).

    Args:
        df: DataFrame containing a ``Subject ID`` column.

    Returns:
        The same DataFrame with an added ``is_repeat_subject`` column.
    """
    if "Subject ID" not in df.columns:
        logger.warning("Column 'Subject ID' not found; skipping repeat flag.")
        return df

    counts = df["Subject ID"].value_counts()
    df["is_repeat_subject"] = (
        df["Subject ID"].map(counts).fillna(1).gt(1).astype(int)
    )
    logger.info(
        "Created is_repeat_subject (repeat subjects: %d).",
        df["is_repeat_subject"].sum(),
    )
    return df
