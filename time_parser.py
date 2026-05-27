import pandas as pd
import regex as re


def time_parser(df: pd.DataFrame, date_col_name: str, year_bin: int = None, month_bin: int = None):
    """
    Parse YYYY or YYYY-MM dates and create time bins as Timestamps.
    
    - year_bin:  group years into N-year windows  (e.g. 5 → 1920, 1925, 1930…)
    - month_bin: group months into N-month windows (e.g. 3 → quarterly)
    - If both are None, bin = the exact year/month as a Timestamp.
    
    The resulting `time_bin` column always contains pd.Timestamp objects
    anchored to the *start* of each bin — safe for sorting, arithmetic,
    and matplotlib date axes.
    """

    def extract(date_str):
        match = re.match(r"^(\d{4})\D*(\d{1,2})?$", str(date_str).strip())
        if not match:
            return (None, None)
        year  = int(match.group(1))
        month = int(match.group(2)) if match.group(2) else None
        return (year, month)

    df = df.copy()
    df["_year"], df["_month"] = zip(*df[date_col_name].apply(extract))

    has_months = df["_month"].notna().any()

    if has_months:
        # ── YYYY-MM data ──────────────────────────────────────────────
        if month_bin and month_bin > 1:
            # Snap each month back to the first month of its N-month bucket
            # e.g. month_bin=3: Jan/Feb/Mar → Jan, Apr/May/Jun → Apr, etc.
            df["time_bin"] = df.apply(
                lambda r: pd.Timestamp(
                    year=int(r["_year"]),
                    month=int(((r["_month"] - 1) // month_bin) * month_bin + 1),
                    day=1,
                ),
                axis=1,
            )
        else:
            # Exact month precision
            df["time_bin"] = df.apply(
                lambda r: pd.Timestamp(year=int(r["_year"]), month=int(r["_month"]), day=1),
                axis=1,
            )

    else:
        # ── YYYY-only data ────────────────────────────────────────────
        if year_bin and year_bin > 1:
            min_year = int(df["_year"].min())
            # Same floor logic you had originally — but returns a Timestamp
            df["time_bin"] = df["_year"].apply(
                lambda y: pd.Timestamp(
                    year=int(y) - (int(y) - min_year) % year_bin,
                    month=1,
                    day=1,
                )
            )
        else:
            # One bin per calendar year
            df["time_bin"] = df["_year"].apply(
                lambda y: pd.Timestamp(year=int(y), month=1, day=1)
            )

    return df.drop(columns=["_year", "_month"])