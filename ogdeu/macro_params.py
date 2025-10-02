from pathlib import Path
import re
import pandas as pd
import statsmodels.api as sm

DATA_DIR = Path(__file__).parent / "data"

def read_bundesbank_data(filename, value_name):
    p = DATA_DIR / filename

    # erste Datenzeile (YYYY-MM) finden
    with open(p, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    start = next((i for i, ln in enumerate(lines) if re.match(r"^\d{4}-\d{2}", ln.strip())), None)
    if start is None:
        raise ValueError(f"Keine Datenzeile gefunden in {filename} (YYYY-MM).")

   
    df = pd.read_csv(
        p,
        sep=";",
        header=None,
        names=["date", value_name],
        skiprows=start,
        decimal=",",
        encoding="utf-8-sig",
        engine="python",
        usecols=[0, 1],
        on_bad_lines="skip",
    )

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m", errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

def load_de_yields(start_date="2005-01-01"):
    corp = read_bundesbank_data("corp_de.csv", "corp_y_pct")
    bund = read_bundesbank_data("bund_de.csv", "bund_y_pct")
    df = corp.join(bund, how="inner").sort_index()
    min_overlap = max(corp.index.min(), bund.index.min(), pd.Timestamp(start_date))
    df = df[df.index >= min_overlap].dropna().copy()
    if df.empty:
        raise ValueError(
            f"Kein Overlap nach {min_overlap.date()}. "
            f"corp: {corp.index.min().date()}..{corp.index.max().date()}, "
            f"bund: {bund.index.min().date()}..{bund.index.max().date()}"
        )
    return df

def estimate_rgov_params(df):
    X = sm.add_constant(df["corp_y_pct"].values)  # Regression in %
    y = df["bund_y_pct"].values
    res = sm.OLS(y, X).fit()
    c, b = res.params[0], res.params[1]
    return float(b), float(-c / 100.0), float(res.rsquared)

#-----------ergebnis in baseline_overrides.json übertragen: -----------------
#"r_gov_scale": [{"value": 0.944888}],
#"r_gov_shift": [{"value": 0.014632}]

