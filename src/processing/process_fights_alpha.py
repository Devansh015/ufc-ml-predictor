import pandas as pd
import numpy as np
from pathlib import Path
import re

DETAILS = "data/processed/fight_details.csv"
TOTALS  = "data/processed/fighter_totals.csv"  # unused in this variant (kept for compatibility)
ROUND_STATS = "data/raw/round_stats.csv"
FIGHTS_CLEAN = "data/raw/fights_clean.csv"  # Clean data with reliable winner labels
EVENTS  = "data/raw/events.csv"

OUT_DIR = Path("data/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.7
ROUND_LEN_SEC = 300  # 5 minutes


# -----------------------------
# Base + Derived feature sets
# -----------------------------
BASE_STATS = [
    "kd",
    "sig_landed", "sig_attempted",
    "tot_landed", "tot_attempted",
    "td_landed", "td_attempted",
    "sub_att", "rev",
    "ctrl_sec"
]

DERIVED_STATS = [
    # rates
    "sig_landed_per_min",
    "sig_attempted_per_min",
    "td_landed_per_15",
    "td_attempted_per_15",

    # accuracies
    "sig_acc",
    "td_acc",

    # style balance
    "strike_share",
    "grapple_share",

    # momentum (raw delta vs previous EWMA raw)
    "sig_landed_trend",
    "td_landed_trend",
]

ALL_STATS = BASE_STATS + DERIVED_STATS


def ewma(prev, value, alpha=ALPHA):
    if prev is None:
        return value
    return alpha * value + (1 - alpha) * prev


def safe_div(a, b, default=0.0):
    try:
        if a is None or (isinstance(a, float) and pd.isna(a)):
            return default
        if b is None or b == 0 or (isinstance(b, float) and pd.isna(b)):
            return default
        return float(a) / float(b)
    except Exception:
        return default


def _safe_sub(a, b):
    a_val = 0.0 if (a is None or (isinstance(a, float) and pd.isna(a))) else float(a)
    b_val = 0.0 if (b is None or (isinstance(b, float) and pd.isna(b))) else float(b)
    return a_val - b_val


def parse_mmss_to_seconds(s):
    """Parse 'M:SS' -> seconds."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    m = re.match(r"^(\d+):(\d{2})$", s)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def build_fight_seconds_map(details_df: pd.DataFrame):
    """
    Build fight_url -> total_seconds using columns if present.
    Fallback to 900 seconds (3 rounds) when unknown.
    """
    details = details_df.copy()

    round_cols = ["finish_round", "round", "last_round", "ending_round"]
    time_cols  = ["finish_time", "time", "last_round_time", "ending_time"]

    rcol = next((c for c in round_cols if c in details.columns), None)
    tcol = next((c for c in time_cols if c in details.columns), None)

    seconds_map = {}

    for _, row in details.iterrows():
        fight_url = row.get("fight_url")
        if not isinstance(fight_url, str):
            continue

        total_sec = None

        if rcol and tcol:
            r = row.get(rcol)
            t = row.get(tcol)

            try:
                r_int = int(r) if pd.notna(r) else None
            except Exception:
                r_int = None

            t_sec = parse_mmss_to_seconds(t) if pd.notna(t) else None

            if r_int is not None and t_sec is not None:
                total_sec = (r_int - 1) * ROUND_LEN_SEC + t_sec

        if total_sec is None or total_sec <= 0:
            total_sec = 900  # default: 3 rounds

        seconds_map[fight_url] = total_sec

    return seconds_map


def _parse_two_of_pairs(value):
    """Parse strings like '5 of 10 6 of 10' -> (5,10,6,10)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    nums = re.findall(r"\d+", text)
    if len(nums) < 4:
        return None
    a1, b1, a2, b2 = (int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3]))
    return a1, b1, a2, b2


def _infer_pair_order(fighter_pair_text, red_name, blue_name):
    """Return True if pair is in red-blue order, False if blue-red, else None."""
    if fighter_pair_text is None or (isinstance(fighter_pair_text, float) and pd.isna(fighter_pair_text)):
        return None
    pair = str(fighter_pair_text)
    red = str(red_name)
    blue = str(blue_name)
    if pair.startswith(red) and blue in pair:
        return True
    if pair.startswith(blue) and red in pair:
        return False
    return None


def _build_per_fight_fighter_stats_from_rounds(fights_df, rounds_df):
    """
    Build per-fight per-fighter aggregates from round rows.

    NOTE: Your raw round rows encode BOTH fighters inside the 'fighter' column
    and store both fighters' landed/attempted values inside 'sig_pct' and 'td_pct'.
    We aggregate across rounds to get per-fight totals for each fighter:
      - sig_landed, sig_attempted
      - td_landed, td_attempted
    Other base stats remain unavailable in this file format (left as None).
    """
    fights_dedup = fights_df.drop_duplicates(subset=["fight_url"], keep="first")
    fights_map = fights_dedup.set_index("fight_url")[["red_fighter", "blue_fighter"]].to_dict("index")
    out = {}

    for fight_url, grp in rounds_df.groupby("fight_url"):
        if fight_url not in fights_map:
            continue

        red = fights_map[fight_url]["red_fighter"]
        blue = fights_map[fight_url]["blue_fighter"]

        order_is_red_blue = _infer_pair_order(grp["fighter"].iloc[0], red, blue)

        red_sig_landed = 0
        red_sig_attempted = 0
        blue_sig_landed = 0
        blue_sig_attempted = 0

        red_td_landed = 0
        red_td_attempted = 0
        blue_td_landed = 0
        blue_td_attempted = 0

        have_any = False

        for _, row in grp.iterrows():
            sig = _parse_two_of_pairs(row.get("sig_pct"))
            td = _parse_two_of_pairs(row.get("td_pct"))

            if sig is not None:
                have_any = True
                a1, b1, a2, b2 = sig
                if order_is_red_blue is False:
                    # pair is blue-red
                    blue_sig_landed += a1
                    blue_sig_attempted += b1
                    red_sig_landed += a2
                    red_sig_attempted += b2
                else:
                    # assume red-blue
                    red_sig_landed += a1
                    red_sig_attempted += b1
                    blue_sig_landed += a2
                    blue_sig_attempted += b2

            if td is not None:
                have_any = True
                a1, b1, a2, b2 = td
                if order_is_red_blue is False:
                    blue_td_landed += a1
                    blue_td_attempted += b1
                    red_td_landed += a2
                    red_td_attempted += b2
                else:
                    red_td_landed += a1
                    red_td_attempted += b1
                    blue_td_landed += a2
                    blue_td_attempted += b2

        if not have_any:
            continue

        # Only populate what we can derive; keep other base stats missing.
        out[(fight_url, red)] = {
            "sig_landed": float(red_sig_landed),
            "sig_attempted": float(red_sig_attempted),
            "td_landed": float(red_td_landed),
            "td_attempted": float(red_td_attempted),
        }
        out[(fight_url, blue)] = {
            "sig_landed": float(blue_sig_landed),
            "sig_attempted": float(blue_sig_attempted),
            "td_landed": float(blue_td_landed),
            "td_attempted": float(blue_td_attempted),
        }

    return out


def main():
    fights = pd.read_csv(FIGHTS_CLEAN)
    events = pd.read_csv(EVENTS)
    rounds = pd.read_csv(ROUND_STATS)

    # fight_details used ONLY for duration mapping (rates/per-minute)
    details = None
    fight_seconds_map = {}
    try:
        details = pd.read_csv(DETAILS)
        fight_seconds_map = build_fight_seconds_map(details)
    except Exception:
        fight_seconds_map = {}

    # Filter to fights with known winners (binary only)
    fights = fights[fights["winner"].isin(["red", "blue"])].copy()

    # ---------------------------
    # Fix event dates (authoritative)
    # ---------------------------
    event_date_map = dict(zip(events["event_url"], events["event_date"]))
    fights["event_date"] = fights["event_url"].map(event_date_map)

    fights["event_date"] = pd.to_datetime(
        fights["event_date"],
        format="%B %d, %Y",
        errors="coerce"
    )

    fights = fights.dropna(subset=["event_date"])
    fights = fights.sort_values("event_date").reset_index(drop=True)

    # ---------------------------
    # Build per-fight per-fighter stats from rounds
    # ---------------------------
    per_fight_stats = _build_per_fight_fighter_stats_from_rounds(fights, rounds)

    fighter_state = {}
    rows = []

    for _, fight in fights.iterrows():
        fight_url = fight["fight_url"]

        red = fight["red_fighter"]
        blue = fight["blue_fighter"]

        # Must have derived per-fighter round aggregates for both
        if (fight_url, red) not in per_fight_stats:
            continue
        if (fight_url, blue) not in per_fight_stats:
            continue

        # Initialize fighter states
        for fighter in [red, blue]:
            if fighter not in fighter_state:
                fighter_state[fighter] = {s: None for s in ALL_STATS}
                fighter_state[fighter]["fights"] = 0

        # ---------------------------
        # Build feature row (uses PRE-fight state only)
        # ---------------------------
        feature_row = {
            "fight_url": fight_url,
            "red_fighter": red,
            "blue_fighter": blue,
            "red_fights": fighter_state[red]["fights"],
            "blue_fights": fighter_state[blue]["fights"],
        }

        for stat in ALL_STATS:
            r = fighter_state[red][stat]
            b = fighter_state[blue][stat]
            feature_row[f"red_{stat}"] = r
            feature_row[f"blue_{stat}"] = b
            feature_row[f"delta_{stat}"] = _safe_sub(r, b)

        # Label (from fights_clean winner)
        feature_row["red_win"] = 1 if fight["winner"] == "red" else 0
        # preserve original winner label to allow correct swapping
        feature_row["winner"] = fight["winner"]
        rows.append(feature_row)

        # ---------------------------
        # Update fighter states AFTER fight (raw + derived)
        # ---------------------------
        fight_sec = fight_seconds_map.get(fight_url, 900)
        fight_min = max(fight_sec / 60.0, 1e-6)

        for fighter in [red, blue]:
            cur = per_fight_stats[(fight_url, fighter)]

            # raw values we can derive from round_stats
            sig_l = cur.get("sig_landed")
            sig_a = cur.get("sig_attempted")
            td_l  = cur.get("td_landed")
            td_a  = cur.get("td_attempted")

            # per-minute/per-15 rates
            sig_l_per_min = safe_div(sig_l, fight_min)
            sig_a_per_min = safe_div(sig_a, fight_min)
            td_l_per_15 = safe_div(td_l, fight_min) * 15.0
            td_a_per_15 = safe_div(td_a, fight_min) * 15.0

            # accuracies
            sig_acc = safe_div(sig_l, sig_a, default=0.0)
            td_acc  = safe_div(td_l, td_a, default=0.0)

            # style balance (based on attempts)
            total_attempt_load = (float(sig_a) if sig_a is not None else 0.0) + (float(td_a) if td_a is not None else 0.0)
            strike_share = safe_div((float(sig_a) if sig_a is not None else 0.0), total_attempt_load, default=0.0)
            grapple_share = safe_div((float(td_a) if td_a is not None else 0.0), total_attempt_load, default=0.0)

            # trends vs previous EWMA raw
            prev_sig_l_ewma = fighter_state[fighter]["sig_landed"]
            prev_td_l_ewma  = fighter_state[fighter]["td_landed"]

            sig_l_trend = (float(sig_l) - float(prev_sig_l_ewma)) if (sig_l is not None and prev_sig_l_ewma is not None) else 0.0
            td_l_trend  = (float(td_l)  - float(prev_td_l_ewma))  if (td_l  is not None and prev_td_l_ewma  is not None) else 0.0

            # Update only what we have
            updates = {
                # raw that exists
                "sig_landed": sig_l,
                "sig_attempted": sig_a,
                "td_landed": td_l,
                "td_attempted": td_a,

                # derived
                "sig_landed_per_min": sig_l_per_min,
                "sig_attempted_per_min": sig_a_per_min,
                "td_landed_per_15": td_l_per_15,
                "td_attempted_per_15": td_a_per_15,
                "sig_acc": sig_acc,
                "td_acc": td_acc,
                "strike_share": strike_share,
                "grapple_share": grapple_share,
                "sig_landed_trend": sig_l_trend,
                "td_landed_trend": td_l_trend,
            }

            for k, v in updates.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                fighter_state[fighter][k] = ewma(fighter_state[fighter][k], float(v))

            fighter_state[fighter]["fights"] += 1

    df = pd.DataFrame(rows)

    if df.empty or ("red_fights" not in df.columns) or ("blue_fights" not in df.columns):
        print("No feature rows were generated.")
        print("This usually means per-fight per-fighter stats could not be derived from the scraped CSVs.")
        print("Check data/raw/round_stats.csv for 'sig_pct'/'td_pct' formatting and fight_url coverage.")
        return

    # Filter: both fighters must have at least 1 prior fight
    df = df[(df["red_fights"] >= 1) & (df["blue_fights"] >= 1)]

    # ---- Remove leaked / dead columns ----
    # 'winner' is the raw label string — must not be in the feature file
    # 'blue_win' was a derivative of red_win — direct target leakage
    leak_cols = [c for c in ["winner", "blue_win"] if c in df.columns]
    if leak_cols:
        df = df.drop(columns=leak_cols)
        print(f"Dropped leaked columns: {leak_cols}")

    # Drop columns that are 100% NaN (stats we can't derive from round_stats)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
        print(f"Dropped all-NaN columns ({len(all_nan_cols)}): {all_nan_cols}")

    # Drop numeric columns that are entirely zero (useless deltas of NaN stats)
    all_zero_cols = [
        c for c in df.select_dtypes(include="number").columns
        if (df[c] == 0).all() and c not in ["red_win"]
    ]
    if all_zero_cols:
        df = df.drop(columns=all_zero_cols)
        print(f"Dropped all-zero columns ({len(all_zero_cols)}): {all_zero_cols}")

    # NOTE: Symmetry augmentation is now done at training time to prevent
    # both versions of a fight leaking across the train/test split.

    out_path = OUT_DIR / "fight_features_alpha.csv"
    df.to_csv(out_path, index=False)

    print("Columns:", df.columns.tolist())
    print(f"Saved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()