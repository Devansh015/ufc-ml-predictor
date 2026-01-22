import pandas as pd
import numpy as np
from pathlib import Path
import re

DETAILS = "data/processed/fight_details.csv"
TOTALS  = "data/processed/fighter_totals.csv"
ROUND_STATS = "data/raw/round_stats.csv"
FIGHTS  = "data/raw/fights.csv"
EVENTS  = "data/raw/events.csv"

OUT_DIR = Path("data/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.7

BASE_STATS = [
    "kd",
    "sig_landed", "sig_attempted",
    "tot_landed", "tot_attempted",
    "td_landed", "td_attempted",
    "sub_att", "rev",
    "ctrl_sec"
]

def ewma(prev, value, alpha=ALPHA):
    if prev is None:
        return value
    return alpha * value + (1 - alpha) * prev


def _num_or_none(value):
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return None if pd.isna(value) else float(value)
    return None


def _as_float_or_none(value):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


def _safe_sub(a, b):
    a_val = 0.0 if (a is None or (isinstance(a, float) and pd.isna(a))) else float(a)
    b_val = 0.0 if (b is None or (isinstance(b, float) and pd.isna(b))) else float(b)
    return a_val - b_val


def _parse_two_of_pairs(value):
    """Parse strings like '5 of 10 6 of 10' -> (5, 10, 6, 10)."""
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
    """Build stats dict keyed by (fight_url, fighter_name) from raw rounds.

    NOTE: Current raw round rows encode BOTH fighters inside the 'fighter' column
    and store both fighters' landed/attempted values inside 'sig_pct' and 'td_pct'.
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
    fights  = pd.read_csv(FIGHTS)
    # fighter_totals in this repo currently encodes BOTH fighters in one row;
    # per-fighter stats are derived from ROUND_STATS below.
    events  = pd.read_csv(EVENTS)
    rounds  = pd.read_csv(ROUND_STATS)

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

        # Require we can derive *some* per-fighter stats for this fight
        if (fight_url, red) not in per_fight_stats:
            continue
        if (fight_url, blue) not in per_fight_stats:
            continue



        # Initialize fighter states
        for fighter in [red, blue]:
            if fighter not in fighter_state:
                fighter_state[fighter] = {s: None for s in BASE_STATS}
                fighter_state[fighter]["fights"] = 0

        # ---------------------------
        # Build feature row
        # ---------------------------
        feature_row = {
            "fight_url": fight_url,
            "red_fighter": red,
            "blue_fighter": blue,
            "red_fights": fighter_state[red]["fights"],
            "blue_fights": fighter_state[blue]["fights"],
        }

        for stat in BASE_STATS:
            r = fighter_state[red][stat]
            b = fighter_state[blue][stat]

            feature_row[f"red_{stat}"] = r
            feature_row[f"blue_{stat}"] = b
            feature_row[f"delta_{stat}"] = _safe_sub(r, b)

        # ---------------------------
        # Label (winner inference)
        # ---------------------------
        winner_text = str(fight["result_marker"]).lower()
        feature_row["red_win"] = 1 if red.lower() in winner_text else 0

        rows.append(feature_row)

        # ---------------------------
        # Update fighter states AFTER fight
        # ---------------------------
        for fighter in [red, blue]:
            for stat in BASE_STATS:
                val = per_fight_stats[(fight_url, fighter)].get(stat)
                if pd.notna(val):
                    fighter_state[fighter][stat] = ewma(
                        fighter_state[fighter][stat],
                        val
                    )
            fighter_state[fighter]["fights"] += 1

    df = pd.DataFrame(rows)

    if df.empty or ("red_fights" not in df.columns) or ("blue_fights" not in df.columns):
        print("No feature rows were generated.")
        print("This usually means per-fight per-fighter stats could not be derived from the scraped CSVs.")
        print("Check data/raw/round_stats.csv for 'sig_pct'/'td_pct' formatting and fight_url coverage.")
        return

    # ---------------------------
    # Filter: both fighters must have history
    # ---------------------------
    df = df[
        (df["red_fights"] >= 1) &
        (df["blue_fights"] >= 1)
    ]

    out_path = OUT_DIR / "fight_features_alpha.csv"
    df.to_csv(out_path, index=False)

    print("Columns:", df.columns.tolist())
    print(f"Saved {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    main()

