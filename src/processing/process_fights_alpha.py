import pandas as pd
import numpy as np
from pathlib import Path

DETAILS = "data/processed/fight_details.csv"
TOTALS  = "data/processed/fighter_totals.csv"
FIGHTS  = "data/raw/fights.csv"

OUT_DIR = Path("data/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.7  # recency weight (higher = recent fights matter more)

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

def main():
    fights = pd.read_csv(FIGHTS)
    totals = pd.read_csv(TOTALS)
    details = pd.read_csv(DETAILS)

    # Sort fights chronologically
    fights["event_date"] = pd.to_datetime(fights["event_date"])
    fights = fights.sort_values("event_date").reset_index(drop=True)

    # Index totals by fight_url + fighter
    totals_idx = totals.set_index(["fight_url", "fighter"])

    # Fighter state store
    fighter_state = {}

    rows = []

    for _, fight in fights.iterrows():
        fight_url = fight["fight_url"]
        red = fight["red_fighter"]
        blue = fight["blue_fighter"]

        if (fight_url, red) not in totals_idx.index or (fight_url, blue) not in totals_idx.index:
            continue

        # Initialize fighter state if missing
        for fighter in [red, blue]:
            if fighter not in fighter_state:
                fighter_state[fighter] = {s: None for s in BASE_STATS}
                fighter_state[fighter]["fights"] = 0

        # Build feature row
        feature_row = {
            "fight_url": fight_url,
            "red_fighter": red,
            "blue_fighter": blue,
        }

        # Red / Blue stats
        for stat in BASE_STATS:
            feature_row[f"red_{stat}"] = fighter_state[red][stat]
            feature_row[f"blue_{stat}"] = fighter_state[blue][stat]
            feature_row[f"delta_{stat}"] = (
                (fighter_state[red][stat] or 0)
                - (fighter_state[blue][stat] or 0)
            )

        # Label
        feature_row["red_win"] = 1 if fight["winner"] == "win" else 0
        rows.append(feature_row)

        # === UPDATE fighter states AFTER fight ===
        for fighter in [red, blue]:
            stats = totals_idx.loc[(fight_url, fighter)]
            for stat in BASE_STATS:
                val = stats[stat]
                if pd.notna(val):
                    fighter_state[fighter][stat] = ewma(
                        fighter_state[fighter][stat],
                        val
                    )
            fighter_state[fighter]["fights"] += 1

    df = pd.DataFrame(rows)

    # Drop early fights with no history
    df = df.dropna()

    out_path = OUT_DIR / "fight_features_alpha.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    main()
