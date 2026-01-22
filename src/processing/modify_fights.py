import pandas as pd
from pathlib import Path

RAW_DETAILS = "data/raw/fight_details.csv"
RAW_TOTALS  = "data/raw/fighter_totals.csv"
RAW_ROUNDS  = "data/raw/round_stats.csv"

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    details = pd.read_csv(RAW_DETAILS)
    totals  = pd.read_csv(RAW_TOTALS)
    rounds  = pd.read_csv(RAW_ROUNDS)

    # Deduplicate
    details = details.drop_duplicates(subset=["fight_url"])
    totals  = totals.drop_duplicates(subset=["fight_url", "fighter"])
    rounds  = rounds.drop_duplicates(subset=["fight_url", "round", "fighter"])

    # Drop unusable rows
    details = details.dropna(subset=["fight_url"])
    totals  = totals.dropna(subset=["fight_url", "fighter"])

    # Coerce numerics
    for c in ["kd","sig_landed","sig_attempted","tot_landed","tot_attempted",
              "td_landed","td_attempted","sub_att","rev","ctrl_sec"]:
        if c in totals.columns:
            totals[c] = pd.to_numeric(totals[c], errors="coerce")

    for c in ["round","kd","sig_landed","sig_attempted","tot_landed","tot_attempted",
              "td_landed","td_attempted","sub_att","rev","ctrl_sec"]:
        if c in rounds.columns:
            rounds[c] = pd.to_numeric(rounds[c], errors="coerce")

    # Save processed
    details.to_csv(OUT_DIR / "fight_details.csv", index=False)
    totals.to_csv(OUT_DIR / "fighter_totals.csv", index=False)
    rounds.to_csv(OUT_DIR / "round_stats.csv", index=False)

    print("Saved processed files to data/processed/")
    print("fight_details:", len(details), "unique fights:", details["fight_url"].nunique())
    print("fighter_totals:", len(totals), "unique fights:", totals["fight_url"].nunique())
    print("round_stats:", len(rounds), "unique fights:", rounds["fight_url"].nunique())

if __name__ == "__main__":
    main()