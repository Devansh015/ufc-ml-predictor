"""
Scrape winner/loser status from individual fight-details pages.

UFCStats encodes W/L/D/NC in `i.b-fight-details__person-status` elements
on each fight-details page. This is the ONLY reliable source for outcome labels.

Output columns:
- fight_url: unique identifier
- fighter1_name, fighter1_status: first person block (W/L/D/NC)
- fighter2_name, fighter2_status: second person block (W/L/D/NC)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time
import sys

FIGHTS_CSV = "data/raw/fights.csv"
OUT_CSV = "data/raw/fight_outcomes.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}


def scrape_fight_outcome(fight_url: str) -> dict | None:
    """Scrape a single fight-details page for fighter names and W/L/D/NC status."""
    try:
        res = requests.get(fight_url, headers=HEADERS, timeout=30)
        res.raise_for_status()
    except Exception as e:
        print(f"  ERROR fetching {fight_url}: {e}")
        return None

    soup = BeautifulSoup(res.text, "lxml")

    # Find the two person blocks
    persons = soup.select("div.b-fight-details__person")
    if len(persons) < 2:
        return None

    results = {"fight_url": fight_url}

    for i, person in enumerate(persons[:2], start=1):
        # Status: W, L, D, NC, or empty
        status_el = person.select_one("i.b-fight-details__person-status")
        status = status_el.get_text(strip=True) if status_el else ""

        # Fighter name
        name_el = person.select_one("a.b-fight-details__person-link")
        name = name_el.get_text(strip=True) if name_el else ""

        results[f"fighter{i}_name"] = name
        results[f"fighter{i}_status"] = status

    return results


def main():
    fights = pd.read_csv(FIGHTS_CSV)
    fight_urls = fights["fight_url"].dropna().unique().tolist()

    print(f"Scraping outcomes for {len(fight_urls)} unique fights...")

    rows = []
    for i, url in enumerate(fight_urls):
        result = scrape_fight_outcome(url)
        if result:
            rows.append(result)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(fight_urls)} fights...")

        time.sleep(0.15)  # polite rate limiting

    df = pd.DataFrame(rows)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Stats
    print(f"\nSaved {len(df)} fight outcomes -> {OUT_CSV}")
    print(f"Fighter1 status distribution:\n{df['fighter1_status'].value_counts()}")
    print(f"Fighter2 status distribution:\n{df['fighter2_status'].value_counts()}")


if __name__ == "__main__":
    main()
