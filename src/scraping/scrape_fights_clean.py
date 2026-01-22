"""
Scrape UFC fights with RELIABLE winner labels from individual fight-details pages.

IMPORTANT: Event listing pages show fighters in WINNER-FIRST order, NOT red/blue corner order!
Fight-details pages show fighters in red (left/first) vs blue (right/second) corner order.

This scraper gets everything from fight-details pages:
- Red corner = first person (left side)
- Blue corner = second person (right side)
- W/L/D/NC status from person-status elements
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

EVENTS_CSV = "data/raw/events.csv"
OUT_CSV = "data/raw/fights_clean.csv"
CHECKPOINT_CSV = "data/raw/.fights_checkpoint.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}

MAX_WORKERS = 10
REQUEST_TIMEOUT = 20


def get_fight_urls_from_event(event_url: str) -> list[str]:
    """Get all fight URLs from an event page (just URLs, not fighter names)."""
    try:
        res = requests.get(event_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(res.text, "lxml")
    urls = []

    for tr in soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover"):
        fight_url = tr.get("data-link")
        if fight_url:
            urls.append(fight_url)

    return urls


def scrape_fight_details(fight_url: str, event_info: dict) -> dict | None:
    """
    Scrape a fight-details page for red/blue fighters and W/L/D/NC status.
    
    On fight-details pages:
    - First person (left) = RED corner
    - Second person (right) = BLUE corner
    """
    try:
        res = requests.get(fight_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(res.text, "lxml")
    persons = soup.select("div.b-fight-details__person")
    
    if len(persons) < 2:
        return None

    # Person 1 = RED corner (left), Person 2 = BLUE corner (right)
    red_status_el = persons[0].select_one("i.b-fight-details__person-status")
    red_name_el = persons[0].select_one("a.b-fight-details__person-link")
    
    blue_status_el = persons[1].select_one("i.b-fight-details__person-status")
    blue_name_el = persons[1].select_one("a.b-fight-details__person-link")

    red_name = red_name_el.get_text(strip=True) if red_name_el else ""
    blue_name = blue_name_el.get_text(strip=True) if blue_name_el else ""
    
    red_status = red_status_el.get_text(strip=True).upper() if red_status_el else ""
    blue_status = blue_status_el.get_text(strip=True).upper() if blue_status_el else ""

    if not red_name or not blue_name:
        return None

    # Determine winner
    winner = None
    if red_status == "W":
        winner = "red"
    elif blue_status == "W":
        winner = "blue"
    elif red_status == "D" and blue_status == "D":
        winner = "draw"
    elif red_status == "NC" or blue_status == "NC":
        winner = "nc"

    return {
        "fight_url": fight_url,
        "red_fighter": red_name,
        "blue_fighter": blue_name,
        "red_outcome": red_status,
        "blue_outcome": blue_status,
        "winner": winner,
        **event_info,
    }


def main():
    events = pd.read_csv(EVENTS_CSV)
    print(f"Processing {len(events)} events...")

    # Step 1: Collect all fight URLs from events
    all_fights = []
    for i, event in events.iterrows():
        event_info = {
            "event_name": event["event_name"],
            "event_date": event["event_date"],
            "event_url": event["event_url"],
        }
        
        fight_urls = get_fight_urls_from_event(event["event_url"])
        for url in fight_urls:
            all_fights.append({"fight_url": url, **event_info})

        if (i + 1) % 100 == 0:
            print(f"  Events: {i+1}/{len(events)} ({len(all_fights)} fights)")
        time.sleep(0.05)

    print(f"\nFound {len(all_fights)} fights. Scraping details with {MAX_WORKERS} workers...")

    # Load checkpoint if exists
    done_urls = set()
    rows = []
    if Path(CHECKPOINT_CSV).exists():
        checkpoint = pd.read_csv(CHECKPOINT_CSV)
        rows = checkpoint.to_dict("records")
        done_urls = set(checkpoint["fight_url"])
        print(f"  Resuming from checkpoint: {len(done_urls)} already done")

    remaining = [f for f in all_fights if f["fight_url"] not in done_urls]
    print(f"  Remaining to scrape: {len(remaining)}")

    # Step 2: Scrape fight details concurrently
    def scrape_one(fight_info):
        return scrape_fight_details(
            fight_info["fight_url"],
            {k: v for k, v in fight_info.items() if k != "fight_url"}
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scrape_one, f): f for f in remaining}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                rows.append(result)

            if (i + 1) % 500 == 0:
                df = pd.DataFrame(rows)
                df.to_csv(CHECKPOINT_CSV, index=False)
                print(f"  Progress: {i+1}/{len(remaining)} (saved checkpoint)")

    # Final save
    df = pd.DataFrame(rows)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Cleanup checkpoint
    if Path(CHECKPOINT_CSV).exists():
        Path(CHECKPOINT_CSV).unlink()

    print(f"\n{'='*50}")
    print(f"Saved {len(df)} fights -> {OUT_CSV}")
    print(f"\nWinner distribution:")
    print(df["winner"].value_counts(dropna=False))
    print(f"\nRed wins: {(df['winner'] == 'red').sum()}")
    print(f"Blue wins: {(df['winner'] == 'blue').sum()}")


if __name__ == "__main__":
    main()
