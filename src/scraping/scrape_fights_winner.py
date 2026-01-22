import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time

EVENTS_CSV = "data/raw/events.csv"
OUT_CSV = "data/raw/fights_winner.csv"

def scrape_fights():
    events = pd.read_csv(EVENTS_CSV)
    rows = []

    for i, event in events.iterrows():
        res = requests.get(event["event_url"], timeout=30)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")

        for tr in soup.select("tr.b-fight-details__table-row"):
            fight_link = tr.select_one("a[href*='fight-details']")
            if not fight_link:
                continue

            fight_url = fight_link["href"]

            # UFCStats uses TWO left-aligned name columns (red then blue)
            name_cells = tr.select("td.b-fight-details__table-col.l-page_align_left")

            if len(name_cells) < 2:
                continue  # malformed row

            red_cell = name_cells[0]
            blue_cell = name_cells[1]

            red_a = red_cell.select_one("a")
            blue_a = blue_cell.select_one("a")

            red = red_a.get_text(strip=True) if red_a else None
            blue = blue_a.get_text(strip=True) if blue_a else None

            # Winner flag 'W' appears inside the winning fighter's name cell
            winner_name = None
            red_flag = red_cell.select_one("i.b-flag__text")
            blue_flag = blue_cell.select_one("i.b-flag__text")

            if red_flag and red_flag.get_text(strip=True) == "W":
                winner_name = red
            elif blue_flag and blue_flag.get_text(strip=True) == "W":
                winner_name = blue

            rows.append({
                "event_name": event["event_name"],
                "event_url": event["event_url"],
                "event_date": event["event_date"],
                "fight_url": fight_url,
                "red_fighter": red,
                "blue_fighter": blue,
                "winner_name": winner_name,
            })

        if (i + 1) % 25 == 0:
            print(f"Processed {i+1}/{len(events)} events...")

        time.sleep(0.25)

    df = pd.DataFrame(rows)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} fights -> {OUT_CSV}")

if __name__ == "__main__":
    scrape_fights()

