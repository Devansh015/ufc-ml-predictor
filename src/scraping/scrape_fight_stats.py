import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time

EVENTS_CSV = "data/raw/events.csv"
OUT_CSV = "data/raw/fights.csv"

def scrape_fights():
    events = pd.read_csv(EVENTS_CSV)

    rows = []
    for i, event in events.iterrows():
        res = requests.get(event["event_url"], timeout=30)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")

        # Each fight is a row; fight link in first col
        for tr in soup.select("tr.b-fight-details__table-row"):
            a = tr.select_one("a[href*='fight-details']")
            if not a:
                continue

            fight_url = a["href"]

            # Red / Blue fighter names are in these corner cells
            red_a = tr.select_one("td.b-fight-details__table-col.l-page_align_left a")
            blue_a = tr.select_one("td.b-fight-details__table-col.l-page_align_right a")

            red = red_a.get_text(strip=True) if red_a else None
            blue = blue_a.get_text(strip=True) if blue_a else None

            # Winner is marked by a 'W' in the W/L column per fighter
            # UFCStats uses i.b-flag__text with 'W' next to winner
            win_flag = tr.select_one("i.b-flag__text")
            winner_name = None
            if win_flag and win_flag.get_text(strip=True) == "W":
                # If a W flag exists, it's on the winner's side of the row.
                # Determine which side contains the flag by checking which fighter cell contains it.
                # Safer approach: check if the left name cell has a win flag inside.
                left_flag = tr.select_one("td.b-fight-details__table-col.l-page_align_left i.b-flag__text")
                right_flag = tr.select_one("td.b-fight-details__table-col.l-page_align_right i.b-flag__text")
                if left_flag and left_flag.get_text(strip=True) == "W":
                    winner_name = red
                elif right_flag and right_flag.get_text(strip=True) == "W":
                    winner_name = blue

            rows.append({
                "event_name": event["event_name"],
                "event_url": event["event_url"],
                "event_date": event["event_date"],
                "fight_url": fight_url,
                "red_fighter": red,
                "blue_fighter": blue,
                "winner_name": winner_name
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