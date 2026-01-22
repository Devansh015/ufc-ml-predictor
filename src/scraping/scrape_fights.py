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
        url = event["event_url"]
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "lxml")

        # Each row is a fight; link points to fight-details page
        for a in soup.select("a[href*='fight-details']"):
            fight_url = a["href"]
            tr = a.find_parent("tr")

            # Fighter names typically appear in the red/blue name cells
            name_links = tr.select("td.b-fight-details__table-col a")
            # fallback: any <a> inside row
            if len(name_links) < 2:
                name_links = tr.select("a")

            red = name_links[0].get_text(strip=True) if len(name_links) > 0 else None
            blue = name_links[1].get_text(strip=True) if len(name_links) > 1 else None

            # Winner is stored as a "W/L" marker in first column; ufcstats varies a bit
            # We'll store result text and you can normalize later in cleaning.
            first_col = tr.select_one("td.b-fight-details__table-col")
            result_marker = first_col.get_text(" ", strip=True) if first_col else None

            rows.append({
                "event_name": event["event_name"],
                "event_date": event["event_date"],
                "event_url": url,
                "fight_url": fight_url,
                "red_fighter": red,
                "blue_fighter": blue,
                "result_marker": result_marker,
            })

        # be polite to the site
        if (i + 1) % 25 == 0:
            print(f"Processed {i+1}/{len(events)} events...")
        time.sleep(0.25)

    df = pd.DataFrame(rows)

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} fights -> {OUT_CSV}")

if __name__ == "__main__":
    scrape_fights()
