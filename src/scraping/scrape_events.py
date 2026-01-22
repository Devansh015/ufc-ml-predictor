import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

BASE_URL = "http://ufcstats.com/statistics/events/completed?page=all"
OUT_CSV = "data/raw/events.csv"

def scrape_events():
    res = requests.get(BASE_URL, timeout=30)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "lxml")

    rows = []
    for a in soup.select("a[href*='event-details']"):
        event_name = a.get_text(strip=True)
        event_url = a["href"]

        tr = a.find_parent("tr")
        if not tr:
            continue

        # ✅ these spans exist on the events listing page
        date_span = tr.select_one("span.b-statistics__date")
        loc_span = tr.select_one("span.b-statistics__location")

        event_date = date_span.get_text(strip=True) if date_span else None
        event_location = loc_span.get_text(strip=True) if loc_span else None

        # skip header/empty rows
        if not event_date:
            continue

        rows.append({
            "event_name": event_name,
            "event_date": event_date,
            "event_location": event_location,
            "event_url": event_url
        })

    df = pd.DataFrame(rows)

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} events -> {OUT_CSV}")

if __name__ == "__main__":
    scrape_events()
