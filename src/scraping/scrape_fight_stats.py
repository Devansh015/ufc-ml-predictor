import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time
import re

FIGHTS_CSV = "data/raw/fights.csv"
OUT_DETAILS = "data/raw/fight_details.csv"
OUT_TOTALS  = "data/raw/fighter_totals.csv"
OUT_ROUNDS  = "data/raw/round_stats.csv"

def parse_x_of_y(s: str):
    """
    '10 of 25' -> (10, 25)
    Returns (None, None) if missing or '--'
    """
    if not isinstance(s, str):
        return (None, None)

    s = " ".join(s.split())  # normalize whitespace
    if s in ["--", ""]:
        return (None, None)

    m = re.match(r"^(\d+)\s+of\s+(\d+)$", s)
    if not m:
        return (None, None)

    return (int(m.group(1)), int(m.group(2)))


def parse_ctrl_time(s: str):
    """
    Converts control time strings to seconds.
    Handles '5:32', '0:05', '--', '', and weird strings safely.
    Uses the LAST two colon-separated parts as mm:ss if extra ':' exist.
    """
    if not isinstance(s, str):
        return 0

    s = s.strip()
    if s in ["--", "", "NULL", "None"]:
        return 0

    # Extract a mm:ss pattern anywhere in the string
    m = re.search(r"(\d+):(\d{2})", s)
    if not m:
        return 0

    mm = int(m.group(1))
    ss = int(m.group(2))
    return mm * 60 + ss


def scrape_one_fight(fight_url: str):
    res = requests.get(fight_url, timeout=30)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "lxml")

    # --- Fight meta ---
    # Weight class + method + round/time etc live in the "details" section
    title = soup.select_one("h2.b-content__title")
    fight_title = title.get_text(" ", strip=True) if title else None

    details_box = soup.select_one("i.b-fight-details__fight-title")
    fight_subtitle = details_box.get_text(" ", strip=True) if details_box else None

    # parse details list items
    # Example: "Weight: Lightweight", "Method: KO/TKO", "Round: 2", "Time: 3:41"
    meta = {"fight_url": fight_url, "fight_title": fight_title, "fight_subtitle": fight_subtitle}

    for li in soup.select("div.b-fight-details__fight i.b-fight-details__text-item"):
        txt = li.get_text(" ", strip=True)
        if ":" in txt:
            k, v = txt.split(":", 1)
            meta[k.strip().lower().replace(" ", "_")] = v.strip()

    # --- Totals table (per fighter) ---
    totals_table = soup.select_one("table.b-fight-details__table.js-fight-table")
    totals_rows = []
    if totals_table:
        trs = totals_table.select("tbody tr")
        for tr in trs:
            tds = [td.get_text(" ", strip=True) for td in tr.select("td")]
            # UFCStats totals columns order (commonly):
            # 0: fighter, 1: kd, 2: sig str, 3: sig str %, 4: total str, 5: td, 6: td %, 7: sub att, 8: rev, 9: ctrl
            # But sometimes includes extra columns, so we map by index safely.
            fighter = tds[0] if len(tds) > 0 else None
            kd = tds[1] if len(tds) > 1 else None
            sig = tds[2] if len(tds) > 2 else None
            sig_pct = tds[3] if len(tds) > 3 else None
            tot = tds[4] if len(tds) > 4 else None
            td = tds[5] if len(tds) > 5 else None
            td_pct = tds[6] if len(tds) > 6 else None
            sub_att = tds[7] if len(tds) > 7 else None
            rev = tds[8] if len(tds) > 8 else None
            ctrl = tds[9] if len(tds) > 9 else None

            sig_l, sig_a = parse_x_of_y(sig)
            tot_l, tot_a = parse_x_of_y(tot)
            td_l, td_a = parse_x_of_y(td)
            ctrl_sec = parse_ctrl_time(ctrl)

            totals_rows.append({
                "fight_url": fight_url,
                "fighter": fighter,
                "kd": int(kd) if str(kd).isdigit() else None,
                "sig_landed": sig_l,
                "sig_attempted": sig_a,
                "sig_pct": sig_pct,
                "tot_landed": tot_l,
                "tot_attempted": tot_a,
                "td_landed": td_l,
                "td_attempted": td_a,
                "td_pct": td_pct,
                "sub_att": int(sub_att) if str(sub_att).isdigit() else None,
                "rev": int(rev) if str(rev).isdigit() else None,
                "ctrl_sec": ctrl_sec,
            })

    # --- Per-round tables ---
    round_rows = []
    # Each round table has class js-fight-table for a specific round section
    # UFCStats uses multiple tables; round-by-round is typically in tables with class js-fight-table and data-round attr.
    for table in soup.select("table.b-fight-details__table.js-fight-table"):
        # first one is totals; subsequent ones are rounds
        # We detect round tables by checking if header contains "Round"
        thead = table.select_one("thead")
        if not thead:
            continue
        header_text = thead.get_text(" ", strip=True).lower()
        if "round" not in header_text:
            continue

        # Try to grab round number from the table's preceding label
        # Often there's an h2/h3 label like "Round 1"
        round_label = table.find_previous(string=re.compile(r"Round\s+\d+", re.IGNORECASE))
        rnd = None
        if round_label:
            m = re.search(r"Round\s+(\d+)", str(round_label), re.IGNORECASE)
            if m:
                rnd = int(m.group(1))

        for tr in table.select("tbody tr"):
            tds = [td.get_text(" ", strip=True) for td in tr.select("td")]
            fighter = tds[0] if len(tds) > 0 else None
            kd = tds[1] if len(tds) > 1 else None
            sig = tds[2] if len(tds) > 2 else None
            sig_pct = tds[3] if len(tds) > 3 else None
            tot = tds[4] if len(tds) > 4 else None
            td = tds[5] if len(tds) > 5 else None
            td_pct = tds[6] if len(tds) > 6 else None
            sub_att = tds[7] if len(tds) > 7 else None
            rev = tds[8] if len(tds) > 8 else None
            ctrl = tds[9] if len(tds) > 9 else None

            sig_l, sig_a = parse_x_of_y(sig)
            tot_l, tot_a = parse_x_of_y(tot)
            td_l, td_a = parse_x_of_y(td)
            ctrl_sec = parse_ctrl_time(ctrl)

            round_rows.append({
                "fight_url": fight_url,
                "round": rnd,
                "fighter": fighter,
                "kd": int(kd) if str(kd).isdigit() else None,
                "sig_landed": sig_l,
                "sig_attempted": sig_a,
                "sig_pct": sig_pct,
                "tot_landed": tot_l,
                "tot_attempted": tot_a,
                "td_landed": td_l,
                "td_attempted": td_a,
                "td_pct": td_pct,
                "sub_att": int(sub_att) if str(sub_att).isdigit() else None,
                "rev": int(rev) if str(rev).isdigit() else None,
                "ctrl_sec": ctrl_sec,
            })

    return meta, totals_rows, round_rows

def main():
    fights = pd.read_csv(FIGHTS_CSV)

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    details = []
    totals_all = []
    rounds_all = []

    for idx, row in fights.iterrows():
        fight_url = row["fight_url"]
        try:
            meta, totals_rows, round_rows = scrape_one_fight(fight_url)
            details.append(meta)
            totals_all.extend(totals_rows)
            rounds_all.extend(round_rows)
        except Exception as e:
            print(f"[WARN] Failed {fight_url}: {e}")

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1}/{len(fights)} fights...")

        time.sleep(0.25)

    pd.DataFrame(details).to_csv(OUT_DETAILS, index=False)
    pd.DataFrame(totals_all).to_csv(OUT_TOTALS, index=False)
    pd.DataFrame(rounds_all).to_csv(OUT_ROUNDS, index=False)

    print(f"Saved -> {OUT_DETAILS}")
    print(f"Saved -> {OUT_TOTALS}")
    print(f"Saved -> {OUT_ROUNDS}")

if __name__ == "__main__":
    main()
