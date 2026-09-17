"""
One-time data ACQUISITION script: fetches three overlapping <=5-year windows of
"flu symptoms" (US) Google Trends interest (each returned at weekly resolution)
and caches them as raw, UNSTITCHED CSVs.

This script does NOT do any rescaling/normalization — that step is done inside
flu_symptoms_analysis.ipynb itself (Step 1) so it is visible in the executed
notebook/PDF. This script only exists because pytrends is rate-limit-prone for
repeated live calls, so the raw chunks are cached here as a fallback data source
for the notebook to load if a live fetch attempt fails.
"""
import time
import pandas as pd
from pytrends.request import TrendReq

CHUNKS = [
    ('2015-01-01', '2019-12-31', 'flu_symptoms_chunk1_raw.csv'),
    ('2019-07-01', '2024-06-30', 'flu_symptoms_chunk2_raw.csv'),
    ('2024-01-01', '2026-07-08', 'flu_symptoms_chunk3_raw.csv'),
]

def fetch_chunk(start, end, retries=4, sleep_between=45):
    pytrends = TrendReq(hl='en-US', tz=360)
    for attempt in range(retries):
        try:
            pytrends.build_payload(['flu symptoms'], cat=0, timeframe=f'{start} {end}', geo='US', gprop='')
            raw = pytrends.interest_over_time()
            raw = raw[raw['isPartial'] == False][['flu symptoms']].copy()
            raw.columns = ['search_interest']
            raw.index.name = 'date'
            print(f'  fetched {start} to {end}: {raw.shape[0]} weekly points')
            return raw
        except Exception as e:
            print(f'  attempt {attempt+1} failed ({e}); backing off...')
            time.sleep(sleep_between * (attempt + 1))
    raise RuntimeError(f'Could not fetch chunk {start} to {end} after {retries} attempts')

for i, (start, end, path) in enumerate(CHUNKS, start=1):
    print(f'Fetching chunk {i} ({start} to {end})...')
    chunk = fetch_chunk(start, end)
    chunk.to_csv(path)
    print(f'  saved to {path}')
    if i < len(CHUNKS):
        time.sleep(30)

print('Done. Raw chunks cached; normalization/stitching happens inside the notebook.')
