import requests
import pandas as pd
import signal
from data_baron.data_baron import data_baron
from concurrent.futures import ThreadPoolExecutor, as_completed, thread
import time

# constants
API_KEY = 'YOUR API KEY'
MAX_WORKERS = 10
REQUEST_INTERVAL = 0.21
DATA_FILE = 'mastery_data_1m.feather'
PUUID_FILE = 'puuids/puuids_sliced_1m.txt'
BATCH_SIZE = 50000

results = []


def get_champion_mastery(puuid):
    """retrieves champion mastery data for a given puuid"""
    url = f'https://euw1.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}?api_key={API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f'error retrieving data for {puuid}: {str(e)}')
        return []


def process_mastery_data(puuid):
    """processes champion mastery data for a given puuid"""
    time.sleep(REQUEST_INTERVAL)
    mastery_data = get_champion_mastery(puuid)
    champion_points = {'PUUID': puuid}
    for champ_data in mastery_data:
        champion_name = data_baron.get(str(champ_data['championId']), 'Unknown Champion')
        champion_points[champion_name] = champ_data.get('championPoints', 0)
    return champion_points


def load_puuids():
    """loads puuids from file"""
    with open(PUUID_FILE, 'r') as file:
        puuids = [line.strip() for line in file]
    return puuids


def save_dataframe():
    """saves data to disk"""
    if results:
        df = pd.DataFrame(results)
        df.fillna(0, inplace=True)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(int)
        df.to_feather(DATA_FILE, index=False)
        print('data saved to disk')


def signal_handler(signal_received, frame):
    """handles SIGINT signal"""
    print('SIGINT or CTRL-C detected. exiting gracefully')
    save_dataframe()
    exit(0)


def main():
    global results
    signal.signal(signal.SIGINT, signal_handler)
    puuids = load_puuids()

    processed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_mastery_data, puuid) for puuid in puuids]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
                processed_count += 1
                print(f'{processed_count} / {len(puuids)} ({processed_count / len(puuids) * 100:.2f}%) processed')
                # if processed_count % BATCH_SIZE == 0:
                    # save_dataframe()

    save_dataframe()
    print('all data has been processed and saved')


if __name__ == "__main__":
    main()