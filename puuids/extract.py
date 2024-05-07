import json
import time


def extract(input_json_path, output_text_path):
    """
    extracts puuids from a json file and writes them to a text file
    """
    with open(input_json_path, 'r') as file:
        data = json.load(file)

    puuids = []
    for item in data:
        participants = item['Info']['Participants']
        for participant in participants:
            puuids.append(participant['Puuid'])

    with open(output_text_path, 'w') as file:
        for i, puuid in enumerate(puuids, start=1):
            file.write(puuid + '\n')
    print(f"extracted {len(puuids)} puuids in {time.time() - start:.2f} seconds")


input_json_path = 'riotapi.matches.json'
output_text_path = 'puuids.txt'
start = time.time()
extract(input_json_path, output_text_path)

"""
extracted 20306296 puuids in 74.76 seconds

Process finished with exit code 0 
"""