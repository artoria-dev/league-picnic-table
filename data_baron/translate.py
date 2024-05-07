import requests

response = requests.get(f"https://ddragon.leagueoflegends.com/cdn/14.9.1/data/en_US/champion.json")
c_data = response.json()
c_dict = {}
for champion in c_data['data']:
    c_dict[c_data['data'][champion]['key']] = c_data['data'][champion]['name']
for key in c_dict:
    print(f'"{key}": "{c_dict[key]}"')