from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import json
import time
import requests

list = ["Yangon", "Mandalay", "Magway", 
        "Bago", "Pyay", "Taungoo", 
        "Naypyidaw", "Meiktila", "Pyin Oo Lwin", 
        "Hinthada", "Yamethin", "Pyawbwe", 
        "Tatkone", "Myingyan", "Bagan", 
        "Pakokku", "Taungtha", "Taunggyi", 
        "Phyu", "Kyaukpadaung", "Yesagyo", 
        "Monywa", "Taunggyi", "Kyaikto", 
        "Hpa-An", "Thaton", "Kyaukse"]

data = {
    "Yangon": {"Bago", "Hinthada"},
    "Mandalay": {"Pyawbwe", "Kyaukse", "Myingyan", "Pyin Oo Lwin"},
    "Magway": {"Pyay", "Pakokku", "Myingyan"},
    "Bago": {"Yangon", "Taungoo", "Pyay", "Kyaikto"},
    "Pyay": {"Bago", "Magway", "Hinthada"},
    "Taungoo": {"Bago", "Phyu", "Naypyidaw"},
    "Naypyidaw": {"Taungoo", "Phyu", "Tatkone", "Yamethin"},
    "Meiktila": {"Yamethin", "Pyawbwe", "Kyaukse"},
    "Pyin Oo Lwin": {"Mandalay"},
    "Hinthada": {"Yangon", "Pyay"},
    "Yamethin": {"Naypyidaw", "Tatkone", "Meiktila", "Pyawbwe"},
    "Pyawbwe": {"Yamethin", "Meiktila", "Mandalay"},
    "Tatkone": {"Naypyidaw", "Yamethin"},
    "Myingyan": {"Mandalay", "Magway", "Taungtha", "Kyaukpadaung"},
    "Bagan": {"Kyaukpadaung", "Pakokku"},
    "Pakokku": {"Bagan", "Magway", "Yesagyo"},
    "Taungtha": {"Myingyan", "Kyaukpadaung"},
    "Taunggyi": {"Kyaukse"},
    "Phyu": {"Taungoo", "Naypyidaw"},
    "Kyaukpadaung": {"Myingyan", "Taungtha", "Bagan"},
    "Yesagyo": {"Pakokku", "Monywa"},
    "Monywa": {"Yesagyo"},
    "Kyaikto": {"Bago", "Thaton"},
    "Hpa-An": {"Thaton"},
    "Thaton": {"Kyaikto", "Hpa-An"},
    "Kyaukse": {"Mandalay", "Meiktila", "Taunggyi"}
}


geolocator = Nominatim(user_agent="myGeoLocator")
coordinates = {}
graph = {}
for cities in list:
    try:
        location = geolocator.geocode(f"{cities}, Myanmar", timeout=10)
        if location:
            latitude, longitude = location.latitude, location.longitude
            coordinates[cities] = (latitude, longitude)
            graph[cities] = []
        else:
            print(f"Location not found for city: {cities}")

    except GeocoderTimedOut:
        print(f"Timeout occurred for city: {cities}")
    time.sleep(1)



for cities in data:
      for neighbor in data[cities]:
            lat_start, lon_start = coordinates[cities]
            lat_end, lon_end = coordinates[neighbor]
            url =  f"http://router.project-osrm.org/route/v1/driving/{lon_start},{lat_start};{lon_end},{lat_end}?overview=false"
            response = requests.get(url)
            result = response.json()
            graph[cities].append((neighbor, result['routes'][0]['distance']))
print(graph)
map_data = {"coordinates": coordinates, "graph": graph}

with open("map_data.json","w") as f:
    json.dump(map_data,f,indent=4)

