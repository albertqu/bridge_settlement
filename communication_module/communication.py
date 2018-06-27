import requests
from requests import Session
import json
from datetime import datetime

BRIDGE_NAME = 'san-mateo'
dest = "http://127.0.0.1:8000/sensors/"+BRIDGE_NAME+"/update/"
PASSWORD = "djioewfj34jod2jdoi3jr0jl983jsa"


def send_data_json(x, y, z):
    val = {'x': x, 'y': y, 'z': z, "pw": PASSWORD}
    #payload = json.dumps(val)
    headers = {'TIME': str(datetime.now())}
    cookies = {'csrftoken': '7HfotuHJteyTQkYomyo1oe72bWXRtlZyNsT7MnSmJsnFBVICyT6VYvZYJKwI4EDa'}
    r = requests.post(dest, json=val, cookies=cookies)
    print(r.status_code)
    print(r.headers)
    print("cookies: ", r.cookies)
    print(r.text)


def send_data_form(x, y, z):
    payload = {'x': x, 'y': y, 'z': z, "pw": PASSWORD}
    cookies = {'csrftoken': '7HfotuHJteyTQkYomyo1oe72bWXRtlZyNsT7MnSmJsnFBVICyT6VYvZYJKwI4EDa', 'TIME': datetime.now()}
    r = requests.post(dest, data=payload, cookies=cookies)
    print(r.status_code)
    print(r.headers)
    print("cookies: ", r.cookies)
    print(r.text)


send_data_json(1.0, 2.0, 3.0)
#send_data_form(1.0, 2.0, 3.0)
#cookies = {'csrftoken': "djioewfj34jod2jdoi3jr0jl983jsa", 'TIME': datetime.now()}
