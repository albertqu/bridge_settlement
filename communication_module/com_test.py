import requests
from requests import Session
import json
from datetime import datetime
from dateutil import parser
from random import randint

BRIDGE_NAME = 'dummy'
dest = "http://127.0.0.1:8000/sensors/"+BRIDGE_NAME+"/update/"
PASSWORD = "djioewfj34jod2jdoi3jr0jl983jsa"


def send_data_to_server(x=0.0, y=0.0, z=0.0, theta=0.0, phi=0.0, psi=0.0):
    payload = {'x': x, 'y': y, 'z': z, 'theta': theta, 'phi': phi, 'psi': psi}
    dt = datetime.now()
    cookies = {'csrftoken': encode(dt), 'time': str(dt)}
    r = requests.post(dest, data=payload, cookies=cookies)
    print(r.status_code)
    if r.status_code == 403:
        raise RuntimeWarning("CSRF ALERT ON, INSPECT YOUR SYSTEM")
    else:
        return r.text


def encode(time_sign):
    len_pw = len(PASSWORD)
    index = (time_sign.year + time_sign.month * 100 + time_sign.day + time_sign.hour * time_sign.minute
             * time_sign.second) % len_pw
    repl = chr(randint(97, 122))
    return PASSWORD[:index] + repl + PASSWORD[index+1:]


print(send_data_to_server(0.5, 1.5, 1.5))
#send_data_form(1.0, 2.0, 3.0)
#cookies = {'csrftoken': "djioewfj34jod2jdoi3jr0jl983jsa", 'TIME': datetime.now()}
