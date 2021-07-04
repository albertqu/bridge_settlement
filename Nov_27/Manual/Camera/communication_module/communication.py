import requests, sys
from requests import Session
import json
from datetime import datetime
from dateutil import parser
from random import randint
sys.path.insert(0,"/home/pi/Desktop/Full_Demo/transfer_test_correct_config")
from SETTINGS import PASSWORD, SERVER

dest = SERVER

def send_data_to_server(x=0.0, y=0.0, z=0.0, theta=0.0, phi=0.0, psi=0.0):
    payload = {'x': x, 'y': y, 'z': z, 'theta': theta, 'phi': phi, 'psi': psi}
    dt = datetime.now()
    cookies = {'csrftoken': encode(dt), 'time': str(dt)}
    r = requests.post(dest, data=payload, cookies=cookies)
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