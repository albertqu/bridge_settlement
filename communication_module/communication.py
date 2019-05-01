import requests
from datetime import datetime
from random import randint

BRIDGE_NAME = 'dummy'
SERVER = "http://apps2.peer.berkeley.edu/caltrans/sensors/"+BRIDGE_NAME+"/update/"
PASSWORD = "djioewfj34jod2jdoi3jr0jl983jsa"


def send_data_to_server(x=0.0, y=0.0, z=0.0, theta=0.0, phi=0.0, psi=0.0, errors=None, counter=-1):
    payload = {'x': x, 'y': y, 'z': z, 'theta': theta, 'phi': phi, 'psi': psi, 'errors': errors, 'counter': counter}
    dt = datetime.now()
    cookies = {'csrftoken': encode(dt), 'time': str(dt)}
    r = requests.post(SERVER, data=payload, cookies=cookies)
    if r.status_code == 403:
        raise RuntimeWarning("CSRF ALERT ON, INSPECT YOUR SYSTEM")
    elif r.status_code != 200:
        raise RuntimeWarning("Server Error, Code: {}".format(r.status_code))
    else:
        return r.text


def encode(time_sign):
    len_pw = len(PASSWORD)
    index = (time_sign.year + time_sign.month * 100 + time_sign.day + time_sign.hour * time_sign.minute
             * time_sign.second) % len_pw
    repl = chr(randint(97, 122))
    return PASSWORD[:index] + repl + PASSWORD[index+1:]
