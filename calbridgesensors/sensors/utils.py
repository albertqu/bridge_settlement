import re
import json
from .apps import CONNECTION_PASSWORD
from dateutil import parser


def name_validate(s):
    target = r'[^ A-Za-z0-9]'
    return re.sub(target, " ", s).title()


def check_abnomaly(r1, r2):
    return True


def minv(a, b):
    return a if a < b else b


def maxv(a, b):
    return a if a > b else b


def min_val(a, b, c):
    return minv(a, minv(b, c))


def max_val(a, b, c):
    return maxv(a, maxv(b, c))


def verify_request(ck):
    time_sign = parser.parse(ck['time'])
    len_pw = len(CONNECTION_PASSWORD)
    quest = ck['csrftoken']
    index = code_expr(time_sign) % len_pw
    for i in range(len_pw):
        if i != index and quest[i] != CONNECTION_PASSWORD[i]:
            return False
    return True


def code_expr(time_sign):
    return time_sign.year + time_sign.month * 100 + time_sign.day \
           + time_sign.hour * time_sign.minute * time_sign.second


def succinct_time_str(dt):
    return str(dt)[:19]




