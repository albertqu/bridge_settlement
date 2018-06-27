import re


def name_validate(s):
    target = r'[^ A-Za-z0-9]'
    return re.sub(target, " ", s).title()


def check_abnomaly(r1, r2):
    return True
