from django.apps import AppConfig
from decimal import Decimal


class SensorsConfig(AppConfig):
    name = 'sensors'

CONNECTION_PASSWORD = "djioewfj34jod2jdoi3jr0jl983jsa"
BUFFER_TIME = 3
RECENT_PERIOD = 10
THRESHOLD_DIS = 1
THRESHOLD_ROT = 1
CALIB_VAL = Decimal(166.1861)
