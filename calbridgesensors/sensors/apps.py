from django.apps import AppConfig
from decimal import Decimal
import datetime


class SensorsConfig(AppConfig):
    name = 'sensors'


CONNECTION_PASSWORD = "djioewfj34jod2jdoi3jr0jl983jsa"
BUFFER_TIME = datetime.timedelta(days=3)
BUFFER_MEAS = 3 # EXCLUSIVE OF THE FIRST DAMAGE READING
RECENT_PERIOD = 10
THRESHOLD_DIS = 1
THRESHOLD_ROT = 1
CALIB_VAL = Decimal(166.1861)


