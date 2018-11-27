import os

BASE_DIR = "/home/pi/Desktop/Full/"

IMG_DIR = os.path.join(BASE_DIR, "images")
ACCELEROMETER = os.path.join(BASE_DIR, "accel_correct_config")
IMG_REC = os.path.join(BASE_DIR, "img_rec_module")
IMG_CAM = os.path.join(BASE_DIR, "RPiCam")
COMM = os.path.join(BASE_DIR,"communication_module")
RUN_LOG_PATH = os.path.join(BASE_DIR, "transfer_test_correct_config/run_log.txt")
NAME_SCHEME = "img"
FILE_FORMAT = "png"
RAW_IMAGE = False
NUM_SAMPLES = 5
NAME_FORMAT = IMG_DIR + NAME_SCHEME + "_{0}" + FILE_FORMAT

#------- IMPORT THESE THREE ----
BRIDGE_NAME = 'dummy'
SERVER = "http://192.168.15.101:8000/sensors/"+BRIDGE_NAME+"/update/"
PASSWORD = "djioewfj34jod2jdoi3jr0jl983jsa"
#-------- ABOVE ------
#For RPI_C.py

RPI_L_IP = "192.168.15.103" #when hooked to Netgear Switch (Phil)
RPI_C_IP = "0.0.0.0"
RPI_C_Port = 1234
marker = "?"
units = "mm"
span = 10*1000
pi = 3.1415926535897932
query_time = 5
re_sync = 3

pMap = {3:2,5:3,7:4,8:14,10:15,11:17,12:18,13:27,15:22,16:23,18:24,19:10,21:9,22:25,23:11,24:8,26:7}
