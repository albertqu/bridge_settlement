import os

BASE_DIR = '/home/pi/Desktop/Full/'

ACCELEROMETER = os.path.join(BASE_DIR, "accel_correct_config")
RUN_LOG_PATH = os.path.join(BASE_DIR, "transfer_test_correct_config/run_log.txt")

#For RPI_L.py

#RPI_L_IP = "192.168.3.3" #address of cam module (client)
RPI_C_IP = "192.168.15.104" #when hooked to Netgear Switch (Phil)
RPI_L_IP = "192.168.15.103" #when hooked to Netgear Switch (Phil)
RPI_L_Port = 1234
RPI_C_Port = 1234
Laser_Port = 11
marker = "?"
query_time = 5
re_sync = 3
pi = 3.1415926535897932

pMap = {3:2,5:3,7:4,8:14,10:15,11:17,12:18,13:27,15:22,16:23,18:24,19:10,21:9,22:25,23:11,24:8,26:7}
