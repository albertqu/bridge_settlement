#Code for communications, sensor actions, and calculations for RPi on camera module
#Assume that RPI_C and RPI_L codes are implemented at same time (synced)
#Below code will execute at startup of RPI_C

from SETTINGS import *
import subprocess
import RPi.GPIO as GPIO
import sys
import socket
import time
from Read_Accel import read_accel
sys.path.insert(0,IMG_REC)
from sig_proc import center_detect
sys.path.insert(0,IMG_CAM)
from img_capture import ImgCollector
sys.path.insert(0,COMM)
from communication import send_data_to_server
from server_socket_connect_new import server_socket
from pinMasterSleep import pinMasterSleep
import traceback

#Givens

#See SETTINGS.py

time.sleep(10) #to avoid pins being set during flickering

def main():

    running = True

    print("run_log_path: " + RUN_LOG_PATH)
    if os.path.exists(RUN_LOG_PATH):
        rfile = open(RUN_LOG_PATH, "r")
        try:
            run_counter = int(rfile.read())
        except ValueError:
            run_counter = -1
    else:
        rfile = open(RUN_LOG_PATH, "w")
        run_counter = 0
        rfile.write(str(run_counter))
    rfile.close()

    print("run _counter: " + str(run_counter))
    if run_counter > 0 and run_counter % re_sync != 0:
        #Power Pins (The same actions would be performed in pinMasterSleep if run_counter = 0)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pMap[12], GPIO.OUT)
        GPIO.output(pMap[12], GPIO.HIGH)
        print("RPICam: 12 High")

        #Other initializations
        pos = []
        pitch_1 = []
        pitch_2 = []
        ds = []
        state = "BASELINE"
    elif run_counter == -1:
        print("RPICam: Issue with Run_File Read, Proceeding to Shutdown")
        state = "SHUTDOWN"
    else:
        state = "SYNC"

    while running == True:

        try:
            if state == "BASELINE":
                #Listen for Connection from RPI_L
                print("RPICam: Waiting for RPI_L to Wake")
                try:
                    rsv_msg_L = server_socket("RPI_C_AWAKE?")
                    if rsv_msg_L == '':
                        print("RPICam: Connection with RPI_L Not Successful")
                        print("RPICam: Proceeding to Safe Shutdown")
                        state = "SHUTDOWN"
                    joined_msg = ''.join(rsv_msg_L)
                    print("RPICam: Received: " + joined_msg)
                    if joined_msg == "LaserOn":
                        print("RPICam: Laser is On, Continuing to Accel Acquisition")
                        state = "ACCEL"
                    else:
                        print("RPICam: ERROR: Laser Apparently Not On")
                        print("RPICam: Will Proceed to Safe Shutdown")
                        state = "SHUTDOWN"
                except:
                    print("RPICam: BASELINE ERROR")
                    state = "SHUTDOWN"

            elif state == "SYNC":
                #Run Master or Slave code
                print("RPICam: Performing Time Sync with RPI_L")
                try:
                    t_sleep_sec = pinMasterSleep(pMap)
                except:
                    print("RPICam: SYNC ERROR")
                    t_sleep_sec = -1

                state = "SYNC_SHUTDOWN"

            elif state == "ACCEL":
                try:
                    print("RPICam: Taking Acceleration 1 Readings at Camera Module")
                    pitch_1 = read_accel()
                    print("pitch_1: " + str(pitch_1*180/pi))
                    print("RPICam: Continue to Image Acquisition")
                    state = "IMAGE"
                except:
                    print("RPICam: ACCEL ERROR")
                    state = "SHUTDOWN"

            elif state == "IMAGE":
                try:
                    img_collector = ImgCollector(IMG_DIR, NAME_SCHEME, FILE_FORMAT, RAW_IMAGE, NUM_SAMPLES)
                    print("RPICam: Taking Camera Picture of Laser Beam")
                    img_collector.capture()
                    time.sleep(2)
                    img_collector.shutdown()

                    rsv_msg_L = server_socket("TURN_OFF_LASER?")
                    joined_msg = ''.join(rsv_msg_L)
                    if joined_msg == "RPI_L_WAITING":
                        print("RPICam: Performing Image Processing of Image")
                        pos = center_detect(img_collector.get_last_meas(), NUM_SAMPLES)
                        print("pos = [" + str(pos[0]) + "," + str(pos[1]) + "]")
                        print("RPICam: Proceeding to Settlement Estimation")
                        state = "ESTIMATE"
                    else:
                        print("RPICam: RPI_L Communication Error")
                        state = "SHUTDOWN"
                except:
                    traceback.print_exc(file = sys.stdout)
                    print("RPICam: IMAGE ERROR")
                    state = "SHUTDOWN"

            elif state == "ESTIMATE":
                print("RPICam: Retrieving Acceleration Value from Laser Module")
                try:
                    accel_2_rsv = server_socket("GOT_ACCEL_2?")
                    joined_msg = ''.join(accel_2_rsv)
                    print("RPICam: Received: " + joined_msg)
                    if joined_msg == "ERROR_ACCEL":
                        print("RPICam: ERROR: Accel_2 Not Valid")
                        state = "SHUTDOWN"
                    else:
                        pitch_2 = float(''.join(accel_2_rsv))
                        print("pitch_2: " + str(pitch_2 * 180 / pi))

                        print("RPI_Cam: Performing Settlement Estimation")
                        ds = pos[1]
                        print("Settlement = " + str(ds) + " " + units)
                        print("RPI_Cam: Proceeding to Server Connection")
                        state = "SERVER"
                except:
                    traceback.print_exc(file = sys.stdout)
                    print("RPICam: ESTIMATE ERROR")
                    state = "SHUTDOWN"


            elif state == "SERVER":
                print("Sending Data to Server")
                try:
                    print(send_data_to_server(x = pos[0], y = ds, phi = pitch_1))
                    print("RPI_Cam: Value pushed, proceeding to safe shutdown")
                except RuntimeWarning as e:
                    print(e)
                    print("RPI_Cam: RunTime Warning: Value Not Pushed to Server")
                except:
                    print("RPI_Cam: Value Not Pushed to Server")
                time.sleep(1)
                state = "SHUTDOWN"

            elif state == "SHUTDOWN":
                run_counter = run_counter + 1
                wfile = open(RUN_LOG_PATH, "w")
                wfile.write(str(run_counter))
                wfile.close()

                img_collector.shutdown()
                print("RPI_Cam: Defer to Arduino for Power Control")
                time.sleep(10)
                subprocess.call(["shutdown", "-h", "now"])

            elif state == "SYNC_SHUTDOWN":
                if t_sleep_sec == -1:
                    print("RPICam: Will Reattempt Sync")
                else:
                    run_counter = run_counter + 1
                    wfile = open(RUN_LOG_PATH, "w")
                    wfile.write(str(run_counter))
                    wfile.close()
                    time.sleep(t_sleep_sec)
                GPIO.output(pMap[16], GPIO.HIGH)
                print("RPI_Cam: 16 high")
                time.sleep(1)
                subprocess.call(['shutdown', '-h', 'now'])

        except KeyboardInterrupt:
            img_collector.shutdown()
            print('User KeyBoard Interrupt, Proceeding to Shutdown')
            time.sleep(10)
            subprocess.call(["shutdown", "-h", "now"])

if __name__ == "__main__":
    main()







