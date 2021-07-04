#Code for communications, sensor actions, and calculations for RPi on laser module
#Assume that RPI_C and RPI_L codes are implemented at same time (synced)
#Below code will execute at startup of RPI_L

from SETTINGS import *
import sys
import subprocess
import socket
import RPi.GPIO as GPIO
import time
from Read_Accel import read_accel
from client_socket_connect_new import client_socket
from slaveSleep import slaveSleep

#Givens

# See: SETTINGS.py

time.sleep(10) #to avoid setting pins when pins are flickering
 
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
    print("run_counter: " +  str(run_counter))
    if run_counter > 0 and run_counter % re_sync != 0:
        #Power Pins (The same actions would be performed in slaveSleep if run_counter = 0)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pMap[12], GPIO.OUT)
        GPIO.output(pMap[12], GPIO.HIGH)
        print("RPI_L: 12 High")

        #Other initializations
        pitch_2 = []
        state = "BASELINE"
    elif run_counter == -1:
        print("RPI_L: Issue with Run_File Read, Proceeding to Shutdown")
        state = "SHUTDOWN"
    else:
        state = "SYNC"

    while running == True:

        try:
            if state == "BASELINE":

                print("RPI_L: Turning on Laser")
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(pMap[Laser_Port], GPIO.OUT)
                GPIO.output(pMap[Laser_Port], True)
                print("RPI_L: Waiting for RPI_Cam to Connect")

                try:
                    rsv_msg_C = client_socket("LaserOn?")
                    joined_msg = ''.join(rsv_msg_C)
                    print("RPI_L: Received: " + joined_msg)
                    if joined_msg == '':
                        print("RPI_L: Connection with RPI_C Not Successful")
                        print("RPI_L: Will Proceed to Safe Shutdown")
                        GPIO.output(pMap(Laser_Port),False)
                        state = "SHUTDOWN"
                    if joined_msg == "RPI_C_AWAKE":
                        print("RPI_L: RPICam Awake, Will Turn off Laser Upon Notice")
                        state = "WAIT"
                    else:
                        print("RPI_L: ERROR: RPICam Apparently Not On")
                        print("RPI_L: Will Proceed to Safe Shutdown")
                        GPIO.output(pMap(Laser_Port),False)
                        state = "SHUTDOWN"
                except:
                    print("RPI_L: BASELINE ERROR")
                    state = "SHUTDOWN"

            elif state == "WAIT":
                try:
                    time.sleep(query_time * 5) # give RPI_C a chance to close its socket
                    rsv_msg_C = client_socket("RPI_L_WAITING?")
                    joined_msg = ''.join(rsv_msg_C)
                    print("RPI_L: Received: " + joined_msg)

                    if joined_msg == "TURN_OFF_LASER":
                        print("RPI_L: RPICam Took Photo, Will Turn Off Laser")
                        try:
                            GPIO.output(pMap[Laser_Port], False)
                            state = "ACCEL"
                            time.sleep(2)
                        except:
                            print("RPI_L: ERROR: Cannot Turn off Laser, Proceed to Shutdown")
                            state = "SHUTDOWN"
                    else:
                        print("RPI_L: ERROR: RPICam Apparently Did Not Take Photo. Exiting Program")
                        state = "SHUTDOWN"
                except:
                    print("RPI_L: WAIT ERROR")
                    state = "SHUTDOWN"

            elif state == "SYNC":
                #Run Master or Slave code
                print("RPI_L: Performing Time Sync with RPICam")
                try:
                    t_sleep_sec = slaveSleep(pMap)
                except:
                    print("RPI_L: SYNC ERROR")
                    t_sleep_sec = -1

                state = "SYNC_SHUTDOWN"

            elif state == "ACCEL":
                print("RPI_L: Taking Acceleration Reading at Laser Module")
                try:
                    pitch_2 = read_accel() #[rads]
                    print("pitch_2: " + str(pitch_2*180/pi))
                    state = "SEND_ACCEL"
                except:
                    print("RPI_L: ACCEL ERROR")
                    state = "SHUTDOWN"

            elif state == "SEND_ACCEL":
                    try:
                        rsv_msg_C = client_socket(str(pitch_2) + "?")
                        joined_msg = ''.join(rsv_msg_C)

                        if joined_msg == "GOT_ACCEL_2":
                            print("RPI_L: RPICam Received Accel_2, Proceed to Safe Shutdown")
                            state = "SHUTDOWN"
                        else:
                            print("RPI_L: ERROR: Accel_2 Not Acquired, Proceed to Safe Shutdown")
                            state = "SHUTDOWN"
                    except:
                        print("RPI_L: ACCEL ERROR")
                        state = "SHUTDOWN"

            elif state == "SHUTDOWN":
                run_counter = run_counter + 1
                wfile = open(RUN_LOG_PATH, "w")
                wfile.write(str(run_counter))
                wfile.close()

                print("RPI_L: Defer to Arduino for Power Control")
                time.sleep(10)
                subprocess.call(["shutdown", "-h", "now"])

            elif state == "SYNC_SHUTDOWN":
                if t_sleep_sec == -1:
                    print("RPI_L: Will Reattempt Sync")
                else:
                    run_counter = run_counter + 1
                    wfile = open(RUN_LOG_PATH, "w")
                    wfile.write(str(run_counter))
                    wfile.close()
                    time.sleep(t_sleep_sec)
                GPIO.output(pMap[16], GPIO.HIGH)
                print("RPI_L: pin 16 HIGH")
                time.sleep(1)
                subprocess.call(['shutdown', '-h', 'now'])

        except KeyboardInterrupt:
            print('User KeyBoard Interrupt, Proceeding to Shutdown')
            time.sleep(10)
            subprocess.call(["shutdown", "-h", "now"])


if __name__ == "__main__":
    main()










