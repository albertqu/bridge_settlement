# Author:  Alex R. Mead
# Date: May 2016
# Description:
# This code will run on the Raspberry Pi 2 with Camera Module (RPiCM) to interact with the MacBookPro (MBP). It will
# wait for the MBP to request a measurement, take a measurement, then wait for the MBP to request the latest
# measurement, at which time it will send the measurement to the MBP.

# Same as the MBP this system is designed to run like finite state machine.

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
    )

import sys
import socket
import io
import subprocess
import picamera
import numpy as np
import gc
from time import sleep

# Constants
#IP = '127.0.0.1' # Testing on a single machine
IP = '0.0.0.0' # Actual run time.
port1 = 1234
port2 = 2345
marker = "?"

MAX_PIXEL_VALUE = 1023 # 2^10 = 1024
PIXEL_HEIGHT = 1944
PIXEL_WIDTH = 2592

# Takes in the Camera object instantiated at the beginning of the run and takes the desired SS expsoures with it.
def HDRI(SS,camera):

    size = len(SS)
    print("Taking %i measurements.\n",size)
    names = []

    # Loop through for each shutter speed requested
    for index, shutterSpeed in enumerate(SS):
        fileName = str(shutterSpeed) + ".data"
        camera.shutter_speed = shutterSpeed
        sleep(1)
        print(str(camera.shutter_speed))
        camera.capture(fileName,format='jpeg',bayer=True)
        names.append(fileName)

    # Return the list of filenames to the calling function.
    return names

def main():

    STATE = "idle"
    RUNNING = [True,0]

    # First declare the camera object and configure it as needed
    camera = picamera.PiCamera()
    camera.framerate=1
    sleep(1.0)

    camera.led = False
    camera.iso = 100
    sleep(10.0)
    # This longer sleep is needed so analog_gain and digital_gains settle so when I turn them off in the next step
    # they will have settled and not be in their initial 'low' values.

    camera.exposure_mode = 'off' # Fixes analog_gain and digital_gain values
    g = camera.awb_gains
    camera.aws_mode = 'off'
    camera.awb_gains = [1.0,1.0] # typical values 0.9-1.9 according to documentation, I picked these for consistency.
    camera.rotation = 180
    sleep(2.0)

    while(RUNNING[0]):

        if("idle" == STATE):
            print("We're in idle state waiting to hear from the MacBookPro...")

            # Start up a connection listening for the MBP.
            sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind((IP,port1))
            sock.listen(1)
            connection, client_address = sock.accept()

            rcv_msg = []
            while True:
                data = connection.recv(1)
                if marker in data:
                    rcv_msg.append(data[:data.find(marker)])
                    break
                rcv_msg.append(data)

            # Check the message received from the MBP, if correct, continue, else exit.
            rpl = ''.join(rcv_msg)
            if rpl == "True":
                print("Now going to acknowledge measurement.")

            # elif used to teminate remotely.
            elif rpl == "c":
                print("Big John called it for me...")
                sys.exit(0)

            # Acknowledge the MBP so it knows a measurement will be taking place.
            connection.send("True?")
            connection.close()

            # Change state: Continue on and take measurement
            STATE ="Measuring"

        elif ("Measuring" == STATE):
            print("Engaging the Camera Module to measure the output of the CFS under test...")
            # Here is the function call to take the measurement of the CFS inside CUBE2.0

            # List of the shutter speeds in microseconds to be used
            #SS = [1000000, 100000, 10000, 1000, 100, 10];
            SS = [1000000, 100000, 10000];

            # Take the measurements with the above stated shutter speeds
            files = HDRI(SS,camera)

            print("...Measurement taken of CFS.\n")

            # Continue to the next state and wait for MBP to query for the 145 measurements.
            STATE = "waitToSend"

        elif("waitToSend" == STATE):
            print("Waiting to send to MBP...")
            # Setup connection to wait for the request from the MBP
            # This maybe a short wait as the MBP may be trying to connect already if the Measurememt was a long time.
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((IP, port2))
            sock.listen(1)
            connection, client_address = sock.accept()

            rcv_msg = []
            while True:
                data = connection.recv(1)
                if marker in data:
                    rcv_msg.append(data[:data.find(marker)])
                    break
                rcv_msg.append(data)

            # Check the message received from the MBP, if correct, continue, else exit.
            rpl = ''.join(rcv_msg)
            if rpl == "True":
                print("Send measurement.")

            # Now use the same connection to let the MBP know it can grab the .jpg+RAW
            # files and run RPiCM_C.c locally over there, which is MUCH faster.
            connection.sendall("True?")
            connection.close()

            print("...measurment has been sent.\n")

            # Change state: measurement has been sent, change state back to idle
            STATE = "idle"

        # Cycle counter for the finite state machine
        RUNNING[1] = RUNNING[1] + 1
        print("Cycle number: " + str(RUNNING[1]))

    print("Little John called it...")
    sys.exit(0)

if __name__ == "__main__":
    main()

