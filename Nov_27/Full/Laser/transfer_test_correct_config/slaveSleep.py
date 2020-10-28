import socket, time, sys, subprocess
import RPi.GPIO as GPIO
from SETTINGS import *

def slaveSleep(pMap):
    max_count = 30

    # Configure board pin numbering system
    GPIO.setmode(GPIO.BCM)
    # Setup pin 12 to high so upon shutdown the pin drops and notifies the Arduino to cut power
    GPIO.setup(pMap[12],GPIO.OUT)
    GPIO.output(pMap[12],GPIO.HIGH)
    # Set up pin 16 as clock output
    GPIO.setup(pMap[16],GPIO.OUT)
    GPIO.output(pMap[16],GPIO.LOW)
    print("RPI_L: 16 Low, 12 High")

    # Set up socket and connect to the server
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client.setblocking(0)
    client.settimeout(query_time*max_count)
    #client.connect(('192.168.15.104',1234))

    for i in range(0, max_count - 1):
        try:
            client.connect((RPI_C_IP, RPI_C_Port))
            break
        except:
            time.sleep(query_time)

    # Receive t1 from Master, at t1p
    t1b = client.recv(16) # Receive t1 value
    t2p = time.time() # Get time of delivery of t1, which is t2p

    t3p = time.time() # Get time t3p, send notification to Master
    client.send(str.encode(str(int(time.time()*10**6))))

    t4b = client.recv(16)

    # Now we have all times, t1, t2p, t3p, t4

    t1 = int(t1b.decode())
    t2p = int(t2p*10**6)
    t3p = int(t3p*10**6)
    t4 = int(t4b)


    print('t1: ' + str(t1))
    print('t2p: ' + str(t2p))
    print('t3p: ' + str(t3p))
    print('t4: ' + str(t4))

    r = 30 # transmission time, microseconds

    e = t2p - t1 - (r / 2)

    print('Error Estimate: ' + str(e))


    #print(str(t1 - t2p))

    # This is the synchronization code for the CalTrans bridge start up
    t_awake = int(client.recv(16))

    t_sleep = t_awake - int(time.time()*10**6) + e

    print('t_sleep: ' + str(t_sleep))

    t_sleep_sec = t_sleep / 10.0**6

    print('sleep time: ' + str(t_sleep_sec))

    client.close()

    return t_sleep_sec


