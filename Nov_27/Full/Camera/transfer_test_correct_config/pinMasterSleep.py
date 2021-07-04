import socket, time
import RPi.GPIO as GPIO
from SETTINGS import *

def pinMasterSleep(pMap):
    max_count = 30
    # Configure board pin numbering system
    GPIO.setmode(GPIO.BCM)
    # Setup pin 12 to high so upon shutdown the pin drops and notifies the Arduino to cut power
    GPIO.setup(pMap[12],GPIO.OUT)
    GPIO.output(pMap[12],GPIO.HIGH)
    # Set up pin 16 as clock output
    GPIO.setup(pMap[16],GPIO.OUT)
    GPIO.setup(pMap[16],GPIO.LOW)
    print("RPI_Cam: 16 Low, 12 High")

    # Set up the server socket
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.settimeout(query_time*max_count)
    server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    server.bind((RPI_C_IP, RPI_C_Port))

    # Listen to the socket for client connection
    #server.bind(('localhost',1234))
    #server.bind(('0.0.0.0',1234))
    server.listen(1)
    toClient, client_address = server.accept()


    # Send time t1 to slave
    toClient.send(str.encode(str(int(time.time()*10**6))))

    # Get msg from Slave, delivered at t4
    BURN = toClient.recv(16)
    t4 = time.time()
    t4 = str.encode(str(int(t4*10**6)))

    # Send t4 to slave
    toClient.send(t4)

    print('Done with the code')

    # Now determine the clock coordination code.

    t_now = int(time.time()*10**6)

    #t_awake = t_now + 0.25*60*10**6
    t_awake = t_now + 15*10**6

    toClient.send(str.encode(str(t_awake)))

    t_sleep = t_awake - int(time.time()*10**6)

    t_sleep_sec = t_sleep / 10.0**6

    print('sleep time: ' + str(t_sleep_sec))

    server.close()

    return t_sleep_sec
