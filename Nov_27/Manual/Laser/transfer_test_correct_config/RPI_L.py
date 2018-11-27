#Code for communications, sensor actions, and calculations for RPi on laser module
#Assume that RPI_C and RPI_L codes are implemented at same time (synced)
#Below code will execute at startup of RPI_L

from SETTINGS import *
import sys
import RPi.GPIO as GPIO
import time
from Read_Accel import read_accel

#Givens

# See: SETTINGS.py

def main():

    try:
        while True:
            las_o = raw_input('Press [y] to turn on laser: ')
            if las_o == 'y':
                try:
                    print('Turning on laser...')
                    GPIO.setmode(GPIO.BCM)
                    GPIO.setup(pMap[Laser_Port], GPIO.OUT)
                    GPIO.output(pMap[Laser_Port], True)
                except:
                    print('Error turning on laser')
                finally:
                    break
            else:
                print('Please try again')


        while True:
            las = raw_input('Press [y] to turn off laser: ')
            if las == "y":
                try:
                    print('Turning off laser...')
                    GPIO.output(pMap[Laser_Port],False)
                    GPIO.cleanup()
                except:
                    print('Error turning off laser')
                finally:
                    break
            else:
                print('Please try again')

        while True:
            accel = raw_input('Press [y] to continue to accel_2 taking: ')
            if accel == 'y':
                try:
                    print('Taking Accel_1 Value...')
                    Ax, Ay, Az, Pitch = read_accel()
                    print("Ax: " + str(Ax) + " Ay: " + str(Ay) + "Az: " + str(Az) + "Pitch: " + str(Pitch*180/pi))
                except:
                    print('Error taking Accel_1 value')
                finally:
                    break
            else:
                print('Please try again')

        print('Program Done, Proceeding to Program Exit...')
        GPIO.cleanup()
        sys.exit(0)

    except KeyboardInterrupt:
        print('User KeyBoard Interrupt, Proceeding to Program Exit')
        GPIO.cleanup()
        time.sleep(2)
        sys.exit(0)


if __name__ == "__main__":
    main()
