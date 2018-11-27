#Code for communications, sensor actions, and calculations for RPi on camera module
#Assume that RPI_C and RPI_L codes are implemented at same time (synced)
#Below code will execute at startup of RPI_C

from SETTINGS import *
import RPi.GPIO as GPIO
import sys, traceback
import time
from Read_Accel import read_accel
sys.path.insert(0,IMG_REC)
from sig_proc import center_detect
sys.path.insert(0,IMG_CAM)
from img_capture import ImgCollector
sys.path.insert(0,COMM)
from communication import send_data_to_server

#See SETTINGS.py

def main():

    try:
        try:
            img_collector = ImgCollector(IMG_DIR, NAME_SCHEME, FILE_FORMAT, RAW_IMAGE, NUM_SAMPLES)
        except:
            print('Error: Camera Not Setup')
        while True:
            pic = raw_input('Press [y] to continue to picture taking [Ambient]: ')
            if pic == 'y':
                try:
                    print('Taking Picture...')
                    img_collector.capture()
                    time.sleep(2)
                    print('Picture Taken, img_collector shutdown')
                except:
                    print('Error Taking Photo...')
                    traceback.print_exc(file=sys.stdout)
                finally:
                    break
            else:
                print('Please try again')

        while True:
            pic = raw_input('Press [y] to continue to picture taking [Laser]: ')
            if pic == 'y':
                try:
                    print('Taking Picture...')
                    img_collector.capture()
                    time.sleep(2)
                    img_collector.shutdown()
                    print('Picture Taken, img_collector shutdown')
                except:
                    print('Error Taking Photo...')
                    traceback.print_exc(file=sys.stdout)
                finally:
                    break
            else:
                print('Please try again')

        while True:
            accel = raw_input('Press [y] to continue to accel_1 taking: ')
            if accel == 'y':
                try:
                    print('Taking Accel_1 Value...')
                    Ax,Ay,Az,pitch = read_accel()
                    print("Ax: " + str(Ax) + " Ay: " + str(Ay) + "Az: " + str(Az) + "Pitch [deg]: " + str(pitch*180/pi))
                except:
                    print('Error Taking Accel_1 Value...')
                finally:
                    break
            else:
                print('Please try again')

        print('Program Done, Proceeding to Program Exit...')
        GPIO.cleanup()
        sys.exit(0)

    except KeyboardInterrupt:
        img_collector.shutdown()
        GPIO.cleanup()
        print('User KeyBoard Interrupt, Proceeding to Program Exit')
        time.sleep(2)
        sys.exit(0)

if __name__ == "__main__":
    main()
