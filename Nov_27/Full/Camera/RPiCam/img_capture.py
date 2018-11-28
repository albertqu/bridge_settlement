import picamera
import os
from fractions import Fraction
import sys, traceback
import time


class ImgCollector:

    def __init__(self, dir='', ns='img', form='png', raw=False, num=1, serialize=True, step=True):
        if dir:
            if not os.path.exists(dir):
                os.mkdir(dir)
            if dir[len(dir)-1] != '/':
                dir += '/'

        self.name_scheme = dir + ns + '_{0}.' + form
        self._dir = dir
        self._ns = ns
        self._form = form
        self._raw = raw
        self._num = num
        self._step = step
        self.init_cam()
        if serialize:
            if os.path.exists("img_log.txt"):
                rfile = open("img_log.txt", "r")
                try:
                    self.counter = int(rfile.read())
                except ValueError:
                    self.counter = 1
            else:
                rfile = open("img_log.txt", "w")
                self.counter = 1
            rfile.close()
        else:
            self.counter = 1

    def change_ns(self, ns):
        self._ns = ns
        self.name_scheme = self._dir + self._ns + '_{0}.' + self._form

    def change_format(self, form):
        self._form = form
        self.name_scheme = self._dir + self._ns + '_{0}.' + self._form

    def change_dir(self, dir):
        self._dir = dir
        self.name_scheme = self._dir + self._ns + '_{0}.' + self._form

    def change_num(self, num):
        self._num = num
        if self._num == 1:
            self.capture = self.uni_capture
        else:
            self.capture = self.multi_capture

    def init_cam(self):
        self.cam = picamera.PiCamera(resolution=(640, 480))
        time.sleep(2)
        self.cam.led = False
        self.cam.framerate = Fraction(1, 6)
        self.cam.rotation = 180
        self.cam.shutter_speed = 800000
        self.cam.iso = 100
        self.cam.exposure_mode = 'off'
        time.sleep(3)
        if not self._step and self._num == 1:
            self.capture = self.uni_capture
        else:
            self.capture = self.multi_capture

    def get_last_meas(self):
        if self._num == 1:
            return self.name_scheme.format(self.counter - 1)
        else:
            return self._dir + self._ns + '_%d_{0}.' % (self.counter - 1) + self._form

    def shutdown(self):
        self.cam.close()
        wfile = open("img_log.txt", "w")
        wfile.write(str(self.counter))
        wfile.close()

    def uni_capture(self):
        self.cam.capture(self.name_scheme.format(self.counter), bayer=self._raw)
        self.counter += 1

    def multi_capture(self, ambi=False):
        if ambi:
            self.cam.capture(self.name_scheme.format("%d_%d" % (self.counter, 0)), bayer=self._raw)
        else:
            self.multi_capture_base()

    def multi_capture_base(self):
        file_list = [self.name_scheme.format("%d_%d" % (self.counter, i)) for i in range(1, self._num + 1)]
        self.cam.capture_sequence(file_list, bayer=self._raw)
        self.counter += 1


def main():
    prompt = input("Welcome to the RPiCam Module. Type q for quick test, d for debug, or f / [other inputs] for full test.\n")
    global recur
    if prompt == 'q':
        directory = 'quick_test'
        name_pattern = 'img'
        pic_format = 'png'
        raw_image = False
        num_meas = 1
        recur = True
    elif prompt == 'd':
        camera_debug()
        sys.exit()
    else:
        directory = input("Input a directory:\n")
        name_pattern = input("Input a name pattern:\n")
        pic_format = input("Input a picture format:\n")
        raw_image = input("Raw image?[y/n]\n") in ['y', 'yes']
        num_meas = int(input("Number of image samples for one measurement?\n"))
        recur = False

    while True:
        try:
            ic = ImgCollector(dir=directory, ns=name_pattern, form=pic_format, raw=raw_image, num=num_meas)
            break
        except:
            directory = input("Ill-formated directory, type in another one: ")
            traceback.print_exc(file=sys.stdout)

    while True:
        if recur:
            ic.shutdown()
            break

        option = input("Type in an action or h for help:\n")
        if option == 'h':
            print("m: take measurement\n" +
                  "r: show raw image status\n" +
                  "cr: change raw image status\n" +
                  "cf: change image format\n" +
                  "cd: change directory\n" +
                  "cn: change name\n" +
                  "cm: change number of measurement\n" +
                  "e: end the program")
        elif option == 'm':
            ic.capture()
        elif option == 'r':
            print(ic._raw)
        elif option == 'cr':
            ic._raw = input("Raw image?[y/n]\n") in ['y', 'yes']
        elif option == 'cf':
            ic.change_format(input("Input a picture format:\n"))
        elif option == 'cd':
            ic.change_dir(input("Input a directory:\n"))
        elif option == 'cn':
            ic.change_ns(input("Input a name pattern:\n"))
        elif option == 'cm':
            ic.change_num(int(input("Number of image samples for one measurement?\n")))
        elif option == 'e':
            ic.shutdown()
            break


def camera_debug():
    DEBUG_DIR = 'debug/'
    filename = DEBUG_DIR + 'shutter_{0}_iso_{1}_bright_{2}.jpeg'
    analog = 'analog_gain: '
    awb_gain = 'awb_gain: '
    expo_modes = ['night', 'night_preview', 'very_long']
    #flash_modes = ['off', 'redeye']
    flash_modes = ['off']
    shutter = [600000, 800000, 1000000]
    with picamera.PiCamera(resolution=(640, 480)) as cam:
        time.sleep(1)
        print(analog + str(cam.analog_gain))
        print(awb_gain + str(cam.awb_gains))
        cam.framerate = Fraction(1, 6)
        for ss in shutter:
            #for br in brightness:
            time.sleep(0.1)
            cam.shutter_speed = ss
            cam.brightness = 50
            cam.capture(filename.format(ss), bayer=False)


if __name__ == "__main__":
    recur = True
    while recur:
        main()


