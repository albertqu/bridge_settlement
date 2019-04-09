import picamera
import os
from fractions import Fraction
import time


class ImgCollector:

    def __init__(self, dir='', ns='img', ext='png', raw=False, ss=3000000, iso=100, num=2, serialize=True):
        """ Constructor for ImgCollector
        NOTE: Currently ImgCollector only works in Linux/Linux-like system
        :param dir: directory for putting images, as well as the log file if serialize is True (_dir)
        :param ns: name scheme (_ns)
        :param ext: file extension, e.g. png (_ext)
        :param raw: whether using raw images or not (_raw)
        :param ss: shutter_speed for the camera (ss)
        :param iso: iso value for camera (iso)
        :param num: total number of pictures in a sequence (_num)
        :param serialize: Whether logging counter to a text file. (ser)

        Instance variable:
            logfile: path of the logfile if serialize is True
            counter: correspond to first dynamic part of image naming
            curr: second dynamic part of image naming

        """
        if dir:
            if not os.path.exists(dir):
                os.makedirs(dir)
            if dir[len(dir)-1] != '/':
                dir += '/'

        self.name_scheme = dir + ns + '_{0}.' + ext
        self._dir = dir
        self._ns = ns
        self._ext = ext
        self._raw = raw
        self._num = num
        self.ss = ss
        self.iso = iso
        self.ser = serialize
        self.cam = None
        self.curr = 0
        self.init_cam()
        if serialize:
            self.logfile = self._dir + "img_log.txt"
            if os.path.exists(self.logfile):
                rfile = open(self.logfile, "r")
                try:
                    self.counter = int(rfile.read())
                except ValueError:
                    self.counter = 1
            else:
                rfile = open(self.logfile, "w")
                self.counter = 1
            rfile.close()
        else:
            self.counter = 1

    def change_ns(self, ns):
        self._ns = ns
        self.name_scheme = self._dir + self._ns + '_{0}.' + self._ext

    def change_extension(self, ext):
        self._ext = ext
        self.name_scheme = self._dir + self._ns + '_{0}.' + self._ext

    def change_dir(self, dir):
        self._dir = dir
        self.name_scheme = self._dir + self._ns + '_{0}.' + self._ext

    def change_num(self, num):
        self._num = num

    def change_iso(self, iso):
        self.iso = iso

    def change_ss(self, ss):
        self.ss = ss

    def init_cam(self):
        """Initiates the camera according to the iso, and shutterspeed set up upon initiation"""
        self.cam = picamera.PiCamera(resolution=(2592, 1944), framerate=Fraction(1, 6), sensor_mode=3)
        time.sleep(1)
        self.cam.led = False
        self.cam.shutter_speed = self.ss
        self.cam.iso = self.iso
        time.sleep(10)
        self.cam.rotation = 180
        self.cam.exposure_mode = 'off'
        self.cam.awb_mode = 'off'
        self.cam.awb_gains = (Fraction(63, 128), Fraction(93, 64))

    def get_last_meas(self):
        if self._num == 1:
            return self.name_scheme.format(self.counter - 1)
        else:
            return self._dir + self._ns + '_%d_{0}.' % (self.counter - 1) + self._ext

    def shutdown(self):
        self.cam.framerate = Fraction(1, 1)
        time.sleep(6)
        self.cam.close()
        if self.ser:
            wfile = open(self.logfile, "w")
            wfile.write(str(self.counter))
            wfile.close()

    def capture(self):
        self.cam.capture(self.name_scheme.format("%d_%d" % (self.counter, self.curr)), bayer=self._raw)
        self.curr += 1
        if self.curr == self._num:
            self.counter += 1
            self.curr = 0


if __name__ == "__main__":
    recur = True
