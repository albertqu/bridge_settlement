import picamera
import os

class ImgCollector:

    def __init__(self, dir='', ns='img', form='png', raw=False, num=1):
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
        self.counter = 1
        self.init_cam()

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
        self.cam = picamera.PiCamera()
        self.cam.led = False
        self.cam.iso = 100
        if self._num == 1:
            self.capture = self.uni_capture
        else:
            self.capture = self.multi_capture

    def shutdown(self):
        self.cam.close()

    def uni_capture(self):
        self.cam.capture(self.name_scheme.format(self.counter), bayer=self._raw)
        self.counter += 1

    def multi_capture(self):
        file_list = [self.name_scheme.format("%d_%d" % (self.counter, i)) for i in range(1, self._num+1)]
        self.cam.capture_sequence(file_list, bayer=self._raw)
        self.counter += 1


def main():
    directory = input("Input a directory:\n")
    name_pattern = input("Input a name pattern:\n")
    pic_format = input("Input a picture format:\n")
    raw_image = input("Raw image?[y/n]\n") in ['y', 'yes']
    num_meas = int(input("Number of image samples for one measurement?\n"))

    while True:
        try:
            ic = ImgCollector(dir=directory, ns=name_pattern, form=pic_format, raw=raw_image, num=num_meas)
            break
        except:
            directory = input("Ill-formated directory, type in another one: ")

    while True:
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
            exit(0)

if __name__ == "__main__":
    main()



