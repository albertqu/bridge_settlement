import picamera
from fractions import Fraction

cam = picamera.PiCamera(resolution=(640, 480))
cam.exposure_mode = 'off'
DEBUG_DIR = 'debug/'
filename = DEBUG_DIR + 'shutter_{0}_iso_{1}_expo_{2}.png'
cam.capture("base.png")
cam.framerate = Fraction(1, 6)
while True:
    try:
        exp = input("Run experiment?[y/n] ")
        if exp == 'y':
            ss = input("shutter speed:  ")
            iso = input("iso:  ")
            options = {'a': 'off', 'b': 'auto', 'c': 'night', 'd': 'night_preview', 'e': 'sports'}
            expo = input("exposure mode: " + str(options) + "   ")
            #bright = input("brightness:  ")
            ss, iso, opt = int(ss), int(iso), options[expo]
            cam.shutter_speed = ss
            cam.iso = iso
            cam.exposure_mode = options[expo]
            #cam.brightness = bright
            cam.capture(filename.format(ss, iso, opt))
        else:
            cam.close()
            break
    except ValueError:
        pass



