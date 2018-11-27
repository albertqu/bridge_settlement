from SETTINGS import *
import sys
from math import atan,sqrt,sin,cos
sys.path.insert(0,ACCELEROMETER)
from accel import extract_meas

grav = 9.81

def read_accel():
    x_vec,y_vec,z_vec = extract_meas()
    x = float(sum(x_vec))/float(len(x_vec))
    y = float(sum(y_vec))/float(len(y_vec))
    z = float(sum(z_vec))/float(len(z_vec))
    pitch = atan(x/(sqrt(z**2 + y**2)))  #angle forward from vert [x,y flat, z vert]
    #roll = atan(y/(sqrt(x**2 + z**2))) #angle side from vert [x,y flat, z vert]
    return pitch



