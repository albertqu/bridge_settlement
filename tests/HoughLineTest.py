from cv_module.sig_proc_test_v0 import HoughLine, angle_interp, normalize_angle
import numpy as np
from random import uniform
from math import sin, cos, asin, acos, degrees


def angle_interp_test(sample):
    for t in range(4):
        angle = np.pi * t / 2
        angle_test(angle)
    for i in range(sample):
        angle = uniform(0, 2 * np.pi)
        angle_test(angle)
    print("All tests passed!")


def angle_test(angle):
    s, c = sin(angle), cos(angle)
    aint = angle_interp(s, c)
    e = abs(aint - angle)
    assert e < 10 ** (-8), "angle: {0} aint: {3} sin: {1}, cos: {2}".format(degrees(angle), s, c, degrees(aint))


data1 = [(50, 93.58985799408744), (100, 94.83752561512769), (150, 93.46476622047003), (200, 92.12757475618413), (250, 91.21899822237681), (300, 94.66610143151713), (350, 93.39700158152765), (400, 93.55400856320203), (450, 94.81488837189904)]
data2 = [(50, 220.68898677024384), (150, 207.0369850844262), (200, 219.1873113806197), (250, 218.4221525438996), (300, 215.83141259108666), (350, 216.1097540779057), (400, 214.30630223604018), (450, 211.84789180370862), (500, 211.03929058602466), (550, 208.37556376057634), (600, 208.1358916890448)]


def diagnose():
    x1 = np.array([d[1] for d in data1])
    y1 = np.array([d[0] for d in data1])
    line1 = HoughLine(x=x1, data=y1)
    x2 = np.array([d[0] for d in data2])
    y2 = np.array([d[1] for d in data2])
    line2 = HoughLine(x=x2, data=y2)
    xmean = sum(x1) / len(x1)
    ymean = sum(y2) / len(y2)
    print(line1._r, line1._t)
    print(line2._r, line2._t)
    print((xmean, ymean), HoughLine.intersect(line1, line2))


def normalize_angle_test(sample):
    for t in range(4):
        angle = np.pi * t / 2
        test_normalize(angle)
    for i in range(sample):
        angle = uniform(0, 1000000)
        test_normalize(angle)
    print('All tests passed!')


def test_normalize(angle):
    s, c = sin(angle), cos(angle)
    norm_ang = normalize_angle(angle)
    ns, nc = sin(norm_ang), cos(norm_ang)
    case1 = equal_within_error(s, ns) and equal_within_error(c, nc)
    case2 = equal_within_error(s, -ns) and equal_within_error(c, -nc)
    assert case1 or case2, 's:{0} c:{1}, ns:{2}, nc:{3}, angle:{4}, norm:{5}'.format(s, c, ns, nc, angle, norm_ang)



def equal_within_error(v1, v2):
    e = abs(v1 - v2)
    return e < 10 ** (-8)


if __name__ == '__main__':
    #angle_interp_test(100000000)
    #diagnose()
    normalize_angle_test(1000000)