from img_rec_module.sig_proc_test_v0 import FM
import numpy as np
from random import randint
import traceback
import sys

matrix = np.random.rand(20, 30)


def segmentation_test(sample):
    ax = randint(0, 1)
    target = randint(0, 19) if ax == 1 else randint(0, 29)
    fmat = FM(matrix, ax, target)
    assert fmat.irange() == 20 if ax == 1 else fmat.irange() == 30
    cap = len(fmat)
    for i in range(sample):
        start = randint(0, cap - 1)
        end = randint(start + 1, cap)
        fmat.segmentize(start, end)
        assert len(fmat.extract_array()) == end - start, "start: {0}, end: {1}".format(start, end)
        elem = randint(0, len(fmat) - 1)
        assert fmat[elem] == matrix.item(target, elem + start) if ax == 1 \
            else fmat[elem] == matrix.item(elem + start, target)
    print("Test 1 passed!")
    return fmat.copy()


def shape_change_test(sample):
    fmat = mat
    print(len(fmat), "start: {0}, end: {1}, ax: {2}, index: {3}".format(fmat.start, fmat.end, fmat._ax, fmat._index))
    for i in range(sample):
        ax = randint(0, 1)
        target = randint(0, 19) if ax == 1 else randint(0, 29)
        try:
            fmat = fmat.copy(ax, target)
            elem = randint(0, len(fmat) - 1)
            assert fmat[elem] == matrix.item(target, elem) if ax == 1 else fmat[elem] == matrix.item(elem, target), "start: {0}, end: {1}, ax: {2}, index: {3}, changed:{4}, elem: {5}".format(fmat.start, fmat.end, ax, fmat._index, target, elem)
        except IndexError:
            print("Index Error, start: {0}, end: {1}, ax: {2}, index: {3}".format(fmat.start, fmat.end, fmat._ax, fmat._index))
            traceback.print_exc(file=sys.stderr)
            sys.exit(0)
    print("Test 2 passed!")

mat = segmentation_test(1000000)
shape_change_test(1000000)