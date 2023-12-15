import cv2 as cv
from qtools import ROOT
from qtools.image.view import show_im

im2test = cv.imread((ROOT / 'tests/assets/bus.jpg').as_posix())

def test_show_im():
    # show_im(im2test)
    ...

