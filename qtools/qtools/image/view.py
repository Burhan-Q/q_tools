import cv2 as cv

def show_im(img):
    cv.namedWindow('Preview', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
    cv.imshow('Preview', img)
    cv.waitKey(0)
    cv.destroyAllWindows()