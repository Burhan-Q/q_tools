from pathlib import Path

import cv2 as cv
import numpy as np

from qtools.image.img_data import im2lbl, get_data
from qtools.image.im_utils import pad_image, load_img

def show_im(img:str|Path|np.ndarray):
    img = img if isinstance(img, np.ndarray) else (load_img(img))
    cv.imshow('', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def show_win(img:str|Path|np.ndarray):
    img = img if isinstance(img, np.ndarray) else (load_img(img))
    cv.namedWindow('Preview', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
    cv.imshow('Preview', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def draw_mosaic9(ims:list[str|Path]):
    size = 9
    dims = [cv.imread(str(im)).shape[:2] for im in ims if Path(im).exists()]
    
    if len(dims) < len(ims):
        ipad = len(ims) - len(dims)
        _ = [dims.append(dims[0]) for _ in range(ipad)]

    if len(ims) < size:
        _ = [dims.append(dims[0]) for _ in range(size - len(ims))]
        _ = [ims.append(None) for _ in range(size - len(ims))]
    # elif len(ims) > size:
    #     # chunk data
    #     ...
    imgsz = np.array(dims).max() # may not be good for small and large images
    ims_2plot = list()
    for im in ims:
        if im is None:
            img2plot = np.zeros(imgsz, imgsz, 3)
        else:
            ## Calculate and pad image to make square
            img = cv.imread(str(im))
            im_h, im_w = img.shape[:2]
            dH, dW = imgsz - im_h, imgsz - im_w
            evenH, evenW = dH % 2 == 0, dW % 2 == 0
            padT = (dH // 2 if evenH else np.ceil(dH / 2).astype(int))
            padB = (dH // 2 if evenH else np.floor(dH / 2).astype(int))
            padR = (dW // 2 if evenW else np.ceil(dW / 2).astype(int))
            padL = (dW // 2 if evenW else np.floor(dW / 2).astype(int))
            img2plot = pad_image(img, padT, padB, padL, padR)
        ims_2plot.append(np.copy(img2plot))

    arr = np.array([im.shape[:2] for im in ims_2plot]).reshape(3, -1, 2)
    j = arr[...,None, 1::2].max(1).squeeze() # widths
    i = arr[...,None, ::2].max(1).squeeze() # heights
    lims = list(zip(i,j))
    H = arr[:,:,0:1].max() * 3
    W = arr[:,:,1:2].max() * 3

    canvas = np.zeros((H, W, 3), np.uint8) # all color images

    # n = [s for si,s in enumerate(_sizes) if min(len(dims), len(ims)) % s == 0]
    # N = n.pop(0) if any(n) else ...
    for n,im in enumerate(ims_2plot):
        m = min(0 + n, n % 3)
        y, x = lims[m]
        offsets = np.array(lims[:m])
        offset_y = offsets[...,0].sum() if offsets.any() else 0
        # offset_x = offsets[...,1].sum() * (m - n % 3) if offsets.any() else 0
        offset_x = imgsz * (n - m) // 3
        # img = cv.imread(str(im)) if im is not None else np.zeros((*dims[n], 3), np.uint8)
        img = np.copy(im)
        h, w = img.shape[:2]
        y, x = min(h, y) if h < y else h, min(w, x) if w < x else x
        # (offset_y, offset_y+y), (offset_x, offset_x+x)
        canvas[offset_y:offset_y+y, offset_x:offset_x+x, :] = np.copy(img)
    # show_im(canvas)
    return canvas
    

def draw_obb_img(im:Path|str|np.ndarray, obb_lbl:Path|str, keep_in_bounds:bool=True) -> np.ndarray:
    """
    Generates annotated image for YOLO-formatted oriented bounding box (OBB) data, given an image and OBB-text file.
    """
    img = cv.imread(str(im)) if not isinstance(im, np.ndarray) else np.copy(im)
    ih, iw = img.shape[:2]
    
    obb_lbl = Path(obb_lbl) if obb_lbl is not None and Path(obb_lbl).exists() else im2lbl(im)
    
    lines = get_data(Path(obb_lbl))
    unique_labels = sorted(set([l[0] for l in lines]))
    N = len(lines)
    
    # Randomize colors based on label-index
    colors = {
            k:(
                min((max(k % 7, k % 2) * 80.0), 255.0),
                min(min((k * 2) % 3, (k * 5) % 2) * 47, 255.0),
                min((k % 3 * k % 2) * 130, 255.0),
                ) for k in unique_labels
        }
    
    draw = np.copy(img)
    
    for li, line in enumerate(lines):
        label, *points = line
        arr = np.array(points).reshape(-1, 2)
        arr[..., 0:1] *= iw
        arr[..., 1:2] *= ih

        rot_rect = cv.RotatedRect(*cv.minAreaRect(arr.astype(int)))  # ((xc, yc), (w, h), angle)
        obb_points = rot_rect.points()
        obb_points[..., 0:1] /= iw
        obb_points[..., 1:2] /= ih
        
        # Use normal bounding box when any point is out of bounds
        if not ((0 <= obb_points) & (obb_points <= 1)).all() and keep_in_bounds:
            bbox_x1, bbox_y1, bbox_w, bbox_h = rot_rect.boundingRect() # x1y1wh
            bbox_x2, bbox_y2 = bbox_x1 + bbox_w, bbox_y1 + bbox_h
            
            obb_points = np.array(
                (bbox_x1, bbox_y1, bbox_x2, bbox_y1, bbox_x2, bbox_y2, bbox_x1, bbox_y2,)
                ).reshape(-1,2)

        else:
            # Keep OBB but clip if keeping in bounds
            obb_points = obb_points if not keep_in_bounds else obb_points.clip(0,1)
            # Convert to normalized coordinates
            obb_points[..., 0:1] *= iw
            obb_points[..., 1:2] *= ih

        draw = cv.drawContours(draw, [obb_points.astype(int)], -1, colors.get(label, (0,0,255)), 2)
        obb_points = None
    return draw

