from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2 as cv

@dataclass
class CropCoords:
    """Dataclass with `coords()` method to return ``tuple`` of crop coordinates and attributes `xmin`, `ymin`, `xmax`, `ymax`, and `crop` (cropped image)."""
    xmin:int
    ymin:int
    xmax:int
    ymax:int
    crop:np.ndarray

    def __iter__(self):
        return iter(vars(self).values())
    
    def coords(self) -> tuple:
        """Returns ``tuple`` of cropped location from source image in order (`xmin`, `ymin`, `xmax`, `ymax`)."""
        return tuple(v for k,v in vars(self).items() if k not in ['crop'])

def load_img(im:str|Path|np.ndarray, *args, **kwargs) -> np.ndarray:
    """Function to handle multiple types of image input will always be `np.ndarray`."""
    if isinstance(im, (str,Path)):
        im_file = Path(im)
        assert im_file.exists(), f"Unable to locate image file at {im_file.as_posix()}"
        img = cv.imread(str(im_file), *args, **kwargs)
    elif isinstance(im, np.ndarray):
        img = np.copy(im)
    return img

def im_info(im:str|Path|np.ndarray, *args, **kwargs):
    """Outputs image height, width, channel count (if > 1, otherwise skipped) and `dtype` as ``tuple`` with length 4 if multi-channel or length 3 if single channel."""
    img = load_img(im)
    return tuple(d for d in [*img.shape, img.dtype])

def calc_img_padding(shape:tuple|np.ndarray|list, divisor:int) -> tuple[int,int,int,int]:
    """Calculates image padding based on equal number of splits."""
    assert all([isinstance(i,int) for i in shape]), f""
    h, w = shape
    top = bot = left = right = 0
    pad_H = h % divisor != 0
    pad_W = w % divisor != 0
    same = h == w
    
    if pad_H or pad_W:
        
        if pad_H:
            nearest_mul = (h // divisor) + 1
            nearest_int = divisor * nearest_mul
            isodd = (nearest_int - h) % 2 != 0
            
            if not isodd:
                top = bot = int((nearest_int - h) // 2)
                
            elif isodd:
                top = int(((nearest_int - h) - 1) // 2)
                bot = int((nearest_int -h) - top)
        
        if pad_W and same:
            left, right = top, bot
        
        elif pad_W and not same:
            nearest_mul = (w // divisor) + 1
            nearest_int = divisor * nearest_mul
            isodd = (nearest_int - w) % 2 != 0
            
            if not isodd:
                left = right = int((nearest_int - w) // 2)
                
            elif isodd:
                left = int(((nearest_int - w) - 1) // 2)
                right = int((nearest_int - w) - left)

    elif not pad_H and not pad_W:
        top = bot = left = right = 0
        
    return top, bot, left, right

def pad_image(img_in:str|Path|np.ndarray,
              padT:int,
              padB:int,
              padL:int,
              padR:int,
              padtype:int=cv.BORDER_CONSTANT,
              padval:int|float|tuple=0) -> np.ndarray:
    """Loads image file or passes `np.ndarray` and passes arguments to `cv2.copyMakeBorder`."""
    img = load_img(img_in)
    return cv.copyMakeBorder(np.copy(img), padT, padB, padL, padR, padtype, value=padval)

def equal_crops(img_in:str|Path|np.ndarray,
                nsplit:int,
                vert_only:bool = True,
                horz_only:bool = True,
                pad_color:int|tuple = 0) -> np.ndarray:
    """
    Usage
    ---
    Creates image crops all equally sized, includes padding as required, slicing image both horizontally and vertically by default.

    Arguments
    ---
    img_in : input image as ``str``, ``pathlib.Path``, or ``np.ndarray``
    nsplit : number as ``int`` of horizontal and/or vertical crops; EX: `nsplit=4` -> 16 crops
    vert_only : when ``True`` and `horz_only` is ``False``, outputs `nsplit` number of crops all full image width. Default is ``True``
    horz_only : when ``True`` and `vert_only` is ``False``, outputs `nsplit` number of crops all full image height. Default is ``True``
    pad_color : color to use for pixel padding, when ``int`` value repeated for all channels, BGR color when ``tuple``. Default is 0 (black border pixels)
    """
    horz_and_vert = vert_only and horz_only
    assert horz_and_vert or vert_only or horz_only, f"Must provide at least one ``True`` value for `vert_only` or `horz_only` arguments is required."
    nsplit = abs(int(round(nsplit))) # ensure no negative values
    
    img = load_img(img_in)
    
    h, w = img.shape[:2]
    c = img.shape[-1] if img.ndim == 3 else 1
    in_dtype = img.dtype
    
    assert nsplit != np.nan and 1 < nsplit < min(h,w), f"Number of splits must be positive integer less than the smallest image dimension."
    
    padding = calc_img_padding((h,w), nsplit)
    pad_color = pad_color if isinstance(pad_color, (tuple, list)) else (pad_color,) * c
    img2split = pad_image(np.copy(img), *padding, padval=pad_color)
    h, w = img2split.shape[:2]
    
    split_idx = np.array([[i,j] for i in range(nsplit) for j in range(nsplit)])
    
    if not horz_and_vert and (vert_only or horz_only):
        total_out = nsplit
        split_idx = split_idx[::nsplit] if horz_only else split_idx[:nsplit:]
        h_split = round(h / nsplit) if horz_only else h # full width, y-axis slicing
        v_split = round(w/ nsplit) if vert_only else w # full width, x-axis slicing
    elif horz_and_vert:
        total_out = nsplit ** 2
        h_split = round(h / nsplit) # y-axis slicing
        v_split = round(w / nsplit) # x-axis slicing
        
    crops = np.zeros([total_out, h_split, v_split, c]).squeeze()
    
    for ic, (y, x) in enumerate(split_idx):
        y_slice, x_slice = y * h_split, x * v_split
        x1, y1, x2, y2 = x_slice, y_slice, x_slice + v_split, y_slice + h_split
        crops[ic] = img2split[y1:y2, x1:x2]
    
    return crops.astype(in_dtype)

def eq_crops_w_coords(img_in:str|Path|np.ndarray,
                        nsplit:int,
                        vert_only:bool = True,
                        horz_only:bool = True,
                        pad_color:int|tuple = 0) -> dict[CropCoords]:
    """
    Usage
    ---
    Creates image crops all equally sized, includes padding as required, slicing image both horizontally and vertically by default. Output includes coordinates from source (or padded source) image for each crop.

    Arguments
    ---
    img_in : input image as ``str``, ``pathlib.Path``, or ``np.ndarray``
    nsplit : number as ``int`` of horizontal and/or vertical crops; EX: `nsplit=4` -> 16 crops
    vert_only : when ``True`` and `horz_only` is ``False``, outputs `nsplit` number of crops all full image width. Default is ``True``
    horz_only : when ``True`` and `vert_only` is ``False``, outputs `nsplit` number of crops all full image height. Default is ``True``
    pad_color : color to use for pixel padding, when ``int`` value repeated for all channels, BGR color when ``tuple``. Default is 0 (black border pixels)

    Returns
    ---
    Output is a ``dict`` with keys matching indices of crops, ordered top-left [0] to bottom-right [`nsplit` ^ 2 - 1], with values as ``CropCoords`` (``dataclass``) objects.
    """
    horz_and_vert = vert_only and horz_only
    assert horz_and_vert or vert_only or horz_only, f"Must provide at least one ``True`` value for `vert_only` or `horz_only` arguments is required."
    nsplit = abs(int(round(nsplit))) # ensure no negative values
    
    img = load_img(img_in)
    i_info = im_info(img)
    h, w, c, in_dtype = i_info if len(i_info) == 4 else (*i_info[:2], 1, i_info[-1])
    
    assert nsplit != np.nan and 1 < nsplit < min(h,w), f"Number of splits must be positive integer less than the smallest image dimension."
    
    padding = calc_img_padding((h,w), nsplit)
    pad_color = pad_color if isinstance(pad_color, (tuple, list)) else (pad_color,) * c
    img2split = pad_image(np.copy(img), *padding, padval=pad_color)
    h, w = img2split.shape[:2]
    
    split_idx = np.array([[i,j] for i in range(nsplit) for j in range(nsplit)])
    
    if not horz_and_vert and (vert_only or horz_only):
        total_out = nsplit
        split_idx = split_idx[::nsplit] if horz_only else split_idx[:nsplit:]
        h_split = round(h / nsplit) if horz_only else h # full width, split on y-axis
        v_split = round(w/ nsplit) if vert_only else w # full height, split on x-axis
    elif horz_and_vert:
        total_out = nsplit ** 2
        h_split = round(h / nsplit) # y-axis slicing
        v_split = round(w / nsplit) # x-axis slicing
        
    out = {i:None for i in range(total_out)}
    
    for ic, (y, x) in enumerate(split_idx):
        y_slice, x_slice = y * h_split, x * v_split
        x1, y1, x2, y2 = x_slice, y_slice, x_slice + v_split, y_slice + h_split
        out[ic] = CropCoords(x1, y1, x2, y2, img2split[y1:y2, x1:x2].astype(in_dtype))
    
    return out

def clip2region(xmin:int, ymin:int, xmax:int, ymax:int, boxes:np.ndarray):
    # TODO write docstring
    b1, b2, b3, b4 = np.split(boxes, 4, 1)

    keep_b1, keep_b3 = b1.clip(xmin,xmax), b3.clip(xmin,xmax)
    keep_b2, keep_b4 = b2.clip(ymin,ymax), b4.clip(ymin,ymax)
    out = np.hstack([keep_b1, keep_b2, keep_b3, keep_b4])
    
    return out[np.where((np.diff(out[...,::2]) != 0) & (np.diff(out[...,1::2]) != 0))[0]]

def eq_crops_w_boxes(img_in:str|Path|np.ndarray,
                    nsplit:int,
                    boxes:np.ndarray=None, # only uses xyxy format
                    vert_slice:bool = True,
                    horz_slice:bool = True,
                    pad_color:int|tuple = 0):
    """
    Usage
    ---
    Slice or crop an image into multiple sub-images and retains bounding box information for annotations contained within each sliced region. May be useful for dataset training with high-resolution images that need to be sliced into smaller sections.
    
    ⚠ CAUTION ⚠
    ---
    This function has no awareness of object extent, so an annotation may be present for an object that is not visible in a slice. This can happen when an object is extended at one end (like someone walking with an outstretched leg).

    Arguments
    ---
    img_in : input image as file path ``str`` or ``pathlib.Path`` or as ``np.ndarray``
    nsplit : ``int`` number of slices to make for image. See NOTE below
    boxes : bounding boxes as ``np.ndarray`` using `xyxy` (xy-min, xy-max) format of all object annotations for `img_in`
    vert_slice : ``bool`` to enable vertical image slicing (full height). Default is ``True``
    horz_slice : ``bool`` to enable horizontal image slicing (full width). Default is ``True``
    
        NOTE when `vert_slice == horz_slice == True` image will be cropped equally in both directions and output with consist of `nsplit^2` slices

    pad_color : color to use for image padding as ``int`` or ``tuple``. When using ``int``, values are repeated for all channels for `img_in`. Default is 0 and will pad with `(0, 0, 0)` for 3-channel images.

    Returns
    ---
    Outputs a nested ``dict`` containing ``int`` keys equal to the total number of slices/crops with a ``dict`` value. The ``dict`` value will contain a key `crop` for the cropped/sliced image ``np.ndarray`` and `boxes` for the bounding boxes as ``np.ndarray`` captured by the corresponding image slice/crop.
    """
    horz_and_vert = vert_slice and horz_slice
    assert horz_and_vert or vert_slice or horz_slice, f"Must provide at least one ``True`` value for `vert_only` or `horz_only` arguments is required."
    nsplit = abs(int(round(nsplit))) # ensure no negative values
    # TODO add check for `boxes` data and type
    
    img = load_img(img_in)
    i_info = im_info(img)
    h, w, c, in_dtype = i_info if len(i_info) == 4 else (*i_info[:2], 1, i_info[-1])
    
    assert nsplit != np.nan and 1 < nsplit < min(h,w), f"Number of splits must be positive integer less than the smallest image dimension."
    
    # Pad image for slicing as needed (determined by image dimensions)
    padding = calc_img_padding((h,w), nsplit)
    pad_color = pad_color if isinstance(pad_color, (tuple, list)) else (pad_color,) * c
    img2split = pad_image(np.copy(img), *padding, padval=pad_color)
    h, w = img2split.shape[:2]
    
    split_idx = np.array([[i,j] for i in range(nsplit) for j in range(nsplit)])
    
    # Slice image
    if not horz_and_vert and (vert_slice or horz_slice): # slice single direction
        total_out = nsplit
        split_idx = split_idx[::nsplit] if horz_slice else split_idx[:nsplit:]
        h_split = round(h / nsplit) if horz_slice else h # full width, split on y-axis 
        v_split = round(w/ nsplit) if vert_slice else w # full height, split on x-axis
    elif horz_and_vert: # slice both directions
        total_out = nsplit ** 2
        h_split = round(h / nsplit) # y-axis slicing
        v_split = round(w / nsplit) # x-axis slicing
    
    crops = {i:None for i in range(total_out)}
    
    # Slice image and boxes
    for ic, (y, x) in enumerate(split_idx):
        c = {'crop':None, 'boxes':None}
        # Slice/crop image
        y_slice, x_slice = y * h_split, x * v_split
        x1, y1, x2, y2 = x_slice, y_slice, x_slice + v_split, y_slice + h_split
        c['crop'] = img2split[y1:y2, x1:x2].astype(in_dtype)
        # Convert boxes for crop
        c_boxes = clip2region(x1, y1, x2, y2, boxes)
        c_boxes[...,::1], c_boxes[...,1::2] = c_boxes[...,::1] - x1, c_boxes[...,1::2] - y1
        c['boxes'] = c_boxes
        
        crops[ic] = c.copy()
    
    return crops

# TESTING

# from qtools.qtools.image.view import show_im
# from qtools.qtools.data.boxes import ncxywh2xyxy

# txt = r"C:\Users\User\python_proj\pytorch_and_yolo\runs\detect\predict5\labels\bus.txt"
# im = r"C:\Users\User\python_proj\pytorch_and_yolo\bus.jpg"
# img = cv.imread(im)
# h, w, = img.shape[:2]
# ncxywh = np.loadtxt(txt)

# xyxy = ncxywh2xyxy(np.loadtxt(txt)[...,1:5], h, w)
# out = eq_crops_w_boxes(img, 5, xyxy) # NOTE currently only supports xyxy format

# for c in out.values():
#     c_boxes = c.get('boxes')
#     c_crop = c.get('crop')
#     c_draw = np.copy(c_crop)
#     for b in c_boxes:
#         _ = cv.rectangle(c_draw, (b[0],b[1]), (b[2],b[3]), (0,255,0), 2, cv.LINE_AA)
#     show_im(c_draw)