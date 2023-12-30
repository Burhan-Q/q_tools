from pathlib import Path

import numpy as np

BOX_TYPES = ('cxywh', 'ncxywh', 'xyxy', 'nxyxy', 'xywh', 'nxywh')
BOX_SHAPES = [(4,), (1,4),]

# TODO add doc strings
# TODO verify / update formatting
# TODO write unit tests

def min2center(minP:float|int, size:float|int) -> float|int:
    normalized = isinstance(size, (float, np.float_)) or size.dtype in (float, np.float_)
    res = (np.floor((minP + (minP + size)) / 2)) if not normalized else ((minP + (minP + size)) / 2)
    return res.astype(size.dtype) if isinstance(size,np.ndarray) else type(minP)(res)

def center2min(centerP:float|int, size:float|int) -> float|int:
    normalized = isinstance(size, (float, np.float_)) or size.dtype in (float, np.float_)
    res = (centerP - np.floor(size / 2)) if not normalized else (centerP - (size / 2))
    return res.astype(size.dtype) if isinstance(size,np.ndarray) else type(centerP)(res)

def box_dims(box:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Calculates height and width of box input, assumes box form is xyxy or nxyxy"""
    box = box.reshape(-1,4) if not box_chck(box) else box
    box_W, box_H = np.diff(box[..., ::2]), np.diff(box[..., 1::2])
    return (box_W, box_H)

def box_parts(box:np.ndarray) -> tuple:
    if box.any() and len(box > 1):
        b1, b2, b3, b4 = np.split(box, 4, 1)
    elif box.any() and len(box) == 1:
        b1, b2, b3, b4 = box.squeeze()
    else:
        b1, b2, b3, b4 = np.array([[]]*4, dtype=box.dtype)
    return b1, b2, b3, b4

def box_area(box:np.ndarray, format:str):
    assert format.lower() in BOX_TYPES, f"Must use one of {BOX_TYPES} for bounding box format"
    box = box.reshape(-1,4) if not box_chck(box) else box
    
    #if format in ['xywh', 'nxywh', 'cxywh', 'ncxywh']:
    if 'wh' in format.lower():
        area = np.prod(box[..., 2:])
        # area = box[..., 2:3] * box[..., 3:4]

    elif format.lower() in ['xyxy', 'nxyxy']:
        w, h = box_dims(box)
        area = w * h
        
    else: # this shouldn't occur
        area = np.array([[-1]] * box.shape[0])
        
    return area

def box_chck(box:np.ndarray) -> bool:
    n = box.shape[0]
    return box.shape in BOX_SHAPES + [(n,4)]

def clip_bbox(box:np.ndarray, format:str, imH:int|float, imW:int|float):
    assert format.lower() in BOX_TYPES, f""
    normalized = format.lower().startswith('n')
    dtype_in = box.dtype
    imH, imW = (1.0, 1.0) if normalized else (imH - 1, imW - 1) # offset for 0-index
    
    if box_inbounds(box, format, imH, imW):
        return box
    
    if format.lower() in ['cxywh', 'ncxywh']:
        xmin = box[...,0:1] - (box[...,2:3] / 2 if normalized else (box[...,2:3] // 2))
        ymin = box[...,1:2] - (box[...,2:3] / 2 if normalized else (box[...,3:4] // 2))
        xmax = box[...,0:1] + (box[...,2:3] / 2 if normalized else (box[...,2:3] // 2))
        ymax = box[...,1:2] + (box[...,2:3] / 2 if normalized else (box[...,3:4] // 2))
        
    elif format.lower() in ['xywh', 'nxywh']:
        xmin, ymin = np.split(box[...,:2], 2, 1)
        xmax = xmin + box[...,2:3]
        ymax = ymin + box[...,3:4]
        
    elif format.lower() in ['xyxy', 'nxyxy']:
        xmin, ymin, xmax, ymax = np.split(box,4,1)
        
    out = np.hstack([xmin.clip(0,imW), ymin.clip(0,imH), xmax.clip(0,imW), ymax.clip(0,imH)], dtype = dtype_in)
    
    # NOTE maybe clip x/y-min to max - 1 to ensure no issues with zero width or height boxes
    
    # Reformat for output
    ## NOTE could use conversion functions instead? Concerned about reference loopback
    
    if format.lower() in ['cxywh', 'ncxywh']:
        outW, outH = box_dims(out)
        out_xc = out[...,0:1] + (outW / 2 if normalized else np.ceil(outW / 2).astype(dtype_in))
        out_yc = out[...,1:2] + (outH / 2 if normalized else np.ceil(outH / 2).astype(dtype_in))
        out = np.hstack([out_xc, out_yc, outW, outH],dtype=dtype_in)
        
    elif format.lower() in ['xywh', 'nxywh']:
        outW, outH = box_dims(out)
        out = np.hstack([out[...,:2], outW, outH], dtype=dtype_in)
    
    return out

def box_inbounds(box:np.ndarray, format:str, imH:int|float, imW:int|float, clip:bool=False) -> tuple[np.ndarray,bool]|bool:
    
    assert format.lower() in BOX_TYPES, f""
    normalized = format.lower().startswith('n')
    imH, imW = (1.0, 1.0) if normalized else (imH - 1, imW - 1)
    out = np.zeros(box.shape)
    inbounds = False
    
    if clip:
        inbounds = clip
        out = clip_bbox(box, format, imH, imW)
        
    else:
        
        if 'c' in format.lower():
            xmin = box[...,0:1] - (box[...,2:3] / 2 if normalized else (box[...,2:3] // 2))
            ymin = box[...,1:2] - (box[...,2:3] / 2 if normalized else (box[...,3:4] // 2))
            xmax = box[...,0:1] + (box[...,2:3] / 2 if normalized else (box[...,2:3] // 2))
            ymax = box[...,1:2] + (box[...,2:3] / 2 if normalized else (box[...,3:4] // 2))
            
        elif format.lower() in ['xywh', 'nxywh']:
            xmin, ymin = box[...,0:1], box[...,1:2]
            xmax = xmin + box[...,2:3]
            ymax = ymin + box[...,3:4]
        
        elif format.lower() in ['xyxy', 'nxyxy']:
            xmin, ymin, xmax, ymax = box_parts(box)
        
        inbounds = all([b.all() for b in [(xmin >= 0), (ymin >= 0), (xmax <= imW), (ymax <= imH)]])
        
    return (out, inbounds) if (out.any() and clip) else inbounds

# Conversions

## cxywh (xy-centerpoint, width, height)

def cxywh2xywh(box:np.ndarray, imH:int=None, imW:int=None, clip:bool=False) -> np.ndarray:
    if clip:
        assert imH is not None and imW is not None, f""
    *_, boxW, boxH = box_parts(box)
    xmin, ymin = center2min(box[...,0:1],boxW), center2min(box[...,1:2],boxH)
    out = np.hstack([xmin,ymin,boxW,boxH])
    return out if not clip else clip_bbox(out,'xywh', imH, imW)

def cxywh2xyxy(box:np.ndarray, imH:int=None, imW:int=None, clip:bool=False) -> np.ndarray:
    if clip:
        assert imH is not None and imW is not None, f""
    *_, boxW, boxH = box_parts(box)
    xmin, ymin = center2min(box[...,0:1],boxW), center2min(box[...,1:2],boxH)
    xmax, ymax = xmin + boxW, ymin + boxH
    out = np.hstack([xmin,ymin,xmax,ymax])
    return out if not clip else clip_bbox(out,'xyxy',imH,imW)

def cxywh2ncxywh(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    out = np.zeros(box.shape)
    out[...,::2], out[...,1::2] = box[...,::2] / imW, box[...,1::2] / imH
    return out if not clip else clip_bbox(out, 'ncxywh', 1.0, 1.0)
    
def cxywh2nxyxy(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    return cxywh2xyxy(cxywh2ncxywh(box,imH,imW,clip))

def cxywh2nxywh(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    return cxywh2xywh(cxywh2ncxywh(box,imH,imW,clip))

def ncxywh2cxywh(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    out = np.zeros(box.shape)
    box = clip_bbox(box, 'ncxywh', 1.0, 1.0) if clip else box
    out[...,::2], out[...,1::2] = box[...,::2] * imW, box[...,1::2] * imH
    return out.astype(np.int_)

def ncxywh2xyxy(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    return cxywh2xyxy(ncxywh2cxywh(box, imH, imW, clip), imH, imW, clip)

## xywh (xmin, ymin, width, height)

def xywh2cxywh(box:np.ndarray, imH:int=None, imW:int=None, clip:bool=False) -> np.ndarray:
    if clip:
        assert imH is not None and imW is not None, f""
    xmin, ymin, w, h = box_parts(box)
    xc, yc = min2center(xmin,w), min2center(ymin,h)
    out = np.hstack([xc, yc, w, h], dtype=box.dtype) # NOTE maybe remove dtype, not certain if required
    return out if not clip else clip_bbox(out, 'cxywh', imH, imW)

def xywh2xyxy(box:np.ndarray, imH:int=None, imW:int=None, clip:bool=False) -> np.ndarray:
    if clip:
        assert imH is not None and imW is not None, f""
    xmin, ymin, w, h = box_parts(box)
    xmax, ymax = xmin + w, ymin + h
    out = np.hstack([xmin,ymin,xmax,ymax], dtype=box.dtype) # NOTE dtype needed?
    return out if not clip else clip_bbox(out, 'xyxy', imH, imW)

def xywh2nxywh(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    out = np.zeros(box.shape)
    out[...,::2], out[...,1::2] = box[...,::2] / imW, box[...,1::2] / imH
    return out if not clip else clip_bbox(out, 'nxywh', 1.0, 1.0)

def xywh2ncxywh(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    return xywh2cxywh(xywh2nxywh(box,imH,imW,clip))

def xywh2nxyxy(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    return xywh2xyxy(xywh2nxywh(box,imH,imW,clip))

def nxywh2xywh(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    out = np.zeros(box.shape)
    box = clip_bbox(box, 'nxywh', 1.0, 1.0) if clip else box
    out[...,::2],out[...,1::2] = box[...,::2] * imW, box[...,1::2] * imH
    return out.astype(np.int_)

# xyxy (xmin, ymin, xmax, ymax)

def xyxy2cxywh(box:np.ndarray, imH:int=None, imW:int=None, clip:bool=False) -> np.ndarray:
    if clip:
        assert imH is not None and imW is not None, f""
    w, h = box_dims(box)
    xc = min2center(box[...,0:1],w)
    yc = min2center(box[...,1:2],h)
    out = np.hstack([xc,yc,w,h])
    return out if not clip else clip_bbox(out, 'cxywh', imH, imW)

def xyxy2xywh(box:np.ndarray, imH:int=None, imW:int=None, clip:bool=False) -> np.ndarray:
    if clip:
        assert imH is not None and imW is not None, f""
    w, h = box_dims(box)
    xmin, ymin, *_ = box_parts(box)
    out = np.hstack([xmin,ymin,w,h],dtype=box.dtype) # NOTE keep dtype?
    return out if not clip else clip_bbox(out, 'xywh', imH, imW)

def xyxy2nxyxy(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    out = np.zeros(box.shape)
    out[...,::2],out[...,1::2] = box[...,::2] / imW, box[...,1::2] / imH
    return out if not clip else clip_bbox(out, 'nxyxy', 1.0, 1.0)

def xyxy2ncxywh(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    return xyxy2cxywh(xyxy2nxyxy(box,imH,imW,clip))

def xyxy2nxywh(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    return xyxy2xywh(xyxy2nxyxy(box,imH,imW,clip))

def nxyxy2xyxy(box:np.ndarray, imH:int, imW:int, clip:bool=False) -> np.ndarray:
    out = np.zeros(box.shape)
    box = clip_bbox(box, 'nxywh', 1.0, 1.0) if clip else box
    out[...,::2], out[...,1::2] = box[...,::2] * imW, box[...,1::2] * imH
    return out.astype(np.int_)

# File read to bounding box
# NOTE could use `np.loadtxt(txt_file)` instead

def file2box(im_file:str|Path=None, txt_file:str|Path=None) -> None|np.ndarray:
    """Assumes text file located in same directory as image file, and one bounding box per line with coordinates seperated by spaces."""
    assert im_file or txt_file, f""
    txt_file = Path(txt_file) if txt_file is not None else Path(im_file).with_suffix('.txt')
    
    if not txt_file.exists():
        return None
    
    return np.array([[c for c in l.split(' ') if c != ''] for l in txt_file.read_text('utf-8').split('\n') if l != '' and not l.startswith('#')],dtype=np.float_)