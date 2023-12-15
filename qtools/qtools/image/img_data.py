import random
from pathlib import Path
from dataclasses import dataclass

import yaml

FORMATS = 'YOLO', 'VOC', 'COCO'
IMAGES = '.png', '.jpeg', '.jpg', '.bmp', '.tif', '.tiff'

@dataclass
class Formats:
    yolo = '.txt'
    voc = '.xml'
    coco = '.json'

def imgs_N_labels(imgs:list[Path], lbls:list[Path]) -> list[tuple[Path,Path|None]]:
    """Pairs matching image and label files, if no label file is found, returns image file with `None` instead."""
    pairs = list()
    longr = max(len(imgs), len(lbls))
    N = len(imgs) if longr == len(imgs) else len(lbls)
    for n in range(N):
        i, l = imgs[n], lbls[n]
        m = i.stem == l.stem
        if not m:
            p = [L for L in lbls if L.stem == i.stem]
            l = p.pop() if any(p) else None # no matching label found
        
        _ = pairs.append((i,l))
    return pairs

class ImgData:
    def __init__(self, path:str|Path, data_yaml:str|Path=None, data_form:str='YOLO', img_ext:tuple[str]=('.png',),) -> None:
        self.path = Path(str(path))
        self.data_yaml = Path(str(data_yaml))
        self.format = str(data_form).upper()
        self.img_ext = tuple(img_ext) if isinstance(img_ext, (list,tuple,set)) else tuple(str(img_ext),)
        self.error = self._checks()
        self.find_imgs()
    
    def _checks(self):
        """Post-init checks"""
        self.total = self._n = 0
        _path_found = self.path.exists()
        _data_yaml_found = self.data_yaml.exists() and self.data_yaml.suffix in ['.yml', '.yaml']
        if not _path_found and _data_yaml_found:
            self.yaml = yaml.safe_load(self.data_yaml.read_text('utf-8'))
            _p = Path(str(self.yaml.get('path')))
            _path_found = _p.exists()
            self.path = _p if _path_found else None
        _formats_known = self.format in FORMATS
        _ext_known = all(e.lower() in IMAGES for e in self.img_ext)
        self.frmt_ext = getattr(Formats, self.format.lower()) if _formats_known else '.txt'
        return not( _path_found or _data_yaml_found and _formats_known and _ext_known )
    
    def _len(self):
        if not self.error and hasattr(self,'dataset'):
            self._n = self._n if self._n != 0 else len(self.tolist())
        return self._n

    def find_imgs(self):
        if not self.error:
            self._all_imgs = sorted(i for g in [self.path.rglob(f'*{e.lower()}') for e in self.img_ext] for i in g)
            self._all_lbls = sorted(l for l in self.path.rglob(f'*{self.frmt_ext}'))
            self.total = self._n = len(self._all_imgs)
            self._ds = imgs_N_labels(self._all_imgs, self._all_lbls)
            self.dataset = (p for p in self._ds)
    
    def idx_files(self):
        """Creates dataset dictionary using indices from generator as keys."""
        if not self.error and hasattr(self, 'dataset'):
            self.idx_dataset = {ix:d for ix,d in enumerate(self._ds)}
            self.dataset = (v for v in self.idx_dataset.values())

    def idx_shuffle(self):
        """Randomly shuffle order of indexed dataset, creates index if one doesn't exist yet."""
        if not self.error:
            _ = None if hasattr(self, 'idx_dataset') else self._idx_files()
            idx_keys = list(self.idx_dataset.keys())
            random.shuffle(idx_keys)
            self.idx_dataset = {n:self.idx_dataset.get(k) for n,k in enumerate(idx_keys)}
    
    def tolist(self):
        return self._ds[(self.total - self._n):] if not self.error and hasattr(self, 'dataset') else []
    
    def __next__(self):
        self._n -= 1 if not self.error and self._n > 0 and hasattr(self,'dataset') else 0
        return next(self.dataset) if not self.error and hasattr(self,'dataset') else (None, None)
    
    # def __iter__(self):
    #     return iter(self.dataset) if not self.error and hasattr(self,'dataset') else (None, None)
    
    def __len__(self):
        return self._len() if not self.error and hasattr(self,'dataset') else 0
    
    def __getitem__(self, key:int):
        return self.idx_dataset.get(int(key)) if not self.error and hasattr(self, 'idx_dataset') else (None, None) 

home = Path.home()
# path = home / "python_proj/datasets/VisDrone"
# path = Path(r"Q:\datasets\VisDrone\HUB_VisDrone")
path = home / "python_proj/datasets/VisDrone/VisDrone2019-DET-train"

import time

t0 = time.time()
idata = ImgData(path,img_ext=('.jpg','.jpeg'))
time.time() - t0
# NOTE fewer directories will help run significantly faster
# found ~1000x difference when using '/datasets/VisDrone' vs '/datasets/VisDrone/VisDrone2019-DET-train'
# 

