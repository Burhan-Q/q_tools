'''
Author: Burhan Qaddoumi
Source: github.com/Burhan-Q

Requires: PyYaml, joblib
'''
import random
from pathlib import Path
from dataclasses import dataclass

import yaml
from joblib import Parallel, delayed

NP = 4
FORMATS = 'YOLO', 'VOC', 'COCO'
IMAGES = '.png', '.jpeg', '.jpg', '.bmp', '.tif', '.tiff'

@dataclass
class Formats:
    yolo = '.txt'
    voc = '.xml'
    coco = '.json'

def file_ext(ext:str):
    """Ensures text string for file extensions start with `.` character."""
    return str(ext) if str(ext).startswith('.') else '.' + str(ext)

def im2lbl(im:Path, idir:str='images', ldir:str='labels', l_ext:str='.txt'):
    """Uses heuristic of swapping image directory `idir` (default='images') name with label directory `ldir` (default='labels') and label file extension `l_ext` (default='.txt') to specify a label file for a particular image file."""
    return Path(str(im).replace(f'{idir}', f'{ldir}')).with_suffix(f'{l_ext}')

def look4lbl_w_img(img:Path, l_ext:str='.txt'):
    """Takes an image ``pathlib.Path`` object and will check for any label files."""
    labels = list(Path(img).parent.glob(f"*{l_ext}"))
    return labels if any(labels) else []

def get_data(file:Path):
        """Loads annoation file, splits new lines and splits again on space-character, casts all data as ``float`` during split."""
        return [[float(v) for v in l.split(" ")] for l in file.read_text("utf-8").split("\n") if l != ""]

def im_lbl_pair(ims:list[Path],
                lbls:list[Path],
                ipath:str='images',
                lpath:str='labels',
                lext:str='.txt',
                msgs:bool=True,
                ) -> list[tuple[Path,Path|None]]: # ~35s matching 117266 images + labels
    """
    Usage
    ---
    Matches image and label files from the provided lists of `pathlib.Path` objects.
    
    Parameters
    ---
    ims   : ``list[Path]``
        Sequence of `pathlib.Path` objects for image files.
    lbls  : ``list[Path]``
        Sequence of `pathlib.Path` objects for label files, label files must all use same extension as provided in `lext` parameter.
    ipath : ``str``
        Common directory name where all image-files will be found, default is `'images'` swapped as heuristic to match with label file.
    lpath : ``str``
        Common directory name where all label-files will be found, default is `'labels'` swapped as heuristic to match with image file.
    lext  : ``str``
        File extension used for all label files found in `lbls`, default is `'.txt'`
    
    Returns
    ---
    Sequence of matched image and label files as a ``list`` of ``tuple``. Each ``tuple`` contains ``pathlib.Path`` object for the image file (@index 1) and either ``pathlib.Path`` object for matching label file or ``None`` if no match was found (@index 2).
    """
    ims, lbls = sorted(ims), sorted(lbls)
    _ims = ims.copy()
    paired, unpaired = list(), list()
    for _ in range(len(ims)):
        i = _ims.pop(0)
        l = im2lbl(i,ipath,lpath,lext)
        _ = paired.append((i,l)) if l.exists() and l.stem == i.stem else None
        _ = unpaired.append(i) if not (l.exists() and l.stem == i.stem) else None
        
        # Reduce list of labels if found
        _ = lbls.pop(lbls.index(l)) if l in lbls else None
    unpaired = unpaired + (_ims if any(_ims) else [])
    
    # Check for unpaired matches (slower)
    if any(unpaired):
        if msgs:
            print(f"now searching for {len(unpaired)} unpaired labels")
        for u in unpaired: # NOTE this doesn't seem to be adding anything
            _ = [paired.append((u,ll if u.stem == ll.stem else None)) for ll in lbls]
    if msgs:
        print(f"found {len(paired)} img-label pairs")
    
    return list(set(paired)) if len(paired) > 1 else paired.pop()

class ImgData:
    """
    Usage
    ---
    Match image files with their respective label files. Requires label files to use same filename as image filenames. Will operate fastest when files use form `path/to/images/*.img` and `path/to/labels/*.lbl`. Utilizes multiprocessing to expedite file matching.

    Parameters
    ---
    path      : ``str`` | ``pathlib.Path``
        Top-level directory where image and label files are located
    data_yaml : ``str`` | ``pathlib.Path`` | ``None``
        For YOLO datasets, optionally include the `data.yaml`, default is ``None``
    data_form : ``str``
        One of `YOLO`, `VOC`, or `COCO` to denote which label format is used, default is `'YOLO'`.
    img_ext   : ``tuple[str]``
        A ``tuple`` containing ``str`` of all image file formats to retrieve, default is `('.png',)`
    img_dir   : ``str``
        The name of the directory where all image files should be expected to use, default is `'images'`
    lbls_dir  : ``str``
        The name of the directory where all label files should be expected to use, default is `'labels'`
    n_proc    : ``int``
        Number of jobs/processes to use when matching label and image files.

    Example Directory/File Structure
    ---
    ```
    . # example 1           . # example 2
    ├───images              ├───test2017   
    │   ├───test2017        │   ├───images           
    │   ├───train2017       │   └───labels           
    │   └───val2017         ├───train2017       
    └───labels              │   ├───images   
        ├───train2017       │   └───labels           
        ├───val2017         └───val2017       
        └───val2_test           ├───images 
                                └───labels          
    ```

    Attributes
    ---
    dataset     : ``generator``
        Generator that returns a ``tuple`` with image file (@index 1) and label file (@index 2). Both will be ``pathlib.Path`` objects, unless no label file is found, then will be (``Path``, ``None``).
    error       : ``bool``
        If any errors were encountered during process, this will be ``True`` otherwise expected to be ``False``
    format      : ``str``
        Same as the parameter `data_form` using a different alias.
    frmt_ext    : ``str``
        The file extension for label files matching the `format` attribute.
    idx_dataset : ``dict``
        All ``tuple`` from `dataset` with keys using their original sorted indices the recursive file search.
    total       : ``int``
        Total number of image-label pairs in the `dataset`.

    Methods
    ---
    build_ds( )    : populates the `dataset` attribute and will reindex all entries for `idx_dataset`, runs after `init`.
    idx_shuffle( ) : in place shuffling of `idx_dataset` attribute, resets when `build_ds()` is used, runs after `init`.
    tolist( )      : return a ``list`` object for remaining items in `dataset` attribute.

    """
    def __init__(self, path:str|Path,
                 data_yaml:str|Path=None,
                 data_form:str='YOLO',
                 img_ext:tuple[str]=('.png',),
                 img_dir:str='images',
                 lbls_dir:str='labels',
                 n_proc:int=NP,
                 ) -> None:
        self.path = Path(str(path))
        self.NP = int(round(n_proc))
        self.data_yaml = Path(str(data_yaml))
        self.img_dir = img_dir
        self.lbl_dir = lbls_dir
        self.format = str(data_form).upper()
        self.img_ext = tuple(file_ext(e) for e in img_ext) if isinstance(img_ext, (list,tuple,set)) else tuple(file_ext(img_ext),)
        self.error = self._checks()
        self._find_files()
        self.build_ds()
    
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
        self._all_imgs = []
        _formats_known = self.format in FORMATS
        _ext_known = all(e.lower() in IMAGES for e in self.img_ext)
        self.frmt_ext = getattr(Formats, self.format.lower()) if _formats_known else '.txt'
        return not( _path_found or _data_yaml_found and _formats_known and _ext_known )
    
    def _len(self):
        if not self.error and hasattr(self,'dataset'):
            # self._n = self._n if self._n != 0 else len(self.tolist())
            self.dataset = list(self.dataset)
            self._n = len(self.dataset)
            self.dataset = (p for p in self.dataset)
        return self._n
        
    def build_ds(self):
        """Matches all image files with their corresponding label file, and populates `.dataset` attribute as a ``generator`` object."""
        if not (self.error and any(self._all_imgs)):
            self._find_files()
        if not self.error and any(self._all_imgs):
            self.total = self._n = len(self._all_imgs)
            self._ds = Parallel(n_jobs=self.NP)(delayed(im_lbl_pair)([i], [Path('')], self.img_dir, self.lbl_dir, lext=self.frmt_ext, msgs=False) for i in self._all_imgs)
            # self._ds = im_lbl_pair(self._all_imgs, self._all_lbls, lext=self.frmt_ext) # ~30s w/o Parallel for 100k images
            self.dataset = (p for p in self._ds)
            self._idx_files()
    
    def _find_files(self):
        if not self.error and not self.data_yaml.exists():
            # Find all unique files
            self._all_imgs = sorted(set(i for g in [self.path.rglob(f'*{e.lower()}') for e in self.img_ext] for i in g))
            # self._all_lbls = sorted(set(l for l in self.path.rglob(f'*{self.frmt_ext}')))
            # if not (any(self._all_imgs) and any(self._all_lbls)):
            if not any(self._all_imgs):
                print(f"No images with extension(s) {tuple(e for e in self.img_ext)} found in {str(self.path)}.")
                self.error = True
        elif not self.error and self.data_yaml.exists():
            ... # TODO
    
    def _idx_files(self):
        """Creates dataset dictionary using indices as keys, populated using the full dataset."""
        if not self.error and hasattr(self, 'dataset'):
            self.idx_dataset = {ix:d for ix,d in enumerate(self._ds)}
            # self.dataset = (v for v in self.idx_dataset.values())

    def idx_shuffle(self):
        """Randomly shuffle order of indexed dataset, creates index if one doesn't exist yet."""
        if not self.error:
            _ = None if hasattr(self, 'idx_dataset') else self._idx_files()
            idx_keys = list(self.idx_dataset.keys())
            random.shuffle(idx_keys)
            self.idx_dataset = {n:self.idx_dataset.get(k) for n,k in enumerate(idx_keys)}
    
    def tolist(self):
        """Creates list from remaining data in the `.dataset` generator"""
        return self._ds[(self.total - self._n):] if not self.error and self._len() > 0 else []
    
    def __next__(self):
        self._n -= 1 if not self.error and self._n > 0 and hasattr(self,'dataset') else 0
        return next(self.dataset) if not self.error and self._len() > 0 else (None, None)
    
    def __iter__(self):
        return iter(self.dataset) if not self.error and self._len() > 0 else (None, None)
    
    def __len__(self):
        return self._len() if not self.error and self._len() > 0 else 0
    
    def __getitem__(self, key:int): # TODO decide if indexing should use offset
        key = key if 0 < key else self.total + key
        # offset = self.total - self._n
        # key += offset
        # return self.idx_dataset.get(int(key)) if not self.error and hasattr(self, 'idx_dataset') else (None, None)
        return self.idx_dataset.get(int(key)) if not self.error and self._len() > 0 else (None, None)
