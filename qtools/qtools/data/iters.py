'''
Author: Burhan Qaddoumi
Source: github.com/Burhan-Q

Requires: ...
'''

# Examples / Testing
d = {'a':'A', 'b':'B', 'c':{'d':'D', 'e':'E', 'f':{'g':'G','h':'H','i':'I','j':'J','k':{'l':'L','m':'M','n':'N'}}}}
d2 = {1: 'A', 'b': 'B', 'c': {2: 'C', 'd': 'D', 'e': 'E', 'f': {3: 'F', 'g': 'G', 'h': 'H', 'i': {4: 'I', 'j': 'J', 'k': 'K', 'l': 'L'}}}}

def flatten_dict(indict:dict, keys:list=None, sep:str=None):
    """Flattens multi-depth dictionary and uses separator '.' to build key-hierarchy from original dictionary (Left-to-Right key-hierarchy)."""
    keys = [] if keys is None else list(keys)
    sep = '.' if sep is None else str(sep)
    if isinstance(indict, dict):
        d2 = indict.copy()
        for k,v in indict.items():
            if isinstance(v,dict):
                _ = [d2.update({kk:vv}) for kk,vv in flatten_dict(v.copy(), keys=keys+[str(k)]).items()]
                _ = d2.pop(k)
            else:
                _ = d2.pop(k)
                new_k = sep.join(keys + [str(k)])
                d2[new_k] = v
    else:
        d2 = indict
    return d2

def get_nested_value(ndict:dict, *args):
    """
    Usage
    ---
    Retrieves the value(s) from the keys specified by any number of `args` (in order provided), will return ``None`` if any key doesn't exist in nesting (as an example, when keys are specified in the wrong order).

    Example
    ---
    ```py
    # Nested dictionary, can use any type for keys but ``str`` used here 
    d = {
        'a':'A', 'b':'B', 'c':
            {
            'd':'D', 'e':'E', 'f':
                {
                'g':'G','h':'H','i':'I','j':'J','k':
                    {
                    'l':'L','m':'M','n':'N'
                    }
                }
            }
        }
    get_nested_value(d, 'c', 'f', 'k', 'm')
    >>> 'M'
    get_nested_value(d, *['c', 'f', 'k', 'm'])
    >>> 'M'
    get_nested_value(d, 'c', 'f', 'k')
    >>> {'l': 'L', 'm': 'M', 'n': 'N'}
    ```
    """
    keys = list(args)
    _keys = list(keys).copy()
    _d = None
    for _ in keys:
        k = _keys.pop(0)
        _d = ndict.get(k) if k and _d is None else _d.copy().get(k)
        if isinstance(_d, dict) and _keys and _keys[0] in _d.keys():
            continue
    return _d

def delist_dict(in_obj:list, out:dict=None) -> dict:
    """Creates nested dictionaries if dictionaries contain list of dictionaries."""
    out = out if out is not None else dict()
    if isinstance(in_obj, list):
        _ = [out.update(delist_dict(x)) for x in in_obj]
    elif isinstance(in_obj, str):
        pass
    elif isinstance(in_obj, dict):
        for k,v in in_obj.items():
            out.update({k:delist_dict(v)} if isinstance(v, list) else {k:v})
    return out
