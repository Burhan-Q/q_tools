'''
Author: Burhan Qaddoumi
Source: github.com/Burhan-Q

Requires: ...
'''

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

# TODO need to get working; not currently working
## NOTE might be better to rewrite a different function
def get_nest_dict(ndict:dict, keys:tuple|list):
    """Gets nested dictionary using known keys"""
    for ik,k in enumerate(keys):
        mdict = ndict[k].copy() if isinstance(ndict[k], dict) else None

        if mdict is None:
            continue
        elif isinstance(mdict, dict) and ik+1 < len(keys) and keys[ik+1] in mdict:
            odict = get_nest_dict(ndict[k], keys[ik + 1:])
        else:
            odict = ndict[k].copy()
    return odict

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