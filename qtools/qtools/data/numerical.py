'''
Author: Burhan Qaddoumi
Source: github.com/Burhan-Q

Requires: ...
'''

def magnitude(num:int):
    """Calculate order of magnitude for input number. For number `0 < x < 1` returns negative magnitude, for number for `1 < x < ∞` returns positive magnitude."""
    # Zeroth order
    if 1 < abs(num) < 10:
        return 0
    n = 0
    loop = 10 <= abs(num)
    # Positive magnitude
    if loop:
        while loop:
            num /= 10
            loop = 10 <= abs(num)
            n += 1
            if not loop:
                break
    # Negative magnitude
    elif not loop and abs(num) < 1:
        loop = True
        while loop:
            num *= 10
            loop = abs(num) < 1
            n += 1
            if not loop:
                n *= -1
                break
    return n