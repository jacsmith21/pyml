def approx(exp, act, places=6):
    if places <= 0:
        raise ValueError('place cannot be <= 0: {}'.format(places))

    return abs(exp - act) < 1 / places
