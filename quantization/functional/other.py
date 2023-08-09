

def Q_MIN(b: int, sign: int = True) -> int:
    return -(1 << (b - 1)) if sign else 0


def Q_MAX(b: int, sign: int = True) -> int:
    return ((1 << (b - 1)) - 1) if sign else ((1 << b) - 1)
