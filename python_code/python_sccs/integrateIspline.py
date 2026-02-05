from .ispline import ispline
from .ispline1 import ispline1
from .ispline2 import ispline2
from .ispline3 import ispline3  # Assuming ispline3 is similar

def integrateIspline(x, knots1, m, int):
    """
    Evaluate integral of an I-spline (first, second, or third integral).
    
    Parameters:
    x : array-like
        Values to evaluate at.
    knots1 : array-like
        Knots.
    m : int
        Order.
    int : int
        Level of integration (1, 2, or 3).
    
    Returns:
    result : ndarray
        Integrated I-spline values.
    """
    if int == 1:
        return ispline1(x, knots1, m)
    elif int == 2:
        return ispline2(x, knots1, m)
    else:
        return ispline3(x, knots1, m)