import numpy as np
from points import Point, Points

# FRAME_RATE = 30
# TIME_DELTA = float(1/FRAME_RATE)


def regress(pts:Points, degree=2):
    """
    This function assumes pts is a class Points containing points with a dimension of 2 and a time t
    """
    """
    Need the coefficients from 0 to t-1 and from 1 to t.
    This will produce two polynomials fitted from the positions which will be used to 
    get a prediction

    The coefs will be stored as such: {"x":(poly1, poly2), "y":(poly1, poly2) ... "n":(poly1, poly2)}
    """
    if len(pts.keys()) != 2:
        raise ValueError
    coefs = dict()
    k1 = pts.keys()[0]
    k2 = pts.keys()[1]
    time = pts._time
    t1 = time[0:len(time)-1]
    t2 = time[1:len(time)]
    coefs[k1] = (
            np.polyfit(
                t1,
                pts[k1][0:len(pts)-1],
                degree
                ),
            np.polyfit(
                t2,
                pts[k1][1:len(pts)],
                degree
                )
            )
    coefs[k2] = (
            np.polyfit(
                t1,
                pts[k2][0:len(pts)-1],
                degree
                ),
            np.polyfit(
                t2,
                pts[k2][1:len(pts)],
                degree
                )
            )

    model_time = np.array(time + [time[-1]+1])
    model_k1 = (
            np.polyval(coefs[k1][0], model_time),
            np.polyval(coefs[k1][1], model_time)
            )
    model_k2 = (
            np.polyval(coefs[k2][0], model_time),
            np.polyval(coefs[k2][1], model_time)
            )
    print(model_k1)
    print(model_k2)
    print(coefs[k1])
    print(coefs[k2])
    
    return {'mk1': model_k1, 'mk2': model_k2, 'k1':k1, 'k2':k2, 'len': len(time), 'time': time}

def estimate(model):
    """
    The regression model retrned from regress contains the two models for
    the first two axis lines supplied in Points.

    The model for point estimation is presented as such:
    E = (2 * xfinal_2 - xfinal_1, 2 * yfinal_2 - yfinal_1)
    where E is a point (or vector) in space and xfinal, yfinal
    are the end points from the regression models with
    their respective model numbers
    """
    next_time = model['time'][-1] + 1


    y_Quad_2 = model['mk2'][1][-1]
    y_Quad_1 = model['mk2'][0][-1]
    x_Quad_2 = model['mk1'][1][-1]
    x_Quad_1 = model['mk1'][0][-1]
    prediction = Point(
            {
            model['k1']:2*x_Quad_2 - x_Quad_1,
            model['k2']:2*y_Quad_2 - y_Quad_1
            },
        next_time
        )

    return prediction

