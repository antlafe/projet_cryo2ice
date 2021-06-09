import numpy as np
from numpy import where, dstack, diff, meshgrid
"""
Give, two x,y curves this gives intersection points,
autor: Sukhbinder
5 April 2017
Based on: http://uk.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections
"""


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # a piece of a prolate cycloid, and am going to find
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)

    x2 = phi
    y2 = np.sin(phi)+2
    x, y = intersection(x1, y1, x2, y2)
    plt.plot(x1, y1, c='r')
    plt.plot(x2, y2, c='g')
    plt.plot(x, y, '*k')
    plt.show()


#
MAX_DIFF_DEGREES =  2 # degrees
CHECK_FLAG=False

def find_intersections(A, B, verbose):

    #print("A", A.shape)                                                                             
    #print("B", B.shape)                                                                             

    # min, max and all for arrays                                                                    
    amin = lambda x1, x2: where(x1<x2, x1, x2)
    amax = lambda x1, x2: where(x1>x2, x1, x2)
    aall = lambda abools: dstack(abools).all(axis=2)
    slope = lambda line: (lambda d: d[:,1]/d[:,0])(diff(line, axis=0))

    x11, x21 = meshgrid(A[:-1, 0], B[:-1, 0])
    x12, x22 = meshgrid(A[1:, 0], B[1:, 0])
    y11, y21 = meshgrid(A[:-1, 1], B[:-1, 1])
    y12, y22 = meshgrid(A[1:, 1], B[1:, 1])

    m1, m2 = meshgrid(slope(A), slope(B))
    m1inv, m2inv = 1/m1, 1/m2

    yi = (m1*(x21-x11-m2inv*y21) + y11)/(1 - m1*m2inv)
    xi = (yi - y21)*m2inv + x21

    xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12),
              amin(x21, x22) < xi, xi <= amax(x21, x22) )
    yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
              amin(y21, y22) < yi, yi <= amax(y21, y22) )

    # sara: get indices list                                                                         
    xings_inds = np.where(aall(xconds))
    #inds_x = np.where(aall(xconds))                                                              
    #inds_y = np.where(aall(yconds))                                                              

    # No xings                                                                                       
    if (len(xings_inds) == 0 or len(xings_inds[0])==0):
        if verbose>1:
            ctoh_print_info("INFO: nothing found")
        return None,None,None,None

    # check nb of xings                                                                              
    nb_xings = len(xings_inds[0])
    if nb_xings != 1:
        if verbose>1:
            ctoh_print_info("nb_xings=%d" % nb_xings)

    # select/check the correct one                                                                   
    # (because intersect also when lon cross 0 <-> 360 virtual line)                                 
    IND_LON=0
    IND_LAT=1
    i=0
    ind_inter_B = -1
    for i in range(nb_xings):
        ind_inter_A = xings_inds[1][i]
        pt1_trA = A[ind_inter_A]
        pt2_trA = A[ind_inter_A+1]
        diff_lona = pt1_trA[IND_LON] - pt2_trA[IND_LON]
        if diff_lona < MAX_DIFF_DEGREES and diff_lona > -MAX_DIFF_DEGREES:
            ind_inter_B = xings_inds[0][i]
            pt1_trB = B[ind_inter_B]
            pt2_trB = B[ind_inter_B+1]
            diff_lonb = pt1_trB[IND_LON] - pt2_trB[IND_LON]
            if diff_lonb < MAX_DIFF_DEGREES and diff_lonb > -MAX_DIFF_DEGREES:
                break
    if i == nb_xings or ind_inter_B==-1:
        if verbose>1:
            ctoh_print_info("find_intersections: no xings (MAX_DIFF_DEGREES=%d)" % MAX_DIFF_DEGREES)
        return None,None,None,None

    #y_list = yi[xings_inds]                                                                         
    if nb_xings == 1:
        resX = xi[xings_inds][0]
        resY = yi[xings_inds][0]
    else:
        resX = xi[xings_inds[0][i]][xings_inds[1][i]]
        resY = yi[xings_inds[0][i]][xings_inds[1][i]]
        #inter_ind = [xings_inds[0][i],xings_inds[1][i]]

    res = [resX, resY]

    # check point inside polygon                                                                     
    if CHECK_FLAG and not point_inside_polygon(resX, resY, [pt1_trA, pt1_trB,
                                                            pt2_trA, pt2_trB]):

            # print for debug                                                                        
            print("A1 %s" % pt1_trA)
            print("A2 %s" % pt2_trA)
            print("NOT include? %s %s" % (resX, resY))
            print("B2 %s" % pt2_trB)
            print("B1 %s" % pt1_trB)

    else:
        if verbose>1:
            print("OK lon,lat= %s %s ind A,B = %s %s" % (resX, resY, ind_inter_A, ind_inter_B))
    return (resX, resY, ind_inter_A, ind_inter_B)


# ------------------------------------------------------------                                       
#                                                                                                    
# determine if a point is inside a given polygon or not                                              
# Polygon is a list of (x,y) pairs.                                                                  
#                                                                                                    
def point_inside_polygon(x,y,poly):

    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside





