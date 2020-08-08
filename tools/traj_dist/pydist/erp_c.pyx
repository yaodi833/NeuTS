from libc.math cimport fmin,fabs
from libc.math cimport sqrt

cimport numpy as np
import numpy as np
from numpy.math cimport INFINITY

def c_eucl_dist_2d(double x1,double y1,double x2,double y2):
    cdef double d,dx,dy
    dx=(x1-x2)
    dy=(y1-y2)
    d=sqrt(dx*dx+dy*dy)
    return d

def c_eucl_dist_1d(double x1,double x2):
    cdef double d,dx
    dx=(x1-x2)
    d=sqrt(dx*dx)
    return d

def c_eucl_dist_3d(double x1,double y1, double z1, double x2,double y2,double z2):
    cdef double d,dx,dy,dz
    dx=(x1-x2)
    dy=(y1-y2)
    dz=(z1-z2)
    d=sqrt(dx*dx+dy*dy+dz*dz)
    return d

def c_eucl_dist(np.ndarray[np.float64_t,ndim=1] p1,np.ndarray[np
.float64_t,ndim=1] p2):
    if p1.shape[0] == 1:
        return c_eucl_dist_1d(p1[0],p2[0])
    elif p1.shape[0] == 2:
        return c_eucl_dist_2d(p1[0],p1[1],p2[0],p2[1])
    elif p1.shape[0] == 3:
        return c_eucl_dist_3d(p1[0],p1[1],p1[2],p2[0],p2[1],p2[2])

###############
#### ERP ######
###############

def e_erp(np.ndarray[np.float64_t,ndim=2] t0, np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=1] g):
    """
    Usage
    -----
    The Edit distance with Real Penalty between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array

    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    """
    cdef int n0,n1,i,j
    cdef np.ndarray[np.float64_t,ndim=2] C
    cdef double dtw,edgei,edgej


    n0 = len(t0)+1
    n1 = len(t1)+1
    C=np.zeros((n0,n1))
    edgei=0
    for i from 1 <= i < n0:
        edgei += fabs(c_eucl_dist(t0[i-1,:],g))
    C[1:,0]=edgei

    edgej=0
    for j from 1 <= j < n1:
        edgej += fabs(c_eucl_dist(t1[j-1,:],g))
    C[0,1:]=edgej

    for i from 1 <= i < n0:
        for j from 1 <= j < n1:
            derp0 = C[i-1,j] + c_eucl_dist(t0[i-1,:],g)
            derp1 = C[i,j-1] + c_eucl_dist(g,t1[j-1,:])
            derp01 = C[i-1,j-1] + c_eucl_dist(t0[i-1,:],t1[j-1,:])
            C[i,j] = fmin(derp0,fmin(derp1,derp01))
    erp = C[n0-1,n1-1]
    return erp


