from libc.math cimport fmin
from libc.math cimport fmax
from libc.math cimport sqrt

cimport numpy as np
import numpy as np

from cpython cimport bool

def c_eucl_dist_2d(double x1,double y1,double x2,double y2):
    """
    Usage
    -----
    L2-norm between point (x1,y1) and (x2,y2)

    Parameters
    ----------
    param x1 : float
    param y1 : float
    param x2 : float
    param y2 : float

    Returns
    -------
    dist : float
           L2-norm between (x1,y1) and (x2,y2)
    """
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

cdef double _c( np.ndarray[np.float64_t,ndim=2] ca,int i, int j, np.ndarray[np.float64_t,ndim=2] P,np.ndarray[np
.float64_t,ndim=2] Q):

    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = c_eucl_dist(P[0,:],Q[0,:])
    elif i > 0 and j == 0:
        ca[i,j] = fmax(_c(ca,i-1,0,P,Q),c_eucl_dist(P[i,:],Q[0,:]))
    elif i == 0 and j > 0:
        ca[i,j] = fmax(_c(ca,0,j-1,P,Q),c_eucl_dist(P[0,:],Q[j,:]))
    elif i > 0 and j > 0:
        ca[i,j] = fmax(fmin(_c(ca,i-1,j,P,Q),fmin(_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q))),c_eucl_dist(P[i,:],Q[j,:]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]

def discret_frechet(P,Q):
    """
    Usage
    -----
    Compute the discret frechet distance between trajectories P and Q

    Parameters
    ----------
    param P : px2 array_like, Trajectory P
    param Q : qx2 array_like, Trajectory Q

    Returns
    -------
    frech : float, the discret frechet distance between trajectories P and Q
    """
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    return _c(ca,len(P)-1,len(Q)-1,P,Q)

