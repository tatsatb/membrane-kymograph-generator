"""
Core helper functions for kymograph processing.
"""

from typing import Tuple
import numpy as np
from scipy.interpolate import pchip_interpolate

#
def smooth_boundary(bdy: np.ndarray, mfilter: int) -> np.ndarray:
    """
    Smooth a boundary using a moving average with circular shifts.
    
    Parameters
    ----------
    bdy : np.ndarray
        1D array of boundary coordinates
    mfilter : int
        Size of the moving average filter
    
    Returns
    -------
    np.ndarray
        Smoothed boundary array of same shape as input
    """
    s = mfilter
    sbdy = bdy.copy()
    
    for counter in range(s):
        sbdy = sbdy + np.roll(bdy, counter, axis=0) + np.roll(bdy, -counter, axis=0)
        
    sbdy = sbdy / (2 * s + 1)
    return sbdy


def interpboundary(bdy: np.ndarray, subpixel: float = 0.25) -> np.ndarray:
    """
    Interpolate boundary points to achieve subpixel resolution.
    
    Parameters
    ----------
    bdy : np.ndarray
        2D array of shape (n_points, 2) representing boundary coordinates
    subpixel : float, optional
        Subpixel resolution for interpolation (default: 0.25)
    
    Returns
    -------
    np.ndarray
        Interpolated boundary with subpixel resolution
    """

    if not np.array_equal(bdy[0], bdy[-1]):
        bdy = np.vstack([bdy, bdy[0]])
        

    d = np.diff(bdy, axis=0)
    dist = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
    cumdist = np.concatenate([[0], np.cumsum(dist)])

    perim = subpixel * round(cumdist[-1] / subpixel)
    interp_points = np.arange(0, perim, subpixel)
    
    x = pchip_interpolate(cumdist, bdy[:, 0], interp_points)
    y = pchip_interpolate(cumdist, bdy[:, 1], interp_points)

    ibdy = np.column_stack((x, y))
    ibdy = ibdy[:-1]  # Remove duplicate last point
    ibdy = ibdy[::int(1 / subpixel)]  # Subsample to pixel resolution
    
    return ibdy


def aligninitboundary(bdy: np.ndarray, z0: np.ndarray, theta0: float) -> Tuple[np.ndarray, int]:
    """
    Align initial boundary to start at a specific angle.
    
    Parameters
    ----------
    bdy : np.ndarray
        2D array of boundary coordinates
    z0 : np.ndarray
        Center point [x, y]
    theta0 : float
        Desired starting angle in degrees
    
    Returns
    -------
    tuple
        Aligned boundary and shift index
    """
    x0, y0 = z0
    
    if theta0 > 180:
        theta0 = theta0 - 180
    else:
        theta0 = 180 - theta0
        

    theta = 180 - np.degrees(np.arctan2(bdy[:, 1] - y0, bdy[:, 0] - x0))
    

    idx = np.argmin(np.abs(theta - theta0))
    

    aligned_bdy = np.roll(bdy, -idx, axis=0)
    
    return aligned_bdy, idx


def alignboundary(newbdy: np.ndarray, oldbdy: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Align new boundary to best match old boundary (legacy logic).
    
    Parameters
    ----------
    newbdy : np.ndarray
        New boundary to align
    oldbdy : np.ndarray
        Reference boundary
    
    Returns
    -------
    tuple
        Aligned boundary and minimum distance
    """
    sw = len(newbdy) > len(oldbdy)
    if sw:
        b1 = newbdy
        b2 = oldbdy
    else:
        b1 = oldbdy
        b2 = newbdy
        
    n1 = len(b1)
    n2 = len(b2)
    d = np.zeros(n2)
    idx = []
    

    while len(idx) < n1 - n2:
        deltan = n1 - n2 - len(idx)
        nidx = np.floor(n1 * np.random.rand(deltan))
        idx = np.unique(np.sort(np.concatenate((idx, nidx.astype(int))))).astype(int)
    
    kidx = np.sort(np.setdiff1d(np.arange(0, n1), idx))
    b1s = b1[kidx]
    
    # Try all possible alignments
    dst = np.zeros(len(b1s))
    for i in range(len(b1s)):
        d = b2 - np.roll(b1s, i, axis=0)
        dst[i] = np.mean(np.sqrt(np.sum(d ** 2, axis=1)))
        
    mindst, minidx = np.min(dst), np.argmin(dst)
    
    # Apply alignment
    if sw:
        alignedbdy = np.roll(newbdy, minidx - 1, axis=0)
    else:
        alignedbdy = np.roll(newbdy, 1 - minidx, axis=0)
    
    idx = []
    return alignedbdy, mindst