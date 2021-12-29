"""
Analysis Tools
==============

Provides
  1. Loading and manipulating images and stacks of images
  2. Measuring number and distribution of ommatidia in compound eye images.
  3. Measuring ommatidia in 3D image stacks of compound eyes.

Classes
-------
ColorSelector
    A GUI for generating a boolean mask based on user input.
LSqEllipse
    From https://doi.org/10.5281/zenodo.3723294, fit ellipse to 2D points.
Layer
    An image loaded from file or 2D array.
Eye
    A Layer child specifically for processing images of compound eyes.
Stack
    A stack of images at different depths for making a focus stack.
EyeStack
    A special stack for handling a focus stack of fly eye images.
CTStack
    A special stack for handling a CT stack of compound eyes.

Functions
---------
rgb_2_gray(rgb) : np.ndarray
    Converts from image with red, green, and blue channels into grayscale.

Note: for cropping, first load the mask 


"""
import h5py
from .interfaces import *
import math
import matplotlib
import numpy as np
import os
import pandas as pd
import PIL
from PIL import Image
import pickle
import seaborn as sbn
import subprocess
import sys
from tempfile import mkdtemp

from matplotlib import colors, mlab
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import skimage
from skimage.draw import ellipse as Ellipse
from skimage.feature import peak_local_max
from sklearn import cluster, mixture

import scipy
from scipy import interpolate, optimize, ndimage, signal, spatial, stats
from scipy.optimize import minimize
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass

blue, green, yellow, orange, red, purple = [(0.30, 0.45, 0.69), (0.33, 0.66, 0.41), (
    0.83, 0.74, 0.37), (0.78, 0.50, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]


def print_progress(part, whole):
    import sys
    prop = float(part)/float(whole)
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ("="*int(20*prop), 100*prop))
    sys.stdout.flush()

def load_image(fn):
    """Import an image as a numpy array using the PIL."""
    return np.asarray(PIL.Image.open(fn))

def save_image(fn, arr):
    """Save an image using the PIL."""
    img = PIL.Image.fromarray(arr)
    if os.path.exists(fn):
        os.remove(fn)
    return img.save(fn)

def rgb_2_gray(rgb):
    """Convert image from RGB to grayscale."""
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def rgb_to_hsv(rgb):
    """Convert image from RGB to HSV."""
    if rgb.ndim == 3:
        ret = matplotlib.colors.rgb_to_hsv(rgb)
    else:
        l, w = rgb.shape
        ret = np.repeat(rgb, 3, axis=-1)
    return ret

def rectangular_to_spherical(vals, center=[0, 0, 0]):
    """Convert 3D pts from rectangular to spherical coordinates.


    Parameters
    ----------
    vals : np.ndarray, shape (N, 3)
        3D points to be converted.
    center : array-like, shape (3)
        Center point to use for spherical conversion.
    
    Returns
    -------
    polar, shape (N, 3)
        The [inclination, azimuth, radius] per coordinate in vals.
    """
    pts = np.copy(vals)
    center = np.asarray(center)
    # center the points
    pts -= center[np.newaxis]
    xs, ys, zs = pts.T
    # rotate points so that 
    # get polar transformation
    radius = np.linalg.norm(pts, axis=-1)
    inclination = np.arccos(pts[:, 2] / radius) # theta - [  0, pi]
    azimuth = np.arctan2(pts[:, 1], pts[:, 0])  # phi   - [-pi, pi]
    polar = np.array([inclination, azimuth, radius]).T
    return polar

def spherical_to_rectangular(vals):
    """Convert 3D pts from rectangular to spherical coordinates.


    Parameters
    ----------
    vals : np.ndarray, shape (N, 3)
        3D points to be converted.
    
    Returns
    -------
    coords, shape (N, 3)
        The [x, y, z] per polar coordinate in vals.
    """
    pts = np.copy(vals)
    # center the points
    inclination, azimuth, radius = pts.T # theta, phi, radii
    # get polar transformation
    xs = radius * np.cos(azimuth) * np.sin(inclination)
    ys = radius * np.sin(azimuth) * np.sin(inclination)
    zs = radius * np.cos(inclination)
    # combine into one array
    coords = np.array([xs, ys, zs]).T
    return coords

def rotate(arr, theta, axis=0):
    """Generate a rotation matrix and rotate input array along a single axis."""
    if axis == 0:
        rot_matrix = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]])
    elif axis == 1:
        rot_matrix = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 2:
        rot_matrix = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]])
    nx, ny, nz = np.dot(arr, rot_matrix).T
    nx = np.squeeze(nx)
    ny = np.squeeze(ny)
    nz = np.squeeze(nz)
    return np.array([nx, ny, nz])

def rotate_compound(arr, yaw=0, pitch=0, roll=0):
    """Rotate the arr of coordinates along all three axes.

    
    Parameters
    ----------
    arr : array-like, shape=(N, 3)
        The array of 3D points to rotate.
    yaw : float
        The angle to rotate about the z-axis.
    pitch : float
        The angle to rotate about the x-axis.
    roll : float
        The angle to rotate about the y-axis.
    """
    yaw_arr = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    pitch_arr = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    roll_arr = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll) ]])
    rotation_matrix = yaw_arr @ pitch_arr @ roll_arr
    return arr @ rotation_matrix


def fit_line(data):             # fit 3d line to 3d data
    """Use singular value decomposition (SVD) to find the best fitting vector to the data.


    Keyword arguments:
    data -- input data points
    component -- the order of the axis used to decompose the data (default 0 => the first component vector which represents the plurality of the data
    """

    m = data.mean(0)
    max_val = np.round(2*abs(data - m).max()).astype(int)
    uu, dd, vv = np.linalg.svd(data - m)
    return vv


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def positive_fit(predictor, outcome):
    """Uses non-linear minimization to fit a polynomial to data

    
    Parameters
    ----------
    predictor : np.ndarray
        Array of values used to model the outcome.
    outcome : np.ndarray
        Array of values being predicted by the predictor.

    Returns
    -------
    final_func : function
        The final fitted function, producing estimate of outcome.
    residuals_normalized : float
        The squared residuals divided by the number of samples.
    """
    # if predictor and outcome have different shapes, make sure they
    # have the same dimension
    if predictor.shape != outcome.shape:
        # check dimensions
        if predictor.ndim != outcome.ndim:
            max_dims = max(predictor.ndim, outcome.ndim)
            if predictor.ndim < max_dims:
                predictor = np.expand_dims(predictor, axis=-1)
            if outcome.ndim < max_dims:
                outcome = np.expand_dims(outcome, axis=-1)
    # this only works if there are more predictors than outcome variables
    assert outcome.shape[-1] == 1, ("There should only be one outcome variable.")
    # iterates through orders of polynomial degree and find best fit
    def func(predictor, pars):
        return np.polyval(pars, predictor)
    def resid(pars):
        predicted_vals = func(predictor, pars)
        # consider overdetermined case
        if predictor.shape[-1] > outcome.shape[-1]:
            predicted_vals = predicted_vals.sum(-1)
        return ((outcome-predicted_vals)**2).sum()
    predictor_range = predictor.ptp(0)
    def constr(pars):
        new_predictor = np.linspace(predictor.min(0) - predictor_range/2,
                                    predictor.max(0) + predictor_range/2,
                                    1000)
        pred_vals = func(new_predictor, pars)
        deriv = np.diff(pred_vals)
        return min(deriv)
    con1 = {'type': 'ineq', 'fun': constr}
    min_resids = np.inf
    model = None
    for deg in np.arange(2, 5, 1):
        pars = np.zeros(deg)
        pars[0] = .1
        # res = minimize(resid, pars, method='cobyla',
        #                constraints=con1, options={'maxiter':50000})
        res = minimize(resid, pars, method='cobyla', options={'maxiter':50000})
        # print(res)
        new_predictor = np.linspace(min(predictor), max(predictor))
        pred_vals = func(new_predictor, res.x)
        resids = resid(res.x)
        if resids < .95 * min_resids:
            # plt.scatter(predictor, outcome)
            # plt.plot(new_predictor, pred_vals)
            # plt.gca().set_aspect('equal')
            # plt.show()
            model = res
            min_resids = resids
    pred_vals = func(new_predictor, res.x)
    def final_func(x):
        return np.polyval(model.x, x)
    # normalize the residuals by the the number of samples used
    residuals_normalized = resid(model.x)/len(predictor)
    return final_func, residuals_normalized


def angle_fitter(pts, lbls, angle_deviation_limit=np.pi/3, display=False):
    """


    (1) Import 3D coordinates and clusterd by lbls. (2) Using the cluster 
    centers, fit a circle in order to do a polar transformation. (3) Use the 
    SVD of each cluster in 3D, projected onto the 2D plane, and then regress 
    the direction vectors on polar angle using robust linear modelling to account
    for noisy SVDs."""
    # (1) import data
    lbls_set = sorted(set(lbls))
    # pts should have shape = N x (x, y, z) where z is the axis to be ignored
    pts = np.array(pts)#[np.argsort(lbls)]
    xs, ys, zs = pts.T
    # lbls.sort()
    centers = np.zeros((len(lbls), 3))
    # find center per cluster
    group_centers = np.zeros((len(set(lbls)), 3))
    for num, lbl in enumerate(lbls_set):
        ind = lbls == lbl
        center = pts[ind].mean(0)
        centers[ind] = center
        group_centers[num] = center
    # (2) rays from center of sphere
    rays = group_centers[:, :2]
    norms = np.linalg.norm(rays, axis=1)
    rays = rays / np.linalg.norm(rays, axis=1)[:, np.newaxis]
    # get polar position of each cluster center (polar transformation)
    group_polar_angles = np.arctan2(rays[:, 1], rays[:, 0])
    # (3) get svds, project onto 2D plane, and regress onto group_polar_angles
    # using RLM
    svds = []
    lengths = []
    for lbl, ray, center in zip(lbls_set, rays, group_centers[:, :2]):
        ind = lbls == lbl
        sub_pts = pts[ind]
        svd = np.linalg.svd(sub_pts - sub_pts.mean(0))
        svds += [svd[-1][0]]
        lengths += [svd[1][0]]
        # get angle difference using dot product
        l = 5
    #     p1, p2 = center - l * svd[-1][0][:2], center + l * svd[-1][0][:2]
    #     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=red)
    #     p1, p2 = center - l * ray, center + l * ray
    #     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=green)
    #     plt.scatter(sub_pts[:, 0], sub_pts[:, 1], marker='.', zorder=0)
    # plt.gca().set_aspect('equal')
    # plt.show()
    svds = np.array(svds)
    lengths = np.array(lengths)
    # rotate svds so they are between 0 and pi
    neg_svds = svds[:, 1] > 0
    svds[neg_svds, :2] *= -1
    # replace any svds where angle difference > 60 degs with nan
    # ang_diffs = []
    # for svd, ray in zip(svds, rays):
    #     dot = np.dot(svd[:2], ray)
    #     ang_diff = np.arccos(dot)
    #     ang_diffs += [ang_diff]
    # ang_diffs = np.array(ang_diffs)
    # include = ang_diffs <= angle_deviation_limit
    include = np.ones(len(rays))
    # svds[include == False] = np.nan
    # rotate svds so they're centered at pi
    xmean, ymean = svds[:, :2].mean(0)
    angles = np.arctan2(svds[:, 1], svds[:, 0])
    # plt.scatter(group_polar_angles[include],
    #             angles[include], label='original', alpha=.5)
    mean_ang = np.arctan2(ymean, xmean)
    rot_ang = mean_ang - np.pi/2
    svds_rotated = rotate(svds, rot_ang, axis=2).T
    angles = np.arctan2(svds_rotated[:, 1], svds_rotated[:, 0])
    wls_mod = None
    aic = np.inf
    # use polyfit instead and iterate until there is a plateau in p values
    # or use curve_fit and set constraint on first derivative to be positive
    if include.sum() > 2:
        # mod, resids = positive_fit(group_polar_angles[include], angles[include])
        mod, resids = positive_fit(group_polar_angles, angles)
        wls_mod = mod
        new_xs = np.linspace(group_polar_angles.min(),
                             group_polar_angles.max(), 100) 
        new_ys = mod(new_xs)
        new_angs = mod(group_polar_angles)
        # new_angs = wls_mod.predict({'xs':group_polar_angles}).values
        new_svds_rotated = np.copy(svds_rotated)
        new_svds_rotated[:, 1] = np.tan(new_angs)
        new_svds_rotated[:, 0] = 1
        new_svds = rotate(new_svds_rotated, -rot_ang, axis=2).T
        norms = np.linalg.norm(new_svds[:, :2], axis=-1)
        new_svds /= norms[:, np.newaxis]
        if display:
            # plot the clusters
            fig = plt.figure()
            # scramble lbls
            new_lbls = dict()
            new_vals = np.arange(len(set(lbls))) + 1
            for lbl, new_val in zip(set(lbls), new_vals):
                new_lbls[lbl] = new_val
            clbls = []
            for lbl in lbls:
                clbls += [new_lbls[lbl]]
            plt.scatter(pts[:, 0], pts[:, 1], c=clbls, cmap='tab20')
            # plot the centers
            for lbl, ray, svd, center in zip(lbls_set, rays, new_svds, group_centers[:, :2]):
                ind = lbls == lbl
                sub_pts = pts[ind]
                # get angle difference using dot product
                l = 30
                # plot the vectors
                p1, p2 = center - l * svd[:2], center + l * svd[:2]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=red)
                p1, p2 = center - l * ray, center + l * ray
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=green)
                plt.scatter(sub_pts[:, 0], sub_pts[:, 1], marker='.', zorder=0)
            plt.gca().set_aspect('equal')
            plt.show()
    else:
        new_svds = np.zeros((len(lbls_set), 3))
        new_svds.fill(np.nan)
        resids = np.inf
    return lbls_set, new_svds[:, :2], resids
    # return lbls_set, rays, resids


def fit_circle(spX, spY):
    """Find best fitting sphere to x, y, and z coordinates using OLS."""
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX**2) + (spY**2)
    A = np.zeros((len(spX), 3))
    A[:, 0] = spX*2
    A[:, 1] = spY*2
    A[:, 2] = 1
    C, residuals, rank, sigval = np.linalg.lstsq(A, f, rcond=None)
    #   solve for the radius
    t = C[0]**2 + C[1]**2 + C[2]
    radius = math.sqrt(t)
    return radius, np.squeeze(C[:-1]), residuals


class SphereFit():
    """Fit sphere to points to find center and radius.


    Attributes
    ----------
    pts : np.ndarray, shape (N, 3)
        The points to fit the sphere to.
    center : np.ndarray, len (3)
        The resulting center point.
    radius : float
        The average distance of pts to center.
    """
    def __init__(self, pts):
        """Fit sphere equation to 3D points using scipy.optimize.minimize.


        Parameters
        ----------
        pts : np.ndarray, shape (N, 3)
            The array of 3D points to be fitted.
        """
        self.pts = np.copy(pts)
        # store the original pts for posterity
        self.original_pts = np.copy(self.pts)
        self.xs, self.ys, self.zs = self.pts.T
        # find the point closest to the center of the points
        # construct the outcome matrix
        outcome = (self.pts ** 2).sum(1)
        outcome = outcome[:, np.newaxis]
        # construct coefficient matrix
        coefficients = np.ones((len(self.xs), 4))
        coefficients[:, :3] = self.pts * 2
        # solve using numpy
        solution, sum_sq_residuals, rank, singular = np.linalg.lstsq(
            coefficients, outcome, rcond=None)
        # get radius
        self.center = solution[:-1, 0]
        self.radii = np.linalg.norm(self.pts - self.center[np.newaxis], axis=-1)
        self.radius = np.mean(self.radii)
        # center the pts 
        self.pts -= self.center
        # rotate the points about the center until com is ideal
        self.center_com()
        # then perform spherical transformation
        self.get_polar()

    def center_com(self):
        # rotate points using the center of mass:
        # 1. find center of mass
        com = self.pts.mean(0)
        # 2. rotate com along x axis (com[0]) until z (com[2]) = 0
        ang1 = np.arctan2(com[2], com[1])
        com1 = rotate(com, ang1, axis=0)
        rot1 = rotate(self.pts, ang1, axis=0).T
        # 3. rotate com along z axis (com[2]) until y (com[1]) = 0
        ang2 = np.arctan2(com1[1], com1[0])
        rot2 = rotate(rot1, ang2, axis=2).T
        self.pts = rot2

    def get_polar(self):
        """Transform self.pts to polar coordinates using sphere center.


        Attributes
        ----------
        polar : np.ndarray, shape=(N,3)
            The list of coordinates transformed into spherical coordinates.
        """
        xs, ys, zs = self.pts.T
        # get polar transformation
        radius = np.linalg.norm(self.pts, axis=-1)
        inclination = np.arccos(zs / radius)
        azimuth = np.arctan2(ys, xs)
        self.polar = np.array([inclination, azimuth, radius]).T




def colorbar_histogram(colorvals, vmin, vmax, ax=None, bin_number=100,
                       fill_color='k', line_color='w', colormap='viridis'):
    """Plot a colorbar with a histogram skyline superimposed.


    Parameters
    ----------
    colorvals : array-like
        List of values corresponding to colors drawn from the colormap.
    vmin : float
        Minimum colorvalue to include in the histogram and colorbar.
    vmin : float
        Maximum colorvalue to include in the histogram and colorbar.
    ax : matplotlib.axes._subplots.AxesSubplot
        The pyplot axis in which to plot the histogram and colorbar.
    bin_number : int, default=100
        The number of bins to use in plotting the histogram.
    fill_color : matplotlib color, default='k'
        Color for filling the space under the histogram. Default is black.
    line_color : matplotlib color, default='w'
        Color for the histogram skyline.
    colormap : matplotlib colormap, default='viridis'
        Colormap of colorvals to colors.
    """
    assert all([vmin < np.inf, vmax < np.inf, not np.isnan(vmin), not np.isnan(vmax)]), (
        "Input vmin and vmax should be finite floats")
    if ax is None:
        ax = plt.gca()
    if not isinstance(colorvals, np.ndarray):
        colorvals = np.asarray(colorvals)
    # use evenly spaced bins and counts
    bins = np.linspace(vmin, vmax, bin_number + 1)
    counts, bin_edges = np.histogram(colorvals, bins=bins)
    # use seaborn distplot to get a histogram skyline
    # histogram = sbn.distplot(colorvals, kde=False, color=fill_color,
    #                          ax=ax, vertical=True, bins=bins,
    #                          axlabel=False)
    # plot the histogram skyline 
    bin_edges = np.repeat(bins, 2)[1:-1]
    heights = np.repeat(counts, 2)
    ax.plot(heights, bin_edges, color=line_color)
    # color under the skyline
    ax.fill_betweenx(bin_edges, heights, color=fill_color, alpha=.3)
    # plot the color gradient
    vals = np.linspace(vmin, vmax)
    C = vals
    X = np.array([0, counts.max()])
    Y = np.repeat(vals[:, np.newaxis], 2, axis=-1)
    ax.pcolormesh(X, C, Y, cmap=colormap,
                  zorder=0, vmin=vmin, vmax=vmax,
                  shading='nearest')
    # formatting
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # sbn.despine(ax=ax, bottom=False) # remove spines
    ax.set_xticks([])                # remove xticks
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylim(vmin, vmax)
    ax.set_xlim(0, counts.max())


class Points():
    """Coordinate data in cartesian and spherical coordinates.


    Attributes
    ----------
    pts : array_like, shape=(N, 3)
        Array of 3D coordinates.
    original_pts : array_like, shape=(N, 3)
        Array of the input 3D coordinates before any rotations or 
        translations.
    shape : tuple, default=(N, 3)
        Shape of the 3D coordinates.
    center : array_like, default=[0, 0, 0]
        The 3D coordinate of the center point.
    raster : array_like, default=None
        The 2D raster image of the 3D coordinates.
    xvals, yvals : array_like, default=None
        The boundaries of the pixels in self.raster.
    sphere_model : SphereFit
        Model fitting a sphere to 3D points using OLS.
    radius : float
        Radius of the fitted sphere.
    center : array_like
        3D enter of the fitted sphere.
    polar : array_like, shape=(N, 3)
        Polar coordinates of self.pts with respect to the input center.
    theta, phi, radii : array_like, shape=(N, 1)
        The azimuth, elevation, and radial distance from self.polar.
    residuals : array_like, shape=(N, 1)
        The differences between the radii and the fitted radius.
    raster : np.ndarray
        The 2D histogram of the points, optionally weighted by self.vals.
    surface : array_like
        The resulting surface.

    Methods
    -------
    spherical(center=None):
        Perform the spherical transformation.
    rasterize(polar=True, axes=[0, 1], pixel_length=.01, weights=None):
        Rasterize coordinates onto a grid defined by min and max vals.
    fit_surface(polar=True, outcome_axis=0, image_size=10**4):
        Cubic interpolate surface of one axis using the other two.
    get_polar_cross_section(thickness=.1, pixel_length=.01):
        Find best fitting surface of radii using phis and thetas.
    save(fn):
        Save using pickle.
    """

    def __init__(self, arr, center=[0, 0, 0], polar=None,
                 sphere_fit=True, spherical_conversion=True,
                 rotate_com=True, vals=None):
        """Import array of rectangular coordinates with some options.


        Parameters
        ----------
        arr : np.ndarray, shape (N, 3)
            The input array of 3D points.
        center_points : bool, default=True
            Whether to center the input points.
        polar : np.ndarr, default=None
            Option to input the polar coordinates, to avoid recentering.
        sphere_fit : bool, default=True
            Whether to fit a sphere to the coordinates and center.
        spherical_conversion : bool, default=Trued 
            Whether to calculate polar coordinates.
        rotate_com : bool, default=True
            Whether to rotate input coordinates so that the center of 
            mass is centered in terms of azimuth and inclination.
        vals : np.ndarray, shape (N)
            Values associated with each point in arr.

        Attributes
        ----------
        pts : array_like, shape=(N, 3)
            Array of 3D coordinates.
        original_pts : array_like, shape=(N, 3)
            Array of the input 3D coordinates before any rotations or 
            translations.
        shape : tuple, default=(N, 3)
            Shape of the 3D coordinates.
        center : array_like, default=[0, 0, 0]
            The 3D coordinate of the center point.
        raster : array_like, default=None
            The 2D raster image of the 3D coordinates.
        xvals, yvals : array_like, default=None
            The boundaries of the pixels in self.raster.
        polar : array_like, default=None
            Custom input polar coordinates (optional).
        sphere_model : SphereFit
            Model fitting a sphere to 3D points using OLS.
        radius : float
            Radius of the fitted sphere.
        center : array_like
            3D enter of the fitted sphere.
        """
        self.pts = np.array(arr)
        if vals is None:
            self.vals = np.ones(len(self.pts))
        else:
            self.vals = vals
        if arr.ndim > 1:
            assert self.pts.shape[1] == 3, (
                "Input array should have shape N x 3. Instead it has "
                "shape {} x {}.".format(self.pts.shape[0], self.pts.shape[1]))
        else:
            assert self.pts.shape[0] == 3, (
                "Input array should have shape 3 or N x 3. Instead it has "
                "shape {}.".format(self.pts.shape))
            self.pts = self.pts.reshape((1, -1))
        self.original_pts = self.pts
        self.shape = self.pts.shape
        self.center = np.asarray(center)
        self.raster = None
        self.xvals, self.yvals = None, None
        self.polar = None
        if polar is not None:
            self.polar = polar
            self.theta, self.phi, self.radii = self.polar.T
        if sphere_fit:
            # fit sphere
            self.sphere_model = SphereFit(self.pts)
            self.radius = self.sphere_model.radius
            self.center = self.sphere_model.center
            # center points using the center of that sphere
            self.pts -= self.center
            self.center = self.center - self.center
        if spherical_conversion:
            # optionally:
            if rotate_com:
                # find svd of points 
                # use a small sample of points
                sample_inds = np.arange(len(self.pts))
                num_points = 1000
                sample_inds = np.random.choice(sample_inds, num_points)
                # get svd solutions
                # uu, dd, vv = np.linalg.svd(self.pts[sample_inds])
                # new_pts = np.dot(self.pts, vv)[:, [1, 2, 0]]
                # self.pts = new_pts
                # rotate points using the center of mass:
                # 1. find center of mass
                com = self.pts.mean(0)
                # 2. rotate along x axis (com[0]) until z (com[2]) = 0
                ang1 = np.arctan2(com[2], com[1])
                com1 = rotate(com, ang1, axis=0)
                rot1 = rotate(self.pts, ang1, axis=0).T
                # 3. rotate along z axis (com[2]) until y (com[1]) = 0
                ang2 = np.arctan2(com1[1], com1[0])
                rot2 = rotate(rot1, ang2, axis=2).T
                self.pts = rot2
            # grab spherical coordinates of centered points
            self.spherical()
        self.x, self.y, self.z = self.pts.T

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        out = Points(self.pts[key], polar=self.polar[key],
                     rotate_com=False, spherical_conversion=False,
                     vals=self.vals[key])
        return out

    def spherical(self, center=None):
        """Perform the spherical transformation.


        Parameters
        ----------
        center : bool, default=None
            Option to input custom center point.

        Attributes
        ----------
        polar : array_like, shape=(N, 3)
            The polar coordinates of self.pts with respect to the input center.
        theta, phi, radii : array_like, shape=(N, 1)
            The azimuth, elevation, and radial distance from self.polar.
        residuals : array_like, shape=(N, 1)
            The differences between the radii and the fitted radius.
        """
        if center is None:
            center = self.center
        self.polar = rectangular_to_spherical(self.pts, center=center)
        self.theta, self.phi, self.radii = self.polar.T
        if "radius" in dir(self):
            self.residuals = self.radii - self.radius

    def rasterize(self, polar=True, axes=[0, 1], image_size=10**4,
                  weights=None, pixel_length=None):
        """Rasterize coordinates onto a grid defined by min and max vals.


        Parameters
        ----------
        polar : bool, default=True
            Whether to rasterize polar (vs. rectangular) coordinates.
        image_size : int, default=1e4
            The number of pixels in the image.
        weights : list, shape=(N, 1), default=None
            Optional weights associated with each point.
        pixel_length : float, default=None
            Alternative method of specifying the grid based on the pixel length, as 
            opposed to image size. This overrides the image size.

        Returns
        -------
        raster : np.ndarray
            The 2D histogram of the points, optionally weighted by self.vals.
        (xs, ys) : tuple
            The x and y coordinates marking the boundaries of each pixel. 
            Useful for rendering as a pyplot.pcolormesh.
        """
        if polar:
            arr = self.polar
        else:
            arr = self.pts
        x, y = arr.T[axes]
        # get coordinate ranges for the appropriate aspect ratio
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        # figure out side lengths needed for input image size
        ratio = y_range / x_range
        if pixel_length is None:
            x_len = int(np.round(np.sqrt(image_size/ratio)))
            # get x and y ranges corresponding to image size
            xs = np.linspace(x.min(), x.max(), x_len)
            self.raster_pixel_length = xs[1] - xs[0]
        else:
            self.raster_pixel_length = pixel_length            
            xs = np.arange(x.min(), x.max(), self.raster_pixel_length)
        ys = np.arange(y.min(), y.max(), self.raster_pixel_length)
        if weights is None:
            # a simple 2D histogram of the x and y coordinates
            avg = np.histogram2d(x, y, bins=(xs, ys))[0] # histogram image
        else:
            # a weighted 2D histogram if values were provided for 
            avg = np.histogram2d(x, y, bins=(xs, ys), weights = weights)[0]
        self.raster = avg
        # use raster pixel length to get the x and y axes for the raster image
        xs = xs[:-1] + (self.raster_pixel_length / 2.)
        ys = ys[:-1] + (self.raster_pixel_length / 2.)
        self.xvals, self.yvals = xs, ys
        return self.raster, (xs, ys)

    def fit_surface(self, polar=True, outcome_axis=0, image_size=1e4):
        """Cubic interpolate surface of one axis using the other two.


        Parameters
        ----------
        polar : bool, default=True
            Whether to fit a surface using polar coordinates.
        outcome_axis : int, default=0
            The axis to use as the outcome of the other axes.
        image_size : int, default=1e4
            The number of pixels in the image.

        Attributes
        ----------
        avg : array_like
            The rolling average
        """
        if polar:
            arr = self.polar
        else:
            arr = self.pts

        x, y, z = arr.T
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        # figure out side lengths needed for input image size
        ratio = y_range / x_range
        x_len = int(np.round(np.sqrt(image_size/ratio)))
        y_len = int(np.round(ratio * x_len))
        # reduce data using a 2D rolling average
        # xs = np.arange(x.min(), x.max(), pixel_length)
        # ys = np.arange(y.min(), y.max(), pixel_length)
        xs = np.linspace(x.min(), x.max(), x_len)
        ys = np.linspace(y.min(), y.max(), y_len)
        avg = []
        for col_num, (x1, x2) in enumerate(zip(xs[:-1], xs[1:])):
            col = []
            in_column = np.logical_and(x >= x1, x < x2)
            in_column = arr[in_column]
            for row_num, (y1, y2) in enumerate(zip(ys[:-1], ys[1:])):
                in_row = np.logical_and(
                    in_column[:, 1] >= y1, in_column[:, 1] < y2)
                if any(in_row):
                    avg += [np.mean(in_column[in_row], axis=0)]
                    # vals = in_column[in_row][:, -1]
                    # xvals, yvals, zvals = in_column[in_row].T
                    # avg += [[x1, y1, np.median(in_column[in_row][:, -1])]]
            print_progress(col_num, len(xs) - 1)
        print()
        avg = np.array(avg)
        # filter outlier points by using bootstraped 95% confidence band (not of the mean)
        # low, high = np.percentile(avg[:, 2], [.5, 99.5])
        # self.avg = avg[np.logical_and(avg[:, 2] >= low, avg[:, 2] < high)]
        self.avg = avg
        self.avg_x, self.avg_y, self.avg_z = self.avg.T

    def surface_predict(self, xvals=None, yvals=None, polar=True, image_size=1e4):
        """Find the approximate zvalue given arbitrary x and y values."""
        # if xvals is not None:
        #     breakpoint()
        if "avg_x" not in dir(self):
            self.fit_surface(polar=polar, image_size=image_size)
        if (xvals is None) or (yvals is None):
            if polar:
                arr = self.polar
            else:
                arr = self.pts
            xvals, yvals, zvals = arr.T
        points = np.array([xvals, yvals]).T
        sort_inds = np.argsort(points[:, 0])
        self.surface = np.zeros(len(points), dtype=float)
        self.surface[sort_inds] = interpolate.griddata(
            self.avg[:, :2], self.avg_z, points[sort_inds], method='nearest')
        cubic_surface = interpolate.griddata(
            self.avg[:, :2], self.avg_z, points[sort_inds], method='cubic')
        # cubic interpolation can lead to NaNs
        # replace these using the nearest method
        no_nans = np.isnan(cubic_surface) == False
        self.surface[sort_inds][no_nans] = cubic_surface[no_nans]
        


    def get_polar_cross_section(self, thickness=.1, pixel_length=.01):
        """Find best fitting surface of radii using phis and thetas."""
        # self.fit_surface(mode='polar', pixel_length=pixel_length)
        self.surface_predict(polar=True)
        # find distance of datapoints from surface (ie. residuals)
        self.residuals = self.radii - self.surface
        # choose points within 'thickness' proportion of residuals
        no_nans = np.isnan(self.residuals) == False
        self.cross_section_thickness = np.percentile(
            abs(self.residuals[no_nans]), thickness * 100)
        self.surface_lower_bound = self.surface - self.cross_section_thickness
        self.surface_upper_bound = self.surface + self.cross_section_thickness
        cross_section_inds = np.logical_and(
            self.radii <= self.surface_upper_bound,
            self.radii > self.surface_lower_bound)
        self.cross_section = self[cross_section_inds]

    def save(self, fn):
        """Save using pickle."""
        with open(fn, "wb") as pickle_file:
            pickle.dump(self, pickle_file)


class ColorSelector():
    """GUI for masking the image based on user selection statistics. 

    
    Uses the selected distribution of hues, saturations, and values to find 
    pixel regions that fall within those distributions. This is useful for 
    chromatic keying or background subtraction.

    Atributes
    ---------
    hsv : np.ndarray
        The hues, saturations, and values per pixel of the filtered image.
    hue_dist : list
        The bin size and values of the hues in the sample.
    sat_dist : list
        The bin size and values of the saturations in the sample.
    val_dist : list
        The bin size and values of the values in the sample.
    lows : np.ndarray
        Minimum hue, saturation, and value of the region distribution.
    highs : np.ndarray
        Maximum hue, saturation, and value of the region distribution.
    mask : np.ndarray
        2D masking boolean array of image using selected color range.

    Methods
    -------
    get_color_stats()
        Calculate the histograms of the hues, saturations, and values.
    plot_color_stats(init=False)
        Initialize or update the plots for hues, saturations, and values.
    select_color(dilate_iters=5)
        Generate a mask based on the selected colors and dilated.
    """

    def __init__(self, image, bw=False, hue_only=False):
        """Initialize the ColorSelector GUI.


        Make a pyplot an interactive figure with the original image to be 
        sampled, the processed image based on the sample region, and the hues,
        saturations, and values of the sample region.

        Parameters
        ----------
        image : np.ndarray
            The 2D image we want to filter.
        bw : bool, default=False
            Whether the image is grayscale.
        hue_only : bool, default=False
            Whether to use only the hue channel.
        """
        # store options
        self.bw = bw            # True -> grayscale image
        self.hue_only = hue_only # True -> only use hue data
        # if image is a filename, load the file
        if isinstance(image, str):
            image = ndimage.imread(image)
        self.image = image
        self.image_hsv = rgb_to_hsv(self.image)
        # begin with an all-inclusive mask
        self.mask = np.ones(self.image.shape[:2], dtype=bool)
        # and an all-inclusive color range
        self.color_range = np.array([[0, 0, 0], [1, 1, 255]]) # low, high in hsv
        # Setup the figure:
        self.fig = matplotlib.pyplot.figure(
            figsize=(8, 8), num="Color Selector")
        self.grid = matplotlib.gridspec.GridSpec(
            6, 2, width_ratios=[1, 3]) # 6 rows X 2 cols grid organization
        # Setup axes for:
        ## 1. the original image:
        self.original_image_ax = self.fig.add_subplot(self.grid[:3, 1])
        # formatting
        self.original_image_ax.set_xticks([])
        self.original_image_ax.set_yticks([])
        matplotlib.pyplot.title("Original Image")
        matplotlib.pyplot.imshow(self.image.astype('uint8'))
        ## 2. the masked image:
        self.masked_image_ax = self.fig.add_subplot(self.grid[3:, 1])
        # formatting
        self.masked_image_ax.set_xticks([])
        self.masked_image_ax.set_yticks([])
        matplotlib.pyplot.title("Masked Image")
        self.masked_im = self.masked_image_ax.imshow(
            self.image.astype('uint8')) 
        ## 3. and plot the hues, saturations, and values:
        self.plot_color_stats(init=True)

    def get_color_stats(self):
        """Calculate the histograms of the hues, saturations, and values.

        
        Atributes
        ---------
        hsv : np.ndarray
            The hues, saturations, and values per pixel of the filtered image.
        hue_dist : list
            The bin size and values of the hues in the sample.
        sat_dist : list
            The bin size and values of the saturations in the sample.
        val_dist : list
            The bin size and values of the values in the sample.
        """
        self.sample_hsv = self.image_hsv[self.mask]
        # the distribution of hues
        self.hue_dist = list(np.histogram(
            self.sample_hsv[:, 0], 255, range=(0, 1), density=True))
        self.hue_dist[0] = np.append(self.hue_dist[0], self.hue_dist[0][0])
        # the distribution of saturations
        self.sat_dist = np.histogram(
            self.sample_hsv[:, 1], 255, range=(0, 1), density=True)
        # the distribution of values
        self.val_dist = np.histogram(
            self.sample_hsv[:, 2], 255, range=(0, 255), density=True)

    def plot_color_stats(self, init=False):
        """Initialize or update the plots for hues, saturations, and values.


        Parameters
        ----------
        init : bool, default=False
            Whether to initialize the plots.
        """
        # get hue, saturation, and value statistics
        self.get_color_stats()
        if init:
            # On the first round, initialize the plots for
            # Hues:
            self.hues = self.fig.add_subplot(self.grid[0:2, 0], polar=True)
            matplotlib.pyplot.title("Hues")
            # colormap in background for reference
            radii, theta = np.array([0, self.image.size]), np.linspace(0, 2*np.pi, 256)
            colorvals = np.arange(256)/256
            colorvals = np.array([colorvals, colorvals])
            self.hues.pcolormesh(theta, radii, colorvals, cmap='hsv',
                                 shading='nearest')
            self.hues.set_xticks([])
            self.hues.set_xticklabels([])
            self.hues.set_rticks([])
            # Saturations: 
            self.sats = self.fig.add_subplot(self.grid[2:4, 0])
            self.sats.set_xticks([0, .5, 1])
            self.sats.set_yticks([])
            matplotlib.pyplot.title("Saturations")
            # colormap in background for reference
            xs, ys = self.sat_dist[1], np.array([0, self.image.size])
            self.sats.pcolormesh(xs, ys, colorvals, cmap='Blues',
                                 shading='nearest')
            # and Values:
            self.vals = self.fig.add_subplot(self.grid[4:, 0])
            self.vals.set_xticks([0, 128, 255])
            self.vals.set_yticks([])
            matplotlib.pyplot.title("Values")
            # background for reference
            xs, ys = self.val_dist[1], np.array([0, self.image.size])
            self.vals.pcolormesh(xs, ys, colorvals[:, ::-1],
                                 cmap='Greys',
                                 shading='nearest')
            # plot the skyline histogram for hues
            self.h_line, = self.hues.plot(
                2*np.pi*self.hue_dist[1], self.hue_dist[0], "k")
            # saturations
            self.s_line, = self.sats.plot(
                self.sat_dist[1][1:], self.sat_dist[0], "r")
            self.sats.set_xlim(0, 1)
            # and values
            self.v_line, = self.vals.plot(
                self.val_dist[1][1:], self.val_dist[0], "r")
            self.vals.set_xlim(0, 255)
            # indicate the regions included in the mask
            # self.huespan = self.hues.axvspan(
            #     0, 2*np.pi,
            #     color="k", alpha=.3, ymin=0, ymax=self.image.size)
            self.satspan = self.sats.axvspan(
                self.color_range[0][1], self.color_range[1][1],
                color="k", alpha=.3)
            self.valspan = self.vals.axvspan(
                self.color_range[0][2], self.color_range[1][2],
                color="k", alpha=.3)
            # remove extra spacing
            self.fig.tight_layout()
        else:
            # if already initialized, update the lines and spans
            self.h_line.set_ydata(self.hue_dist[0])
            self.s_line.set_ydata(self.sat_dist[0])
            self.v_line.set_ydata(self.val_dist[0])
            # self.huespan.set_xy(
            #     self.set_radius_span(self.color_range[0][0] * 2 * np.pi,
            #                          self.color_range[1][0] * 2 * np.pi))
            self.satspan.set_xy(
                self.get_axvspan(self.color_range[0][1],
                                 self.color_range[1][1]))
            self.valspan.set_xy(
                self.get_axvspan(self.color_range[0][2],
                                 self.color_range[1][2]))
        # general formatting to keep statistics in range
        self.hues.set_rlim(
            rmin=-.5*self.hue_dist[0].max(),
            rmax=1*self.hue_dist[0].max())
        self.sats.set_ylim(ymin=0, ymax=self.sat_dist[0].max())
        self.vals.set_ylim(ymin=0, ymax=self.val_dist[0].max())
        
    def select_color(self, dilate_iters=5):
        """Generate a mask based on the selected colors and dilated.


        Parameters
        ----------
        dilate_iters : int, default=5
            Number of iterations to apply the binary dilation to the mask.

        Attributes
        ----------
        lows : np.ndarray
            Minimum hue, saturation, and value of the region distribution.
        highs : np.ndarray
            Maximum hue, saturation, and value of the region distribution.
        mask : np.ndarray
            2D masking boolean array of image using selected color range.

        Returns
        -------
        keyed : np.ndarray
            The image including only the pixels within the selected color range.
        """
        # grag low and high values of the selected color range
        self.lows, self.highs = self.color_range.min(0), self.color_range.max(0)
        hue_low, hue_high= self.lows[0], self.highs[0]
        # create a boolean mask for each channel
        include = np.logical_and(
            self.image_hsv > self.lows[np.newaxis, np.newaxis],
            self.image_hsv < self.highs[np.newaxis, np.newaxis])
        # hues are circular and so we should allow ranges across 0
        if hue_low < 0:    # if range overlaps 0, use or logic
            # to do this, include two regions, below minimum or above maximum
            hue_low = 1 + hue_low
            include[..., 0] = np.logical_or(
                self.image_hsv[..., 0] > hue_low,
                self.image_hsv[..., 0] < hue_high)
        # if the image is being treated as greyscale, use only the values
        if self.bw:
            self.mask = vals
        else:
            # if we specified to use only the hues, use only hues
            if self.hue_only:
                self.mask = hues
            # otherwise use the intersection of the 3 conditions
            else:
                self.mask = np.product(include, axis=-1).astype(bool)
        # dilate the mask to fill gaps and smooth the outline
        if dilate_iters > 0:
            self.mask = ndimage.morphology.binary_dilation(
                self.mask,
                iterations=dilate_iters).astype(bool)
        keyed = self.image.copy()
        keyed[self.mask == False] = [0, 0, 0]
        return keyed

    def onselect(self, eclick, erelease):
        """Update image based on rectangle between eclick and erelease."""
        # get the region of the image within the selection box
        self.select = self.image[
            int(eclick.ydata):int(erelease.ydata),
            int(eclick.xdata):int(erelease.xdata)]
        # get the hsv values for that region
        self.select_hsv = self.image_hsv[
            int(eclick.ydata):int(erelease.ydata),
            int(eclick.xdata):int(erelease.xdata)]
        # if a nontrivial region is selected:
        if self.select.shape[0] != 0 and self.select.shape[1] != 0:
            # assume we want the mean +- 3 standard devations of the selection
            means = self.select_hsv.mean((0, 1))
            standard_dev = self.select_hsv.std((0, 1))
            # use circular statistics for the hues
            h_mean = stats.circmean(self.select_hsv[..., 0].flatten(), 0, 1)
            h_std = stats.circstd(self.select_hsv[..., 0].flatten(), 0, 1)
            means[0], standard_dev[0] = h_mean, h_std
            # define the color range based on means +- 3 standard deviations
            self.color_range = np.array([
                means-3*standard_dev, means+3*standard_dev])
            self.masked_image = self.select_color()
            # update the plots
            self.masked_im.set_array(self.masked_image.astype('uint8'))
            self.plot_color_stats()
            self.fig.canvas.draw()

    def toggle_selector(self, event):
        """Keyboard shortcuts to close the window and toggle the selector."""
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and self.RS.active:
            matplotlib.pyplot.close()
        if event.key in ['A', 'a'] and not self.RS.active:
            print(' RectangleSelector activated.')
            self.RS.set_active(True)

    def get_axvspan(self, x1, x2):
        """Get corners for updating the axvspans."""
        return np.array([
            [x1, 0.],
            [x1, 1.],
            [x2, 1.],
            [x2, 0.],
            [x1, 0.]])

    def displaying(self):
        """True if the GUI is currently displayed."""
        return matplotlib.pyplot.fignum_exists(self.fig.number)

    def start_up(self):
        """Run when ready to display."""
        # from matplotlib.widgets import RectangleSelector
        self.RS = matplotlib.widgets.RectangleSelector(
            self.original_image_ax, self.onselect, drawtype="box")
        matplotlib.pyplot.connect('key_press_event', self.toggle_selector)
        matplotlib.pyplot.show()


class LSqEllipse():
    """Fits an ellipse to the 2D outline of points.


    From:
    @software{ben_hammel_2020_3723294,
          author       = {Ben Hammel and Nick Sullivan-Molina},
          title        = {bdhammel/least-squares-ellipse-fitting: v2.0.0},
          month        = mar,
          year         = 2020,
          publisher    = {Zenodo},
          version      = {v2.0.0},
          doi          = {10.5281/zenodo.3723294},
          url          = {https://doi.org/10.5281/zenodo.3723294}
        }
    """
    def fit(self, data):
        """Lest Squares fitting algorithm

        Theory taken from (*)
        Solving equation Sa=lCa. with a = |a b c d f g> and a1 = |a b c>
            a2 = |d f g>

        Args
        ----
        data (list:list:float): list of two lists containing the x and y data of the
            ellipse. of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]

        Returns
        ------
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
        """
        x, y = np.asarray(data, dtype=float)

        # Quadratic part of design matrix [eqn. 15] from (*)
        D1 = np.mat(np.vstack([x**2, x*y, y**2])).T
        # Linear part of design matrix [eqn. 16] from (*)
        D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T

        # forming scatter matrix [eqn. 17] from (*)
        S1 = D1.T*D1
        S2 = D1.T*D2
        S3 = D2.T*D2

        # Constraint matrix [eqn. 18]
        C1 = np.mat('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')

        # Reduced scatter matrix [eqn. 29]
        M = C1.I*(S1-S2*S3.I*S2.T)

        # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this equation [eqn. 28]
        eval, evec = np.linalg.eig(M)

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4*np.multiply(evec[0, :], evec[2, :]) - \
            np.power(evec[1, :], 2)
        a1 = evec[:, np.nonzero(cond.A > 0)[1]]

        # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -S3.I*S2.T*a1

        # eigenvectors |a b c d f g>
        self.coef = np.vstack([a1, a2])
        self._save_parameters()

    def _save_parameters(self):
        """finds the important parameters of the fitted ellipse

        Theory taken form http://mathworld.wolfram

        Args
        -----
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g

        Returns
        _______
        center (List): of the form [x0, y0]
        width (float): major axis
        height (float): minor axis
        phi (float): rotation of major axis form the x-axis in radians
        """

        # eigenvectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 [eqn. 15) from (**) or (***)
        a = self.coef[0, 0]
        b = self.coef[1, 0]/2.
        c = self.coef[2, 0]
        d = self.coef[3, 0]/2.
        f = self.coef[4, 0]/2.
        g = self.coef[5, 0]

        # finding center of ellipse [eqn.19 and 20] from (**)
        x0 = (c*d-b*f)/(b**2.-a*c)
        y0 = (a*f-b*d)/(b**2.-a*c)

        # Find the semi-axes lengths [eqn. 21 and 22] from (**)
        numerator = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        denominator1 = (b*b-a*c) * \
            ((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        denominator2 = (b*b-a*c) * \
            ((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        width = np.sqrt(numerator/denominator1)
        height = np.sqrt(numerator/denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from (**)
        # or [eqn. 26] from (***).
        phi = .5*np.arctan((2.*b)/(a-c))

        self._center = [x0, y0]
        self._width = width
        self._height = height
        self._phi = phi

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def phi(self):
        """angle of counterclockwise rotation of major-axis of ellipse to x-axis
        [eqn. 23] from (**)
        """
        return self._phi

    def parameters(self):
        return self.center, self.width, self.height, self.phi


class Layer():
    """An image loaded from file or numpy array.


    Attributes
    ----------
    filename : str, default=None
        Path to the image file.
    arr : array_like, default=None
        Input image as a 2D array.
    bw : bool, default=False
        Whether the image should be treated as greyscale.
    image : np.ndarray
        2D array of the image.
    grad : np.ndarray
        2D gradient magnitude of the image. Corresponds to local sharpness.
    color_selector : ColorSelector
        Matplotlib GUI generating a silhouetting mask based on user input.
    mask : np.ndarray
        2D boolean mask indicating the pixels consisting of the eye.

    Methods
    -------
    load()
        Load image if not loaded yet.
    get_gradient(smooth=0)
        Measure the relative focus of each pixel using numpy.gradient.
    color_key(hue_only=False)
        Generate a 2D boolean sillouetting mask based on user input.
    load_mask(mask_fn=None, mask_arr=None)
        Generate a 2D boolean sillouetting mask based on an image file.    
    """
    def __init__(self, filename=None, arr=None, bw=False):
        """Initialize for processing single images.


        Parameters
        ----------
        filename : str, default=None
            Path to the image file.
        arr : array_like, default=None
            Input image as a 2D array.
        bw : bool, default=False
            Whether the image is greyscale.

        Returns
        -------
        out : Layer
              An general image object with various methods for image processing.
        """
        self.filename = filename # the image can be loaded from a file
        self.image = arr           # or it can be stored directly
        self.bw = bw
        self.gradient = None
        self.color_selector = None
        self.mask = None

    def load(self):
        """Load image using PIL.


        Returns
        -------
        self.image : np.ndarray
            The loaded or directly specified image.
        """
        # load from input array as ndarray
        if self.image is not None:
            if not isinstance(self.image, np.ndarray):
                self.image = np.asarray(self.image)
            # check assumptions:
            assert self.image.ndim > 1, (
                "Input array should be at least 2D")
        # load from file
        if self.image is None and self.filename is not None:
            assert isinstance(self.filename, str), (
                "Input filename should be a string.")
            self.image = np.asarray(PIL.Image.open(self.filename))
        # assume greyscale if there's no color channel
        if self.image.ndim == 2:
            self.bw = True
        # if there's a color channel:
        if self.image.ndim == 3:
            # and if the color channel is unneccessary, remove it
            if self.image.shape[-1] == 1:
                self.image = np.squeeze(self.image)
            # and if the color channel has more than 3 values, use first 3
            elif self.image.shape[-1] > 3:
                self.image = self.image[..., :-1]
        # if all three channels are equivalent, use just one bw image
        if (self.image[..., 0] == self.image.mean(-1)).mean() == 1:
            self.image = self.image[..., 0]
            self.bw = True
        # save a greyscale version of all images, bw or not
        if self.bw:
            self.image_bw = self.image.astype('uint8')
        else:
            self.image_bw = rgb_2_gray(self.image.astype('uint8'))
        return self.image

    def load_memmap(self, filename=None):
        """Load image and store as a numpy memmap, deleting the local copy.


        Returns
        -------
        self.image : np.memmap
            The loaded or directly specified image stored to memory.
        """
        self.load()
        # if filename is not specified, use a temporary file
        if self.filename is None and filename is None:
            memmap_fn = os.path.join(mkdtemp(), 'temp_img.memmap')
        # otherwise, use the filename
        else:
            if filename is not None:
                file_ext = "." + filename.split(".")[-1]
                memmap_fn = filename.replace(file_ext, ".memmap")
            elif self.filename is not None:
                file_ext = "." + self.filename.split(".")[-1]
                memmap_fn = self.filename.replace(file_ext, ".memmap")
        if os.path.exists(memmap_fn):
            # load
            memmap = np.memmap(memmap_fn, mode='r+', shape=self.image.shape)
        else:
            # make the memmap and store it
            memmap = np.memmap(
                memmap_fn, dtype='uint8', mode='w+', shape=self.image.shape)
            memmap[:] = self.image[:]
        self.image = memmap
        

    def save(self, pickle_fn):
        """Save using pickle.


        Parameters
        ----------
        pickle_fn : str
            Filename of the pickle file to save.
        """
        self.pickle_fn = pickle_fn
        with open(pickle_fn, "wb") as pickle_file:
            pickle.dump(self, pickle_file)

    def get_gradient(self, smooth=0):
        """Measure the relative focus of each pixel using numpy.gradient.


        Parameters
        ----------
        smooth : float, default=0
            standard devation of 2D gaussian filter applied to the gradient.

        Returns
        -------
        self.gradient : np.ndarray
            2D array of the magnitude of the gradient image.
        """
        assert self.image is not None, (
            f"No image loaded. Try running {self.load} or {self.load_memmap}")
        # grab image
        if not self.bw:
            gray = rgb_2_gray(self.image)
        else:
            gray = self.image
        # use numpy gradient tool
        grad_0 = np.gradient(gray, axis=0)
        grad_1 = np.gradient(gray, axis=1)
        self.gradient = np.linalg.norm(np.array([grad_0, grad_1]), axis=0)
        # if there's a smoothing factor, apply gaussian filter
        if smooth > 0:
            self.gradient = ndimage.filters.gaussian_filter(self.gradient, sigma=smooth)
        return self.gradient

    def color_key(self, hue_only=False):
        """Use ColorSelector to apply a mask based on color statistics.
        

        Parameters
        ----------
        hue_only : bool
            Whehter to yse only the hue channel of the images.

        Returns
        -------
        self.mask : np.ndarray
            2D array of the sillouetting mask.
        """
        if self.image is None:  # load image
            self.load()
        # initialize the GUI for selecting color information
        self.color_selector = ColorSelector(
            self.image, bw=self.bw, hue_only=hue_only)
        self.color_selector.start_up()
        self.mask = self.color_selector.mask # save as attribute
        return self.mask

    def load_mask(self, mask_fn=None, mask_arr=None):
        """Load a 2D sillhouetting mask from an image file or array.

        
        If the image isn't boolean, we assume pixels > mean == True. 
        You can either load from an image file or directly as an array.

        Parameters
        ----------
        mask_fn : str, default=None
            Path to the masking image file. 
        mask_arr : array_like bool, default=None
            2D boolean masking array. 
        """
        # load the mask using a temporary Layer instance
        if mask_fn is not None:
            assert isinstance(mask_fn, str), (
                "Input mask_fn should be a string.")
            if os.path.exists(mask_fn):
                layer = Layer(mask_fn, bw=True)
                self.mask = layer.load()
        # or load directly as a numpy array
        elif mask_arr is not None:
            self.mask = np.asarray(mask_arr)
            assert self.mask.ndim > 1, (
                "Input mask_arr should be at least 2D")
        # if it loaded properly:
        if self.mask is not None:
            # and its not a boolean array, threshold using the mean value 
            if self.mask.dtype is not np.dtype('bool'):
                # self.mask = self.mask > self.mask.mean()
                self.mask = self.mask > self.mask.max()/2
            # assume the mask matches the shape of the image
            assert self.mask.shape == self.image.shape[:2], (
                "input mask should have the same shape as input image. "
                f"input shape = {self.mask.shape}, image shape = {self.image.shape[:2]}")
            # assume the mask isn't empty
            assert self.mask.mean() > 0, "input mask is empty"


class Eye(Layer):
    """A class specifically for processing images of compound eyes. 
    
    
    Could be modified for other eyes (square lattice, for instance). Input 
    mask should be either a boolean array where True => points included in
    the eye or a filename pointing to such an array.

    Attributes
    ----------
    pixel_size : float
        Actual length of the side of each pixel.
    mask_fn : str
        Path to the image of boolean mask.
    mask_arr : array_like
        Image of the boolean mask.
    pickle_fn : str
        Path to the pickle file for loading a previously saved Eye object.
    eye_contour : np.ndarray
        2D coordinates of N points on the eye contour with shape N x 2.
    eye_mask : np.ndarray
        2D masking image of the eye smoothed and filled.
    ellipse : LSqEllipse
        Ellipse object with properties like center, width, and height.
    filtered_image : np.ndarray
        The filtered image made by inverse transforming the filtered 2D fft.
    ommatidial_diameter_fft : float
        The average wavelength of the fundamental frequencies, 
        corresponding to the ommatidial diameters.
    ommatidial_inds : np.ndarray
        2D indices of the N ommatidia with shape N x 2.
    ommatidia : np.ndarray
        2D coordinates of N ommatidia with shape N x 2.
    
    Methods
    -------
    get_eye_outline(hue_only=False, smooth_factor=11)
        Get the outline of the eye based on an eye mask.
    get_eye_dimensions(display=False)
        Assuming an elliptical eye, get length, width, and area.
    crop_eye(padding=1.05, use_ellipse_fit=False)
        Crop the image so that the frame is filled by the eye with padding.
    get_ommatidia(bright_peak=True, min_count=500, max_count=50000, 
        fft_smoothing=5, square_lattice=False, high_pass=False)
        Detect ommatidia coordinates assuming hex or square lattice.
    measure_ommatidia(num_neighbors=3, sample_size=100)
        Measure ommatidial diameter using the ommatidia coordinates.
    ommatidia_detecting_algorithm(bright_peak=True, fft_smoothing=5, 
        square_lattice=False, high_pass=False, num_neighbors=3, 
        sample_size=100, plot=False, plot_fn=None)
        The complete algorithm for measuring ommatidia in images.
    """
    def __init__(self, filename=None, arr=None, bw=False, pixel_size=1,
                 mask_fn=None, mask_arr=None):
        """Initialize the eye object, which is a child class of Layer.
        

        Parameters
        ----------
        filename : str
            The file path to the eye image.
        bw : bool
            Whether the image in greyscale.
        pixel_size : float, default = 1
            The actual length of the side of one pixel.
        mask_fn : str, default = "mask.jpg"
            The path to the sillhouetting mask image file.
        mask : array_like, default = None
            Boolean masking image with the same shape as the input image array.
        """
        Layer.__init__(self, filename=filename, arr=arr,
                       bw=bw) # initialize parent class
        self.eye_contour = None
        self.ellipse = None
        self.ommatidia = None
        self.pixel_size = pixel_size
        self.mask_fn = mask_fn
        self.mask_arr = mask_arr
        self.load()
        self.pickle_fn = None
        self.load_mask(mask_fn=self.mask_fn, mask_arr=self.mask_arr)
        self.oda = self.ommatidia_detecting_algorithm

    def get_eye_outline(self, hue_only=False, smooth_factor=11):
        """Get the outline of the eye based on an eye mask.


        Parameters
        ----------
        hue_only : bool, default=False
            Whether to filter using only the hue values.
        smooth_factor : int, default=11
            Size of 2D median filter to smooth outline. smooth_factor=0 -> 
            no smoothing.

        Attributes
        ----------
        eye_outline : np.ndarray
            2D coordinates of N points on the eye contour with shape N x 2.
        eye_mask : np.ndarray
            2D masking image of the eye smoothed and filled.
        """
        assert self.mask is not None, (
            f"No boolean mask loaded. First try running {self.load_mask}")
        # add border of zeros to find contour even in all True image
        mask = np.zeros(np.array(self.mask.shape)+2, dtype=bool)
        mask[1:-1, 1:-1] = self.mask
        # find the contours in the mask image
        contour = skimage.measure.find_contours(
            (255/mask.max()) * mask.astype(int), 256/2)
        # escape if no contours were found
        if len(contour) == 0:
            breakpoint()
        assert len(contour) > 0, "could not find enough points in the contour"
        # use the longest contour
        contour = max(contour, key=len).astype(float)
        # recenter points
        contour -= 1
        # contour -= contour.min()
        contour = np.round(contour).astype(int)
        self.eye_outline = contour # pixel coords
        # make a new mask by filling the contour
        new_mask = np.zeros(self.mask.shape, dtype=int)
        new_mask[contour[:, 0], contour[:, 1]] = 1
        ndimage.binary_fill_holes(new_mask, output=new_mask)
        # smooth the shape by applying a rolling median 2D window
        if smooth_factor > 0:
            new_mask = signal.medfilt2d(
                new_mask.astype('uint8'), smooth_factor).astype(bool)
        self.eye_mask = new_mask

    def get_eye_dimensions(self, display=False):
        """Assuming an elliptical eye, get its length, width, and area.


        Parameters
        ----------
        display : bool, default=False
            Whether to plot the eye with the ellipse superimposed.

        Attributes
        ----------
        ellipse : LSqEllipse
            Ellipse class that uses OLS to fit an ellipse to contour data.
        eye_length : float
            Major diameter of the fitted ellipse.
        eye_width : float
            Minor diameter of the fitted ellipse.
        eye_area : float
            Area of the fitted ellipse
        """
        # check that there is an eye contour
        assert self.eye_outline is not None, f"first run {self.get_eye_outline}"
        # fit an ellipse to the contour using OLS
        least_sqr_ellipse = LSqEllipse()
        least_sqr_ellipse.fit(self.eye_outline.T)
        self.ellipse = least_sqr_ellipse
        # store the eye center, width, and height based on the fitted ellipse
        center, width, height, phi = self.ellipse.parameters()
        self.eye_length = 2 * self.pixel_size * max(width, height)
        self.eye_width = 2 * self.pixel_size * min(width, height)
        self.eye_area = np.pi * self.eye_length / 2 * self.eye_width / 2
        # if display selected, plot the image with superimposed eye contour
        if display:
            plt.imshow(self.image)
            plt.plot(self.eye_outline[:, 1], self.eye_outline[:, 0])
            plt.show()

    def crop_eye(self, padding=1.05, use_ellipse_fit=False):
        """Crop the image so that the frame is filled by the eye with padding.


        Parameters
        ----------
        padding : float, default=1.05
            Proportion of the length of the eye to include in width and height.
        use_ellipse_fit : bool, default=False
            Whether to use the fitted ellipse to mask the eye.

        Returns
        -------
        self.eye : Eye
            A cropped Eye using the boolean mask.
        """
        out = np.copy(self.image)
        # if we assume the eye outline is an ellipse:
        if use_ellipse_fit:
            # fit an ellipse using OLS
            least_sqr_ellipse = LSqEllipse()
            least_sqr_ellipse.fit(self.eye_outline.T)
            self.ellipse = least_sqr_ellipse
            # get relevant properties of the ellipse
            (x, y), width, height, ang = self.ellipse.parameters()
            self.angle = ang
            w = padding*width
            h = padding*height
            # get pixel coordinates of the ellipse
            # eye_mask_ys and eye_mask_xs used to be .cc and .rr
            ys, xs = Ellipse(
                x, y, w, h, shape=self.image.shape[:2], rotation=ang)
            # use the ellipse as the eye mask
            new_mask = self.mask[min(ys):max(ys), min(xs):max(xs)]
            # generate an eye object using the cropped image
            self.eye = Eye(arr=out[min(ys):max(ys), min(xs):max(xs)],
                           mask_arr=new_mask,
                           pixel_size=self.pixel_size)
        # or just use the exact mask with some padding:
        else:
            xs, ys = np.where(self.mask)
            minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
            minx -= padding / 2
            miny -= padding / 2
            maxx += padding / 2
            maxy += padding / 2
            minx, maxx, miny, maxy = int(round(minx)), int(round(
                maxx)), int(round(miny)), int(round(maxy))
            new_mask = self.mask[minx:maxx, miny:maxy]
            # generate an eye object using the cropped image
            self.eye = Eye(arr=out[minx:maxx, miny:maxy],
                           mask_arr=new_mask,
                           pixel_size=self.pixel_size)
        return self.eye

    def get_ommatidia(self, bright_peak=True, fft_smoothing=5,
                      square_lattice=False, high_pass=False, regular=True):
        """Detect ommatidia coordinates assuming hex or square lattice.


        Use the ommatidia detecting algorithm (ODA) to find the center of
        ommatidia assuming they are arranged in a hexagonal lattice. Note: 
        This can be computationally intensive on larger images so we suggest 
        cropping out irrelevant regions via self.crop_eye().

        Parameters
        ----------
        bright_peak : bool, default=True
            Whether the ommatidia are defined by brighter (vs. darker) peaks.
        fft_smoothing : int, default=5
            The standard deviation of a 2D gaussian filter applied to the 
            reciprocal image before finding peaks.
        square_lattice : bool, default=False
            Whether this a square (rather than a hexagonal) lattice.
        high_pass : bool, default=False
            Whether to also filter frequencies below the fundamental one.
        regular : bool, default=False
            Whether to assume the ommatidial lattice is approximately regular.
        
        Atributes
        ---------
        __freqs :  np.ndarray
            2D image of spatial frequencies corresponding to the reciprocal 
            space of the 2D FFT.
        __orientations : np.ndarray
            2D image of spatial orientations corresponding to the reciprocal 
            space of the 2D FFT.
        __fundamental_frequencies : float
            The set of spatial frequencies determined by the peak frequencies 
            in the reciprocal image.
        __upper_bound : float
            The threshold frequency used in the low-pass filter = 1.25 * 
            max(self.fundamental_frequencies)
        __low_pass_filter : np.ndarray, dtype=bool
            2D boolean mask used as a low-pass filter on the reciprocal image.
        __fft_shifted : np.ndarray, dtype=complex
            The filtered 2D FFT of the image with low frequencies shifted to 
            the center.
        __fft : np.ndarray, dtype=complex
            The filtered 2D FFT of the image.
        filtered_image : np.ndarray
            The filtered image made by inverse transforming the filtered 2D fft.
        ommatidial_diameter_fft : float
            The average wavelength of the fundamental frequencies, 
            corresponding to the ommatidial diameters.
        ommatidial_inds : np.ndarray
            2D indices of the N ommatidia with shape N x 2.
        ommatidia : np.ndarray
            2D coordinates of N ommatidia with shape N x 2.
        reciprocal : np.ndarray
            2D reciprocal image of self.image, correcting for the natural 
            1/(f^2) distribution of spatial frequencies and the low horizontal
            and vertical spatial frequencies corresponding to the vertical
            and horizontal boundaries.
        """
        assert self.eye_outline is not None, (
            f"first run {self.get_eye_dimensions}")
        # get the reciprocal image using the FFT
        # first, get the 2D FFT
        image_bw_centered = self.image_bw - self.image_bw.mean()
        height, width = image_bw_centered.shape
        # todo: try adding a gaussian window
        window_h = signal.windows.gaussian(height, height/3)
        window_w = signal.windows.gaussian(width, height/3)
        window = window_w[np.newaxis, :] * window_h[:, np.newaxis]
        image_windowed = image_bw_centered * window
        fft = np.fft.fft2(image_windowed)
        # fft = np.fft.fft2(image_bw_centered)
        # shift frequencies so that low frequencies are central and high
        # frequencies are peripheral
        fft_shifted = np.fft.fftshift(fft)
        # calculate reciprocal frequencies using their distance to the center
        xfreqs = np.fft.fftfreq(self.image_bw.shape[1], self.pixel_size)
        yfreqs = np.fft.fftfreq(self.image_bw.shape[0], self.pixel_size)
        xgrid, ygrid = np.meshgrid(xfreqs, yfreqs)
        self.__freqs = np.array([xgrid, ygrid])
        self.__freqs = np.array(self.__freqs, dtype=float)
        self.__freqs = np.fft.fftshift(self.__freqs)
        self.__xfreqs = np.fft.fftshift(xfreqs)
        self.__yfreqs = np.fft.fftshift(yfreqs)
        # calculate grating orientations for the reciprocal image
        self.__orientations = np.arctan2(self.__freqs[1], self.__freqs[0])
        self.__freqs = np.linalg.norm(self.__freqs, axis=0)
        i = self.__orientations < 0    # indices of the negative half
        self.__orientations[i] = self.__orientations[i] + np.pi # make positive
        # different method for regular and irregular lattices
        # the reciprocal image is the magnitude of the frequency shifted 2D FFT
        self.reciprocal = abs(fft_shifted)
        height, width = self.reciprocal.shape
        # instead of blurring, just use the autocorrelation
        if regular:
            self.reciprocal = signal.correlate(
                self.reciprocal, self.reciprocal, mode='same', method='fft')
            # find the peaks of the upper half of the reciprocal image
            pos_freqs = self._Eye__freqs > 0
            thresh = 2 * self._Eye__freqs[pos_freqs].min()
            # # TODO: non-linearly fit lattice to peaks instead of finding maxima
            # lattice_fit = LatticeFit(self.reciprocal, self.__xfreqs, self.__yfreqs)
            # lattice_fit.error()
            # breakpoint()
            # peaks = peak_local_max(
            #     self.reciprocal[:round(height/2)], num_peaks=4, min_distance=3)
            peaks = peak_local_max(
                self.reciprocal, num_peaks=10, min_distance=3)
            ys, xs = peaks.T
            key_freqs = self._Eye__freqs[ys, xs]
            # the center is distinct in that it has a frequency ~ 0 and isn't duplicated
            thresh = 3 * np.unique(self._Eye__freqs)[1]
            include = key_freqs > thresh
            key_freqs = key_freqs[include]  # omit very small freqs
            # remove those without duplicates
            key_freq_set, counts = np.unique(key_freqs, return_counts=True)
            include = counts > 1
            key_freqs = key_freq_set[include]
            # use those with highest values
            i = np.argsort(key_freqs)
            if square_lattice:
                # if square, use the 2 fundamental frequencies
                self.__fundamental_frequencies = key_freqs[i][:2]
            else:
                # if hexagonal, use the 3 fundamental frequencies
                self.__fundamental_frequencies = key_freqs[i][:3]
        else:
            # for irregular lattices, we can collapse the 2D reciprocal image
            # into a 1D frequency response curve. The fundamental frequency
            # is just the second peak in this curve
            # product = self.reciprocal * self._Eye__freqs  # normalized by frequency
            product = np.log(self.reciprocal)
            # use frequency values to find mean power function
            freqs = self._Eye__freqs.flatten()
            power_normalized = product.flatten()
            # for each frequency
            hist, xvals, yvals = np.histogram2d(freqs, power_normalized, bins=50)
            # get middle points of bin values for plotting point estimates
            xs = (xvals[:-1] + xvals[:-1])/2
            ys = (yvals[:-1] + yvals[:-1])/2
            # iterate through each column and measure weighted mean of power
            weighted_means = []
            for col in hist:
                weighted_means += [sum(col * ys / col.sum())]
            weighted_means = np.array(weighted_means)
            # plt.pcolormesh(xvals, yvals, hist.T, cmap='Greys')
            # plt.plot(xs, weighted_means)
            # find the frequency corresponding to the maximum power
            xmax = peak_local_max(weighted_means, num_peaks=1)
            if len(xmax) > 0:
                xmax = xmax[0][0]
                peak_frequency = xs[xmax]
                self.__fundamental_frequencies = np.array([peak_frequency])
            else:
                self.__fundamental_frequencies = np.array([])
        # use a minimum distance based on the wavelength of the
        # fundamental grating
        self.ommatidial_diameter_fft = 1 / self.__fundamental_frequencies.mean()
        dist = self.ommatidial_diameter_fft / self.pixel_size
        if len(self.__fundamental_frequencies) > 0 and (dist > 2):
            # set the upper bound as halfway between the fundamental frequency
            # and the next harmonic
            self.__upper_bound = 1.25 * self.__fundamental_frequencies.max()
            # make a 2D image to filter only frequencies less than the upper bound
            in_range = self.__freqs < self.__upper_bound
            self.__low_pass_filter = np.ones(self.__freqs.shape)
            self.__low_pass_filter[in_range == False] = 0
            # if we also want to apply a high pass filter:
            if high_pass:
                in_range = self.__freqs < .75 * self.__fundamental_frequencies.min()
                self.__low_pass_filter[in_range] = 0
            # apply the low pass filter and then invert back to the filtered image
            self.__fft_shifted = np.zeros(fft.shape, dtype=complex)
            self.__fft_shifted[:] = fft_shifted * self.__low_pass_filter
            self.__fft = np.fft.ifftshift(self.__fft_shifted)
            self.filtered_image = np.fft.ifft2(self.__fft).real
            smooth_surface = self.filtered_image
            if not bright_peak:
                smooth_surface = smooth_surface.max() - smooth_surface
            self.ommatidial_inds = peak_local_max(
                smooth_surface, min_distance=int(round(dist/4)),
                exclude_border=False, threshold_abs=1)
            # remove points outside of the mask
            ys, xs = self.ommatidial_inds.T
            self.ommatidial_inds = self.ommatidial_inds[self.mask[ys, xs]]
            if len(self.ommatidial_inds) > 5000:
                breakpoint()
            # store ommatidia coordinates in terms of the pixel size
            self.ommatidia = self.ommatidial_inds * self.pixel_size
        else:
            print("Failed to find fundamental frequencies.")
            empty_img = np.copy(self.image)
            empty_img[:] = empty_img.mean().astype('uint8')
            self.filtered_image = np.copy(empty_img)
            self.ommatidial_diameter_fft = np.nan
            self.ommatidial_inds = np.array([])
            self.ommatidia = np.array([])
            

    def measure_ommatidia(self, num_neighbors=3, sample_size=100):
        """Measure ommatidial diameter using the ommatidia coordinates.


        Once the ommatidia coordinates are measured, we can measure ommatidial
        diameter given the expected number of neighbors

        Parameters
        ----------
        num_neighbors : int, default=6
            The number of neighbors to check for measuring the ommatidial 
            diameter. Defaults to 6, assuming a hexagonal lattice.
        sample_size : int, default=100
            The number of ommatidia near the center to include in diameter
            estimation.

        Atributes
        ---------
        __ommatidial_dists_tree : scipy.spatial.kdtree.KDTree
            K-dimensional tree for efficiently taking distance measurements.
        __ommatidial_dists : np.ndarray
            N X num_neighbors array of the distance to neighboring ommatidia.
        ommatidial_diameters : np.ndarray
            1-D array of average diameter per ommatidium.
        ommatidial_diameter : float
            Average ommatidial diameter of sample near the mask center of mass.
        ommatidial_diameter_SD : float
            Standard deviation of ommatidial diameters in sample.
        """
        assert self.ommatidia is not None, (
            f"first run {self.get_ommatidia}")
        if len(self.ommatidia) > 0:
            # make a k-dimensional tree
            self.__ommatidial_dists_tree = spatial.KDTree(self.ommatidia)
            # find the set of nearest neighbors
            self.__ommatidial_dists , inds = self.__ommatidial_dists_tree.query(
                self.ommatidia, k=num_neighbors+1)
            self.ommatidial_diameters = self.__ommatidial_dists[:, 1:].mean(1)
            # use ommatidia center of mass to grab ommatidia near the center
            com = self.ommatidia.mean(0)
            near_dists, near_center = self.__ommatidial_dists_tree.query(
                com, k=sample_size)
            near_dists, near_center = self._Eye__ommatidial_dists_tree.query(
                com, k=sample_size)
            near_center = near_center[near_dists < np.inf]
            # store the mean and standard deviation of the sample
            self.ommatidial_diameter = self.ommatidial_diameters[near_center].mean()
            self.ommatidial_diameter_SD = self.ommatidial_diameters[near_center].std()
        else:
            self.ommatidial_diameter = np.nan
            self.ommatidial_diameter_SD = np.nan

    def ommatidia_detecting_algorithm(self, bright_peak=True, fft_smoothing=5,
                                      square_lattice=False, high_pass=False,
                                      num_neighbors=3, sample_size=100,
                                      plot=False, plot_fn=None, regular=True):
        """The complete algorithm for measuring ommatidia in images.

        
        Parameters
        ----------
        bright_peak : bool, default=True
            Whether the ommatidia are defined by brighter (vs. darker) peaks.
        fft_smoothing : int, default=5
            The standard deviation of a 2D gaussian filter applied to the 
            reciprocal image before finding peaks.
        square_lattice : bool, default=False
            Whether this a square (rather than a hexagonal) lattice.
        high_pass : bool, default=False
            Whether to also filter frequencies below the fundamental one.
        num_neighbors : int, default=6
            The number of neighbors to check for measuring the ommatidial 
            diameter. Defaults to 6, assuming a hexagonal lattice.
        sample_size : int, default=100
            The number of ommatidia near the center to include in diameter
            estimation.
        plot : bool, default=False
            Whether to plot the eye with ommatidia and diameters superimposed.
        plot_fn : str, default=None
            Filename to save the plotted eye with superimposed ommatidia and 
            their diameters.
        regular : bool, default=False
            Whether to assume the ommatidial lattice is approximately regular.

        Attributes
        ----------
        eye_length : float
            Major diameter of the fitted ellipse.
        eye_width : float
            Minor diameter of the fitted ellipse.
        eye_area : float
            Area of the fitted ellipse
        ommatidia : np.ndarray
            2D coordinates of N ommatidia with shape N x 2.
        ommatidial_diameter_fft : float
            The average wavelength of the fundamental frequencies, 
            corresponding to the ommatidial diameters.
        ommatidial_diameter : float
            Average ommatidial diameter of sample near the mask center of mass.
        ommatidial_diameter_SD : float
            Standard deviation of ommatidial diameters in sample.
        """
        # first, approximate the eye outline using the loaded mask
        self.get_eye_outline()
        # then, calculate length, width, and area of elliptical eye
        self.get_eye_dimensions()
        area, length, width = np.round(
            [self.eye_area, self.eye_length, self.eye_width], 3)
        # print key eye dimensions
        print(f"Eye: \tArea = {area}\tLength = {length}\tWidth = {width}")
        # finally, locate ommatidia using the FFT
        self.get_ommatidia(bright_peak=bright_peak,
                           fft_smoothing=fft_smoothing,
                           square_lattice=square_lattice,
                           high_pass=high_pass, regular=regular)
        # and measure ommatidial diameters
        self.measure_ommatidia(num_neighbors=num_neighbors,
                               sample_size=sample_size)
        # print key ommatidia parameters
        count = len(self.ommatidia)
        sample_diameter = np.round(self.ommatidial_diameter, 4)
        sample_std = np.round(self.ommatidial_diameter_SD, 4)
        fft_diameter = np.round(self.ommatidial_diameter_fft, 4)
        print(
            f"Ommatidia: \tN={count}\tmean={sample_diameter}"
            f"\tstd={sample_std}\tfft={fft_diameter}")
        print()
        if plot or plot_fn is not None:
            fig = plt.figure()
            # use a gridpec to specify a wide image and thin colorbar
            gridspec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[9, 1])
            img_ax = fig.add_subplot(gridspec[0, 0])
            colorbar_ax = fig.add_subplot(gridspec[0, 1])
            # plot the eye image with ommatidia superimposed
            img_ax.imshow(self.image_bw, cmap='gray',
                          vmin=0, vmax=np.iinfo(self.image.dtype).max)
            # img_ax.scatter(xs, ys, marker='.', c=colorvals, s=.5 * dot_areas,
            #                # vmin=colorvals.min(), vmax=colorvals.max(),
            #                vmin=vmin, vmax=vmax,
            #                cmap='plasma')
            img_ax.set_xticks([])
            img_ax.set_yticks([])
            # Keep dots the same size until we get this to work better
            if len(self.ommatidia) > 0:
                dot_radii = (self.ommatidial_diameters / (2 * self.pixel_size))
                dot_areas = np.pi * dot_radii ** 2
                colorvals = self.ommatidial_diameters
                vmin, vmax = np.percentile(colorvals, [.5, 99.5]) # use 99%
                ys, xs = self.ommatidial_inds.T
                img_ax.scatter(xs, ys, marker='.', c=colorvals,
                               # vmin=colorvals.min(), vmax=colorvals.max(),
                               vmin=vmin, vmax=vmax,
                               cmap='plasma')
                # crop image around the x and y coordinates, with .1 padding
                ys, xs = np.where(self.mask)
                width = xs.max() - xs.min()
                height = ys.max() - ys.min()
                xpad, ypad = .05 * width, .05 * height
                img_ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
                img_ax.set_ylim(ys.max() + ypad, ys.min() - ypad)
                # make the colorbar
                # colorbar_histogram(colorvals, colorvals.min(), colorvals.max(),
                if not any([np.isnan(vmin), np.isnan(vmax), np.isinf(vmin), np.isinf(vmax)]): 
                    colorbar_histogram(colorvals, vmin, vmax,
                                       ax=colorbar_ax, bin_number=25, colormap='plasma')
            colorbar_ax.set_ylabel(f"Ommatidial Diameter (N={len(self.ommatidia)})",
                                   rotation=270)
            colorbar_ax.get_yaxis().labelpad = 15
            fig.tight_layout()
            # if a filename is provided, save the figure
            if plot_fn is not None:
                plt.savefig(plot_fn)
            # if the plot option is set, display the figure
            if plot:
                plt.show()
            del fig


class Stack():
    """A stack of images at different depths for making a focus stack.


    Attributes
    ----------
    layers : list
        The list of image layers.
    layer_class : class, default=Layer
        The class to use for each layer.
    img_extension : str
        The file extension to use for importing images.
    fns : list
        The list of filenames included in the Stack.
    images : np.ndarray
        The 4D (image number X height X width X rgb) array of images.
    gradients : np.ndarray
        The 3D (image number X height X width) array of image gradients.
    heights : np.ndarray
        The 2D (height X width) array of image indices maximizing 
        gradient values.
    stack : np.ndarray
        The 2D (height X width X rgb) array of pixels maximizing gradient
        values.
    
    Methods
    -------
    load():
        Load the individual layers.
    get_average(num_samples=5):
        Grab unmoving 'background' by averaging over some layers.
    get_focus_stack(smooth_factor=0):
        Generate a focus stack accross the image layers.
    smooth(sigma)
        A 2d smoothing filter for the heights array.
    """

    def __init__(self, dirname="./", img_extension=".jpg", bw=False,
                 layer_class=Layer, pixel_size=1, depth_size=1):
        """Initialize a stack of images for making a focus stack.


        Parameters
        ----------
        dirname : str
            Path to the directory containing the images to load.
        img_extension : str
            The file extension of the images to load.
        bw : bool
            Whether the images are greyscale.
        pixel_size : float, default=1
            Actual length of the side of a pixel.
        depth_size : float, default=1
            Actual depth interval between individual layers.

        Attributes
        ----------
        layers : list
            The list of image layers.
        layer_class : Layer, Eye
            The class to use for each layer.
        img_extension : str
            The file extension to use for importing images.
        fns : list
            The list of filenames included in the Stack.
        """
        self.dirname = dirname
        self.pixel_size = pixel_size
        self.depth_size = depth_size
        # store the full path of filenames that match img_extension
        self.fns = os.listdir(self.dirname)
        self.fns = sorted([os.path.join(self.dirname, f) for f in self.fns])
        self.fns = [f for f in self.fns if f.endswith(img_extension)]
        self.layers = []
        self.bw = bw
        self.layer_class = layer_class
        self.img_extension = img_extension
        
    def load(self):
        """Load the individual layers."""
        print("loading images: ")
        self.layers = []
        for num, f in enumerate(self.fns):
            layer = self.layer_class(f, bw=self.bw)
            layer.load()
            self.layers += [layer]
            print_progress(num + 1, len(self.fns))
        print()

    def iter_layers(self):
        """Generator yielding Layers in order."""
        for fn in self.fns:
            layer = self.layer_class(fn, bw=self.bw)
            yield layer

    def load_memmaps(self):
        """Load the individual layers as memmaps to free up RAM."""
        print("loading images: ")
        self.layers = []
        for num, f in enumerate(self.fns):
            layer = self.layer_class(f, bw=self.bw)
            layer.load_memmap()
            self.layers += [layer]
            print_progress(num + 1, len(self.fns))
        print()
        
    def load_masks(self, mask_fn=None, mask_arr=None):
        """Load the masks using either their mask file or array.


        Parameters
        ----------
        mask_fn : str, default=None
            Filename of the mask image.
        arr : np.ndarray, default=None
            2D boolean masking array. 
        """
        print("loading masks: ")
        for num, layer in enumerate(self.layers):
            layer.load_mask(mask_fn=mask_fn, mask_arr=mask_arr)
            print_progress(num + 1, len(self.fns))
        print()

    def get_average(self, num_samples=5):
        """Grab unmoving 'background' by averaging over some layers.


        Parameters
        ----------
        num_samples : int, default=5
            Maximum number of samples to average over.
        """
        # use the first image for its shape
        first = self.layers[0].load_image()
        avg = np.zeros(first.shape, dtype=float)
        # use num_samples to calculate the interval size needed
        intervals = len(self.layers)/num_samples
        for layer in self.layers[::int(intervals)]:
            # load the images and add them to the  
            img = layer.load_image().astype(float)
            avg += img
            layer.image = None
        return (avg / num_samples).astype('uint8')

    def get_focus_stack(self, smooth_factor=0):
        """Generate a focus stack accross the image layers.
        

        Parameters
        ----------
        smooth_factor : float, default=0
            The standard deviation of the gaussian 2D filter applied to the 
            approximate heights.

        Attributes
        ----------
        images : np.ndarray
            The 4D (image number, height, width, rgb) array of images.
        gradients : np.ndarray
            The 3D (image number, height, width) array of image gradients.
        heights : np.ndarray
            The 2D (height, width) array of image indices maximizing 
            gradient values.
        stack : np.ndarray
            The 2D (height, width, rgb) array of pixels maximizing gradient
            values.
        """
        # assume the images have been imported
        assert len(self.layers) > 0, (
            f"Images have not been imported. Try running {self.load} or"
            f" {self.load_memmaps}.")
        # go through each layer and store its image and gradient
        first_image = self.layers[0].image
        # make empty arrays for making the focus stack
        self.stack = np.copy(first_image)
        self.max_gradients = np.zeros(
            (first_image.shape[0], first_image.shape[1]), dtype=float)
        self.height_indices = np.zeros(
            (first_image.shape[0], first_image.shape[1]), dtype=int)
        print("generating focus stack:")
        for num, layer in enumerate(self.layers):
            # get the image and gradient
            img = layer.image
            layer.get_gradient(smooth=smooth_factor)
            # find pixels of increased gradient values
            increases = np.greater_equal(layer.gradient, self.max_gradients)
            # replace max_gradients with pixel increases
            self.max_gradients[increases] = layer.gradient[increases]
            del layer.gradient
            self.height_indices[increases] = num
            print_progress(num + 1, len(self.layers))
        print()
        # smooth heights to eliminate suddent jumps in the surface
        if smooth_factor > 0:
            self.height_indices = np.round(ndimage.filters.gaussian_filter(
                self.height_indices, sigma=smooth_factor)).astype(int)
        # run through list of height indices, grabbing corresponding pixels
        for num, layer in enumerate(self.layers):
            # get pixels of maximum gradients
            include = self.height_indices == num
            self.stack[include] = layer.image[include]


        # get actual heights in units of distance
        self.heights = self.depth_size * np.copy(self.height_indices)

    def get_smooth_heights(self, sigma):
        """A 2d smoothing filter for the heights array.


        Parameters
        ----------
        sigma : int
            The standard deviation of the gaussian 2D filter applied to the 
            approximate heights.

        Returns
        -------
        new_heights : np.ndarray, shape=(height, width)
            The heights array smoothed using a fourier gaussian filter.
        """
        new_heights = self.heights.astype("float32")
        new_heights = np.fft.ifft2(
            ndimage.fourier_gaussian(
                np.fft.fft2(new_heights),
                sigma=sigma)).real
        return new_heights


class EyeStack(Stack):
    """A special stack for handling a focus stack of fly eye images.


    Attributes
    ----------
        eye : Eye, default=None
            Eye object created by using the focus stack.
        pixel_size : float, default=1
            The real length of the side of one pixel.
        depth_size : float, default=1
            The real distance between stack layers.
        eye_mask : array-like, default="mask.jpg"
            2D boolean masking array.
        ommatidia_polar : np.ndarray, default=None
            The ommatidia locations in spherical coordinates relative to 
            the best fitting sphere.
        fns : list
            The list of included filenames.
        sphere : SphereFit
            An OLS-fitted sphere to the 3D ommatidia coordinates. 
            Transforms the points into polar coordinates relative to 
            the fitted sphere.
        fov_hull : float
            The field of view of the convex hull of the ommatidia in
            polar coordinates.
        fov_long : float
            The longest angle of view using the long diameter of the
            ellipse fitted to ommatidia in polar coordinates.
        fov_short : float, steradians
            The shortest angle of view using the short diameter of 
            the ellipse fitted to ommatidia in polar coordinates.
        surface_area : float, steradians
            The surface area of the sphere region given fov_hull and
            sphere.radius.
        io_angles : np.ndarray, rads
            The approximate inter-ommatidial angles per ommatidium 
            using eye.ommatidial_diameters and eye radius in rads.
        io_angle : float, rad
            The average approximate inter-ommatidial angle using 
            eye.ommatidial_diameter / self.sphere.radius
        io_angle_fft : float, rad
            The average approximate inter-ommatidial angle using
            eye.ommatidial_diameter_fft / self.sphere.radius

    Methods
    -------
    crop_eyes():
        Crop each layer of the stack.
    get_eye_stack(smooth_factor=0):
        Generate focus stack of images and then crop out the eye.
    get_ommatidia(bright_peak=True, fft_smoothing=5,
        square_lattice=False, high_pass=False, num_neighbors=3,
        sample_size=100, plot=False, plot_fn=None):
        Use Eye object of the eye stack image to detect ommatidia.    
    oda_3d(eye_stack_smoothing=0, bright_peak=True, fft_smoothing=5,
        square_lattice=False, high_pass=False, num_neighbors=3,
        sample_size=100, plot=False, plot_fn=None, use_memmaps=False):
        Detect ommatidia using the 3D surface data.
    """

    def __init__(self, dirname, img_extension=".jpg", bw=False,
                 pixel_size=1, depth_size=1, mask_fn='mask.jpg',
                 mask_arr=None):
        """Import a directory of eye images at different depths.


        Parameters
        ----------
        img_extension : str, default=".jpg"
            The image file extension used to avoid unwanted images.
        bw : bool, default=False
            Whether to treat the image as grayscale.
        pixel_size : float, default=1
            The real length of the side of one pixel in the image. Used for
            converting from pixel into real distances.
        depth_size : float, default=1
            The real distance between stack layers. 
        mask_fn : str, default="mask.jpg"
            The filename of the boolean masking image.
        mask_arr : array-like, default=None
            2D boolean masking array.         
        
        Attributes
        ----------
        eye : Eye
            The Eye object of the focus stack of cropped image layers.
        pixel_size : float, default=1
            The real length of the side of one pixel.
        depth_size : float, default=1
            The real distance between stack layers.
        eye_mask : array-like, default="mask.jpg"
            2D boolean masking array.
        ommatidia_polar : np.ndarray, default=None
            The ommatidia locations in spherical coordinates relative to 
            the best fitting sphere.
        fns : list
            The list of included filenames.
        """
        Stack.__init__(self, dirname, img_extension, bw, layer_class=Eye)
        self.eye = None
        self.pixel_size = pixel_size
        self.depth_size = depth_size
        self.eye_mask_fn = mask_fn
        self.eye_mask = mask_arr
        self.ommatidia_polar = None
        # if mask file provided, remove from list of layer files
        if mask_fn is not None:
            if os.path.exists(mask_fn):
                new_fns = [fn for fn in self.fns if fn != mask_fn]
                self.fns = new_fns

    def crop_eyes(self):
        """Crop each layer."""
        assert len(self.layers) > 0, (
            f"No layers loaded yet. Try running {self.load}.")
        # load the boolean masks
        self.load_masks(mask_fn=self.eye_mask_fn, mask_arr=self.eye_mask)
        new_layers = []
        for layer in self.layers:
            new_layers += [layer.crop_eye()]
        # replace the stack layers with cropped Eye objects
        self.layers = new_layers
        # crop the mask image to avoid shape issues
        self.mask = Eye(filename=self.eye_mask_fn, arr=self.eye_mask)
        self.mask.load_mask(mask_fn=self.eye_mask_fn, mask_arr=self.eye_mask)
        self.mask = self.mask.crop_eye()
        self.mask_arr = self.mask.image.astype('uint8')
        # mask the mask_fn None to avoid loading from file
        self.mask_fn = None
        
    def get_eye_stack(self, smooth_factor=0):
        """Generate focus stack of images and then crop out the eye.


        Parameters
        ----------
        smooth_factor : float, default=0
            The standard deviation of the gaussian 2D filter applied to the 
            approximate heights.

        Attributes
        ----------
        eye : Eye
            The Eye object of the focus stack of cropped image layers.
        """
        assert len(self.layers) > 0, (
            f"No layers loaded. Try running {self.load} and {self.crop_eyes}.")
        # get a focus stack with 3D surface data
        self.get_focus_stack(smooth_factor)
        # store an Eye image using the focus stack
        self.eye = Eye(arr=self.stack.astype('uint8'),
                       mask_arr=self.mask_arr,
                       pixel_size=self.pixel_size)

    def get_ommatidia(self, bright_peak=True, fft_smoothing=5,
                      square_lattice=False, high_pass=False, num_neighbors=3,
                      sample_size=100, plot=False, plot_fn=None):
        """Use Eye object of the eye stack image to detect ommatidia.



        Parameters
        ----------
        (see Eye.ommatidia_detecting_algorithm and self.oda_3d)
        """
        assert isinstance(self.eye, Eye), (
            "The focus stack hasn't been processed yet. Try running " +
            str(self.get_eye_stack))
        # find ommatidia in the focus stack image
        self.eye.ommatidia_detecting_algorithm(
            bright_peak=bright_peak, fft_smoothing=fft_smoothing,
            square_lattice=square_lattice, high_pass=high_pass,
            num_neighbors=num_neighbors, sample_size=sample_size,
            plot=plot, plot_fn=plot_fn)

    def oda_3d(self, eye_stack_smoothing=0, bright_peak=True, fft_smoothing=5,
               square_lattice=False, high_pass=False, num_neighbors=3,
               sample_size=100, plot=False, plot_fn=None, use_memmaps=False):
        """Detect ommatidia using the 3D surface data.


        Parameters
        ----------
        eye_stack_smoothing : float, default=0
            Std deviation of gaussian kernal used to smooth the eye surface.
        bright_peak : bool, default=True
            Whether the ommatidia are defined by brighter (vs. darker) peaks.
        fft_smoothing : int, default=5
            The standard deviation of a 2D gaussian filter applied to the 
            reciprocal image before finding peaks.
        square_lattice : bool, default=False
            Whether this a square---rather than a hexagonal---lattice.
        high_pass : bool, default=False
            Whether to also filter frequencies below the fundamental one.
        num_neighbors : int, default=6
            The number of neighbors to check for measuring the ommatidial 
            diameter. Defaults to 6, assuming a hexagonal lattice.
        sample_size : int, default=100
            The number of ommatidia near the center to include in diameter
            estimation.
        plot : bool, default=False
            Whether to plot the eye with ommatidia and diameters superimposed.
        plot_fn : str, default=None
            Filename to save the plotted eye with superimposed ommatidia and 
            their diameters.
        use_memmaps : bool, default=False
            Whether to use memory maps instead of loading the images to RAM.

        Attributes
        ----------
        sphere : SphereFit
            An OLS-fitted sphere to the 3D ommatidia coordinates. 
            Transforms the points into polar coordinates relative to 
            the fitted sphere.
        fov_hull : float
            The field of view of the convex hull of the ommatidia in
            polar coordinates.
        fov_long : float
            The longest angle of view using the long diameter of the
            ellipse fitted to ommatidia in polar coordinates.
        fov_short : float, steradians
            The shortest angle of view using the short diameter of 
            the ellipse fitted to ommatidia in polar coordinates.
        surface_area : float, steradians
            The surface area of the sphere region given fov_hull and
            sphere.radius.
        io_angles : np.ndarray, rads
            The approximate inter-ommatidial angles per ommatidium 
            using eye.ommatidial_diameters and eye radius in rads.
        io_angle : float, rad
            The average approximate inter-ommatidial angle using 
            eye.ommatidial_diameter / self.sphere.radius
        io_angle_fft : float, rad
            The average approximate inter-ommatidial angle using
            eye.ommatidial_diameter_fft / self.sphere.radius
        """
        # 0. make sure the stack is imported and cropped
        if use_memmaps:
            self.load_memmaps()
        else:
            self.load()
        self.crop_eyes()
        self.get_eye_stack(smooth_factor=eye_stack_smoothing)
        # 1. find ommatidia in the focus stack image
        self.get_ommatidia(bright_peak=bright_peak, fft_smoothing=fft_smoothing,
            square_lattice=square_lattice, high_pass=high_pass,
            num_neighbors=num_neighbors, sample_size=sample_size,
            plot=plot, plot_fn=plot_fn)
        # 2. find their approximate z-coordinates
        ys, xs = self.eye.ommatidial_inds.T
        zs = self.heights[ys, xs] # simple way -- todo: consider other ways
        ys, xs = self.eye.ommatidia.T
        new_ommatidia = np.array([ys, xs, zs]).T
        # add z dimension and recalculate ommatidial diameters
        self.eye.ommatidia = new_ommatidia
        self.eye.measure_ommatidia()
        # 3. fit a sphere to the 3D ommatidial coordinates and convert
        # to spherical coordinates
        self.sphere = SphereFit(self.eye.ommatidia)
        # find the convex hull of the data in polar coordinates
        hull = spatial.ConvexHull(self.sphere.polar[:, :2])
        hull_polar = self.sphere.polar[hull.vertices, :2]
        self.fov_hull = hull.area # in steradians
        plt.scatter(hull_polar[:, 0], hull_polar[:, 1])
        # fit an ellipse to the polar convex hull
        polar_ellipse = LSqEllipse()
        polar_ellipse.fit(hull_polar.T)
        # get relevant properties of the ellipse corresponding to FOV
        (theta_center, phi_center), width, height, ang = polar_ellipse.parameters()
        self.fov_short = 2 * width # rads
        self.fov_long = 2 * height # rads
        # eye surface area using fov and eye radius
        self.surface_area = self.fov_hull * self.sphere.radius ** 2 # units of pixel size ** 2
        # IO angles are the diameters / eye radius
        self.io_angles = self.eye.ommatidial_diameters / self.sphere.radius
        self.io_angle = self.eye.ommatidial_diameter / self.sphere.radius
        self.io_angle_fft = self.eye.ommatidial_diameter_fft / self.sphere.radius
        # print a summary of the 3d-related results
        # whole eye parameters
        area = np.round(self.surface_area, 4)
        fov = np.round(self.fov_hull, 4)
        fov_long, fov_short = np.round(self.fov_long, 4), np.round(self.fov_short, 4)
        # updated ommatidial parameters
        count = len(self.eye.ommatidia)
        sample_diameter = np.round(self.eye.ommatidial_diameter, 4)
        sample_std = np.round(self.eye.ommatidial_diameter_SD, 4)
        diameter_fft = np.round(self.eye.ommatidial_diameter_fft, 4)
        # and interommatidial parameters
        io_angle = np.round(self.io_angle * 180 / np.pi, 4)
        io_angle_std = np.round(self.io_angles.std() * 180 / np.pi, 4)
        io_angle_fft = np.round(self.io_angle_fft * 180 / np.pi, 4)
        print(
            "3D results:\n"
            f"Eye:\tSurface Area={area}\tFOV={fov}\tFOV_l={fov_long}\tFOV_s={fov_short}\n"
            f"Ommatidia:\tmean={sample_diameter}\tstd={sample_std}\tfft={diameter_fft}\n"
            f"IO angles(deg):\tmean={io_angle}\tstd={io_angle_std}\tfft={io_angle_fft}\n")
        print()
        

class CTStack(Stack):
    """A special stack for handling a CT stack of a compound eye.


    Methods
    -------
    __init__(dirname="./", img_extension=".jpg", bw=False, 
        layer_class=Layer, pixel_size=1, depth_size=1)
        Import the image stack using the directory of CT layers.
    prefilter(low=0, high=None, folder='./_prefiltered_stack')
        Filter the layers and then save in a new folder.
    import_stack(low=0, high=None)
        Filter the images including values between low and high.
    get_cross_sections(thickness=.3)
        Use 2D interpolation to model the points' radii as a function 
        of their polar position. Provides an approximate cross-section.
    find_ommatidia_clusters()
        Use the ODA to find the point clusters corresponding to
        distinct crystalline cones.
    measure_visual_parameters()
        Using the point clusters corresponding to seperate crystalline 
        cones, measuring important visual parameters.
    oda_3d()
        Run the pipeline using multiple interfaces to tune parameters 
        for processing CT stacks.
    save(filename)
        Save the relevant variables in an H5 database.

    Attributes
    ----------
    
    """
    def __init__(self, database_fn="_compound_eye_data.h5", **kwargs):
        """Import data from a save H5 database if present.


        Parameters
        ----------
        database_fn : str, default="_compoint_eye_data.h5"
            The filename of the H5 database with loaded values.
        """
        # import the stack
        Stack.__init__(self, **kwargs)
        # load the h5 database
        self.database_fn = os.path.join(self.dirname, database_fn)
        self.load_database()
        
    def __del__(self):
        self.database.close()

    def load_database(self, mode='r+'):
        """Initialize and load the H5 database.


        Parameters
        ----------
        mode : str, default='r+'
            The access privileges of the database.
        """
        # make an H5 database to store large sets of coordinates
        if not os.path.exists(self.database_fn):
            new_database = h5py.File(self.database_fn, 'w')
            new_database.close()
        self.database = h5py.File(self.database_fn, mode)
        # get the datatype from the first layer
        first_layer = Layer(self.fns[0])
        first_layer.load()
        self.dtype = first_layer.image.dtype
        # store points array for loading from a file.
        for key in self.database.keys():
            setattr(self, key, self.database[key])
        # load the ommatidial and interommatidial datasets if the files exist
        files_to_load = [
            os.path.join(self.dirname, "ommatidial_data.pkl"),
            os.path.join(self.dirname, "interommatidial_data.pkl")]
        for var, fn in zip(
                ['ommatidial_data', 'interommatidial_data'],
                files_to_load):
            if var in dir(self):
                delattr(self, var)
            if os.path.exists(fn):
                setattr(self, var, pd.read_pickle(fn))

    def save_database(self):
        """Save the H5PY database."""
        self.database.close()
        self.load_database()

    def prefilter(self, low=0, high=None, folder="_prefiltered_stack"):
        """Filter the layers and then save in a new folder.


        Parameters
        ----------
        low : int, default=0
            The minimum value for an inclusive filter.
        high : int, default=None
            The maximum value for an inclusing filter, defaulting to 
            the maximum.
        folder : str, default="_prefiltered_stack"
            The directory to store the prefiltered image.
        """
        # assume no maximum
        first_layer = Layer(self.fns[0])
        dirname, basename = os.path.split(first_layer.filename)
        if high is None:
            first_layer.load()
            dtype = first_layer.image.dtype
            # get maximum value for that dtype
            high = np.iinfo(dtype).max
        # make the folder if it doesn't already exist
        if not os.path.exists(folder):
            os.mkdir(os.path.join(dirname, folder))
        # go through each file, load 
        for num, layer in enumerate(self.iter_layers()):
            layer.load()
            # make a new image using the low and high values
            include = (layer.image >= low) * (layer.image <= high)
            new_img = np.zeros(layer.image.shape, layer.image.dtype)
            new_img[include] = layer.image[include]
            # save in the prefiltered folder
            basename = os.path.basename(layer.filename)
            new_fn = os.path.join(dirname, folder, basename)
            save_image(new_fn, new_img)
            print_progress(num + 1, len(self.fns))

    def import_stack(self, low=0, high=None):
        """Filter the images including values between low and high.


        Parameters
        ----------
        low : int, default=0
            The minimum value for an inclusive filter.
        high : int, default=None
            The maximum value for an inclusing filter, defaulting to 
            the maximum.
        """
        # set some defaults if not loaded
        if "points" in dir(self):
            del self.points
            del self.database['points']
        self.points = self.database.create_dataset(
            "points", data=np.zeros((0, 3)), dtype=float,
            chunks=True, maxshape=(None, 3))
        # assume no maximum
        first_layer = Layer(self.fns[0])
        dirname, basename = os.path.split(first_layer.filename)
        if high is None:
            first_layer.load()
            dtype = first_layer.image.dtype
            # get maximum value for that dtype
            high = np.iinfo(dtype).max
        # if points are already stored, reset
        if self.points.shape[0] > 0:
            self.points.resize(0, axis=0)
        # get points included in low to high range
        for num, layer in enumerate(self.iter_layers()):
            layer.load()
            include = (layer.image >= low) * (layer.image <= high)
            if np.any(include):
                x, y = np.where(include)
                pts = np.array([
                    np.repeat(
                        float(num) * self.depth_size, len(x)),
                    self.pixel_size * x.astype(float),
                    self.pixel_size * y.astype(float)]).T
                # update the points array size and store values
                self.points.resize(
                    (self.points.shape[0] + len(x), 3))
                self.points[-len(x):] = pts
            print_progress(num, len(self.fns))
        # store the new points to access the original coordinates
        # create the original points dataset
        # if an old version already exists, delete it
        if "points_original" in dir(self):
            del self.points_original
            del self.database["points_original"]
        self.points_original = self.database.create_dataset(
            "points_original", data=self.points)


    def get_cross_sections(self, thickness=1.0, chunk_process=False):
        """Approximate surface splitting the inner and outer sections.


        Uses 2D spline interpolation, modelling point radial distance 
        as a function of its polar coordinate.

        Parameters
        ----------
        thickness : float, default=.3
            Proportion of the residuals to include in the cross section 
            used for the ODA.
        chunk_process : bool, default=False
            Whether to process  polar coordinates in chunks or all at 
            once, relying on RAM.

        Attributes
        ----------
        theta, phi, radii : array-like, dtype=float, shape=(N, 1)
            The azimuth, elevation, and radial distance of the loaded 
            coordinates centered around the center of a fitted sphere.
        residual : array-like, dtype=float, shape=(N, 1)
            Residual distance of points from a fitted interpolated surface.
        """
        # 0. assume points are already loaded
        assert self.points.shape[0] > 0, (
            f"No points have been loaded. Try running {self.import_stack} first.")
        # 1. Fit sphere to the points to find a useful center
        # SphereFit on random indexed subset
        ind_range = range(len(self.points))
        # choose a subset size based on RAM limits
        num_samples = min(len(self.points), int(1e6))
        inds = np.random.choice(ind_range, size=num_samples,
                                replace=False)
        inds.sort()
        # import using chunks about 100 long for RAM concerns
        chunksize = 100
        num_chunks = int(np.ceil(len(inds) / chunksize))
        subset = []
        for num in range(num_chunks):
            subset += [self.points[
                num * chunksize: (num + 1) * chunksize]]
        subset = np.concatenate(subset)
        sphere = SphereFit(subset)
        center = sphere.center
        # store the sphere's center
        self.center = center
        center_dir = center / np.linalg.norm(center)
        # center the points
        self.points[:] = self.points - self.center[np.newaxis]
        subset -= center[np.newaxis, :]
        # 2. Convert points to spherical coordinates
        # make a Points object of the subset
        pts = Points(subset, rotate_com=False)    # performs spherical conversion 
        # 3. Spline interpolate radii as function of theta and phi
        pts.surface_predict(image_size=1e4)
        self.shell = pts
        # 4. find the spherical coordinates of all points
        # delete old entries for this data
        vars_to_check = ['theta', 'phi', 'radii', 'residual']
        for var in vars_to_check:
            if var in dir(self):
                delattr(self, var)
            if var in self.database.keys():
                del self.database[var]
        # store the data
        if chunk_process:
            self.theta = self.database.create_dataset(
                "theta", (len(self.points), ), dtype=float)
            self.phi = self.database.create_dataset(
                "phi", (len(self.points), ), dtype=float)
            self.radii = self.database.create_dataset(
                "radii", (len(self.points), ), dtype=float)
            self.residual = self.database.create_dataset(
                "residual", (len(self.points), ), dtype=float)
            # iterate through chunks to avoid loading too much into RAM
            chunksize = min(1e3, len(self.points))
            num_chunks = len(self.points)
            chunks = range(num_chunks)
            for chunk_num in chunks:
                start = round(chunk_num * chunksize)
                stop = round((chunk_num + 1) * chunksize)
                subset = self.points[start:stop]
                polar = rectangular_to_spherical(subset)
                theta, phi, radii = polar.T
                self.theta[start:stop] = theta
                self.phi[start:stop] = phi
                self.radii[start:stop] = radii
                # check predicted radius
                print_progress(chunk_num, len(chunks))
            pts.surface_predict(xvals=self.theta, yvals=self.phi)
            predicted_radii = pts.surface
            self.residual[:] = self.radii - predicted_radii
        else:
            # store the coordinate arrays
            polar = rectangular_to_spherical(self.points)
            theta, phi, radii = polar.T
            self.theta = self.database.create_dataset(
                "theta", data=theta, dtype=float)
            self.phi = self.database.create_dataset(
                "phi", data=phi, dtype=float)
            self.radii = self.database.create_dataset(
                "radii", data=radii, dtype=float)
            # calculate residuals based on predicted surface
            pts.surface_predict(xvals=self.theta[:], yvals=self.phi[:])
            predicted_radii = pts.surface
            residuals = radii - predicted_radii
            # store residuals
            self.residual = self.database.create_dataset(
                "residual", data=residuals, dtype=float)


    def find_ommatidial_clusters(self, polar_clustering=True,
                                 window_length=np.pi/4,
                                 window_pad=np.pi/20,
                                 image_size=1e4, mask_blur_std=2,
                                 square_lattice=False):
        """2D running window applying ODA to spherical projections.

        
        Parameters
        ----------
        polar_clustering : bool, default=True
            Whether to use polar coordinates for clustering (as 
            opposed to the 3D rectangular coordinates).
        window_length : float, default=pi/4
            The angle of view of the rolling square window.
        window_pad : float, default=pi/20
            The padding of overlap used to avoide border issues.
        image_size : float, default=1e6
            The number of pixels to use in rasterizing the rolling
            window.
        mask_blur_std : float, default=2
            The standard deviation of the gaussian blur used for
            smoothing the mask for the ODA.
        square_lattice : bool, default=False
            Wether the ommatidial lattice is square vs. hexagonal.

        Attributes
        ----------
        include : np.ndarray, dtype=bool
        

        """
        # assume that the shell has been loaded
        assert "theta" in self.database.keys(), (
            "No polar coordinates found. Try running "
            f"{self.get_cross_sections} first or running "
            f"{self.ommatidia_detecting_algorithm}")
        # get elevation and inclination ranges
        theta_min, theta_max = np.percentile(self.theta, [0, 100])
        phi_min, phi_max = np.percentile(self.phi, [0, 100])
        # iterate through the windows +/- padding
        # store a binary filter and cluster labels arrays
        to_store = ["include", "labels"]
        dtypes = [bool, int]
        for var, dtype in zip(to_store, dtypes):
            # if it exists already delete
            if var in dir(self):
                delattr(self, var)
                del self.database[var]
            # store the empty array
            if var not in dir(self):
                dataset = self.database.create_dataset(
                    var, (len(self.points),), dtype=dtype)
                setattr(self, var, dataset)
        # iterate through theta and phi ranges taking steps of window_length
        theta_low = theta_min
        phi_low = phi_min
        # get center points for reorienting the polar coordinates
        theta_center = np.mean([theta_min, theta_max])
        phi_center = np.mean([phi_min, phi_max])
        max_val = 0                 # keep track of max label
        # print the segment number
        segment_number = 1
        # get indices for points within 50% of the residuals 
        low, high = np.percentile(self.residual[:], [25, 75])
        in_cross_section = (self.residual[:] > low)*(self.residual[:] < high)
        # update the user
        print("\nProcessing the polar coordinates in segments:")
        while theta_low < theta_max:
            theta_high = theta_low + window_length
            # get relevant theta values
            theta_center = np.mean([theta_low, theta_high])
            while phi_low < phi_max:
                print(f"Segment #{segment_number}:")
                # get important phi values
                phi_high = phi_low + window_length
                phi_center = np.mean([phi_low, phi_high])
                # store inclusion criteria in database
                self.include[:] = True
                # azimuth filter
                self.include[:] *= (self.theta > theta_low - window_pad)
                self.include[:] *= (self.theta <= theta_high + window_pad)
                # elevation filter
                self.include[:] *= (self.phi > phi_low - window_pad)
                self.include[:] *= (self.phi <= phi_high + window_pad)
                # get indices of all points
                include = np.where(self.include)
                in_shell = in_cross_section[include[0]]
                # calculate angular displacements in order to rotate the center of mass
                if len(include[0]) > 100:
                    # test the rotate function:
                    # goal: rotate center of mass vector until x and y components are 0
                    subset_original = np.array(self.points[include])
                    theta_original, phi_original, radii_original = self.theta[include], self.phi[include], self.radii[include]
                    subset = np.copy(subset_original)
                    com = subset.mean(0)
                    com_polar = rectangular_to_spherical(com[np.newaxis])
                    theta_displacement, phi_displacement, _ = com_polar[0]
                    # rotate subset of points to minimize spherical distortion
                    subset = rotate(subset, phi_center, axis=2).T # center at 0
                    subset = rotate(subset, theta_center - np.pi/2, axis=1).T # center at pi/2
                    # get polar coordinates
                    polar = rectangular_to_spherical(subset)
                    # get the centered polar coordinates
                    segment = Points(subset, sphere_fit=False,
                                     rotate_com=False,
                                     spherical_conversion=False,
                                     polar=polar)
                    # use the segment within the cross section for better
                    # defined centers
                    sub_segment = Points(subset[in_shell], sphere_fit=False,
                                     rotate_com=False,
                                     spherical_conversion=False,
                                     polar=polar)
                    # rasterize an image using the polar 2D histogram
                    # find the smallest distance to record
                    dists_tree = spatial.KDTree(sub_segment.polar[:, :2])
                    dists, inds = dists_tree.query(sub_segment.polar[:, :2], k=2)
                    min_dist = 2*np.mean(dists[:, 1])
                    raster, (theta_vals, phi_vals) = sub_segment.rasterize(
                        image_size=image_size, pixel_length=min_dist)
                    # make an Eye object of the raster image to get
                    # ommatidia centers
                    pixel_size = phi_vals[1] - phi_vals[0] # in rads
                    # make a mask using the raster image
                    mask = raster > 0
                    mask = ndimage.gaussian_filter(mask.astype(float), 2)
                    mask /= mask.max()
                    # thresh = np.percentile(mask[mask > 0], 10)
                    thresh = .1
                    mask = mask > thresh
                    mask = 255 * mask.astype(int)
                    raster = 255 * (raster / raster.max())
                    raster = raster.astype('uint8')
                    # apply the ODA to the raster image
                    try:
                        eye = Eye(arr=raster, pixel_size=pixel_size,
                                  mask_arr=mask, mask_fn=None)
                    except:
                        breakpoint()
                    eye.oda(plot=False, square_lattice=square_lattice)
                    # use the ommatidial centers to find the clusters 
                    centers = eye.ommatidia
                    # check that there are more points than centers, otherwise skip
                    if len(centers) < len(subset):
                        # shift the coordinates using the min theta and phi
                        centers += [theta_vals.min(), phi_vals.min()]
                        # use polar angles for clustering
                        segment.surface_predict(
                            xvals=centers[:, 0], yvals=centers[:, 1])
                        model_radii = segment.surface
                        centers = np.array([
                            centers[:, 0], centers[:, 1], model_radii]).T
                        if polar_clustering:
                            # remove any centers with radii we couldn't model
                            no_nans = np.any(np.isnan(centers), axis=1) == False
                            centers = centers[no_nans]
                            # KMeans to cluster points based on centers
                            clusterer = cluster.KMeans(
                                n_clusters=len(centers), init=centers[:, :2],
                                n_init=1)
                            polar = segment.polar
                            lbls = clusterer.fit_predict(polar[:, :2])
                        else:
                            # use the nearest points as seeds in the KMeans in 3D
                            # remove centers with unknown radii
                            no_nans = np.isnan(centers[:, -1]) == False
                            centers = centers[no_nans]
                            # convert to rectangular coordinates
                            centers_rect = spherical_to_rectangular(centers)
                            # use KMeans with the 3D data
                            clusterer = cluster.KMeans(
                                n_clusters=len(centers), init=centers_rect,
                                n_init=1)
                            lbls = clusterer.fit_predict(subset)
                        lbls_set = np.arange(max(lbls) + 1)
                        # randomize lbls and use 
                        scrambled_lbls = np.random.permutation(lbls_set)
                        new_lbls = scrambled_lbls[lbls.astype(int)]
                        # find centers within bounds
                        # first, un-rotate the centers to the original reference frame
                        centers_rect = spherical_to_rectangular(centers)
                        centers_original = rotate(
                            centers_rect, -(theta_center - np.pi/2), axis=1).T
                        centers_original = rotate(
                            centers_original, -phi_center, axis=2).T
                        centers_original_polar = rectangular_to_spherical(
                            centers_original).T
                        # use original bounds to get centers within window
                        theta, phi, radii = centers_original_polar
                        # azimuth filter
                        in_window = (theta > theta_low)
                        in_window *= (theta <= theta_high)
                        # elevation filter
                        in_window *= (phi > phi_low)
                        in_window *= (phi <= phi_high)
                        # remove non-positive lbls
                        positive = lbls_set > 0
                        in_window *= positive
                        # store all clusters having centers within the window
                        lbl_vals_set = lbls_set[in_window]
                        # get lbl vals in window
                        lbls_in_window = np.in1d(lbls, lbl_vals_set)
                        lbl_vals_in_window = lbls[lbls_in_window]
                        # replace with sorted range of values
                        new_lbl_vals_set = np.arange(len(lbl_vals_set))
                        new_lbl_vals_set = new_lbl_vals_set + max_val + 1
                        max_val += len(lbl_vals_set) + 1
                        # make a lookup table to use original values as indices
                        new_lbl_vals_lookup = np.empty(max(lbls_set)+1)
                        new_lbl_vals_lookup.fill(np.nan)
                        new_lbl_vals_lookup[in_window] = new_lbl_vals_set
                        # convert using the new lookup table
                        new_lbls = new_lbl_vals_lookup[lbls]
                        no_nans = np.isnan(new_lbls) == False
                        new_lbls_in_window = (new_lbls[no_nans]).astype(int)
                        inds = include[0][lbls_in_window * no_nans]
                        # store, ignoring lbls originally set to 0
                        self.labels[inds] = new_lbls_in_window
                        segment_number += 1
                        print()
                # update phi lower bound
                phi_low += window_length
            # update theta lower bound
            theta_low += window_length
            phi_low = phi_min

    def measure_ommatidia(self, square_lattice=False, test=False):
        """Take measurements of ommatidia using the ommatidial clusters.

        
        Parameters
        ----------
        square_lattice : bool
            Whether the ommatidial lattice is square vs. hexagonal.

        Attributes
        ----------
        ommatidial_data : pd.DataFrame
            The dataframe containing data on the ommatidial clusters including
            each position in rectangular (x, y, and z) and polar (theta, phi, 
            radius) coordinates. 
        """
        # begin database, storing data per cluster
        centers = []
        xs, ys, zs, thetas, phis, radii = [], [], [], [], [], []
        size = []
        label_set = np.array(sorted(set(self.labels[:])))
        # iterate through the set of labels
        for num, lbl in enumerate(label_set):
            inds = np.where(self.labels[:] == lbl)[0]
            # store center of cluster
            pts = self.points[inds]
            center = self.points[inds].mean(0)
            centers += [center]
            # store mean x, y, and z and azimuth, elevation, and radii
            x, y, z = center
            xs += [x]
            ys += [y]
            zs += [z]
            thetas += [self.theta[inds].mean()]
            phis += [self.phi[inds].mean()]
            radii += [self.radii[inds].mean()]
            size += [len(inds)]
        # convert to numpy arrays for pandas
        centers = np.array(centers)
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        thetas, phis, radii = np.array(thetas), np.array(phis), np.array(radii)
        size = np.array(size)
        # store measurements to a dictionary
        data_to_save = dict()
        for var, arr in zip(
                ['label', 'x', 'y', 'z', 'theta', 'phi', 'radius', 'size'],
                [label_set, xs, ys, zs, thetas, phis, radii, size]):
            data_to_save[var] = arr
        # make a pandas dataframe using the dictionary
        ommatidial_data = pd.DataFrame(data_to_save)
        # store as a csv spreadsheet
        csv_filename = os.path.join(self.dirname, "ommatidial_data.csv")
        ommatidial_data.to_csv(csv_filename, index=False)
        ommatidial_data.to_pickle(csv_filename.replace(".csv", ".pkl"))
        # use a kernal density tree for finding nearest neighbors
        centers_tree = spatial.KDTree(centers)
        # get the 12 nearest neighbors per cluster and use the KMeans
        # to find the cluster of distances corresponding to diameters
        dists, inds = centers_tree.query(centers, k=13)
        dists = dists[:, 1:]
        upper_limit = np.percentile(dists.flatten(), 99)
        dists = dists[dists < upper_limit].flatten()
        clusterer = cluster.KMeans(
            2, init=np.array([0, 100]).reshape(-1, 1), n_init=1).fit(
                dists[:, np.newaxis])
        distance_groups = clusterer.fit_predict(dists[:, np.newaxis])
        # use the maximum distance in the first distance group
        # as a criterion for diameters
        upper_limit = dists[distance_groups == 0].max()
        # find inter-ommatidial pairs using this upper limit
        if square_lattice:
            num_neighbors = 4
        else:
            num_neighbors = 6
        neighbor_dists, neighbor_lbls = centers_tree.query(
            centers, k=num_neighbors + 1, distance_upper_bound=upper_limit)
        neighbor_dists = neighbor_dists[:, 1:]
        neighbor_lbls = neighbor_lbls[:, 1:]
        # replace infs with nans
        to_replace = neighbor_dists == np.inf
        neighbor_dists[to_replace] = np.nan
        # get a larger neighborhood of points for calculating normal vectors
        big_neighborhood_dists, big_neighborhood_lbls = centers_tree.query(
            centers, k=51)
        # start to store important ommatidial measurements
        lens_area = []
        anatomical_vectors = []
        approx_vectors = []
        skewness = []
        neighborhood = []
        # iterate through the clusters and take measurements
        print("\nProcessing ommatidial data:")
        for num, (center, lbl, neighbor_group,
                  neighbor_dist, big_group, big_dist) in enumerate(zip(
                      centers, label_set,
                      neighbor_lbls, neighbor_dists,
                      big_neighborhood_lbls, big_neighborhood_dists)):
            # get points within the cluster
            inds = np.where(self.labels[:] == lbl)
            pts = self.points[inds]
            # ignore any 'neighbors' with Nan distances
            no_nans = np.isnan(neighbor_dist) == False
            neighbor_group = neighbor_group[no_nans]
            no_nans = np.isnan(big_dist) == False
            big_group = big_group[no_nans]
            # get centers of neighboring clusters, avoiding those with Nan distances
            small_neighborhood = centers[neighbor_group]
            big_neighborhood = centers[big_group]
            # 1. lens area using mean distance to nearest neighbors
            if np.all(np.isnan(neighbor_dist)):
                diam = np.nan
            else:
                diam = np.nanmean(neighbor_dist)
            area = np.pi * (.5 * diam) ** 2
            lens_area += [area]
            # 2. approximate ommatidial axis using average normal
            # vector of plane relative to neighboring ommatidia
            pts = centers[neighbor_group]
            # center about the main ommatidium
            pts_centered = big_neighborhood - center
            pts_centered = pts_centered[1:]
            # find cross product of each pair of neighboring ommatidia
            cross = np.cross(pts_centered[np.newaxis], pts_centered[:, np.newaxis])
            # half of these will be left handed while the other half are right handed
            # because the matrix is negative symmetrical
            # find those that minimize the angle between itself and the vector to the
            # center of the fitted sphere
            magn = np.linalg.norm(cross, axis=-1)
            # normalize the direction vectors into unit vectors
            non_zero = magn != 0 # avoid dividing by zero
            cross[non_zero] /= magn[non_zero, np.newaxis]
            # find angle between cross product unit vectors and sphere center unit vector 
            sphere_vector = -center
            sphere_vector /= np.linalg.norm(sphere_vector)
            angles_between = np.arccos(np.dot(cross, sphere_vector))
            avg_vector = cross[angles_between < np.pi/2]
            approx_vector = avg_vector.mean(0)
            # center
            approx_vectors += [approx_vector]
            # 3. calculate ommatidial axis vector by regressing the cluster data
            anatomical_vector = fit_line(pts_centered)
            anatomical_vector = np.array(
                [anatomical_vector, -anatomical_vector])
            angs = np.arccos(np.dot(anatomical_vector, sphere_vector))
            ind = angs == angs.min()
            anatomical_vector = anatomical_vector[ind].min(0)
            anatomical_vectors += [anatomical_vector]
            # 4. calculaate ommatidial skewness as the inside angle
            # between the approx and anatomical vectors
            inside_ang = angle_between(approx_vector, anatomical_vector)
            skewness += [inside_ang]
            # 5. store neighbor groups
            neighborhood += [label_set[neighbor_group]]
            print_progress(num + 1, len(centers))
        # convert to numpy arrays
        lens_area = np.array(lens_area)
        anatomical_vectors = np.array(anatomical_vectors)
        approx_vectors = np.array(approx_vectors)
        skewness = np.array(skewness) * 180 / np.pi
        # calculate the spherical IO angle using eye radius and ommatidial diameter
        diameter = np.sqrt(lens_area / np.pi)
        radius = np.nanmean(radii)
        spherical_IO_angle = (diameter / radius) * 180 / np.pi
        # add to the dataframe
        for var, arr in zip(
                ['lens_area', 'anatomical_axis', 'approx_axis', 'skewness',
                 'spherical_IOA', 'neighbors'],
                [lens_area, anatomical_vectors.tolist(), approx_vectors.tolist(), skewness,
                 spherical_IO_angle, neighborhood]):
            ommatidial_data[var] = arr
        # save the spreadsheet as a csv and pickle file
        ommatidial_data.to_csv(csv_filename, index=False)
        ommatidial_data.to_pickle(csv_filename.replace(".csv", ".pkl"))
        self.ommatidial_data = ommatidial_data
        labels = self.ommatidial_data.neighbors.values
        cluster_lbls = self.labels[:]
        # process ommatidial pair information based on neighborhood estimates
        # setup empty arrays for storing important measurements
        pairs = []
        orientations = []
        pair_centers = []
        # iterate through clusters and get unique pairs based on neighborhood
        print("\nPre-processing interommatidial pairs:")
        for num, cone in self.ommatidial_data.iterrows():
            lbl = cone.label
            in_cluster = np.where(cluster_lbls == lbl)[0]
            pts = self.points[in_cluster]
            neighbor_labels = cone.neighbors
            for neighbor_ind in neighbor_labels:
                pair = tuple(sorted([lbl, neighbor_ind]))
                # go through each pair once
                if pair not in pairs: 
                    # get the pts in the neighboring cluster
                    if np.any(self.ommatidial_data.label.values == neighbor_ind):
                        inds = np.where(
                            self.ommatidial_data.label.values == neighbor_ind)[0][0]
                        neighbor_cone = self.ommatidial_data.loc[inds]
                        inds = np.where(cluster_lbls == neighbor_ind)[0]
                        neighbor_pts = self.points[inds]
                        # use the 2 polar coordinates to get the interommatidial
                        # orientation
                        theta1, phi1 = cone[['theta', 'phi']].values.T
                        x, y, z = cone[['x', 'y', 'z']]
                        theta2, phi2 = neighbor_cone[['theta', 'phi']].values.T
                        neighbor_x, neighbor_y, neighbor_z = neighbor_cone[['x', 'y', 'z']]
                        # get angle 
                        if theta1 < theta2:
                            orientation = np.array([phi2 - phi1, theta2 - theta1])
                        else:
                            orientation = np.array([phi1 - phi2, theta1 - theta2])
                        orientations += [orientation]
                        # store the pair to avoid reprocessing
                        pairs += [pair]
                        # pair_centers += [((theta1 + theta2)/2,
                        #                   (phi1 + phi2)/2)]
                        pair_centers += [((x + neighbor_x)/2,
                                          (y + neighbor_y)/2,
                                          (z + neighbor_z)/2)]
                        print_progress(num + 1, len(labels))
        # make as arrays
        pair_centers = np.array(list(pair_centers))
        pairs_tested = np.array(list(pairs))
        orientations = np.array(list(orientations))
        # store as datasets
        var = 'pairs_tested'
        if var in dir(self):
            delattr(self, var)
        if var in self.database.keys():
            del self.database[var]
        dataset = self.database.create_dataset(
            var, data=pairs_tested)
        

    def measure_interommatidia(self, square_lattice=False, test=False):
        """Take measurements of all inter-ommatidial pairs and save.


        Parameters
        ----------
        square_lattice : bool, default=False
            Whether the ommatidial lattice is square vs. hexagonal.
        test : bool, default=False
            Whether to perform code blocks related to troubleshooting.
        """
        labels = self.ommatidial_data.neighbors.values
        cluster_lbls = self.labels[:]
        # load the tested pairs data
        pair_centers = self.pair_centers[:]
        pairs_tested = self.pairs_tested[:]
        orientations = self.orientations[:]
        # rotate centers to minimize variance along one dimension
        centers_components = np.linalg.svd(pair_centers[::100] - pair_centers.mean(0))[2]
        centers_rotated = np.dot(pair_centers - pair_centers.mean(0), centers_components)
        rotated_eye = np.dot(self.points[:] - pair_centers.mean(0), centers_components)
        rotated_origin = np.dot(np.array([[0, 0, 0]]) - pair_centers.mean(0), centers_components)
        rotated_ranges = rotated_eye.max(0) - rotated_eye.min(0)
        xind, yind, zind = np.argsort(rotated_ranges)[::-1]
        xs, ys, zs = rotated_eye[:, [xind, yind, zind]].T
        xs_pairs, ys_pairs, zs_pairs = centers_rotated[:, [xind, yind, zind]].T
        # store dict of centers per cluster
        cluster_centers = dict()
        for lbl in set(cluster_lbls):
            if lbl >= 0:
                ind = np.where(cluster_lbls == lbl)[0]
                cluster_centers[lbl] = rotated_eye[ind].mean(0)
        # iterate through vertical slices
        xs_pair, ys_pair = pair_centers[:, :2].T
        ys_vals = np.linspace(ys_pair.min(), ys_pair.max(), 21)
        xs_vals = np.linspace(xs_pair.min(), xs_pair.max(), 21)
        # start arrays to store IOA data 
        vertical_IOAs = np.zeros(pairs_tested.shape[0])
        horizontal_IOAs = np.zeros(pairs_tested.shape[0])
        vertical_IOA_residuals = np.zeros(pairs_tested.shape[0])
        horizontal_IOA_residuals = np.zeros(pairs_tested.shape[0])
        # and the real world projection of the coordinates
        projected_coords = dict()
        for lbl in self.ommatidial_data.label.values:
            projected_coords[lbl] = np.zeros(3, float)
            projected_coords[lbl].fill(np.nan)
        # default to NaNs
        for arr in [vertical_IOAs, horizontal_IOAs,
                    vertical_IOA_residuals, horizontal_IOA_residuals]:
            arr.fill(np.nan)
        # keep track of number of pairs measured
        num = 0
        for (var_vals, var_IOAs, var_IOA_resids,
             var_pairs, other_pairs, var_ind, other_ind, var_lbl) in zip(
                [ys_vals, xs_vals],
                [vertical_IOAs, horizontal_IOAs],
                [vertical_IOA_residuals, horizontal_IOA_residuals],
                [ys_pair, xs_pair],
                [xs_pair, ys_pair],
                [yind, xind],
                [xind, yind],
                ['Y-axis', 'X-axis']):
            for var_start, var_stop in zip(var_vals[:-1], var_vals[1:]):
                # get all centers within this slice
                in_slice = (var_pairs >= var_start) * (var_pairs < var_stop)
                pairs_in_slice = pairs_tested[in_slice].astype(int)
                pairs_in_slice_set = sorted(set(pairs_in_slice.flatten()))
                # get all points within the slice
                pts_inds = np.in1d(cluster_lbls, pairs_in_slice_set)
                pts_inds = np.where(pts_inds)[0]
                pts_in_slice = self.points[pts_inds]
                lbls_in_slice = cluster_lbls[pts_inds]
                # FOR TESTING: scrambled colorvals for plotting
                if test:
                    colorvals = np.arange(max(set(lbls_in_slice)))
                    np.random.shuffle(colorvals)
                    color_conv = dict()
                    for lbl, clbl in zip(
                            sorted(set(lbls_in_slice)), colorvals):
                        color_conv[lbl] = clbl
                    clbls = []
                    for lbl in lbls_in_slice:
                        clbls += [color_conv[lbl]]
                    clbls = np.array(clbls)
                # use angle_fitter to find ommatidial rays
                lbls_set, rays, residuals = angle_fitter(
                    pts=pts_in_slice[:, [other_ind, zind, var_ind]],
                    lbls=lbls_in_slice, display=test)
                zvals = rays[:, 1]
                # store projected coordinates 
                for lbl, ray, zval in zip(
                        np.unique(lbls_in_slice),
                        rays, zvals):
                    try:
                        projected_coords[lbl][other_ind] = ray[0]/zval
                        projected_coords[lbl][zind] = 1
                        projected_coords[lbl][var_ind] = 0
                    except:
                        breakpoint()
                # store pair centers 
                pair_centers_in_slice = []
                for pair in pairs_tested:
                    centers_per_pair = []
                    for val in pair:
                        if val in cluster_centers.keys():
                            center = cluster_centers[val][[other_ind, zind]]
                        else:
                            center = np.repeat(np.nan, 2)
                            # breakpoint()
                            # center = pts_in_slice[lbls_in_slice[ind] == val, [other_ind, zind]].mean(0)
                        centers_per_pair += [center]
                    pair_centers_in_slice += [np.array(centers_per_pair)]
                if test:
                    # plot the cross section with their smoothed ommatidial axes
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
                    ax.scatter(pts_in_slice[:, other_ind], pts_in_slice[:, zind], c=clbls,
                               alpha=.5, edgecolors='none', marker='o', cmap='tab20')
                    ax.set_aspect('equal')
                    xmin, xmax = np.percentile(self.points[:, other_ind], [0, 100])
                    ymin, ymax = np.percentile(self.points[:, zind], [0, 100])
                    x_range, y_range = xmax - xmin, ymax - ymin
                    # add 3 zoomed insets of regions in the left, center, and right
                    # use the polar angles to zoom into 3 key spots
                    angles = np.arctan2(pts_in_slice[:, zind], pts_in_slice[:, other_ind])
                    key_angles = np.percentile(angles, [25, 50, 75])
                    key_pts = []
                    for angle in key_angles:
                        ind = np.argmin(abs(angles - angle))
                        key_pt = pts_in_slice[ind, [other_ind, zind]]
                        key_pts += [key_pt]
                    key_pts = np.array(key_pts)
                    xvals, yvals = key_pts.T
                    inset_axes = []
                    inset_box_width = y_range / 6
                    inset_width = inset_box_width / 2
                    inset_ys = np.repeat(ymax, 3)
                    inset_ys -= inset_box_width * np.arange(1, 4)
                    inset_ys = inset_ys[::-1]
                    inset_xs = np.repeat(xmin, 3)
                    inset_xs -= 1.1*inset_box_width
                    # formate axes
                    ax.set_ylim(ymin - .05 * y_range, ymax + .05 * y_range)
                    ax.set_xlim(inset_xs.min(), xmax + .05 * x_range)
                    sbn.despine(ax=ax, bottom=True, left=True)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot calibration line and annotate with actual length
                    ax.annotate('300 pixels', (xmin, ymin))
                    ax.plot([xmin, xmin + 300],
                            [ymin - .025 * y_range, ymin - .025 * y_range], color='k', linewidth=5)
                    for lbl, xval, yval, ins_x, ins_y in zip(
                            ['C.', 'B.', 'A.'], xvals, yvals, inset_xs, inset_ys):
                        x_range = [xval - inset_width/2, xval + inset_width/2]
                        y_range = [yval - inset_width/2, yval + inset_width/2]
                        included = pts_in_slice[:, other_ind] > min(x_range)
                        included *= pts_in_slice[:, other_ind] < max(x_range)
                        included *= pts_in_slice[:, zind] > min(y_range)
                        included *= pts_in_slice[:, zind] < max(y_range)
                        y_center = pts_in_slice[included, zind].mean()
                        axins = ax.inset_axes(
                            [ins_x, ins_y, inset_box_width, inset_box_width],
                            transform=ax.transData)
                        axins.scatter(pts_in_slice[:, other_ind], pts_in_slice[:, zind], c=clbls,
                                      alpha=.5, edgecolors='none', marker='o', cmap='tab20')
                        axins.set_xlim(min(x_range), max(x_range))
                        axins.set_ylim(y_center - inset_width/2, y_center + inset_width/2)
                        axins.set_xticks([])
                        axins.set_yticks([])
                        # insert a label into the top left corner of both boxes
                        # axins.annotate(lbl, (min(x_range), y_center - inset_width/2),
                        #                # xytext=(min(x_range) + .85*inset_width,
                        #                #         y_center - .9*inset_width/2), fontsize=12)
                        #                xytext=(0, 0))
                        # ax.annotate(lbl, (min(x_range), y_center - inset_width/2),
                        #             xytext=(0, 0))
                        #             # xytext=(min(x_range) + .8*inset_width,
                        #             #         y_center - .9*inset_width/2))
                        rect, lines = ax.indicate_inset_zoom(axins, edgecolor='k')
                        [line.set_visible(False) for line in lines]
                        inset_axes += [axins]
                # calculate interommatidial angles
                included_pair = []
                for pair in pairs_in_slice:
                    included_pair += [pair[0] in lbls_set and pair[1] in lbls_set]
                ioas = []
                for pair in pairs_in_slice:
                    if pair[0] in lbls_set and pair[1] in lbls_set:
                        # get ommatidial axes and centers to calculate the IOA
                        ray1 = rays[lbls_set == pair[0]][0]
                        ray2 = rays[lbls_set == pair[1]][0]
                        center1 = pts_in_slice[lbls_in_slice == pair[0]].mean(0)[[other_ind, zind]]
                        center2 = pts_in_slice[lbls_in_slice == pair[1]].mean(0)[[other_ind, zind]]
                        ioa = np.arccos(np.dot(ray1, ray2.T))
                        # testing:
                        # plot all points from the two clusters
                        # fig = plt.figure()
                        # for lbl, color in zip(pair, [blue, red]):
                        #     ind = lbls_in_slice == lbl
                        #     sub_pts = pts_in_slice[ind][:, [other_ind, zind]]
                        #     plt.scatter(sub_pts[:, 0], sub_pts[:, 1], color=color)
                        # plt.scatter(center1[0], center1[1], color='k', marker='o')
                        # plt.scatter(center2[0], center2[1], color='k', marker='o')
                        # plot intersection
                        dx1, dy1 = np.squeeze(ray1)
                        m1 = dy1/dx1
                        dx2, dy2 = np.squeeze(ray2)
                        m2 = dy2/dx2
                        x1, y1 = center1
                        x2, y2 = center2
                        yint1 = y1 - m1*x1
                        yint2 = y2 - m2*x2
                        intersectx = (yint1 - yint2) / (m2 - m1)
                        intersecty = m1 * intersectx + yint1
                        plt.plot([x1, intersectx, x2], [y1, intersecty, y2],
                                 color='k', alpha=.15)

                        # plt.gca().set_aspect('equal')
                        # plt.show()
                        # if not np.isnan(ray1[0]) and not np.isnan(ray2[0]):
                        #     breakpoint()
                        #     plt.show()
                        length = 10
                        # end testing.
                        # store IOA data
                        ioas += [ioa]
                        pair_ind = np.where(np.all((pairs_tested - pair[np.newaxis]) == 0, axis=1))
                        var_IOAs[pair_ind] = ioa
                        var_IOA_resids[pair_ind] = residuals
                        if test:
                            for center in [center1, center2]:
                                ax.scatter(center[0], center[1], edgecolors='none',
                                            marker='.', color='k')
                                for axins in inset_axes:
                                    axins.scatter(center[0], center[1], edgecolors='none',
                                                  marker='.', color='k')
                            # find intersection given the two centers and their direction vectors
                            dx1, dy1 = np.squeeze(ray1)
                            m1 = dy1/dx1
                            dx2, dy2 = np.squeeze(ray2)
                            m2 = dy2/dx2
                            x1, y1 = center1
                            x2, y2 = center2
                            yint1 = y1 - m1*x1
                            yint2 = y2 - m2*x2
                            intersectx = (yint1 - yint2) / (m2 - m1)
                            intersecty = m1 * intersectx + yint1
                            ax.plot([x1, intersectx, x2], [y1, intersecty, y2],
                                     color='k', alpha=.15)
                            for axins in inset_axes:
                                axins.plot([x1, intersectx, x2], [y1, intersecty, y2],
                                           color='k', alpha=.15)
                            ioa = np.arccos(np.dot(ray1, ray2.T))
                            ioas += [ioa]
                            pair_ind = np.where(np.all((pairs_tested - pair[np.newaxis]) == 0, axis=1))
                            var_IOAs[pair_ind] = ioa
                            var_IOA_resids[pair_ind] = residuals
                            print_progress(num, len(pairs_tested))
                            num += 1
                plt.sca(ax)
                title = f"{var_lbl}=[{np.round(var_start, 2)}, {np.round(var_stop, 2)}]"
                plt.title(title)
                # plt.savefig(os.path.join(self.dirname, title + ".svg"))
                plt.tight_layout()
                plt.show()
                ioas = np.squeeze(np.array(ioas))
                print_progress(num, 2 * len(pairs_tested))
        breakpoint()
        # projected_coords are the direction vectors
        # find the cluster centers: they are the position vectors
        position_vector = np.array([centers[lbl] for lbl in cone_cluster_data.label.values])
        direction_vector = np.array([projected_coords[lbl] for lbl in cone_cluster_data.label.values])
        position_vector -= rotated_origin
        # add position and direction vector data to the ommatidial data spreadsheet
        self.ommatidial_data['position_vector'] = position_vector.tolist()
        self.ommatidial_data['direction_vector'] = direction_vector.tolist()
        self.ommatidial_data.to_pickle(os.path.join(self.dirname, "ommatidial_data.pkl"))
        self.ommatidial_data.to_csv(os.path.join(self.dirname, "ommatidia_data.csv"), index=False)
        # calculate the 'anatomical IOA' as the magnitude of the horizontal and vertical components
        IOA_anatomical = np.sqrt(vertical_IOAs**2 + horizontal_IOAs**2)
        data_to_save = dict()
        cols = ['cluster1', 'cluster2',
                'cluster1_x', 'cluster1_y', 'cluster1_z',
                'cluster2_x', 'cluster2_y', 'cluster2_z', 
                'cluster1_theta', 'cluster1_phi', 'cluster1_radii',
                'cluster2_theta', 'cluster2_phi', 'cluster2_radii',
                'anatomical_angle', 'orientation',
                'horizontal_angle', 'horizontal_IOA_residuals',
                'vertical_angle', 'vertical_IOA_residuals'
                ]
        # store the interommatidial data with one pair per row
        for col in cols:
            data_to_save[col] = []
        for num, (pair, anatomical, orientation,
                  horizontal, horizontal_resids,
                  vertical, vertical_resids) in enumerate(
                zip(pairs_tested, IOA_anatomical, orientations,
                    horizontal_IOAs, horizontal_IOA_residuals,
                    vertical_IOAs, vertical_IOA_residuals)):
            ind1, ind2 = pair
            ind1 = np.where(self.ommatidial_data.label.values == ind1)[0][0]
            ind2 = np.where(self.ommatidial_data.label.values == ind2)[0][0]
            cluster1 = self.ommatidial_data.loc[ind1]
            cluster2 = self.ommatidial_data.loc[ind2]
            for lbl, vals in zip(
                    cols,
                    [ind1, ind2,
                     cluster1.x_center, cluster1.y_center, cluster1.z_center,
                     cluster2.x_center, cluster2.y_center, cluster2.z_center, 
                     cluster1.theta_center, cluster1.phi_center, cluster1.r_center,
                     cluster2.theta_center, cluster2.phi_center, cluster2.r_center,
                     anatomical, orientation, horizontal, horizontal_resids,
                     vertical, vertical_resids]):
                data_to_save[lbl] += [vals]
            print_progress(num, len(pairs_tested))
        print("\n")
        interommatidial_data = pd.DataFrame.from_dict(data_to_save)
        interommatidial_data.to_csv(os.path.join(project_folder, "interommatidial_data.csv"),
                                    index=False)
        interommatidial_data.to_csv(os.path.join(project_folder, "interommatidial_data.pkl"),
                                    index=False)
                    
    def measure_interommatidia_fast(self, test=False, display=False):
        """Use the anatomical axes to quickly measure interommatidial angles.

        
        Parameters
        ----------
        test : bool, default=False
            Whether to run troublshooting options.
        display : bool, default=False
            Whether to display the processed information.
        
        Attributes
        ----------
        
        """
        # model the 3D direction vectors as a function of
        # azimuth and elevation
        # testing: first, plot the horizontal and vertical direction components
        # of each anatomical vector
        # horizontal component
        thetas, phis = self.ommatidial_data[['theta', 'phi']].values.T
        pts = self.ommatidial_data[['x', 'y', 'z']].values
        axes = self.ommatidial_data.anatomical_axis.values
        axes = np.array([ax for ax in axes])
        # get angles
        h_angs = np.arctan2(axes[:, 1], axes[:, 0])
        v_angs = np.arctan2(axes[:, 2], axes[:, 0])
        # center points using sphere center
        pt_model = SphereFit(np.copy(pts))
        center = pt_model.center
        pts -= center
        if test:
            scatter = ScatterPlot3d(pts, size=5)
            scatter.show()
            # plot polar coordinates 
            dth, dph, dr = rectangular_to_spherical(axes).T
            th, ph, r = rectangular_to_spherical(pts).T
            # plot centers
            fig = plt.figure()
            plt.scatter(th, ph, c=r)
            # plot vectors
            l = .005
            lows = [th - l*dth, ph - l*dph]
            highs = [th + l*dth, ph + l*dph]
            plt.plot([lows[0], highs[0]], [lows[1], highs[1]], color='k', alpha=.1)
            plt.gca().set_aspect('equal')
            plt.show()
        # get the svd
        # use the Points class to properly rotate the COM
        # uu, dd, vv = np.linalg.svd(pts)
        # # use vv to rotate pts and axes
        # pts = np.dot(pts, vv)[:, [1, 2, 0]]
        # axes = np.dot(axes, vv)[:, [1, 2, 0]]

        # rotate points using the center of mass:
        # 1. find center of mass
        com = pts.mean(0)
        # 2. rotate along x axis (com[0]) until z (com[2]) = 0
        ang1 = np.arctan2(com[2], com[1])
        com1 = rotate(com, ang1, axis=0)
        rot1 = rotate(pts, ang1, axis=0).T
        axes_rot1 = rotate(axes, ang1, axis=0).T
        # 3. rotate along z axis (com[2]) until y (com[1]) = 0
        ang2 = np.arctan2(com1[1], com1[0])
        pts = rotate(rot1, ang2, axis=2).T
        axes = rotate(rot1, ang2, axis=2).T
        if test:
            # plot scatter plot of all the vectors
            fig, axs = plt.subplots(ncols=2)
            h_ax, v_ax = axs
            random_order = np.random.permutation(np.arange(len(thetas)))
            for ax, vals, title in zip(
                    [h_ax, v_ax],
                    [h_angs, v_angs],
                    ['Horizontal', 'Vertical']):
                ax.scatter(thetas[random_order], phis[random_order],
                           c=vals[random_order], alpha=.1)
                ax.set_aspect('equal')
                ax.set_title(title)
            # plot scatter plots for each component of the vectors
            fig, axs = plt.subplots(ncols=3)
            for ax, vals, title in zip(
                    axs,
                    axes.T,
                    ['x', 'y', 'z']):
                scatter = ax.scatter(thetas[random_order], phis[random_order],
                                     c=vals[random_order], alpha=.1)
                ax.set_aspect('equal')
                ax.set_title(title + ' component')
                plt.colorbar(scatter, ax=ax)
            # plot each component in a narrow band of elevations
            included = np.abs(phis-phis.mean()) < np.pi/128
            fig, axs = plt.subplots(ncols=3)
            for ax, vals, title in zip(
                    axs,
                    axes.T,
                    ['x', 'y', 'z']):
                ax.scatter(thetas[included], vals[included], color='k')
                ax.set_aspect('equal')
                ax.set_title(title + ' component')
            plt.show()
        # give vertical and horizontal slice numbers to each interommatidial pair
        labels = np.copy(self.ommatidial_data.label.values)
        # get interommatidial pairs 
        pair_labels = self.pairs_tested[:]
        # remove any labels that are not in the ommatidial_data dataset
        in_dataset = np.all(np.isin(pair_labels, labels), axis=1)
        pair_labels = pair_labels[in_dataset]
        # grab the relevant cluster centers and ommatidial axes
        pair_inds = np.searchsorted(labels, pair_labels)
        pair_pts = pts[pair_inds]
        pair_axes = axes[pair_inds]
        # calculate centers as mean and orientations as normalized difference
        pair_centers = pair_pts.mean(1)
        # get pair orientation in polar coordinates
        pair_polar = rectangular_to_spherical(np.vstack(pair_pts)).reshape(pair_pts.shape)
        pair_orientations = pair_polar[:, 0, :2] - pair_polar[:, 1, :2]
        pair_diams = np.linalg.norm(pair_pts[:, 0] - pair_pts[:, 1], axis=-1)
        pair_orientations /= np.linalg.norm(pair_orientations, axis=1)[:, np.newaxis]
        pair_orientations = np.arctan2(
            pair_orientations[:, 1], pair_orientations[:, 0])
        # for each section, project anatomical vectors onto the parallel plane
        # and ignore the flattened dimension. For instance, for a vertical
        # section, we flatten the dimension of width, ignoring x values and
        # then find the polar angle of all the flattened points (y, z), rotating
        # to avoid the polar boundary problems. Store the resulting angle
        # between projected axis vectors as a measure of one component of the
        # io angle
        horizontal_angles = {}          # make a dictionary of horizontal angles
        vertical_angles = {}          # make a dictionary of vertical angles
        # make an empy dataset with a row for each interommatidial pair
        cols = ["lbl1", "lbl2",
                "pt1_x", "pt1_y", "pt1_z", "pt1_th", "pt1_ph", "pt1_r",
                "pt1_dx", "pt1_dy", "pt1_dz",
                "pt2_x", "pt2_y", "pt2_z", "pt2_th", "pt2_ph", "pt2_r",
                "pt2_dx", "pt2_dy", "pt2_dz",
                "orientation", "diameter",
                "angle_h", "angle_v", "angle_total"]
        interommatidial_data = pd.DataFrame(
            np.zeros((len(pair_labels), len(cols))), columns=cols)
        # store the pair data
        for col, val in zip(
                cols[:2], [pair_labels[:, 0], pair_labels[:, 1]]):
            interommatidial_data[col] = val
        # store data to identify and locate each interommatidial pair
        for pt, arr, polar, axes in zip(
                ['1', '2'], [pair_pts[:, 0], pair_pts[:, 1]],
                [pair_polar[:, 0], pair_polar[:, 1]],
                [axes[:, 0], axes[:, 1]]):
            for col, vals in zip(
                    [f'pt{pt}_x', f'pt{pt}_y', f'pt{pt}_z'],
                    arr.T):
                interommatidial_data[col] = vals
            for col, vals in zip(
                    [f'pt{pt}_th', f'pt{pt}_ph', f'pt{pt}_r'],
                    polar.T):
                interommatidial_data[col] = vals
            for col, vals in zip(
                    [f'pt{pt}_dx', f'pt{pt}_dy', f'pt{pt}_dz'],
                    axes.T):
                interommatidial_data[col] = vals
            # store ommatidial axis data for each interommatidial pair
        interommatidial_data.orientation = pair_orientations
        num = 0
        # process interommatidial pairs in sections
        print("\nProcessing interommatidial data:")
        for flat_dim, angle_col in zip(
                [1, 2], ['angle_h', 'angle_v']):
            # find limits of the flattened dimension
            flat_vals = pair_centers[..., flat_dim]
            limits = np.linspace(flat_vals.min(), flat_vals.max(), 21)
            # find pair centers within the limits of non-flat dimensions
            for lim_low, lim_high in zip(
                    limits[:-1], limits[1:]):
                include = pair_centers[:, flat_dim] >= lim_low
                include *= pair_centers[:, flat_dim] <= lim_high
                labels_in_slice = pair_labels[include]
                centers_in_slice = pair_centers[include]
                pairs_in_slice = pair_inds[include]
                polar_in_slice = pair_polar[include]
                pts_in_slice = pair_pts[include]
                axes_in_slice = pair_axes[include]
                # get dimensions other than the flat dimension
                other_dims = np.arange(pts.shape[-1])
                other_dims = other_dims[other_dims != flat_dim]
                other_dim, depth = pts_in_slice[..., other_dims].T
                # plot the axes
                x, y = pts_in_slice[..., other_dims].transpose(2, 0, 1)
                dx, dy = axes_in_slice[..., other_dims].transpose(2, 0, 1)
                # convert x and y to polar angle
                theta = np.arctan2(x, y)
                # convert dx and dy to get ommatidial angle
                phi = np.arctan2(dx, dy)
                # check that all theta are above 0, if not wrap around
                theta_neg = theta < 0
                theta[theta_neg] = theta[theta_neg] + 2*np.pi
                # fit a positive polynomial to the axis orientations
                model, resids = positive_fit(
                    theta.flatten(), phi.flatten()[:, np.newaxis])
                new_phi = model(theta)
                # get dy and dx based on new phi
                norm = np.sqrt(dx**2 + dy**2)
                new_dy = norm * np.cos(new_phi)
                new_dx = norm * np.sin(new_phi)
                # new_axes has shape = (num samples, 2 cones, 2 dimensions)
                axes_new = np.array([new_dx, new_dy]).transpose(1, 2, 0) 
                # measure goodness of fit using r squared
                corr_r, corr_p = stats.pearsonr(
                    phi.flatten(), new_phi.flatten())
                if test:
                    l = .05
                    fig, axs = plt.subplots(ncols=2)
                    cart_ax, polar_ax = axs
                    data_line = cart_ax.plot(
                        [x.flatten()-l*dx.flatten(),
                         x.flatten()+l*dx.flatten()],
                        [y.flatten()-l*dy.flatten(),
                         y.flatten()+l*dy.flatten()],
                        color=green, alpha=.5)
                    cart_ax.scatter(x, y)
                    cart_ax.set_aspect('equal')
                    # plot polar coords and projected ommatidial axes
                    polar_ax.scatter(theta, phi)
                    inds = np.argsort(theta.flatten())
                    polar_ax.plot(
                        theta.flatten()[inds],
                        new_phi.flatten()[inds], color='k')
                    polar_ax.set_aspect('equal')
                    polar_ax.set_title(f"r={corr_r}, p={corr_p}")
                    # plot the coords and new phi
                    model_line = cart_ax.plot(
                        [x.flatten()-l*new_dx.flatten(),
                         x.flatten()+l*new_dx.flatten()],
                        [y.flatten()-l*new_dy.flatten(),
                         y.flatten()+l*new_dy.flatten()],
                        color=red, alpha=.5)
                    # cart_ax.scatter(x, y)
                    cart_ax.set_aspect('equal')
                    # cart_ax.legend()
                    plt.show()
                    scatter = ScatterPlot3d(pts, size=1)
                    scatter2 = ScatterPlot3d(pts_in_slice[:, 0],
                                             color=(1, 0, 0, 1),
                                             size=10, window=scatter.window)
                    scatter3 = ScatterPlot3d(pts_in_slice[:, 1],
                                             color=(0, 0, 1, 1),
                                             size=10, window=scatter.window)
                # measure IO angles per pair
                # measure angle between new_axes[:, 0] and new_axes[:, 1]
                angles = []
                inds = []
                for pair, lbls in zip(axes_new, labels_in_slice):
                    angle = angle_between(pair[0], pair[1])
                    # find the row in dataset corresponding to this pair
                    lbl1, lbl2 = lbls
                    i = interommatidial_data.lbl1.values == lbl1
                    i *= interommatidial_data.lbl2.values == lbl2
                    if sum(i) > 0:
                        inds += [np.where(i)[0][0]]
                        angles += [angle]
                interommatidial_data.loc[inds, angle_col] = angles
                num += 1
                print_progress(num, 40)
        # calculate the total io angle
        dx, dy = interommatidial_data.angle_h, interommatidial_data.angle_v
        interommatidial_data.angle_total = np.sqrt(dx**2 + dy**2)
        # 
        if test:
            # rotate all negative orientations
            interommatidial_data.orientation = abs(interommatidial_data.orientation)
            # plot the horizontal and vertical IO angle components
            fig, axes = plt.subplots(ncols=2)
            axes[0].hist2d(interommatidial_data.orientation * 180 / np.pi,
                            interommatidial_data.angle_h * 180 / np.pi,
                            color='k', bins=50, cmap='Greys', edgecolor='w')
            axes[0].set_title("Horizontal Angles ($\degree$)")
            axes[0].set_xlabel("Orientation ($\degree$)")
            axes[1].hist2d(interommatidial_data.orientation * 180 / np.pi,
                            interommatidial_data.angle_v * 180 / np.pi,
                            color='k', bins=50, cmap='Greys', edgecolor='w')
            axes[1].set_title("Vertical Angles ($\degree$)")
            axes[1].set_xlabel("Orientation ($\degree$)")
            plt.tight_layout()
            plt.show()
            # plot the total IO angle per orientation
            fig, axes = plt.subplots(ncols=1)
            axes.hist2d(interommatidial_data.orientation * 180 / np.pi,
                            interommatidial_data.angle_total * 180 / np.pi,
                            color='k', bins=50, cmap='Greys', edgecolor='w')
            axes.set_title("Total IO Angles ($\degree$)")
            axes.set_xlabel("Orientation ($\degree$)")
            plt.tight_layout()
            plt.show()            
        # store the dataset
        csv_filename = os.path.join(self.dirname, "interommatidial_data.csv")
        interommatidial_data.to_csv(csv_filename, index=False)
        interommatidial_data.to_pickle(csv_filename.replace(".csv", ".pkl"))


    def plot_raw_data(self, three_d=False):
        """Function for plotting the imported data with a 3D option.


        Parameters
        ----------
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the data in 3D.
        """
        if three_d:
        # show the 3D scatterplot
            scatter = ScatterPlot3d(self.points[:], title="Imported Stack")
            scatter.show()
        # show the 2D scatterplot, color coding the 3rd dimension
        xs, ys, zs = self.points[:].T
        order = np.argsort(zs)
        scatter = plt.scatter(xs[order], ys[order], c=zs[order],
                              alpha=.1, marker='.')
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.show()


    def plot_cross_section(self, three_d=False, residual_proportion=.5):
        """Plot the points near the cross section fit along the eye surface.


        Parameters
        ----------
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the cross section in 3D.
        residual_proportion : float, default
            Proportion of the residuals to include in cross section, 
            affecting its thickness.
        """
        # show points around residuals
        percentage = residual_proportion * 100 # convert
        low, high = np.percentile(self.residual[:], [50 - percentage/2, 50 + percentage/2])
        include = (self.residual[:] > low) * (self.residual[:] < high)
        include = np.where(include)[0]
        cross_section = self.points[:]
        cross_section = cross_section[include]
        theta = self.theta[include]
        phi = self.phi[include]
        resids = self.residual[:]
        if three_d:
            # in 3d
            scatter = ScatterPlot3d(cross_section, colorvals=resids[include],
                                    title="Cross Section Residuals", size=2)
            scatter.show()
        # in spherical coordinates
        fig = plt.figure()
        gridspec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[9, 1])
        img_ax = fig.add_subplot(gridspec[0, 0])
        colorbar_ax = fig.add_subplot(gridspec[0, 1])
        vmin, vmax = low, high
        img_ax.scatter(self.theta[:], self.phi[:], c=self.residual[:],
                       alpha=.5, marker='.', edgecolor='none')
        img_ax.set_title("Cross Section Residuals")
        colorbar_histogram(resids, vmin=resids.min(), vmax=resids.max(),
                           ax=colorbar_ax, bin_number=25, colormap='viridis')
        colorbar_ax.set_ylabel(f"Residuals (N={len(resids)})",
                               rotation=270)
        colorbar_ax.get_yaxis().labelpad = 15
        img_ax.set_aspect('equal')
        plt.show()


    def plot_ommatidial_clusters(self, three_d=False):
        """Plot the ommatidial clusters, color coded by cluster.


        Parameters
        ----------
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the cross section in 3D.
        """
        # plot the points in 3D, color coded with the new labels
        # scramble the labels to avoid clumping
        lbls = self.labels[:]
        # check for outliers based on first differences
        diffs = np.diff(sorted(lbls))
        # note: lbls often includes a huge number from recoding negative numbers
        lbls_set = np.arange(max(lbls) + 1)
        # randomize lbls and use 
        scrambled_lbls = np.random.permutation(lbls_set)
        new_lbls = scrambled_lbls[lbls]
        if three_d:
            # plot in 3d
            scatter = ScatterPlot3d(
                self.points[:], colorvals=new_lbls, cmap=plt.cm.tab20,
                title="Ommatidial Cluters")
            scatter.show()
        # and in 2d
        plt.scatter(self.theta, self.phi, c=new_lbls, cmap='tab20')
        plt.gca().set_aspect('equal')
        plt.show()


    def plot_ommatidial_data(self, three_d=False, image_size=1e4, scatter=False):
        """Plot the ommatidial data (lens area, IO angle, ...) in 2D histograms.


        Parameters
        ----------
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the cross section in 3D.
        image_size : float, default=1e4
            The size of the 2d histogram used for plotting the variables.
        scatter : bool, default=bool
            Whether to plot the variable as a scatterplot as opposed to a 2D histogram
        """
        data = self.ommatidial_data
        theta, phi, x, y, z = data[['theta', 'phi', 'x', 'y', 'z']].values.T
        pts = np.array([x, y, z]).T
        vars_to_plot = ['size', 'lens_area', 'spherical_IOA', 'skewness']
        # store 3D scatterplots to 
        scatters_3d = []
        for num, var in enumerate(vars_to_plot):
            # in 2D:
            # plot as subplots in a 2x2 grid
            colorvals = data[var].values
            # Remove any nans
            no_nans = np.isnan(colorvals) == False
            # plot subplot
            # ax = plt.subplot(2, 2, num + 1)
            summary = VarSummary(theta[no_nans], phi[no_nans], colorvals,
                                 suptitle=f"{var} (N={no_nans.sum()})",
                                 color_label=var, image_size=image_size, scatter=scatter)
            summary.plot()
            # scatter = ax.scatter(
            #     theta[no_nans], phi[no_nans],
            #     c=colorvals[no_nans], cmap='viridis', marker='.',
            #     edgecolor='none')
            # # formatting
            # ax.set_aspect('equal')
            # plt.colorbar(scatter)
            # sbn.despine(ax=ax, bottom=True, left=True)
            # ax.set_title(var)
            # in 3D:
            if three_d:
                scatter = ScatterPlot3d(
                    pts[no_nans], colorvals=colorvals[no_nans], title=var, size=10)
                scatters_3d += [scatter]
        plt.tight_layout()
        plt.show()
        # plot 3d plots one at a time
        for scatter in scatters_3d:
            scatter.show()


    def plot_interommatidial_data(self, three_d=False):
        """Plot the interommatidial data


        Parameters
        ----------
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the cross section in 3D.
        """
        interommatidial_data = self.interommatidial_data
        orientation = abs(interommatidial_data.orientation)
        CMAP = 'Greys'
        angles = interommatidial_data.angle_total.values * 180 / np.pi
        x1, y1, z1 = interommatidial_data[['pt1_x', 'pt1_y', 'pt1_z']].values.T
        x2, y2, z2 = interommatidial_data[['pt2_x', 'pt2_y', 'pt2_z']].values.T
        x, y, z = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
        arr = np.array([x, y, z]).T
        polar = rectangular_to_spherical(arr)
        theta, phi, radii = polar.T
        BINS = np.linspace(0, angles.max(), 50)
        BINS = 50
        # plot the horizontal and vertical IO angle components
        fig, axes = plt.subplots(ncols=2)
        axes[0].hist2d(orientation * 180 / np.pi,
                       interommatidial_data.angle_h * 180 / np.pi,
                       color='k', bins=BINS, cmap=CMAP, edgecolor='none')
                       # norm=colors.LogNorm())
        axes[0].set_title("Horizontal Angles ($\degree$)")
        axes[0].set_xlabel("Orientation ($\degree$)")
        axes[1].hist2d(orientation * 180 / np.pi,
                       interommatidial_data.angle_v * 180 / np.pi,
                       color='k', bins=BINS, cmap=CMAP, edgecolor='none')
                       # norm=colors.LogNorm())
        axes[1].set_title("Vertical Angles ($\degree$)")
        axes[1].set_xlabel("Orientation ($\degree$)")
        plt.tight_layout()
        plt.show()
        # plot the total IO angle per orientation
        fig = plt.figure()
        gridspec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[4, 1])
        img_ax = fig.add_subplot(gridspec[0, 0])
        colorbar_ax = fig.add_subplot(gridspec[0, 1])
        img_ax.hist2d(orientation * 180 / np.pi,
                      angles,
                      color='k', bins=BINS, cmap=CMAP, edgecolor='none')
                             # norm=colors.LogNorm())
        # sbn.distplot(angles, vertical=True, ax=colorbar_ax, bins=BINS)
        # colorbar_histogram(angles, vmin=angles.min(), vmax=angles.max(),
        #                    bin_number=50, ax=colorbar_ax, colormap=CMAP)
        colorbar_ax.hist(angles, bins=BINS, orientation='horizontal', color='k', alpha=1)
        colorbar_ax.set_ylim(0, angles.max())
        img_ax.set_title("Total IO Angles ($\degree$)")
        img_ax.set_xlabel("Orientation ($\degree$)")
        img_ax.set_ylim(0, angles.max())
        plt.tight_layout()
        plt.show()            
        # plot all of the interommatidial pairs, color coded by the total angle
        # get polar coordinates of the centers
        th1, ph1 = interommatidial_data[['pt1_th', 'pt1_ph']].values.T
        th2, ph2 = interommatidial_data[['pt2_th', 'pt2_ph']].values.T
        th, ph = (th1+th2)/2, (ph1+ph2)/2, 
        fig = plt.figure()
        gridspec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[4, 1])
        # horizontal IO
        img_ax = fig.add_subplot(gridspec[0, 0])
        colorbar_ax = fig.add_subplot(gridspec[0, 1])
        summary = VarSummary(
            th * 180/np.pi, ph * 180 / np.pi, angles, color_label='Total IO Angle',
            suptitle=f"Total IO Angle (N{len(angles)})")
        summary.plot()
        plt.show()
        # and in 3d:
        if three_d:
            scatter = ScatterPlot3d(arr, colorvals=angles, size=5,
                                    title=f"Total IO Angle (N{len(angles)})")
            scatter.show()


    def ommatidia_detecting_algorithm(self, polar_clustering=True,
                                      display=False, test=False, three_d=False):
        """Apply the 3D ommatidia detecting algorithm (ODA-3D).
        

        Parameters
        ----------
        polar_clustering : bool, default=True
            Whether to use spectral clustering or to simply use 
            the nearest cluster center for finding ommatidial clusers.
        display : bool, default=False
            Whether to display the data. Combine with three_d to 
            plot in 3D.
        test : bool, default=False
            Whether to run segments designed for troubleshooting.
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the data in 3D.
        """
        # 0. check the status of this project and offer to skip forward
        stage = 0
        # based on the loaded data
        conditions = {
            'stack': 'points' in dir(self),
            'cross-sections': 'theta' in dir(self),
            'cluster labels':'labels' in dir(self),
            'ommatidial data': 'ommatidial_data' in dir(self),
            'interommatidial data': 'interommatidial_data' in dir(self)}
        if any([cond for cond in conditions.values()]):
            print("The following datasets were found for this stack:")
            choices = []
            for num, (dataset, loaded) in enumerate(conditions.items()):
                if loaded:
                    print(f"{num + 1}. {dataset.capitalize()}")
                    choices += [num + 1]
            stage = None
            while stage not in choices + [0]:
                stage = input(
                    f"Enter the number {choices} to load from that stage "
                    "and continue processing or press 0 to start over: ")
                try:
                    stage = int(stage)
                except:
                    pass
        self.display = display
        # 1. check that the coordinates have been loaded
        if stage < 1:
            # use GUI to select a range of pixel values
            self.gui = StackFilter(self.fns)
            low, high = self.gui.get_limits()
            self.import_stack(low, high)
            self.save_database()
            if display:
                self.plot_raw_data(three_d)
        print("\nStack imported.")
        # 2. get cross sectional shell and 
        if stage < 2:
            self.get_cross_sections(chunk_process=False)
            self.save_database()
            if display:
                # show points within 50% of residuals
                self.plot_cross_section(three_d, residual_proportion=.5)
        print("\nCross-section loaded.")
        # 3. find the clusters corresponding to ommatidia
        if stage < 3:
            self.find_ommatidial_clusters(polar_clustering=polar_clustering)
            self.save_database()
            if display:
                self.plot_ommatidial_clusters(three_d)
        print("\nOmmatidial clusters loaded.")
        # 4. measure the ommatidia using their cluster properties
        if stage < 4:
            self.measure_ommatidia()
            self.save_database()
            if display:
                self.plot_ommatidial_data(three_d)
        print("\nOmmatidial data loaded.")
        # 5. measure interommatidial pairs using their centers and longitudinal axes
        if stage < 5:
            self.measure_interommatidia_fast(test=test)
            self.save_database()
            if display:
                data = self.interommatidial_data
        self.plot_interommatidial_data(three_d=three_d)
            
    
