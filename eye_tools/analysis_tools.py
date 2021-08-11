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
import PIL
from PIL import Image
import pickle
import subprocess
import sys
from tempfile import mkdtemp

from matplotlib import colors, mlab
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import skimage
from skimage.draw import ellipse as Ellipse
from skimage.feature import peak_local_max
from sklearn import cluster

from scipy import interpolate, optimize, ndimage, signal, spatial, stats
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass




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
    inclination = np.arccos(zs / radius)
    azimuth = np.arctan2(ys, xs)
    polar = np.array([inclination, azimuth, radius]).T
    return polar


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
        breakpoint()
        # get radius
        # from the model:
        x0, y0, z0, var = solution
        self.radius_model = np.sqrt(var + x0**2 + y**2 + z**2)
        # empirically:
        self.center = solution[:-1, 0]
        self.radii = np.linalg.norm(self.pts - self.center[np.newaxis], axis=-1)
        self.radius = np.mean(self.radii)
        # center the pts 
        self.pts -= self.center
        self.center[:] = 0
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
        # from the model:
        x0, y0, z0, var = solution
        self.radius_model = np.sqrt(var + x0**2 + y0**2 + z0**2)
        # empirically:
        self.center = solution[:-1, 0]
        self.radii = np.linalg.norm(self.pts - self.center[np.newaxis], axis=-1)
        self.radius = np.mean(self.radii)
        # center the pts 
        self.pts -= self.center
        self.center[:] = 0
        # rotate the points about the center until com is ideal
        self.center_com()
        # then perform spherical transformation
        self.get_polar()

    def center_com(self):
        """Rotate points using the center of mass."""
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
        self.radii = np.linalg.norm(self.pts, axis=-1)
        self.inclination = np.arccos(zs / self.radii)
        self.azimuth = np.arctan2(ys, xs)
        self.polar = np.array([self.inclination, self.azimuth, self.radius]).T

    def rasterize(self, image_size=10**4, weights=None):
        """Rasterize coordinates onto a grid defined by min and max vals.


        Parameters
        ----------
        image_size : int, default=1e4
            The number of pixels in the image.
        weights : list, shape=(N, 1), default=None
            Optional weights associated with each point.

        Returns
        -------
        raster : np.ndarray
            The 2D histogram of the points, optionally weighted by self.vals.
        (xs, ys) : tuple
            The x and y coordinates marking the boundaries of each pixel. 
            Useful for rendering as a pyplot.pcolormesh.
        """
        arr = self.polar
        x, y = arr.T[axes]
        # get coordinate ranges for the appropriate aspect ratio
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        # figure out side lengths needed for input image size
        ratio = y_range / x_range
        x_len = int(np.round(np.sqrt(image_size/ratio)))
        # get x and y ranges corresponding to image size
        xs = np.linspace(x.min(), x.max(), x_len)
        self.raster_pixel_length = xs[1] - xs[0]
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

    def fit_surface(self, image_size=1e4):
        """Cubic interpolate surface of one axis using the other two.


        Parameters
        ----------
        image_size : int, default=1e4
            The number of pixels in the image.

        Attributes
        ----------
        avg : array_like
            The rolling average
        """
        arr = self.polar
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
        low, high = np.percentile(avg[:, 2], [.5, 99.5])
        self.avg = avg[np.logical_and(avg[:, 2] >= low, avg[:, 2] < high)]
        avg, (xs, ys) = self.rasterize(image_size=image_size, weights=self.radii)
        breakpoint()
        self.avg_x, self.avg_y, self.avg_z = self.avg.T

    def surface_predict(self, xvals=None, yvals=None, image_size=1e4):
        """Find the approximate zvalue given arbitrary x and y values."""
        if "avg_x" not in dir(self):
            self.fit_surface(image_size=image_size)
        if (xvals is None) or (yvals is None):
            arr = self.polar
            xvals, yvals, zvals = arr.T
        points = np.array([xvals, yvals]).T
        self.surface = interpolate.griddata(
            self.avg[:, :2], self.avg_z, points, method='cubic')
        return self.surface

    def get_polar_cross_section(self, thickness=.1, pixel_length=.01):
        """Find best fitting surface of radii using phis and thetas."""
        # self.fit_surface(mode='polar', pixel_length=pixel_length)
        self.surface_predict()
        # find distance of datapoints from surface (ie. residuals)
        self.residuals = self.radii - self.surface
        # choose points within 'thickness' proportion of residuals
        self.cross_section_thickness = np.percentile(
            abs(self.residuals), thickness * 100)
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
    assert (vmin < np.inf) and (vmax < np.inf), (
        "Input vmin and vmax should be finite floats")
    if ax is None:
        ax = plt.gca()
    if not isinstance(colorvals, np.ndarray):
        colorvals = np.asarray(colorvals)
    # use evenly spaced bins and counts
    bins = np.linspace(colorvals.min(), colorvals.max(), bin_number + 1)
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
                 rotate_com=True):
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
        self.vals = None
        self.polar = None
        if polar is not None:
            self.polar = polar
            self.theta, self.phi, self.radii = self.polar.T
        if sphere_fit:
            # fit sphere
            self.sphere_model = SphereFit(
                self.pts - self.pts.mean(0))
            self.radius = self.sphere_model.radius
            self.center = self.sphere_model.center
            # center points using the center of that sphere
            self.pts -= self.center
            self.center = self.center - self.center
        if spherical_conversion:
            # optionally:
            if rotate_com:
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
        if self.vals is None:
            breakpoint()
            out = Points(self.pts[key], polar=self.polar[key],
                         rotate_com=False, spherical_conversion=False)
        else:
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
                  weights=None):
        """Rasterize coordinates onto a grid defined by min and max vals.


        Parameters
        ----------
        polar : bool, default=True
            Whether to rasterize polar (vs. rectangular) coordinates.
        image_size : int, default=1e4
            The number of pixels in the image.
        weights : list, shape=(N, 1), default=None
            Optional weights associated with each point.

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
        # x_len = int(np.round(np.sqrt(image_size/ratio)))
        # y_len = int(np.round(ratio * x_len))
        x_len = int(np.round(np.sqrt(image_size/ratio)))
        # get x and y ranges corresponding to image size
        xs = np.linspace(x.min(), x.max(), x_len)
        self.raster_pixel_length = xs[1] - xs[0]
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
        arr = self.polar
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
        low, high = np.percentile(avg[:, 2], [.5, 99.5])
        self.avg = avg[np.logical_and(avg[:, 2] >= low, avg[:, 2] < high)]
        self.avg_x, self.avg_y, self.avg_z = self.avg.T

    def surface_predict(self, xvals=None, yvals=None, image_size=1e4):
        """Find the approximate zvalue given arbitrary x and y values."""
        if "avg_x" not in dir(self):
            self.fit_surface(image_size=image_size)
        if (xvals is None) or (yvals is None):
            arr = self.polar
            xvals, yvals, zvals = arr.T
        points = np.array([xvals, yvals]).T
        self.surface = interpolate.griddata(
            self.avg[:, :2], self.avg_z, points, method='cubic')
        return self.surface

    def get_polar_cross_section(self, thickness=.1, pixel_length=.01):
        """Find best fitting surface of radii using phis and thetas."""
        # self.fit_surface(mode='polar', pixel_length=pixel_length)
        self.surface_predict()
        # find distance of datapoints from surface (ie. residuals)
        self.residuals = self.radii - self.surface
        # choose points within 'thickness' proportion of residuals
        self.cross_section_thickness = np.percentile(
            abs(self.residuals), thickness * 100)
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
                self.mask = self.mask > self.mask.mean()
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
        # find the contours in the mask image
        contour = skimage.measure.find_contours(
            (255/self.mask.max()) * self.mask.astype(int), 256/2)
        # escape if no contours were found
        assert len(contour) > 0, "could not find enough points in the contour"
        # use the longest contour
        contour = max(contour, key=len).astype(int)
        self.eye_outline = np.round(contour).astype(int) # pixel coords
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
                      square_lattice=False, high_pass=False):
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
        fft = np.fft.fft2(self.image_bw)
        # shift frequencies so that low frequencies are central and high
        # frequencies are peripheral
        fft_shifted = np.fft.fftshift(fft)
        # calculate reciprocal frequencies using their distance to the center
        self.__freqs = np.array(np.meshgrid(
            np.fft.fftfreq(self.image_bw.shape[1], self.pixel_size),
            np.fft.fftfreq(self.image_bw.shape[0], self.pixel_size)))
        self.__freqs = np.array(self.__freqs, dtype=float)
        self.__freqs = np.fft.fftshift(self.__freqs)
        # calculate grating orientations for the reciprocal image
        self.__orientations = np.arctan2(self.__freqs[1], self.__freqs[0])
        self.__freqs = np.linalg.norm(self.__freqs, axis=0)
        i = self.__orientations < 0    # indices of the negative half
        self.__orientations[i] = self.__orientations[i] + np.pi # make positive
        # the reciprocal image is the magnitude of the frequency shifted 2D FFT
        self.reciprocal = abs(fft_shifted)
        # remove the low frequency gratings corresponding to the horizontal
        # and vertical limits of the image by replacing with the mean
        midv, midh = np.round(np.array(self.reciprocal.shape)/2).astype(int)
        pad = 2
        newv = self.reciprocal[[midv-pad, midv+pad]].mean(0)
        newh = self.reciprocal[:, [midh-pad, midh+pad]].mean(1)
        self.reciprocal[midv - (pad - 1): midv + pad] = newv[np.newaxis]
        self.reciprocal[:, midh - (pad - 1): midh + pad] = newh[:, np.newaxis]
        height, width = self.reciprocal.shape
        # instead of blurring, just use the autocorrelation
        self.reciprocal = signal.correlate(
            self.reciprocal, self.reciprocal, mode='same', method='fft')
        # apply a guassian blur to exclude the effect of noise
        self.reciprocal = ndimage.gaussian_filter(
            self.reciprocal, sigma=2)
        # find the peaks of the upper half of the reciprocal image
        peaks = peak_local_max(
            self.reciprocal[:round(height/2)], num_peaks=3)
        ys, xs = peaks.T
        peak_vals = self.reciprocal[ys, xs]
        order = np.argsort(peak_vals)[::-1]
        xs = xs[order]
        ys = ys[order]
        # TODO: fix this until it works
        # remove any key frequencies that are too low
        key_freqs = self.__freqs[:round(height/2)][ys, xs]
        key_freqs = key_freqs[key_freqs > .01]
        i = np.argsort(key_freqs)
        if square_lattice:
            # if square, use the 2 fundamental frequencies
            self.__fundamental_frequencies = key_freqs[i][:2]
        else:
            # if hexagonal, use the 3 fundamental frequencies
            self.__fundamental_frequencies = key_freqs[i][:3]
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
        # use a minimum distance based on the wavelength of the
        # fundamental grating
        self.ommatidial_diameter_fft = 1 / self.__fundamental_frequencies.mean()
        dist = self.ommatidial_diameter_fft / self.pixel_size
        dist /= 3
        smooth_surface = self.filtered_image
        if not bright_peak:
            smooth_surface = smooth_surface.max() - smooth_surface
        self.ommatidial_inds = peak_local_max(
            smooth_surface, min_distance=int(round(dist)),
            exclude_border=False)
        # remove points outside of the mask
        ys, xs = self.ommatidial_inds.T
        self.ommatidial_inds = self.ommatidial_inds[self.mask[ys, xs]]
        # store ommatidia coordinates in terms of the pixel size
        self.ommatidia = self.ommatidial_inds * self.pixel_size

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

    def ommatidia_detecting_algorithm(self, bright_peak=True, fft_smoothing=5,
                                      square_lattice=False, high_pass=False,
                                      num_neighbors=3, sample_size=100,
                                      plot=False, plot_fn=None):
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
        print()
        # finally, locate ommatidia using the FFT
        self.get_ommatidia(bright_peak=bright_peak,
                           fft_smoothing=fft_smoothing,
                           square_lattice=square_lattice,
                           high_pass=high_pass)
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
            colorvals = self.ommatidial_diameters
            vmin, vmax = np.percentile(colorvals, [.5, 99.5]) # use 99%
            ys, xs = self.ommatidial_inds.T
            img_ax.imshow(self.image_bw, cmap='gray')
            dot_radii = (self.ommatidial_diameters / (2 * self.pixel_size))
            dot_areas = np.pi * dot_radii ** 2
            img_ax.scatter(xs, ys, marker='.', c=colorvals, s=.5 * dot_areas,
                           # vmin=colorvals.min(), vmax=colorvals.max(),
                           vmin=vmin, vmax=vmax,
                           cmap='plasma')
            img_ax.set_xticks([])
            img_ax.set_yticks([])
            # crop image around the x and y coordinates, with .1 padding
            width = xs.max() - xs.min()
            height = ys.max() - ys.min()
            xpad, ypad = .05 * width, .05 * height
            img_ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
            img_ax.set_ylim(ys.max() + ypad, ys.min() - ypad)
            # make the colorbar
            # colorbar_histogram(colorvals, colorvals.min(), colorvals.max(),
            colorbar_histogram(colorvals, vmin, vmax,
                               ax=colorbar_ax, bin_number=25, colormap='plasma')
            colorbar_ax.set_ylabel(f"Ommatidial Diameter (N={len(xs)})",
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
        self.mask.load_mask(mask_fn=self.eye_mask_fn, arr=self.eye_mask)
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
        self.database_fn = database_fn
        # import the stack
        Stack.__init__(self, **kwargs)
        # load the h5 database
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
        # set some defaults if not loaded
        if "points" not in dir(self):
            self.points = self.database.create_dataset(
                "points", data=np.zeros((0, 3)), dtype=float,
                chunks=True, maxshape=(None, 3))

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
            if num % 4 == 0:
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

    def get_cross_sections(self, thickness=1.0):
        """Approximate surface splitting the inner and outer sections.


        Uses 2D spline interpolation, modelling point radial distance 
        as a function of its polar coordinate.

        Parameters
        ----------
        thickness : float, default=.3
            Proportion of the residuals to include in the cross section 
            used for the ODA.
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
        # # store the sphere's center
        # self.center = center
        # center_dir = center / np.linalg.norm(center)
        # # center the points
        # self.points -= self.center[np.newaxis, :]
        # subset -= center[np.newaxis, :]
        # 2. Convert points to spherical coordinates
        # make a Points object of the subset
        pts = Points(subset)    # performs spherical conversion 
        # 3. Spline interpolate radii as function of theta and phi
        sphere.surface_predict(image_size=1e4)
        self.shell = pts
        # 4. find the spherical coordinates of all points
        if "theta" in dir(self):
            del self.theta, self.phi, self.radii, self.residual
            del self.database['theta'], self.database['phi'], self.database['radii'], self.database['residual']
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
        predicted_radii = pts.surface_predict(
            xvals=self.theta, yvals=self.phi)
        self.residual[:] = self.radii - predicted_radii

    def find_ommatidial_clusters(self, polar_clustering=True,
                                 window_length=np.pi/4,
                                 window_pad=np.pi/20,
                                 image_size=1e5, mask_blur_std=2):
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
        """
        # assume that the shell has been loaded
        assert "theta" in self.database.keys(), (
            "No polar coordinates found. Try running "
            f"{self.get_cross_sections} first or running "
            f"{self.ommatidia_detecting_algorithm}")
        # get elevation and inclination ranges
        theta_min, theta_max = np.percentile(self.theta, [0, 100])
        phi_min, phi_max = np.percentile(self.phi, [0, 100])
        # make 2D images and apply ODA to rotated sections
        self.sections = []
        # iterate through the windows +/- padding
        # if it exists already but is the wrong length, delete and make new
        if "include" in dir(self):
            if len(self.points) != len(self.include):
                del self.include
                del self.database["include"]
        # store a binary filter array
        if "include" not in dir(self):
            self.include = self.database.create_dataset(
                "include", (len(self.points),), dtype=bool)
        # iterate through theta and phi ranges taking steps of window_length
        theta_low = theta_min
        phi_low = phi_min
        while theta_low < theta_max:
            theta_high = theta_low + window_length
            # get relevant theta values
            theta_center = np.mean([theta_low, theta_high])
            theta_displacement = np.pi/2 - theta_center
            while phi_low < phi_max:
                # get important phi values
                phi_high = phi_low + window_length
                phi_center = np.mean([phi_low, phi_high])
                phi_displacement = np.pi - phi_center
                # store inclusion criteria in database
                self.include[:] = True
                # azimuth filter
                self.include[:] *= (self.theta > theta_low - window_pad)
                self.include[:] *= (self.theta <= theta_high + window_pad)
                # elevation filter
                self.include[:] *= (self.phi > phi_low - window_pad)
                self.include[:] *= (self.phi <= phi_high + window_pad)
                # use the filter to get the points within the window
                include = np.where(self.include)
                # calculate angular displacements in order to rotate the center of mass
                if len(include[0]) > 100:
                    # get included subset of points
                    subset = np.array(self.points[include])
                    subset = rotate(subset, phi_displacement, axis=2).T
                    subset = rotate(subset, theta_displacement, axis=1).T
                    polar = rectangular_to_spherical(subset)
                    # get the centered polar coordinates
                    segment = Points(subset, sphere_fit=False, rotate_com=False,
                                     spherical_conversion=False, polar=polar)
                    # plt.scatter(segment.theta, segment.phi)
                    # plt.plot(theta_center, phi_center, "ro")
                    # plt.gca().set_aspect('equal')
                    # # plt.xlim(theta_low - window_pad, theta_high + window_pad)
                    # # plt.ylim(phi_low - window_pad, phi_high + window_pad)
                    # plt.show()
                    # rasterize an image using the polar 2D histogram
                    raster, (theta_vals, phi_vals) = segment.rasterize(
                        image_size=image_size)
                    # make an Eye object of the raster image to get
                    # ommatidia centers
                    pixel_size = phi_vals[1] - phi_vals[0] # in rads
                    # make a mask using the raster image
                    mask = raster > 0
                    mask = ndimage.gaussian_filter(mask.astype(float), 2)
                    mask = mask > .1
                    raster = 255 * (raster / raster.max())
                    raster = raster.astype('uint8')
                    # apply the ODA to the raster image
                    eye = Eye(arr=raster, pixel_size=pixel_size,
                              mask_arr=mask.astype(int), mask_fn=None)
                    eye.oda(plot=False)
                    # use the ommatidial centers to find the clusters 
                    centers = eye.ommatidia
                    # shift the coordinates using the min theta and phi
                    centers += [theta_vals.min(), phi_vals.min()]
                    if polar_clustering:
                        # use polar angles for clustering
                        clusterer = cluster.KMeans(
                            n_clusters=len(centers), init=centers)
                        lbls = clusterer.fit_predict(segment.polar[:, :2])
                        # randomize lbls and use 
                        lbls_set = np.array(sorted(set(lbls)))
                        scrambled_lbls = np.random.permutation(lbls_set)
                        new_lbls = []
                        for lbl in lbls:
                            new_lbls += [scrambled_lbls[lbl]]
                        plt.pcolormesh(theta_vals, phi_vals, raster.T)
                        plt.scatter(segment.theta, segment.phi, c=new_lbls,
                                    marker='.', cmap='tab20')
                        plt.scatter(centers[:, 0], centers[:, 1],
                                    marker='+', color='k')
                        plt.gca().set_aspect('equal')
                        plt.show()
                    else:
                        # use the nearest points as seeds in the KMeans
                        breakpoint()
                # update phi lower bound
                phi_low += window_length
            # update theta lower bound
            theta_low += window_length

    def ommatidia_detecting_algorithm(self, polar_clustering=True):
        """Apply the 3D ommatidia detecting algorithm (ODA-3D).
        

        Parameters
        ----------
        polar_clustering : bool, default=True
            Whether to use spectral clustering or to simply use 
            the nearest cluster center for finding ommatidial clusers.
        """
        # 1. check that the coordinates have been loaded
        import_coordinates = True
        if len(self.points) > 0:
            # if loaded, check if user wants to re-import
            resp = ''
            while resp not in [0, 1]:
                resp = input("Coordinates were loaded previously. "
                             "Do you want to re-import the stack? "
                             "Type 1 for yes, and 0 for no: ")
                resp = int(resp)
            import_coordinates = resp == 1
        if import_coordinates:
            # use GUI to select a range of pixel values
            self.gui = StackFilter(self.fns)
            low, high = self.gui.get_limits()
            self.import_stack(low, high)
        self.save_database()
        print("Stack imported.")
        # 2. once the points load, get cross sectional shell and 
        process_shell = True
        resp = ''
        if "theta" in dir(self):
            while resp not in [0, 1]:
                resp = input("The cross-section was loaded previously. "
                             "Do you want to re-process it? "
                             "Type 1 for yes, and 0 for no: ")
                resp = int(resp)
            process_shell = resp == 1
        if process_shell:
            self.get_cross_sections()
        self.save_database()
        print("Cross-section loaded.")
        # 3. then find the ommatidia's centers
        self.find_ommatidial_clusters(polar_clustering=polar_clustering)
        self.save_database()
