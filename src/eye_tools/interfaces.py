from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.backend_bases import NavigationToolbar2
import numpy as np
import os
import PIL
import scipy
import seaborn as sbn
from tempfile import mkdtemp

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
import pyqtgraph.opengl as gl
import pyqtgraph as pg


if 'app' not in globals():
    app = QApplication([])


# make a class for plotting a heatmap with distplots along the axes
class VarSummary():
    """
    """

    def __init__(self, xs, ys, colorvals, cmap='viridis', image_size=10**5,
                 color='k', center=True, color_label="Color", suptitle="Title",
                 vmin=None, vmax=None, scatter=False):
        self.vmin = vmin
        self.vmax = vmax
        self.x = xs
        self.y = ys
        self.pts = np.array([xs, ys]).T
        self.center = center
        if self.center:
            self.x_offset = self.x.mean()
            self.y_offset = self.y.mean()
            self.x -= self.x_offset
            self.y -= self.y_offset
            self.pts -= self.pts.mean(0)
        self.colorvals = colorvals
        self.image_size = image_size
        self.cmap = cmap
        self.color = color
        self.color_label = color_label
        self.suptitle = suptitle
        self.scatter = scatter

    def plot(self):
        x_range = self.x.max() - self.x.min()
        y_range = self.y.max() - self.y.min()
        # figure out side lengths needed for input image size
        ratio = y_range / x_range
        x_len = int(np.round(np.sqrt(self.image_size/ratio)))
        # get x and y ranges corresponding to image size
        xs = np.linspace(self.x.min(), self.x.max(), x_len)
        self.raster_pixel_length = xs[1] - xs[0]
        ys = np.arange(self.y.min(), self.y.max(), self.raster_pixel_length)
        xs = xs[:-1] + (self.raster_pixel_length / 2.)
        ys = ys[:-1] + (self.raster_pixel_length / 2.)
        xvals, yvals = np.meshgrid(xs, ys)
        self.xmin, self.xmax = xs.min(), xs.max()
        self.ymin, self.ymax = ys.min(), ys.max()
        self.xpad = .1 * (abs(self.xmax - self.xmin))
        self.ypad = .1 * (abs(self.ymax - self.ymin))
        self.xmin, self.xmax = self.xmin - self.xpad, self.xmax + self.xpad
        self.ymin, self.ymax = self.ymin - self.ypad, self.ymax + self.ypad
        # split figure into axes using a grid
        ratio = (self.ymax - self.ymin) / (self.xmax - self.xmin)
        width = 1 + 4 + .3
        height = 1 + 4 * ratio
        scale = 7/width
        self.fig = plt.figure(figsize=(scale * width, scale * height))
        self.gridspec = self.fig.add_gridspec(
            ncols=3, nrows=2,
            width_ratios=[1, 4, .3],
            height_ratios=[4*ratio, 1],
            wspace=0, hspace=0)
        # calculate ideal range for colorvals to share between vertical and horizontal axes
        cmin, cmax = self.colorvals.min(), self.colorvals.max()
        pad = .025 * (cmax - cmin)
        cmin -= pad
        cmax += pad
        self.cmin = cmin
        self.cmax = cmax
        # generate ticks for colorvals
        crange = cmax - cmin
        scale = np.round(np.log10(crange))
        cticks = np.linspace(np.round(cmin,  - int(scale - 1)),
                             np.round(cmax,  - int(scale - 1)), 4)[1:-1]
        # plot 2d heatmap
        self.heatmap_ax = self.fig.add_subplot(self.gridspec[0, 1])
        no_nans = np.isnan(self.colorvals) == False
        grid = scipy.interpolate.griddata(self.pts[no_nans],
                                          self.colorvals[no_nans],
                                          np.array([xvals, yvals]).T,
                                          method='linear')
        no_nans = np.isnan(grid) == False
        self.plot_heatmap(xs, ys, grid)
        # self.heatmap_ax.set_title(self.suptitle)
        # make colorbar/histogram
        self.colorbar_ax = self.fig.add_subplot(self.gridspec[0, 2])
        bins = np.linspace(cmin, cmax, 101)
        counts, bin_edges = np.histogram(self.colorvals, bins=bins)
        self.histogram = sbn.distplot(self.colorvals, kde=False, color=self.color,
                                      ax=self.colorbar_ax, vertical=True, bins=bins,
                                      axlabel=False)
        bin_edges = np.repeat(bins, 2)[1:-1]
        heights = np.repeat(counts, 2)
        self.colorbar_ax.plot(heights, bin_edges, color='w')
        vals = np.linspace(cmin, cmax, 100)
        self.colorbar_ax.pcolormesh([0, counts.max()], vals,
                                    vals[:, np.newaxis], cmap=self.cmap,
                                    zorder=0)
        # self.colorbar_ax.set_xlabel("Count")
        sbn.despine(ax=self.colorbar_ax, bottom=False)
        self.colorbar_ax.set_xticks([])
        self.colorbar_ylabel = self.colorbar_ax.set_ylabel(
            self.color_label, rotation=270, labelpad=20)
        self.colorbar_ax.yaxis.set_label_position("right")
        self.colorbar_ax.yaxis.tick_right()
        # self.colorbar_ylabel.set_rotation(270)
        # plot expected colorvals using bootstrapped CIs along vertical
        self.vertical_ax = self.fig.add_subplot(self.gridspec[0, 0],
                                                sharey=self.heatmap_ax)
        lows, mids, highs = [], [], []
        no_nans = np.isnan(grid.T) == False
        for row, no_nan in zip(grid.T, no_nans):
            # low, mid, high = bootstrapped_CI(row)
            low, mid, high = np.percentile(row[no_nan], [16, 50, 84])
            lows += [low]
            mids += [mid]
            highs += [high]
        # self.vertical_means = np.nanmean(grid, axis=0)
        # self.vertical_stds = np.nanstd(grid, axis=0)
        # self.vertical_std_err = 2 * self.vertical_stds / np.sqrt(no_nans.sum(0))
        # self.vertical_ax.plot(self.vertical_means, ys, color=self.color)
        # self.vertical_ax.fill_betweenx(
        #     ys, self.vertical_means - self.vertical_stds,
        #     self.vertical_means + self.vertical_stds,
        #     alpha=.2, color=self.color, ec="", lw=0)
        self.vertical_ax.plot(mids, ys, color=self.color)
        self.vertical_ax.fill_betweenx(
            ys, lows, highs, alpha=.2, color=self.color, edgecolor="none", lw=0)
        self.vertical_ax.set_ylim(self.ymin, self.ymax)
        self.vertical_ax.set_xlim(cmin, cmax)
        self.vertical_ax.set_xticks(cticks)
        self.vertical_ax.set_ylabel("Elevation ($^\circ$)")
        sbn.despine(ax=self.vertical_ax)
        # self.vertical_ax.set_xlabel(self.color_label)
        # plot expected colorvals using bootstrapped CIs along horizontal
        self.horizontal_ax = self.fig.add_subplot(self.gridspec[1, 1],
                                                  sharex=self.heatmap_ax)
        lows, mids, highs = [], [], []
        no_nans = np.isnan(grid) == False
        for col, no_nan in zip(grid, no_nans):
            # low, mid, high = bootstrapped_CI(col)
            low, mid, high = np.percentile(col[no_nan], [16, 50, 84])
            lows += [low]
            mids += [mid]
            highs += [high]
        # self.horizontal_means = np.nanmean(grid, axis=1)
        # self.horizontal_stds = np.nanstd(grid, axis=1)
        # self.horizontal_std_err = 2 * self.horizontal_stds / np.sqrt(no_nans.sum(1))
        # self.horizontal_ax.plot(xs, self.horizontal_means, color=self.color)
        # self.horizontal_ax.fill_between(
        #     xs, self.horizontal_means - self.horizontal_stds,
        #     self.horizontal_means + self.horizontal_stds,
        #     alpha=.2, color=self.color, ec="", lw=0)
        self.horizontal_ax.plot(xs, mids, color=self.color)
        self.horizontal_ax.fill_between(
            xs, lows, highs, alpha=.2, color=self.color, edgecolor="none", lw=0)
        self.horizontal_ax.set_xlim(self.xmin, self.xmax)
        # self.horizontal_ax.set_ylabel(self.color_label)
        self.horizontal_ax.set_yticks(cticks)
        sbn.despine(ax=self.horizontal_ax)
        self.horizontal_ax.set_ylim(cmin, cmax)
        self.horizontal_ax.set_xlabel("Azimuth ($^\circ$)")
        plt.suptitle(self.suptitle)
        # plt.tight_layout()

    def plot_heatmap(self, xs, ys, grid):
        # self.heatmap = self.heatmap_ax.scatter(
        #     self.x, self.y, c=self.colorvals, cmap=self.cmap,
        #     vmin=self.cmin, vmax=self.cmax)
        if self.vmin is not None:
            cmin = self.vmin
        else:
            cmin = self.cmin
        if self.vmax is not None:
            cmax = self.cmax
        else:
            cmax = self.cmax
        if self.scatter:
            self.heatmap = self.heatmap_ax.scatter(
                self.x, self.y, c=self.colorvals, vmin=cmin, vmax=cmax)
        else:
            self.heatmap = self.heatmap_ax.pcolormesh(
                xs, ys, grid.T, cmap=self.cmap, antialiased=True,
                vmin=cmin, vmax=cmax)
        self.heatmap_ax.set_aspect('equal')
        self.heatmap_ax.set_xlim(self.xmin, self.xmax)
        self.heatmap_ax.set_ylim(self.ymin, self.ymax)
        sbn.despine(ax=self.heatmap_ax, bottom=True, left=True)
        self.heatmap_ax.label_outer()
        self.heatmap_ax.tick_params(axis=u'both', which=u'both',length=0)


class VarSummary_lines(VarSummary):
    def __init__(self, xs, ys, colorvals, cmap='viridis', image_size=10**5,
                 color='k', center=True, color_label="Color", suptitle="Title",
                 xs1=None, xs2=None, ys1=None, ys2=None):
        self.xs1, self.xs2, self.ys1, self.ys2 = xs1, xs2, ys1, ys2
        VarSummary.__init__(self, xs, ys, colorvals, cmap='viridis',
                            image_size=10**5, color='k', center=True,
                            color_label="Color", suptitle="Title")
        if self.center:
            self.xs1 -= self.x_offset
            self.xs2 -= self.x_offset
            self.ys1 -= self.y_offset
            self.ys2 -= self.y_offset

    def plot_heatmap(self, *args):
        cmap = plt.cm.get_cmap(self.cmap)
        self.heatmap = []
        for x1, x2, y1, y2, cval in zip(self.xs1, self.xs2, self.ys1,
                                        self.ys2, self.colorvals):
            prop = (cval - self.cmin)/(self.cmax - self.cmin)
            self.heatmap_ax.plot([x1, x2], [y1, y2], color=cmap(prop))
        self.heatmap_ax.set_aspect('equal')
        self.heatmap_ax.set_xlim(self.xmin, self.xmax)
        self.heatmap_ax.set_ylim(self.ymin, self.ymax)
        sbn.despine(ax=self.heatmap_ax, bottom=True, left=True)
        self.heatmap_ax.label_outer()
        self.heatmap_ax.tick_params(axis=u'both', which=u'both',length=0)


class ScatterPlot3d():
    """Plot 3d datapoints using pyqtgraph's GLScatterPlotItem."""

    def __init__(self, arr, color=None, size=1, window=None,
                 colorvals=None, cmap=plt.cm.viridis, title="3D Scatter Plot"):
        self.title = title
        self.arr = arr
        self.app = app
        self.color = color
        self.cmap = cmap
        self.size = size
        self.window = window
        self.n, self.dim = self.arr.shape
        assert self.dim == 3, ("Input array should have shape "
                               "N x 3. Instead it has "
                               "shape {} x {}.".format(
                                   self.n,
                                   self.dim))
        if colorvals is not None:
            assert len(colorvals) == self.n, print("input colorvals should "
                                                   "have the same lengths as "
                                                   "input array")
            if np.any(colorvals < 0) or np.any(colorvals > 1):
                colorvals = (colorvals - colorvals.min()) / \
                    (colorvals.max() - colorvals.min())
            self.color = np.array([self.cmap(c) for c in colorvals])
        elif color is not None:
            assert len(color) == 4, print("color input should be a list or tuple "
                                          "of RGBA values between 0 and 1")
            if isinstance(self.color, (tuple, list)):
                self.color = np.array(self.color)
            if self.color.max() > 1:
                self.color = self.color / self.color.max()
            self.color = tuple(self.color)
        else:
            self.color = (1, 1, 1, 1)
        self.plot()

    def plot(self):
        if self.window is None:
            self.window = gl.GLViewWidget()
            self.window.setWindowTitle(self.title)
        self.scatter_GUI = gl.GLScatterPlotItem(
            pos=self.arr, size=self.size, color=self.color)
        self.window.addItem(self.scatter_GUI)

    def show(self):
        self.window.show()
        self.app.exec_()


class tracker_window():

    def __init__(self, dirname="./"):
        # m.pyplot.ion()
        self.dirname = dirname
        self.load_filenames()
        self.num_frames = len(self.filenames)
        self.range_frames = np.array(range(self.num_frames))
        self.curr_frame_index = 0
        self.data_changed = False
        # the figure
        self.load_image()
        # figsize = self.image.shape[1]/90, self.image.shape[0]/90
        h, w = self.image.shape[:2]
        if w > h:
            fig_width = 8
            fig_height = h/w * fig_width
        else:
            fig_height = 8
            fig_width = w/h * fig_height
        # start with vmin and vmax at extremes
        self.vmin = 0
        self.vmax = np.iinfo(self.image.dtype).max
        self.vmax_possible = self.vmax
        # self.figure = plt.figure(1, figsize=(
        #     figsize[0]+1, figsize[1]+2), dpi=90)
        self.figure = plt.figure(1, figsize=(fig_width, fig_height), dpi=90)
        # xmarg, ymarg = .2, .1
        # fig_left, fig_bottom, fig_width, fig_height = .15, .1, .75, .85
        fig_left, fig_bottom, fig_width, fig_height = .1, .1, .75, .8
        axim = plt.axes([fig_left, fig_bottom, fig_width, fig_height])
        self.implot = plt.imshow(self.image, cmap='viridis', vmin=self.vmin, vmax=self.vmax)
        self.xlim = self.figure.axes[0].get_xlim()
        self.ylim = self.figure.axes[0].get_ylim()
        self.axis = self.figure.get_axes()[0]
        self.figure.axes[0].set_xlim(*self.xlim)
        self.figure.axes[0].set_ylim(*self.ylim)
        self.image_data = self.axis.images[0]
        # title
        self.title = self.figure.suptitle(
            '%d - %s' % (self.curr_frame_index + 1, self.filenames[self.curr_frame_index].rsplit('/')[-1]))

        # the slider controlling frames
        axframe = plt.axes([fig_left, 0.04, fig_width, 0.02])
        self.curr_frame = Slider(
            axframe, 'frame', 1, self.num_frames, valinit=1, valfmt='%d', color='k')
        self.curr_frame.on_changed(self.change_frame)
        # the vmin slider
        vminframe = plt.axes([fig_left + fig_width + .02, 0.1, .02, .05 + .7])
        self.vmin = Slider(
            vminframe, 'min', 0, self.vmax_possible,
            valinit=0, valfmt='%d', color='k', orientation='vertical')
        self.vmin.on_changed(self.show_image)
        # the vmax slider
        vmaxframe = plt.axes([fig_left + fig_width + .1, 0.1, .02, .05 + .7])
        self.vmax = Slider(
            vmaxframe, 'max', 0, self.vmax_possible, valinit=self.vmax_possible,
            valfmt='%d', color='k', orientation='vertical')
        self.vmax.on_changed(self.show_image)
        # limit both sliders
        self.vmin.slidermax = self.vmax
        self.vmax.slidermin = self.vmin
        # the colorbar in between
        self.cbar_ax = plt.axes([fig_left + fig_width + .06, 0.1, .02, .05 + .7])
        self.colorvals = np.arange(self.vmax_possible)
        self.cbar = self.cbar_ax.pcolormesh([0, 10],
                                            self.colorvals,
                                            self.colorvals[:, np.newaxis],
                                            cmap='viridis', vmin=0, vmax=self.vmax_possible)
        self.cbar_ax.set_xticks([])
        self.cbar_ax.set_yticks([])
        # connect some keys
        # self.cidk = self.figure.canvas.mpl_connect(
        #     'key_release_event', self.on_key_release)
        # self.cidm = self.figure.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        # self.cidm = self.figure.canvas.mpl_connect('', self.on_mouse_release)
        # self.figure.canvas.toolbar.home = self.show_image

        # change the toolbar functions
        NavigationToolbar2.home = self.show_image
        NavigationToolbar2.save = self.save_data

    def load_filenames(self):
        ls = os.listdir(self.dirname)
        self.filenames = []
        img_extensions = ('.png', '.jpg', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff')
        for f in ls:
            if f.lower().endswith(img_extensions) and f[0] not in [".", "_"]:
                self.filenames += [os.path.join(self.dirname, f)]
        self.filenames.sort()

    def load_image(self):
        print(self.curr_frame_index)
        self.image = PIL.Image.open(self.filenames[self.curr_frame_index])
        self.image = np.asarray(self.image)

    def show_image(self, *args):
        print('show_image')
        # first plotthe image
        self.im = np.copy(self.image)
        colorvals = np.copy(self.colorvals)
        # remove values > vmax
        self.im[self.im > self.vmax.val] = 0
        self.figure.axes[0].get_images()[0].set_clim([self.vmin.val, self.vmax.val])
        self.figure.axes[0].get_images()[0].set_data(self.im)
        colorvals[colorvals > self.vmax.val] = 0
        self.cbar.set_array(colorvals)
        self.cbar.set_clim([self.vmin.val, self.vmax.val])
        # and the title
        self.title.set_text('%d - %s' % (self.curr_frame_index + 1,
                                         self.filenames[self.curr_frame_index].rsplit('/')[-1]))
        plt.draw()

    def change_frame(self, new_frame):
        print('change_frame {} {}'.format(new_frame, int(new_frame)))
        self.curr_frame_index = int(new_frame)-1
        self.load_image()
        self.show_image()
        if self.data_changed:
            self.save_data()
            self.data_changed = False

    def nudge(self, direction):
        self.show_image()
        # self.change_frame(mod(self.curr_frame, self.num_frames))
        self.data_changed = True

    def on_key_release(self, event):
        # frame change
        if event.key in ("pageup", "alt+v", "alt+tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index, self.num_frames))
        elif event.key in ("pagedown", "alt+c", "tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 2, self.num_frames))
            print(self.curr_frame_index)
        elif event.key == "alt+pageup":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index - 9, self.num_frames))
        elif event.key == "alt+pagedown":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 11, self.num_frames))
        elif event.key == "home":
            self.curr_frame.set_val(1)
        elif event.key == "end":
            self.curr_frame.set_val(self.num_frames)
        # marker move
        elif event.key == "left":
            self.nudge(-1)
        elif event.key == "right":
            self.nudge(1)
        elif event.key == "up":
            self.nudge(-1j)
        elif event.key == "down":
            self.nudge(1j)
        elif event.key == "alt+left":
            self.nudge(-10)
        elif event.key == "alt+right":
            self.nudge(10)
        elif event.key == "alt+up":
            self.nudge(-10j)
        elif event.key == "alt+down":
            self.nudge(10j)

    def update_sliders(self, val):
        self.show_image()

    def on_mouse_release(self, event):
        self.change_frame(0)

    def save_data(self):
        print('save')
        for fn, val in zip(self.objects_to_save.keys(), self.objects_to_save.values()):
            np.save(fn, val)

class StackFilter():
    """Import image filenames filter images using upper and lower contrast bounds."""

    def __init__(self, fns=os.listdir("./")):
        """Import images using fns, a list of filenames."""
        self.fns = fns
        self.folder = os.path.dirname(self.fns[0])
        self.vals = None
        self.imgs = None

    def load_images(self, low_bound=0, upper_bound=np.inf):
        print("Loading images:\n")
        first_img = None
        for fn in self.fns:
            try:
                first_img = load_image(fn)
                break
            except:
                pass
        breakpoint()
        width, height = first_img.shape
        # self.imgs = np.zeros((len(self.fns), width, height), dtype=first_img.dtype)
        # use a memmap to store the stack of images
        memmap_fn = os.path.join(mkdtemp(), 'temp_volume.dat')
        self.imgs = np.memmap(
            memmap_fn, mode='w+',
            shape=(len(self.fns), width, height), dtype=first_img.dtype)
        for num, fn in enumerate(self.fns):
            try:
                img = np.copy(load_image(fn))
                keep = np.logical_and(img >= low_bound, img <= upper_bound)
                img[keep == False] = 0
                self.imgs[num] = img
            except:
                print(f"{fn} failed to load.")
            print_progress(num, len(self.fns))
        # self.imgs = np.array(imgs, dtype=first_img.dtype)
 
    def contrast_filter(self):
        self.contrast_filter_UI = tracker_window(dirname=self.folder)
        plt.show()
        # grab low and high bounds from UI
        self.low = int(np.round(self.contrast_filter_UI.vmin.val))
        self.high = int(np.round(self.contrast_filter_UI.vmax.val))
        print("Extracting coordinate data: ")
        self.load_images(low_bound=self.low, upper_bound=self.high)
        #  inds_to_remove = np.logical_or(self.imgs <= self.low, self.imgs > self.high)
        # self.imgs[inds_to_remove] = 0
        # np.logical_and(self.imgs <= self.high, self.imgs > self.low, out=self.imgs)
        # self.imgs = self.imgs.astype(bool, copy=False)
        # try:
        #     ys, xs, zs = np.where(self.imgs > 0)
        #     vals = self.imgs[ys, xs, zs]
        # except:
        # print(
        #     "coordinate data is too large. Using a hard drive memory map instead of RAM.")
        self.imgs_memmap = np.memmap(os.path.join(self.folder, "volume.npy"),
                                     mode='w+', shape=self.imgs.shape, dtype=bool)
        self.imgs_memmap[:] = self.imgs[:]
        del self.imgs
        self.imgs = None
        # xs, ys, zs, vals = [], [], [], []
        total_vals = self.imgs_memmap.sum()
        self.arr = np.zeros((total_vals, 3), dtype='uint16')
        self.vals = np.zeros(total_vals, dtype=self.imgs_memmap.dtype)
        last_ind = 0
        for depth, img in enumerate(self.imgs_memmap):
            num_vals = img.sum()
            y, x = np.where(img > 0)
            z = np.repeat(depth, len(x))
            arr = np.array([x, y, z])
            self.arr[last_ind: last_ind + num_vals] = arr.T
            self.vals[last_ind: last_ind + num_vals] = img[y, x]
            last_ind = last_ind + num_vals
            print_progress(depth + 1, len(self.imgs_memmap))
        # self.arr = np.array([xs, ys, zs], dtype=np.uint16).T
        # self.vals = np.array(vals)

    def get_limits(self):
        self.limits_UI = tracker_window(dirname=self.folder)
        plt.show()
        # grab low and high bounds from UI
        self.low = int(np.round(self.limits_UI.vmin.val))
        self.high = int(np.round(self.limits_UI.vmax.val))
        return self.low, self.high

    def pre_filter(self):
        self.contrast_filter_UI = tracker_window(dirname=self.folder)
        plt.show()
        # grab low and high bounds from UI
        self.low = self.contrast_filter_UI.vmin.val
        self.high = self.contrast_filter_UI.vmax.val
        folder = os.path.join(self.folder, 'prefiltered_stack')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        # self.load_images(low_bound=self.low, upper_bound=self.high)
        # inds_to_remove = np.logical_or(self.imgs <= self.low, self.imgs > self.high)
        # self.imgs[inds_to_remove] = 0
        # self.imgs = self.imgs.astype('uint8', copy=False)
        # np.multiply(self.imgs, 255, out=self.imgs)
        print("Saving filtered images:\n")
        # for num, (fn, img) in enumerate(zip(self.fns, self.imgs)):
        #     base = os.path.basename(fn)
        #     new_fn = os.path.join(folder, base)
        #     save_image(new_fn, img)
        #     print_progress(num + 1, len(self.fns))
        for num, fn in enumerate(self.contrast_filter_UI.filenames):
            base = os.path.basename(fn)
            new_fn = os.path.join(folder, base)
            img = np.copy(load_image(fn))
            keep = np.logical_and(img >= self.low, img <= self.high)
            img[keep == False] = 0
            save_image(new_fn, img)
            print_progress(num + 1, len(self.fns))


def load_image(fn):
    """Import an image as a numpy array using the PIL."""
    return np.asarray(PIL.Image.open(fn))

def save_image(fn, arr):
    """Save an image using the PIL."""
    img = PIL.Image.fromarray(arr)
    if os.path.exists(fn):
        os.remove(fn)
    return img.save(fn)

