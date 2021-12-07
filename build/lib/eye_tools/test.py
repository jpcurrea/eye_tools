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
    colorbar_ax.hist(angles, bins=BINS, orientation='horizontal', color='k', alpha=.25)
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

    
    if three_d:
        x1, y1, z1 = interommatidial_data[['pt1_x', 'pt1_y', 'pt1_z']].values.T
        x2, y2, z2 = interommatidial_data[['pt2_x', 'pt2_y', 'pt2_z']].values.T
        x, y, z = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
        arr = np.array([x, y, z]).T
        scatter = ScatterPlot3d(arr, colorvals=angles, size=5)
        scatter.show()

