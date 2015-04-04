import numpy as np
import matplotlib.pyplot as plt

def _tile_plot(imgs, titles, share_ax_with, vertical=False, **kwargs):
    """
    Helper function
    """
    n = len(imgs)
    if vertical:
        nrows = n
        ncols = 1
    else:
        nrows = 1
        ncols = n
    # Create a new figure and plot the three images
    if share_ax_with is not None:
        fig = figure()
        ax = []
        for i in range(n):
            new_ax = fig.add_subplot(nrows, ncols, 1+i, sharex=share_ax_with, sharey=share_ax_with)
            ax.append(new_ax)
    else:
        fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)

    for ii, a in enumerate(ax):
        a.set_axis_off()
        a.imshow(imgs[ii], **kwargs)
        a.set_title(titles[ii])

    return fig


def overlay_slices(L, R, slice_index=None, slice_type=1, ltitle='Left',
                   rtitle='Right', fname=None, axes_shared=None):
    r"""Plot three overlaid slices from the given volumes.

    Creates a figure containing three images: the gray scale k-th slice of
    the first volume (L) to the left, where k=slice_index, the k-th slice of
    the second volume (R) to the right and the k-th slices of the two given
    images on top of each other using the red channel for the first volume and
    the green channel for the second one. It is assumed that both volumes have
    the same shape. The intended use of this function is to visually assess the
    quality of a registration result.

    Parameters
    ----------
    L : array, shape (S, R, C)
        the first volume to extract the slice from, plottet to the left
    R : array, shape (S, R, C)
        the second volume to extract the slice from, plotted to the right
    slice_index : int (optional)
        the index of the slices (along the axis given by slice_type) to be
        overlaid. If None, the slice along the specified axis is used
    slice_type : int (optional)
        the type of slice to be extracted:
        0=sagital, 1=coronal (default), 2=axial.
    ltitle : string (optional)
        the string to be written as title of the left image. By default,
        no title is displayed.
    rtitle : string (optional)
        the string to be written as title of the right image. By default,
        no title is displayed.
    fname : string (optional)
        the name of the file to write the image to. If None (default), the
        figure is not saved to disk.
    """

    # Normalize the intensities to [0,255]
    sh = L.shape
    L = np.asarray(L, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    L = 255 * (L - L.min()) / (L.max() - L.min())
    R = 255 * (R - R.min()) / (R.max() - R.min())

    # Create the color image to draw the overlapped slices into, and extract
    # the slices (note the transpositions)
    if slice_type is 0:
        if slice_index is None:
            slice_index = sh[0]//2
        colorImage = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
        ll = np.asarray(L[slice_index, :, :]).astype(np.uint8).T
        rr = np.asarray(R[slice_index, :, :]).astype(np.uint8).T
    elif slice_type is 1:
        if slice_index is None:
            slice_index = sh[1]//2
        colorImage = np.zeros(shape=(sh[2], sh[0], 3), dtype=np.uint8)
        ll = np.asarray(L[:, slice_index, :]).astype(np.uint8).T
        rr = np.asarray(R[:, slice_index, :]).astype(np.uint8).T
    elif slice_type is 2:
        if slice_index is None:
            slice_index = sh[2]//2
        slice_index = sh[2]//2
        colorImage = np.zeros(shape=(sh[1], sh[0], 3), dtype=np.uint8)
        ll = np.asarray(L[:, :, slice_index]).astype(np.uint8).T
        rr = np.asarray(R[:, :, slice_index]).astype(np.uint8).T
    else:
        print("Slice type must be 0, 1 or 2.")
        return

    # Draw the intensity images to the appropriate channels of the color image
    # The "(ll > ll[0, 0])" condition is just an attempt to eliminate the
    # background when its intensity is not exactly zero (the [0,0] corner is
    # usually background)
    colorImage[..., 0] = ll * (ll > ll[0, 0])
    colorImage[..., 1] = rr * (rr > rr[0, 0])

    fig = _tile_plot([ll, colorImage, rr],
                      [ltitle, 'Overlay', rtitle], share_ax_with=axes_shared,
                      cmap=plt.cm.gray, origin='lower', )

    # Save the figure to disk, if requested
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight')

    return fig
    
    
def overlay_slices_with_contours(L, R, slice_index=None, slice_type=1, ltitle='Left',
                                mid_title="Overlay", rtitle='Right', fname=None, 
                                ncontours=20, axes_shared=None):
    f = overlay_slices(L, R, slice_index, slice_type, ltitle, rtitle, fname, axes_shared=axes_shared)
    a = f.get_axes()
    idx = R.shape[slice_type] // 2
    if slice_type == 0:
        a[1].contour(L[idx,:,:].transpose(), ncontours, colors='white')
    elif slice_type == 1:
        a[1].contour(L[:,idx,:].transpose(), ncontours, colors='white')
    else:
        a[1].contour(L[:,:,idx].transpose(), ncontours, colors='white')
    return f
