import copy
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import palettable
from palettable.colorbrewer.qualitative import (Accent_8, Pastel1_9, Set1_9,
                                                Set3_12)


def maybe_save_fig(fig, save_fig, bbox_inches='tight'):
    """Save the figure to the specified path if the path is not None.

    Parameters
    ----------
    fig : matplotlib Figure

    save_fig : None or str
        If None, then do not save the figure. Otherwise save the figure to the
        path specified by the string `save_fig`.

    bbox_inches : str, optional, default: 'tight'
        The keyword argument in `matplotlib.pyplot.Figure.savefig` that crops
        whitespace out of the figure.
    """
    if save_fig:
        fig.savefig(save_fig, bbox_inches=bbox_inches)


def create_fig_ax(ax, **kwargs):
    """Create a figure and axis if needed, or the figure associated with the
    given axis if it already exists."""
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.get_figure()
    return fig, ax


def convert_None_to_empty_dict_else_copy(d):
    return {} if d is None else copy.copy(d)


def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0,
                      name='shiftedcmap', data=None):
    """Offset the "center" of a colormap.

    Useful for data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Source: https://gist.github.com/phobson/7916777

    Parameters
    ----------
    cmap : The matplotlib colormap to be altered

    start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and 1.0.

    midpoint : The new center of the colormap. Default: 0.5 (no shift)
          Should be between 0.0 and 1.0. If data is not None,
          then midpoint is set to 1 - vmax/(vmax + abs(vmin)) so that zero
          corresponds to the center of the color map.

    stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.

    data : None or numpy array, default: None
        If data is not None, then midpoint is overriden by
        1 - vmax/(vmax + abs(vmin)), where vmin and vmax are the min and max
        of data. For example if your data range from -15.0 to +5.0 and
        you want the center of the colormap at 0.0, `midpoint`
        should be set to  1 - 5/(5 + 15)) or 0.75.

    Returns
    -------
    newcmap : matplotlib colormap
    """
    if data is not None:
        midpoint = midpoint_to_shift_color_to_zero(data)

    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def midpoint_to_shift_color_to_zero(data):
    data_flat = np.array(data).flatten()
    vmax = np.max(data_flat)
    vmin = np.min(data_flat)
    return 1 - vmax / (vmax + np.abs(vmin))


def convert_to_filename(string):
    return (
        (''.join(str(string).split())
         .replace(',', '__')
         .replace('.', 'p')
         .replace('\n', '')))


def make_tableau_color_cycler():
    return itertools.cycle(palettable.tableau.Tableau_20.mpl_colors)


def make_color_cycler():
    palettes = [
        [c for c in Set1_9.mpl_colors if c != Set1_9.mpl_colors[5]],
        [c for c in Accent_8.mpl_colors if c != Accent_8.mpl_colors[3]],
        [c for c in Set3_12.mpl_colors if c != Set3_12.mpl_colors[1]],
        [c for c in Pastel1_9.mpl_colors]]
    lots_of_colors = list(itertools.chain(*palettes))
    return itertools.cycle(lots_of_colors)
