import matplotlib.pyplot as plt
import numpy as np

from ..utils import convert_panel_dataframe
from .utils import create_fig_ax, maybe_save_fig


def histogram_boxplot_of_flattened_data(
        panel, expression_name=None, yscale='log', xscale='linear',
        save_fig=None):
    """Flatten and visualize the values of a panel, and optionally save it to
    a file.

    The expression name becomes the x-axis label; if expression_name is None
    but `panel` has a `name` attribute, then `panel.name` is used to label
    the x-axis.
    """
    flattened_values = panel.values.flatten()
    flattened_values = flattened_values[~np.isnan(flattened_values)]

    fig, ax = plt.subplots(ncols=2)
    counts, bins, patches = ax[0].hist(flattened_values, bins=50)
    ax[0].set_xscale(xscale)
    ax[0].set_yscale(yscale)
    ax[0].set_ylabel('number of observations')

    ax[1].boxplot(flattened_values)
    ax[1].set_yscale(xscale)

    if expression_name is None and hasattr(panel, 'name'):
        expression_name = panel.name
    if expression_name:
        fig.text(.5, .0, expression_name,
                 horizontalalignment='center', verticalalignment='top')
    fig.suptitle('Flattened data')
    maybe_save_fig(fig, save_fig)
    return fig, ax


def column_boxplot(df, feature_labels=None, ax=None,
                   col_name='column', expression_name='value',
                   tick_font_size='auto', figsize=(30, 10),
                   save_fig=None):
    """Make a box plot of the columns of a dataframe.
    """
    x_values = list(range(1, 1 + df.shape[1]))
    fig, ax = create_fig_ax(ax, figsize=figsize)
    if tick_font_size == 'auto':
        tick_font_size = int(30 / (1 + np.log(df.shape[1])))
    ax.boxplot(df, showmeans=True)
    ax.set_xlabel(col_name)
    ax.set_ylabel(expression_name)
    ax.set_xticks(x_values)
    if feature_labels is not None:
        ax.set_xticklabels(feature_labels,
                           rotation=90, fontsize=tick_font_size)
    maybe_save_fig(fig, save_fig)

    return fig, ax


def boxplot_of_panel_minor_axis(panel, expression_name=None):
    if expression_name is None and hasattr(panel, 'name'):
        expression_name = panel.name
    mi_df = convert_panel_dataframe.panel_to_multiindex(panel)
    return column_boxplot(
        mi_df, mi_df.columns.get_level_values(0),
        col_name=mi_df.columns.names[0].replace('_', ' '),
        expression_name=expression_name)
