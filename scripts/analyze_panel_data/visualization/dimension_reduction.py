from math import ceil
import matplotlib.pyplot as plt
import matplotlib as mpl
import textwrap
from statsmodels.tsa import stattools
import os
import numpy as np
import pandas as pd
from .utils import maybe_save_fig, shifted_color_map


def product_heatmap(*a, **kws):
    return feature_loading_heatmap(
        *a, original_features_axis_label='product name', **kws)


def feature_loading_heatmap(
        loadings, feature_labels, font_size=16, tick_font_size=4,
        figsize=(50, 10), major_tick_label_size=6, minor_tick_label_size=0,
        cmap=mpl.cm.PuOr_r, shift_color_map=True, save_fig=None,
        original_features_axis_label='original features'):
    """Plot the loadings on features as a heatmap.

    Parameters
    ----------
    loadings : array of shape
                [number of loadings on features, number of features]
        The vectors of loadings on features, stacked horizontally. For example,
        loadings could be PCA().components_.

    feature_labels : list of strings of length equal to the number of products
        The names of the features

    font_size, tick_font_size, : int, optional, default: 16
        `font_size` is the font size of the labels of the axes and color bar.
        The others are sizes of ticks.

    major_tick_label_size, minor_tick_label_size : int, optional, default 4, 6
        Sizes of the ticks

    cmap : matplotlib color map

    shift_color_map : bool, default: True
        Whether to shift the colormap `cmap` to center it at zero.

    save_fig : string or None, optional, default: None
        If not None, then save the figure to the path specified by `save_fig`
    """
    fig, ax = plt.subplots(figsize=figsize)

    vmin = np.min(loadings)
    vmax = np.max(loadings)
    if shift_color_map:
        midpoint = 1 - vmax / (vmax + np.abs(vmin))
        cmap = shifted_color_map(cmap, midpoint=midpoint)
    axim = ax.matshow(loadings, cmap=cmap, aspect=10,
                      vmin=vmin, vmax=vmax)

    ax.set_xlabel(original_features_axis_label, size=font_size)
    ax.set_ylabel('component', size=font_size)
    ax.tick_params(axis='both', which='major', labelsize=major_tick_label_size)
    ax.tick_params(axis='both', which='minor', labelsize=minor_tick_label_size)
    ax.set_xticks(list(range(loadings.shape[1])))
    ax.set_xticklabels(feature_labels, rotation=90, fontsize=tick_font_size)

    cax, kwargs = mpl.colorbar.make_axes(
        [ax], location='bottom', fraction=0.05,
        aspect=40, pad=.04, label='loading')
    cb = fig.colorbar(axim, cax=cax, **kwargs)
    text = cb.ax.xaxis.label
    font = mpl.font_manager.FontProperties(size=font_size)
    text.set_font_properties(font)
    maybe_save_fig(fig, save_fig)
    return fig, ax


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """Return a colormap with its center shifted.

    Useful for data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Adapted from: http://stackoverflow.com/a/20528097/6301373

    Parameters
    ----------
    cmap : The matplotlib colormap to be altered
    start : Offset from lowest point in the colormap's range.
        Defaults to 0.0 (no lower ofset). Should be between
        0.0 and `midpoint`.
    midpoint : The new center of the colormap. Defaults to
        0.5 (no shift). Should be between 0.0 and 1.0. In
        general, this should be  1 - vmax/(vmax + abs(vmin))
        For example if your data range from -15.0 to +5.0 and
        you want the center of the colormap at 0.0, `midpoint`
        should be set to  1 - 5/(5 + 15)) or 0.75
    stop : Offset from highets point in the colormap's range.
        Defaults to 1.0 (no upper ofset). Should be between
        `midpoint` and 1.0.
    """
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []}

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


def biplot(scores, loadings, axis=None,
           labels=None, figsize=(8, 4),
           xlabel='Score on the first principal component',
           ylabel='Score on the second principal component',
           scatter_size=20,
           label_offset=0.1, fontsize=14, label_font_size=10, scale_loadings=1,
           arrow_color='r', label_color='r', arrow_width=.001, arrow_alpha=0.5,
           scatter_alpha=0.5, score_color='b', tick_color='b',
           save_fig=None):
    """Create a biplot of the principal component analysis of a dataset.

    Parameters
    ----------
    scores : array of shape [number of samples, number of principal components]
        The scores of the data on the principal components. The number of
        principal components should be at least two.

    loadings : numpy array of shape
            [number of features, number of principal components]
        The loadings of features on the principal components

    axis : matplotlib axis, optional, default: None
        The axis on which to attach the biplot. If None, create a new figure.

    labels : list of strings, optional, default: None
        The labels of the features to be plotted next to the arrowheads

    figsize : tuple (width, height), optional, default: (8, 4)
        The size of the figure

    xlabel, ylabel : str or None, optional
        The labels of the axes. If None, then no label is shown.

    scatter_size : float, default 20
        Size of the scatter points in units of points^2

    label_offset : scalar, optional, default: 0.1
        The amount by which to scale the position of the label compared to the
        location of the arrowhead, relative to the 2-norm of the loading.

    fontsize : int, optional, default: 14
        The font size of the axes' labels

    label_font_size : int, optional, default: 14
        The font size of the labels of the arrows (the loading vectors)

    scale_loadings : float, optional, default: 1
        The amount by which to scale the loading vectors to make them easier to
        see.

    arrow_color : string, optional, default: 'r'
        The color of the loading vectors (arrows)

    score_color : string, optional, default: 'b'
        The color of the scores (scatter plot)

    arrow_width : float, optional, default: 0.001
        The width of the loading vectors' arrows

    arrow_alpha, scatter_alpha : float, optional, default: 0.5
        The opacity of the arrows and of the scatter plot

    label_color : string, optional, default: 'r'
        The color of the labels of the loading vectors (arrows)

    tick_color : string, optional, default: 'b'
        The color of the ticks and axis labels

    save_fig : None or a path and file name, optional, default: None
        If save_fig is not False or None, then the figure is saved to this
        file name.
    """
    n = loadings.shape[0]  # number of features to be plotted as arrows

    if scores.shape[1] < 2:
        raise ValueError("The number of principal component scores" +
                         + " must be at least 2.")

    if axis is None:
        fig, axis = plt.subplots(figsize=figsize)
    else:
        fig = axis.figure

    axis.scatter(
        *scores[:, :2].T, alpha=scatter_alpha, color=score_color,
        s=scatter_size)
    if xlabel is not None:
        axis.set_xlabel(xlabel, color=tick_color, fontsize=fontsize)
    if ylabel is not None:
        axis.set_ylabel(ylabel, color=tick_color, fontsize=fontsize)

    for tl in axis.get_xticklabels() + axis.get_yticklabels():
        tl.set_color(tick_color)

    for i in range(n):
        axis.arrow(0, 0,
                   loadings[i, 0] * scale_loadings,
                   loadings[i, 1] * scale_loadings,
                   color=arrow_color, alpha=arrow_alpha, width=arrow_width)
        if labels is not None:
            label_position = np.array([loadings[i, 0],
                                       loadings[i, 1]]) * scale_loadings
            label_position += (label_offset * label_position /
                               np.linalg.norm(label_position))
            axis.text(*label_position, labels[i],
                      color=label_color, ha='center', va='center',
                      wrap=True, fontsize=label_font_size, alpha=1)
    maybe_save_fig(fig, save_fig)
    return


def plot_variance_explained(
        explained_variance_ratio,
        labels_var_explained=[0, 1],
        labels_cumulative_var_explained=[0, 1, 9, 99, 199],
        fig_title='Variance explained by the principal components',
        save_fig=None):
    """Plot the ratio of variance explained by the principal components. Returns
    a row of two plots of the variance explained by each component and of the
    cumulative variance explained up to a certain component.

    Parameters
    ----------
    explained_variance_ratio : array of shape [n_components]
        This is PCA().explained_variance_ratio

    labels_var_explained : list of int's, optional, default: [0, 1]
        Which principal components to label in the plot of variance explained
        by each component.

    labels_cumulative_var_explained : list of int's, optional,
            default: [0, 1, 9, 99, 199]
        Which principal components to label in the plot of cumulative variance
        explained by each component. Including integer `i` in this list means
        label the fraction of variance explained by all components
        0, 1, ..., i, so the label is "i + 1 components".

    fig_title : str, optional, default: 'Variance explained by the principal
    components'
        The figure's title (i.e., its `suptitle`).

    save_fig : None or file path
        If None, do not save the file to disk. Otherwise, save the file to the
        path `save_fig`.

    Returns
    -------
    fig, ax : matplotib Figure and AxesSubplot objects
        The row of two figures showing (1) fraction of variance explained by
        each principal component and (2) the cumulative fraction of variance
        explained by each principal component.
    """
    n_components = len(explained_variance_ratio)
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                             figsize=(8, 4))
    options = {'s': 5, 'marker': 'o'}
    axes[0].scatter(range(n_components), explained_variance_ratio, **options)
    axes[0].set_ylabel('explained variance')
    axes[0].set_xlabel('principal component')
    # Label some of the points
    for i, var in enumerate(explained_variance_ratio[labels_var_explained]):
        axes[0].text(
            i + .03, var + .03,
            ('{var:.1%} (component {n_comp})'.format(var=var, n_comp=i)))

    axes[1].scatter(np.array(range(n_components)),
                    np.cumsum(explained_variance_ratio), **options)
    axes[1].set_ylabel('cumulative explained variance')
    for i, var in enumerate(np.cumsum(explained_variance_ratio)):
        if i in labels_cumulative_var_explained:
            axes[1].text(
                i + 5, var - .06,
                ('{var:.1%} ({n_comp} component'.format(
                    var=var, n_comp=i + 1) +
                 ('s)' if i + 1 != 1 else ')'))
            )
    axes[1].set_xlabel('number of principal components')

    axes[0].set_xlim(-5, n_components + .5)
    axes[0].set_ylim(0, 1)
    fig.suptitle(fig_title, size=14)
    maybe_save_fig(fig, save_fig)

    return fig, axes


def loadings_histograms(
        pca, feature_labels, n_components='all', bins=50,
        n_features_to_label=10, max_text_len=35,
        text_kws={'color': 'white',
                  'bbox': {'facecolor': 'k', 'alpha': 0.7, 'pad': 1}},
        save_fig=None):
    """Plot histograms of the loadings for each principal component.

    Inputs
    ------
    pca : fitted sklearn.decomposition.pca.PCA object

    feature_labels : list of strings of length equal to
    `pca.components_.shape[1]`
        The labels of the features

    bins : int
        The number of bins to use in the histograms

    n_components : 'all' or int
        The number of principal components to show. The components shown are
        0, 1, ..., n_components - 1. If n_components is 'all', then all
        components are shown.

    n_features_to_label : int, default: 10
        The number of most highly and least weighted features to label on the
        histogram. If 0, then do not show any such labels.

    max_text_len : int, default: 35
        The maximum number of characters to use in the labels of the most/least
        weighted features.

    save_fig : None or file path, default: None
        If not None, then save the figure to this path

    text_kws : dict
        The keywrod arguments for the text labels of the features.
    """
    if n_components == 'all':
        pca_weights = pd.DataFrame(pca.components_.T)
    else:
        pca_weights = pd.DataFrame(pca.components_[:n_components].T)
    pca_weights.index = feature_labels
    pca_weights.columns.name = 'principal component'
    exp_var_ratio = pca.explained_variance_ratio_

    fig, axes = plt.subplots(
        nrows=pca_weights.shape[1], sharex=True,
        figsize=(10, 6 * pca_weights.shape[1]))
    for component_num in pca_weights.columns:
        ax = axes[component_num]
        product_weights = (pca_weights.loc[:, component_num]
                                      .sort_values(ascending=True))
        title = 'principal component {num} ({var:.2%} of the variance)'.format(
            num=component_num, var=exp_var_ratio[component_num])
        ax.set_title(title)
        ax.set_xlabel('weights on products')
        ax.hist(product_weights.values, bins=bins)

        if n_features_to_label:
            most_neg_weighted = (
                product_weights.head(n_features_to_label)
                               .sort_values(ascending=False))
            most_pos_weighted = product_weights.tail(n_features_to_label)

            ylim = ax.get_ylim()
            ax.axvline(x=0, ymin=0, ymax=1, linestyle='--', color='k')
            yspan = ylim[1] - ylim[0]
            text_bounds = (ylim[0] + .1 * yspan, ylim[1] - .1 * yspan)
            spacing = ((text_bounds[1] - text_bounds[0]) /
                       len(most_pos_weighted))

            enum_pos = enumerate(most_pos_weighted.iteritems())
            for i, (product, weight) in enum_pos:
                y = text_bounds[0] + i * spacing
                ax.text(weight, y, textwrap.shorten(product, max_text_len),
                        ha='left', va='center', **text_kws)
                ax.axvline(x=weight, ymin=ylim[0], ymax=.03, color='k')

            enum_neg = enumerate(most_neg_weighted.iteritems())
            for i, (product, weight) in enum_neg:
                y = text_bounds[0] + i * spacing
                ax.text(weight, y, textwrap.shorten(product, max_text_len),
                        ha='left', va='center', **text_kws)
                ax.axvline(x=weight, ymin=ylim[0], ymax=.03, color='k')
    maybe_save_fig(fig, save_fig)
    return fig, axes


def plot_histogram_of_loadings(
        ax, loadings, hist_kws=None,
        cmap=mpl.cm.PuOr_r, vmin=None, vmax=None):
    """Plot a colored histogram of loadings (weights) on a given axis."""
    hist_kws = {} if hist_kws is None else hist_kws
    if not vmin:
        vmin = np.min(loadings)
    if not vmax:
        vmax = np.max(loadings)

    n, bins, patches = ax.hist(loadings, **hist_kws)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = mpl.colors.Normalize(vmin=vmin, vmax=vmax)(bin_centers)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmap(c))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params('y', labelleft='off')


def plot_loadings(
        loadings, feature_labels=None,
        feature_label_colors=None,
        figsize=(7., 3.3), cmap=mpl.cm.PuOr_r,
        xlabel='original features', ylabel='component',
        loading_label='loading',
        aspect=40, aspect_colobar=40,
        feature_label_spacing=15, num_characters=40, save_fig=None,
        hist_bottom=.27, hist_top=.54,
        A_label='A', B_label='B',
        A_label_pos=(0.0, .6), B_label_pos=(.91, .6)):
    """Plot loadings on features with histogram of loadings for each component.
    """
    fig = plt.figure(figsize=figsize)
    gs = mpl.gridspec.GridSpec(1, 1, bottom=.15, left=0.06, right=.85)
    ax = plt.subplot(gs[:, 0])

    gs2 = mpl.gridspec.GridSpec(len(loadings), 1)
    gs2.update(left=.86, right=.98, hspace=.05,
               top=hist_top, bottom=hist_bottom)
    ax_hist_0 = plt.subplot(gs2[0, 0])
    histogram_axes = [ax_hist_0]
    histogram_axes += [
        plt.subplot(gs2[i, 0], sharex=ax_hist_0)
        for i in range(1, len(loadings))]

    vmin = np.min(loadings)
    vmax = np.max(loadings)
    midpoint = 1 - vmax / (vmax + np.abs(vmin))
    cmap = shifted_color_map(cmap, midpoint=midpoint)

    for i in range(len(loadings)):
        plot_histogram_of_loadings(
            histogram_axes[i], loadings[i], cmap=cmap, vmin=vmin, vmax=vmax)
        if i == len(loadings) - 1:
            histogram_axes[i].set_xlabel('loading histograms')
        else:
            histogram_axes[i].tick_params('x', labelbottom='off')

    axim = ax.matshow(loadings, cmap=cmap, aspect=aspect)

    font_size = 8
    major_tick_label_size = 6
    minor_tick_label_size = 0
    tick_font_size = 7
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='major', labelsize=major_tick_label_size)
    ax.tick_params(axis='both', which='minor', labelsize=minor_tick_label_size)
    ax.set_xticks(list(range(loadings.shape[1]))[::feature_label_spacing])
    ax.set_yticklabels(['', '1st', '2nd', '3rd', '4th', '5th'])
    ax.tick_params(axis='x', labeltop='off')
    if feature_labels is not None:
        if feature_label_colors is None:
            feature_label_colors = ['k' for label in feature_labels]
        feature_label_colors = feature_label_colors[::feature_label_spacing]
        feature_weight = np.sum(
            np.abs(loadings[1:, ::feature_label_spacing]), axis=0)
        for i, (label, weight, color) in enumerate(
                zip(feature_labels[::feature_label_spacing], feature_weight,
                    feature_label_colors)):
            if weight > np.percentile(feature_weight, 90):
                boldness = 1000
            elif weight > np.percentile(feature_weight, 70):
                boldness = 700
            else:
                boldness = 'normal'
            ax.annotate(textwrap.shorten(label, num_characters,
                                         placeholder='...'),
                        xy=(i / len(feature_weight), 1.05),
                        xytext=(i / len(feature_weight), 1.05),
                        xycoords='axes fraction',
                        textcoords='axes fraction', weight=boldness,
                        rotation=90, fontsize=tick_font_size,
                        ha='left', va='bottom', color=color)

    cax, kwargs = mpl.colorbar.make_axes(
        [ax], location='bottom', fraction=0.04,
        aspect=aspect_colobar, pad=.03)
    cax.text(-0.08, 0.25, loading_label, ha='center', va='center')
    cb = fig.colorbar(axim, cax=cax, **kwargs)

    fig.text(*A_label_pos, A_label, weight='bold', horizontalalignment='left')
    fig.text(*B_label_pos, B_label, weight='bold', horizontalalignment='left')

    maybe_save_fig(fig, save_fig)
    return fig, ax


def plot_increasing_alpha(data, ax, alpha_range, plot_kws=None):
    plot_kws = {} if plot_kws is None else plot_kws
    if alpha_range[0] == alpha_range[1]:
        ax.plot(*data.T, alpha=alpha_range[0], **plot_kws)
    else:
        for i, (traj_now, traj_next) in enumerate(zip(data[:-1], data[1:])):
            frac_idx = i / len(data)
            alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * frac_idx
            ax.plot(
                [traj_now[0], traj_next[0]],
                [traj_now[1], traj_next[1]],
                alpha=alpha, **plot_kws)
