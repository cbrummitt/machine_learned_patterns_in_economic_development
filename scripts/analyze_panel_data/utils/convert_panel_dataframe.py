# Helpful functions for manipulating pandas Panels
# Authors:
#   Charlie Brummitt <brummitt@gmail.com> Github: cbrummitt
#   Andrés Gómez Liévano <andres_gomez@hks.harvard.edu>


def panel_to_multiindex(panel, filter_observations=False):
    """Convert a panel to a MultiIndex DataFrame with the minor_axis of the
    panel as the columns and the items and major_axis as the MultiIndex (levels
    0 and 1, respectively).

    Parameters
    ----------
    panel : a pandas Panel

    filter_observations : boolean, default False
        Whether to drop (major, minor) pairs without a complete set of
        observations across all the items.
    """
    return (panel.transpose(2, 0, 1)
                 .to_frame(filter_observations=filter_observations))


def multiindex_to_panel(multiindex_dataframe):
    """The inverse of `panel_to_multiindex`. Convert a MultIndex DataFrame with
    a MultiIndex, the first level of which becomes the items and second
    level becomes the major_axis. The columns of the DataFrame become the
    minor_axis of the panel.
    """
    return multiindex_dataframe.to_panel().transpose(1, 2, 0)


def panel_to_multiindex_drop_missing(panel, fillna=0):
    """Convert a panel to a MultiIndex DataFrame, drop missing rows, and fill
    remaining missing values.

    The minor_axis of the panel becomes the columns and the items and
    major_axis become the MultiIndex (levels 0 and 1, respectively).
    """
    return panel_to_multiindex(panel).dropna(how='all', axis=0).fillna(fillna)
