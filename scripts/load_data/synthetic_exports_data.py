import pandas as pd
from analyze_panel_data.utils import panel_to_multiindex


def annual_index(start, end):
    return [pd.Period(year, 'A-DEC') for year in range(start, end + 1)]


small_panel = pd.Panel({
    'USA': pd.DataFrame(
        {(1., 'apples'): [2, 3, 9, 12, 15],
         (2., 'bananas'): [17, 5, 1, 2, 4]},
        index=annual_index(1990, 1994), dtype=int),
    'DEU': pd.DataFrame(
        {(1., 'apples'): [15, 3, 29, 10, 5],
         (2., 'bananas'): [12, 15, 19, 20, 3]},
        index=annual_index(1990, 1994), dtype=int)
})
small_panel.items.name = 'country_code'
small_panel.minor_axis.name = 'products'
small_panel.minor_axis.names = ['product_code', 'product_name']
small_panel.major_axis.name = 'year'

medium_panel = pd.Panel({
    'USA': pd.DataFrame(
        {(1., 'apples'): [2, 3, 9, 12, 15],
         (2., 'bananas'): [17, 5, 1, 2, 4]},
        index=annual_index(1990, 1994), dtype=int),
    'DEU': pd.DataFrame(
        {(1., 'apples'): [15, 3, 29, 10, 5],
         (2., 'bananas'): [12, 15, 19, 20, 3]},
        index=annual_index(1990, 1994), dtype=int),
    'CAN': pd.DataFrame(
        {(1., 'apples'): [15, 23, 10, 9],
         (2., 'bananas'): [0, 1, 0, 2]},
        index=annual_index(1990, 1993), dtype=int),
    'MEX': pd.DataFrame(
        {(1., 'apples'): [1, 2, 4],
         (2., 'bananas'): [100, 120, 125]},
        index=annual_index(1992, 1994))})
medium_panel.items.name = 'country_code'
medium_panel.minor_axis.name = 'products'
medium_panel.minor_axis.names = ['product_code', 'product_name']
medium_panel.major_axis.name = 'year'

medium_dataset_long = panel_to_multiindex(medium_panel).stack([0, 1])
medium_dataset_long.name = 'export_value'
medium_dataset_long = medium_dataset_long.reset_index()
