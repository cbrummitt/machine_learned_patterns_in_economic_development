from pandas.util.testing import assert_frame_equal
import pandas as pd
from pandas import Period
import panelyze.model_selection.split_data_target as split_data_target


def annual_index(start, end):
    return pd.PeriodIndex(start=start, end=end, freq='A-DEC', name='year')


small_panel = pd.Panel({
    'USA': pd.DataFrame(
        {'apples': [2, 3, 9, 12, 15],
         'bananas': [17, 5, 1, 2, 4]},
        index=annual_index(1990, 1994), dtype=int),
    'MEX': pd.DataFrame(
        {'apples': [1, 2, 4],
         'bananas': [100, 120, 125]},
        index=annual_index(1992, 1994))})
small_panel.items.name = 'country_code'
small_panel.minor_axis.name = 'product_name'


def test_split_data_target():
    data, target = (
        split_data_target.split_panel_into_data_and_target_dataframes(
            small_panel, 1))

    expected_data = pd.DataFrame.from_dict({
        (1, 'apples'): {
            ('MEX', Period('1993', 'A-DEC')): 1.0,
            ('MEX', Period('1994', 'A-DEC')): 2.0,
            ('USA', Period('1991', 'A-DEC')): 2.0,
            ('USA', Period('1992', 'A-DEC')): 3.0,
            ('USA', Period('1993', 'A-DEC')): 9.0,
            ('USA', Period('1994', 'A-DEC')): 12.0},
        (1, 'bananas'): {
            ('MEX', Period('1993', 'A-DEC')): 100.0,
            ('MEX', Period('1994', 'A-DEC')): 120.0,
            ('USA', Period('1991', 'A-DEC')): 17.0,
            ('USA', Period('1992', 'A-DEC')): 5.0,
            ('USA', Period('1993', 'A-DEC')): 1.0,
            ('USA', Period('1994', 'A-DEC')): 2.0}})
    expected_data.index.names = ['country_code', 'year']
    expected_data.columns.names = ['lag', 'product_name']

    assert_frame_equal(expected_data, data)

    expected_target = pd.DataFrame.from_dict({
        (0, 'apples'): {
            ('MEX', Period('1993', 'A-DEC')): 2.0,
            ('MEX', Period('1994', 'A-DEC')): 4.0,
            ('USA', Period('1991', 'A-DEC')): 3.0,
            ('USA', Period('1992', 'A-DEC')): 9.0,
            ('USA', Period('1993', 'A-DEC')): 12.0,
            ('USA', Period('1994', 'A-DEC')): 15.0},
        (0, 'bananas'): {
            ('MEX', Period('1993', 'A-DEC')): 120.0,
            ('MEX', Period('1994', 'A-DEC')): 125.0,
            ('USA', Period('1991', 'A-DEC')): 5.0,
            ('USA', Period('1992', 'A-DEC')): 1.0,
            ('USA', Period('1993', 'A-DEC')): 2.0,
            ('USA', Period('1994', 'A-DEC')): 4.0}})
    expected_target.index.names = ['country_code', 'year']
    expected_target.columns.names = ['lag', 'product_name']

    assert_frame_equal(expected_target, target)
    print("Tests pass.")
