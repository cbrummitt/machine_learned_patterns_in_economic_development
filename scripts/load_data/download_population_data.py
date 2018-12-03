# Download population data from the World Bank and write it to a JSON file
import logging
import os

import pandas as pd

from pandas_datareader import wb

logger = logging.getLogger(__name__)


PATH_EXPORT_POPULATION_DATA = os.path.join(
    os.pardir, "data", "raw", "population")


def load_population_data_from_World_Bank_and_CID(data_from_CID):
    """Load population data from the World Bank and combine it with the
    population data from the CID dataset."""

    population_data_CID = pd.pivot_table(
        data=data_from_CID,
        index='year', columns='country_code', values='population')

    start_year = data_from_CID.year.min()
    end_year = data_from_CID.year.max()

    population_data_world_bank = load_population_all_countries_from_world_bank(
        start_year, end_year)

    population_data_world_bank_wide = (
        pd.pivot_table(
            population_data_world_bank,
            values='population',
            index='year',
            columns='iso3c')
        .rename_axis('country_code', axis=1))
    population_data_world_bank_wide.index = pd.to_datetime(
        population_data_world_bank_wide.index,
        format="%Y").astype('period[A-DEC]')

    ct_in_WB_data = set(
        (c, y.year)
        for y, c in population_data_world_bank_wide.stack().index.values)
    ct_in_export_data = set(
        (c, y)
        for (c, y) in data_from_CID.loc[:, ['country_code', 'year']].values)

    ct_missing_population = ct_in_export_data - ct_in_WB_data

    logger.info('The World Bank data is missing {} population values'.format(
        len(ct_missing_population)))

    nonzero_pop_data_CID_ct_pairs = set(
        (c, t)
        for c, t in (data_from_CID[data_from_CID.population > 0]
                     .loc[:, ['country_code', 'year']].values))

    samples_we_could_get_pop_for = (
        nonzero_pop_data_CID_ct_pairs & ct_missing_population)

    logger.info('We can fill in {} of those missing values using the CID '
                'dataset'.format(len(samples_we_could_get_pop_for)))

    logger.info(
        'The number of samples for which we cannot get the population'
        ' data from CID is {}'.format(
            len(ct_missing_population - nonzero_pop_data_CID_ct_pairs)))

    population_data_WB_and_CID = population_data_world_bank_wide.copy()
    for c, y in samples_we_could_get_pop_for:
        population_data_WB_and_CID.loc[
            pd.Period(year=y, freq='A'), c] = population_data_CID.loc[y, c]

    return population_data_WB_and_CID


def load_wide_population_data_for_panel(panel):
    try:
        start_year = min(panel.major_axis).year
        end_year = max(panel.major_axis).year
    except AttributeError:
        start_year, end_year = min(panel.major_axis), max(panel.major_axis)

    pop_data = load_population_all_countries_from_world_bank(
        start_year, end_year)

    pop_wide = (
        pd.pivot_table(
            pop_data,
            values='population',
            index='year',
            columns='iso3c')
        .rename_axis('country_code', axis=1))
    pop_wide.index = pd.to_datetime(
        pop_wide.index, format="%Y").astype('period[A-DEC]')
    return pop_wide


def load_population_all_countries_from_world_bank(
        start_year, end_year,
        save_dir=PATH_EXPORT_POPULATION_DATA,
        force_download=False):
    """Load population data for all countries. If it has been downloaded
    already, then it is loaded from the hard disk unless force_download is
    True. Otherwise, population data is downloaded from the World Bank and
    saved to disk as a JSON file in the directory `save_dir`. The file name is
    "WorldBank_PopulationData_1962_to_2014", where 1962 and 2014 in this
    example are the values of start_year and end_year.

    Parameters
    ----------
    start_year, end_year : int
        The start and end dates for which to get population data

    save_dir : path to a directory in which the data should be saved

    force_download : bool, default False
        Whether to download and save the data if a file with the name
        "WorldBank_PopulationData_{start_year}_to_{end_year}" already exists in
        `save_dir`.

    Returns
    -------
    pop_data : pandas DataFrame
        A dataframe containing population, iso3c, iso2c, country_name as
        columns and index given by the years as pandas TimeStamps.
    """
    # Name of the file in which to look for the data or save the data if it is
    # not found in the directory save_dir
    file_name = (
        "WorldBank_PopulationData_{start_year}_to_{end_year}.json".format(
            start_year=start_year, end_year=end_year))
    path = os.path.join(save_dir, file_name)

    if os.path.exists(path) and not force_download:
        logger.info(
            "The data already exists in the path \n\t{path}"
            "\nThat data will be returned. To force a new download of the "
            "data, use `force_download=True`.".format(path=path))
        return read_pop_data_from_path(path)
    else:
        pop_data = download_population_all_countries(start_year, end_year)

        # Write the population data to disk
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print("Created directory {dir}".format(dir=save_dir))
        pop_data.to_json(path, date_format='iso')

        return pop_data


def read_pop_data_from_path(path, convert_dates='year_start'):
    """Read population data from the given path.
    """
    pop_data = pd.read_json(path, convert_dates=convert_dates).sort_index()
    return pop_data


def download_population_all_countries(start_year, end_year):
    """Download population data of all countries from the World Bank, and
    return it as a long DataFrame.
    """
    # Download the population data from the World Bank
    pop_data = (
        wb.download(
            indicator='SP.POP.TOTL', country='all',
            start=start_year, end=end_year)
        .reset_index()
        .rename(columns={'SP.POP.TOTL': 'population',
                         'country': 'country_name'}))

    # Country codes and names
    country_data = wb.get_countries()

    # Remove regions by selecting countries with a nonempty capital city
    country_data = country_data[country_data.capitalCity != '']

    # Merge population data with country data to map country names to
    # country codes
    pop_data = (pd.merge(pop_data, country_data, how='inner',
                         left_on='country_name', right_on='name')
                .reset_index(drop=True))

    pop_data['year_start'] = pd.to_datetime(pop_data.year, format="%Y")

    return pop_data
