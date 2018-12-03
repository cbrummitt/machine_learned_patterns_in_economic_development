import os

import pandas as pd

import load_data.download_population_data as dlpop

PATH_TO_EXPORTS_DATA = os.path.join(
    os.pardir, 'data', 'raw', 'exports', 'S2_final_cpy_all.dta')


def load_and_filter_CID_S2_trade_data(
        path_exports_data=PATH_TO_EXPORTS_DATA,
        min_pop=1.25e6, year_min_pop=2008,
        min_total_exports=1e9, year_min_total_exports=2008,
        exclude_countries=set(('TCD', 'IRQ', 'AFG')),
        min_frac_countries_that_must_export_a_product_in_every_year=0.8,
        min_frac_products_not_exported_some_year_to_exclude_country=.95,
        min_global_exports_each_year=10e6,
        round_to_zero_any_export_value_smaller_than=0.0,
        verbose=1, filter_all_at_once=True,
        min_market_share_quantile=0.05, year_min_market_share=2008,
        fill_na_value=0.0, product_code_digits=2):
    """Return exports data in long and panel format, cleaned and filtered.

    Countries and products are removed as described next.

    By default, these filters are applied all at once
    (`filter_all_at_once=True`), meaning that we first compute the countries
    and products that would be removed by each filter, and then we take the
    union of those sets of countries and products to be removed, and then we
    remove those countries and products all at once. If `filter_all_at_once` is
    `False`, then the filters are applied to the dataset sequentially, which
    introduces path dependence.

    `product_code_digits` is the number of digits in the product codes at which
    to aggregate the data. The finest level, 4-digit product codes, is the
    default. Choosing 1 or 2 means that exports are aggregated at a coarser
    level by summing.

    Filters
    -------
    1. Countries with small populations:

    Select countries with
        population <= `min_pop` in year `year_min_pop`
    according to World Bank data on population. If a country's population is
    not in the World Bank dataset in the year `year_min_pop`, then it is not
    selected by this filter (i.e., it is not removed from the dataset).

    2. Countries with little exports:

    Select countries with total exports smaller than `min_total_exports` in
    year `year_min_total_exports`.

    3. Products not exported at all by too many countries

    Select all products such that, for at least one year in the dataset,
    the product is not exported at all by more than a fraction
    `min_frac_countries_that_must_export_a_product_in_every_year` (default:
    80%) of countries.

    4. Countries with zero exports for too many products

    Select all countries such that, in at least one year, its export value
    is zero for more than
    `min_frac_products_not_exported_some_year_to_exclude_country` (default:
    95%) of all products.

    5. Products with too few global exports

    Select all products such that, in at least one year in the dataset,
    the total global export value of that product is less than
    `min_global_exports_each_year` (default: US$10 million).

    6. Products in the bottom quantile of global exports

    Select all products in the bottom `min_market_share_quantile`th percentile
    (default: 5th percentile) of products by their total global exports in the
    year `year_min_market_share` (default: 2008).

    7. Countries in a certain list.

    Select countries in the list `exclude_countries`. The default is
    set(('TCD', 'IRQ', 'AFG')), which are Chad, Iraq, and Afghanistan.

    8. Products in a certain list.

    Select products in the list `exclude_products`.
    A product is removed if its first 1, 2, 3, or 4 digits is in
    `exclude_products`. The default is to remove all products whose first two
    digits are '96', which corresponds to gold.



    Replacements
    ------------

    After the filters above are applied to the dataset, the following
    replacements are performed:

    1. Missing values (`nan`) are replaced by `fill_na_value` (default: 0.0)

    2. Export values smaller than `round_to_zero_any_export_value_smaller_than`
    are replaced by `0.0`. By default, the parameter
    `round_to_zero_any_export_value_smaller_than` is `0.0`, meaning that no
    export values are set to zero.


    Returns
    -------
    data_from_CID : DataFrame
        The raw data from the file 'S2_final_cpy_all.dta' from the CID website
        https://intl-atlas-downloads.s3.amazonaws.com/CPY/S2_final_cpy_all.dta

    exports_long_format : DataFrame
        A pandas DataFrame with a MultiIndex index containing the country code
        and year, and products on the columns.

    panel_export_value : Panel
        A pandas Panel with countries on the items axis, time on the
        major axis, and products on the minor axis.
    """
    if product_code_digits not in [1, 2, 4]:
        raise ValueError('product_code_digits must be '
                         '1, 2, or 4; got {}'.format(product_code_digits))

    try:
        data_from_CID = pd.read_stata(path_exports_data).rename(columns={
            'commoditycode': 'product_code', 'exporter': 'country_code'})
    except FileNotFoundError as err:
        raise FileNotFoundError('The Stata .dta file was not found at the path'
                                ' {}.'.format(path_exports_data))

    filter_kws = dict(
        min_pop=min_pop, year_min_pop=year_min_pop,
        min_total_exports=min_total_exports,
        year_min_total_exports=year_min_total_exports,
        exclude_countries=exclude_countries,
        min_frac_countries_that_must_export_a_product_in_every_year=(
            min_frac_countries_that_must_export_a_product_in_every_year),
        min_frac_products_not_exported_some_year_to_exclude_country=(
            min_frac_products_not_exported_some_year_to_exclude_country),
        min_global_exports_each_year=(
            min_global_exports_each_year),
        round_to_zero_any_export_value_smaller_than=(
            round_to_zero_any_export_value_smaller_than),
        verbose=verbose,
        filter_all_at_once=filter_all_at_once,
        min_market_share_quantile=min_market_share_quantile,
        year_min_market_share=year_min_market_share)

    exports_long_format = filter_exports(data_from_CID, **filter_kws)

    exports_long_format = clean_missing_exports_long(
        exports_long_format, fill_na_value)

    if product_code_digits in [1, 2]:
        exports_long_format.product_code = (
            exports_long_format.product_code.str[:product_code_digits])
        exports_long_format = (
            exports_long_format
            .groupby(['country_code', 'year', 'product_code'])['export_value']
            .sum().reset_index())
        if verbose:
            m = 'Summing exports at the {} digit level results in {} products'
            print(m.format(product_code_digits,
                           len(exports_long_format.product_code.unique())))

    panel_export_value = (
        create_panel_from_long_data(
            exports_long_format,
            items='country_code',
            major_axis='year',
            minor_axis='product_code',
            values='export_value')
        .pipe(convert_major_axis_to_annual_PeriodIndex))
    panel_export_value.name = 'export value'
    panel_export_value.filename = 'export_value'
    return data_from_CID, exports_long_format, panel_export_value


def clean_missing_exports_long(exports, fill_na_value=0):
    """Clean the exports data in long format by removing records with missing
    product code and then filling missing values with fill_na_value
    (which is 0 by default).
    """
    return (exports.dropna(subset=['product_code'])
                   .fillna(value=fill_na_value))


def convert_major_axis_to_annual_PeriodIndex(panel):
    """Convert a panel's major axis to a pandas PeriodIndex with the period
    dtype equal to 'period[A-DEC]'.

    For more on period dtype for pandas PeriodIndex, see
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#period-dtypes
    """
    major_axis_name = panel.major_axis.name
    panel.major_axis = pd.to_datetime(
        panel.major_axis, format="%Y").astype('period[A-DEC]')
    panel.major_axis.name = major_axis_name
    return panel


def convert_years_to_A_DEC_PeriodIndex(index):
    """Convert an index to an annual period index ending in December."""
    if hasattr(index, 'values'):
        index = index.values
    return (pd.to_datetime(index, format="%Y").astype('period[A-DEC]'))


def create_panel_from_long_data(
        long_data, values, items, major_axis, minor_axis):
    """Create a pandas Panel from a long dataset, with axes determined by the
    given columns.
    """
    return (
        long_data.pivot_table(
            values=values,
            index=[items, major_axis],
            columns=minor_axis)
        .to_panel()
        .transpose('major_axis', 'minor_axis', 'items'))


def filter_exports(
        exports_long,
        min_pop=1.25e6, year_min_pop=2008,
        min_total_exports=1e9, year_min_total_exports=2008,
        exclude_countries=set(('TCD', 'IRQ', 'AFG')),
        exclude_products=set(
            ('93', '94', '95', '96', '97', '32', '33', '34',)),
        min_frac_countries_that_must_export_a_product_in_every_year=0.8,
        min_frac_products_not_exported_some_year_to_exclude_country=.95,
        min_global_exports_each_year=10e6,
        round_to_zero_any_export_value_smaller_than=5000,
        verbose=1, filter_all_at_once=True,
        min_market_share_quantile=0.0, year_min_market_share=2008):
    """Filter exports in long format using the filters similar to those in:
    Albeaik, S., Kaltenberg, M., Alsaleh, M., & Hidalgo, C. A. (2017, July 18).
    Improving the Economic Complexity Index. arXiv.org.

    The column names of `exports_long` should be:
        product_code
        country_code
        export_value
        year
    """
    exports_long = exports_long.assign(
        export_value_is_zero=(exports_long.export_value == 0.0).values)

    country_codes_to_remove = set()
    product_codes_to_remove = set()

    def print_products_with_zero_global_exports(exports_long):
        global_exports_of_each_product = exports_long.pivot_table(
            columns='year', index='product_code',
            values='export_value', aggfunc='sum', fill_value=0.0)
        print(
            'If countries and products were removed now, this is the number '
            'of products with zero global exports in at least one year:',
            (global_exports_of_each_product == 0.0).any(axis=1).sum())
        print('\n\n')

    def _possibly_filter_now(exports_long):
        if verbose:
            print_products_with_zero_global_exports(
                exports_long[
                    (~exports_long.product_code.isin(product_codes_to_remove) &
                     ~exports_long.country_code.isin(country_codes_to_remove))]
            )
        if not filter_all_at_once:
            mask = (~exports_long.product_code.isin(product_codes_to_remove) &
                    ~exports_long.country_code.isin(country_codes_to_remove))
            return exports_long[mask]
        else:
            return exports_long

    country_codes_to_remove.update(get_countries_with_small_population(
        exports_long, min_pop=min_pop, year=year_min_pop, verbose=verbose))
    exports_long = _possibly_filter_now(exports_long)

    country_codes_to_remove.update(countries_with_little_exports(
        exports_long,
        min_total_exports=1e9, year_min_total_exports=2008, verbose=verbose))
    exports_long = _possibly_filter_now(exports_long)

    if verbose and exclude_countries:
        print('Excluding countries:', ', '.join(exclude_countries))
        print()
    country_codes_to_remove.update(set(exclude_countries))
    exports_long = _possibly_filter_now(exports_long)

    if verbose and exclude_products:
        print('Excluding products:', ', '.join(exclude_products))
        print()
    product_codes_to_remove.update(set(
        product_code
        for product_code in set(exports_long.loc[:, 'product_code'])
        if (
            (product_code in exclude_products) or
            (str(product_code)[:3] in exclude_products) or
            (str(product_code)[:2] in exclude_products) or
            (str(product_code)[:1] in exclude_products)
        )
    ))

    #  For each year, we exclude products when the dollar value of exports
    # is equal to zero for more than 80% of the countries
    product_codes_to_remove.update(products_not_exported_by_too_many_countries(
        exports_long,
        min_frac_countries_that_must_export_a_product_in_every_year,
        verbose=verbose))
    exports_long = _possibly_filter_now(exports_long)

    # We also exclude a country if it’s dollar value equals zero for 95% of
    # the products (in 2010 no country would have been excluded)
    country_codes_to_remove.update(
        countries_with_zero_exports_for_too_many_products_some_year(
            exports_long,
            min_frac_products_not_exported_some_year_to_exclude_country,
            verbose=verbose))
    exports_long = _possibly_filter_now(exports_long)

    # We also exclude a product if global exports are less than 10 million
    product_codes_to_remove.update(products_with_too_few_global_exports(
        exports_long,
        min_global_exports_each_year=min_global_exports_each_year,
        verbose=verbose))
    exports_long = _possibly_filter_now(exports_long)

    product_codes_to_remove.update(
        products_with_market_share_below_some_quantile(
            exports_long, min_market_share_quantile=min_market_share_quantile,
            year_min_market_share=year_min_market_share, verbose=verbose))
    exports_long = _possibly_filter_now(exports_long)

    if verbose:
        print('In the end, {} products are removed: {}'.format(
            len(product_codes_to_remove), product_codes_to_remove))
        print('\nAnd {} countries are removed: {}\n'.format(
            len(country_codes_to_remove), country_codes_to_remove))

    final_mask = (
        (~exports_long.country_code.isin(country_codes_to_remove)) &
        (~exports_long.product_code.isin(product_codes_to_remove)))
    exports_filtered = exports_long[final_mask]

    # round to zero any country-product combination that involves less than
    # USD 5,000 in exports.
    threshold = round_to_zero_any_export_value_smaller_than

    def round_to_zero(export_value):
        if export_value < threshold:
            return 0.0
        else:
            return export_value
    if verbose:
        num_thresholded = (exports_filtered.export_value < threshold).sum()
        print(('{} ({:.1%}) of (country, product) pairs have export_value < {}'
               ' and so will get rounded to 0.0').format(
                   num_thresholded,
                   num_thresholded / len(exports_long),
                   threshold))
        print()

    exports_filtered = exports_filtered.assign(
        export_value=exports_filtered.export_value.apply(round_to_zero))

    print('The resulting data has {} countries, {} products, and '
          '{} (country, year) pairs'.format(
              len(exports_filtered.country_code.unique()),
              len(exports_filtered.product_code.unique()),
              len(exports_filtered.loc[
                  :, ['country_code', 'year']].drop_duplicates())))
    return exports_filtered


def get_countries_with_small_population(
        exports_long, min_pop, year, verbose=0):
    # Filter countries with population `<= min_pop` in year `year_min_pop`:
    pop_mask = ((exports_long.year == year) &
                (exports_long.population <= min_pop))
    small_pop_countries = set(exports_long[pop_mask].country_code.values)
    if verbose:
        print('{} countries with population < {} in year {}: {}'.format(
            len(small_pop_countries), min_pop, year,
            sorted(small_pop_countries)))
        print()
    return small_pop_countries


def countries_with_small_population(
        exports_long, min_pop, year,
        population_data_directory='population_data', verbose=0):
    # Load population data from the World Bank
    start_year, end_year = min(exports_long.year), max(exports_long.year)
    pop_data = dlpop.load_population_all_countries(
        start_year, end_year, save_dir=population_data_directory)
    # Filter countries with population `<= min_pop` in year `year_min_pop`:
    pop_mask = ((pop_data.year == year) & (pop_data.population <= min_pop))
    small_pop_countries = set(pop_data[pop_mask].iso3c.values)
    if verbose:
        print('{} countries with population < {} in year {}: {}'.format(
            len(small_pop_countries), min_pop, year,
            sorted(small_pop_countries)))
        print()
    return small_pop_countries


def countries_with_little_exports(
        exports_long, min_total_exports=1e9, year_min_total_exports=2008,
        verbose=0):
    total_exports_in_year = (
        exports_long[exports_long.year == year_min_total_exports]
        .groupby('country_code')['export_value'].sum())

    result = set(total_exports_in_year[
        total_exports_in_year <= min_total_exports].index.unique())

    if verbose:
        print('{} countries with exports < {} in year {}: {}'.format(
            len(result), min_total_exports, year_min_total_exports,
            sorted(result)))
        print()

    return result


def countries_with_zero_exports_for_too_many_products_some_year(
        exports_long, frac=.95, verbose=0):
    frac_products_not_exported_by_a_country = (
        exports_long.pivot_table(
            columns='year', index='country_code',
            values='export_value_is_zero'))
    result = set(frac_products_not_exported_by_a_country[
        (frac_products_not_exported_by_a_country > frac).any(axis=1)].index)
    if verbose:
        print(('{} countries with zero export value for > {}'
               ' products in some year: {}').format(len(result), frac,
                                                    sorted(result)))
        print()
    return result


def products_not_exported_by_too_many_countries(
        exports_long, frac=.80, verbose=0):
    frac_countries_not_exporting_each_product = (
        exports_long.pivot_table(
            columns='year', index='product_code',
            values='export_value_is_zero'))
    result = set(frac_countries_not_exporting_each_product[
        (frac_countries_not_exporting_each_product > frac).any(axis=1)].index)

    if verbose:
        print(('{} products not exported by > {} countries'
               ' in some year: {}').format(len(result), frac, sorted(result)))
        print()
    return result


def products_with_too_few_global_exports(
        exports_long, min_global_exports_each_year=10e6, verbose=0):
    global_exports_of_each_product = exports_long.pivot_table(
        columns='year', index='product_code',
        values='export_value', aggfunc='sum')
    mask = (global_exports_of_each_product < min_global_exports_each_year)
    result = set(global_exports_of_each_product[mask.any(axis=1)].index)
    if verbose:
        print(('{} products with global exports < {}'
               ' in some year: {}').format(
                   len(result), min_global_exports_each_year, sorted(result)))
        print()
    return result


def products_with_market_share_below_some_quantile(
        exports_long, min_market_share_quantile=0.05,
        year_min_market_share=2008, verbose=0):
    """Return the product codes of products whose share of the global export
    market in the given year is less than or equal to the given quantile of all
    such products' market shares in that year.
    """
    if min_market_share_quantile <= 0:
        if verbose:
            print('min_market_share_quantile == 0, so no products will be '
                  'removed due to having too small a market share in year '
                  '{}'.format(year_min_market_share))
            print()
        return set()
    else:
        market_shares_of_products_in_year_of_min_market_share = (
            exports_long[exports_long.year == year_min_market_share]
            .groupby(['product_code'])
            .export_value
            .agg('sum'))

        min_market_share_threshold = (
            market_shares_of_products_in_year_of_min_market_share.quantile(
                min_market_share_quantile))

        product_codes_to_remove = set(
            market_shares_of_products_in_year_of_min_market_share[
                (market_shares_of_products_in_year_of_min_market_share <=
                    min_market_share_threshold)]
            .index
            .get_level_values('product_code'))

        if verbose:
            print('{} many products will be removed because they have market '
                  'share <= the {} quantile in year {}: {}'.format(
                      len(product_codes_to_remove), min_market_share_quantile,
                      year_min_market_share, sorted(product_codes_to_remove)))
            print()

        return product_codes_to_remove
