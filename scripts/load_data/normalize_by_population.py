from collections import OrderedDict
import load_data.download_population_data as dlpop
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
import warnings


class DivideByPopulationPowerLaw(BaseEstimator, TransformerMixin):
    """Normalize exports by a null model based on population and by the same
    expression for the whole world. See  Sec. 4.2.2 'Normalize export values by
    population and by global exports'.
    """

    def __init__(self):
        self.exponents_a = OrderedDict()
        self.exponents_b = OrderedDict()
        self.fit_results_numerator = OrderedDict()
        self.fit_results_denominator = OrderedDict()
        self.linear_regressions_numerator = OrderedDict()
        self.linear_regressions_denominator = OrderedDict()

    def fit(self, panel, y=None, fit_intercept=True,
            wide_population_data=None):
        """Fit linear regressions with the given exports time-series dataframe.
        """
        if wide_population_data is None:
            self.pop_wide = dlpop.load_wide_population_data_for_panel(panel)
        else:
            self.pop_wide = wide_population_data
        pop_long = self.pop_wide.stack()

        try:
            import statsmodels.formula.api as smf
            smf_success = True
        except ImportError:
            warnings.warn('Could not import `statsmodels.formula.api`, so'
                          ' only sklearn.LinearRegression will be used.')
            smf_success = False

        # numerator regressions
        for product_code in panel.minor_axis:
            log_exports_of_1product = (
                panel.loc[:, :, product_code].stack().apply(np.log).dropna())
            log_population_sizes = (
                pop_long.loc[log_exports_of_1product.index].apply(np.log))

            mask = (np.isfinite(log_exports_of_1product).values &
                    log_population_sizes.notnull().values)

            X = pd.DataFrame({'log_population': log_population_sizes[mask]})
            y = pd.Series(
                log_exports_of_1product[mask], name='log_export_value')

            lr = LinearRegression(fit_intercept=fit_intercept)
            lr.fit(X, y)
            self.linear_regressions_numerator[product_code] = lr
            self.exponents_a[product_code] = lr.coef_[0]
            if smf_success:
                self.fit_results_numerator[product_code] = smf.ols(
                    'log_export_value ~ log_population',
                    data=pd.concat((X, y), axis=1)).fit()

        # denominator regressions
        global_exports = panel.sum(axis=0)
        self.global_population = self.pop_wide.sum(axis=1)

        for product_code in panel.minor_axis:
            log_global_exports_1product = (
                global_exports.loc[:, product_code]
                .apply(np.log).dropna())
            log_global_population = (
                self.global_population.loc[log_global_exports_1product.index]
                .apply(np.log))

            mask = (np.isfinite(log_global_exports_1product).values &
                    log_global_population.notnull().values)

            X = pd.DataFrame({
                'log_population': log_global_population[mask]})
            y = pd.Series(
                log_global_exports_1product[mask], name='log_export_value')

            lr = LinearRegression(fit_intercept=fit_intercept)
            lr.fit(X, y)
            self.linear_regressions_denominator[product_code] = lr
            self.exponents_b[product_code] = lr.coef_[0]
            if smf_success:
                self.fit_results_denominator[product_code] = smf.ols(
                    'log_export_value ~ log_population',
                    data=pd.concat((X, y), axis=1)).fit()

        return self

    def transform(self, panel, y=None):
        """Return exports_cpt / pop_ct^a / (global_exports_pt / global_pop_t^b)
        """
        panel_normalized = panel.copy()

        for country in panel.items:
            for product_code in panel.minor_axis:

                if country not in self.pop_wide.columns:
                    panel_normalized.loc[country, :, product_code] = np.nan
                    continue

                country_log_pop = self.pop_wide.loc[:, country].apply(np.log)

                # need to fill missing with zero and then refill with missing
                # because LinearRegression.predict raises an exception on NaN
                mask_na = country_log_pop.isnull()

                lr_num = self.linear_regressions_numerator[product_code]
                export_value_predicted_by_pop = np.exp(
                    lr_num.predict(
                        pd.DataFrame(country_log_pop).fillna(0.)  # fill NaN
                    ))
                # re-insert missing values where they were found earlier
                export_value_predicted_by_pop[mask_na.values] = np.nan

                lr_denom = self.linear_regressions_denominator[product_code]
                predicted_global_exports = np.exp(
                    lr_denom.predict(
                        pd.DataFrame(self.global_population.apply(np.log))))
                normalized_global_exports = (predicted_global_exports /
                                             self.global_population)

                panel_normalized.loc[country, :, product_code] = (
                    panel.loc[country, :, product_code] /
                    export_value_predicted_by_pop /
                    normalized_global_exports)

        if hasattr(panel, 'name'):
            panel_normalized.name = panel.name
        if hasattr(panel, 'filename'):
            panel_normalized.filename = panel.filename

        return panel_normalized

    def inverse_transform(self, panel):
        """Return exports_cpt given a panel of
        exports_cpt / pop_ct^a / (global_exports_pt / global_pop_t^b)."""
        result = panel.copy()

        for country in panel.items:
            for product_code in panel.minor_axis:

                if country not in self.pop_wide.columns:
                    result.loc[country, :, product_code] = np.nan
                    continue

                country_log_pop = self.pop_wide.loc[:, country].apply(np.log)

                # need to fill missing with zero and then refill with missing
                # because LinearRegression.predict raises an exception on NaN
                mask_na = country_log_pop.isnull()

                lr_num = self.linear_regressions_numerator[product_code]
                export_value_predicted_by_pop = np.exp(
                    lr_num.predict(
                        pd.DataFrame(country_log_pop).fillna(0.)  # fill NaN
                    ))

                # re-insert missing values where they were found earlier
                export_value_predicted_by_pop[mask_na.values] = np.nan

                lr_denom = self.linear_regressions_denominator[product_code]
                predicted_global_exports = np.exp(
                    lr_denom.predict(
                        pd.DataFrame(self.global_population.apply(np.log))))
                normalized_global_exports = (predicted_global_exports /
                                             self.global_population)

                result.loc[country, :, product_code] = (
                    panel.loc[country, :, product_code] *
                    export_value_predicted_by_pop *
                    normalized_global_exports)

        return result
