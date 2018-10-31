import logging
import numpy as np
import pandas as pd
from pandas_datareader import wb
import os

logger = logging.getLogger(__name__)


# class GDPperCapitaPredictor():
#     """Class that learns to predict GDPpc from export baskets.

#     Attributes
#     ----------
#     panel_model : SKLearnPanelModel
#         Exports time-series data and a model of it.
#     gdppc_predictor : estimator with `fit` and `predict` methods
#         Model that predicts GDPpc from export baskets
#     gdppc_series : Series of GDPpc data
#         Data on GDP per capita
#     gdppc_predictions_and_empirical_history : DataFrame
#         GDPpc actual history and predicted future
#     gdppc_predictions_of_past_and_future : DataFrame
#         GDPpc predicted history and predicted future (predicted
#         from export baskets)
#     """
#     def __init__(
#             self, panel_model, gdppc_predictor,
#             gdppc_series='log_gdp_per_capita_constant2010USD'):
#         self.panel_model = panel_model
#         self.gdppc_predictor = gdppc_predictor
#         self._initialize_gdppc_series(gdppc_series)
#         self.df_exports_and_gdppc = merge_on_multiindex(
#             panel_model.df_dim_reduced, self.gdppc_series)
#         self.iterated_predictions_df = None
#         self._fitted = False
#         self._have_predicted_gdppc = False

#     def _initialize_gdppc_series(self, gdppc_series):
#         if gdppc_series == 'gdp_per_capita_constant2010USD':
#             self.gdppc_series = load_gdp_per_capita_nonmissing()
#             self.gdppc_series.name = 'gdp_per_capita_constant2010USD'
#         elif gdppc_series == 'log_gdp_per_capita_constant2010USD':
#             self.gdppc_series = load_gdp_per_capita_nonmissing().apply(np.log)
#             self.gdppc_series.name = 'log_gdp_per_capita_constant2010USD'
#         else:
#             self.gdppc_series = gdppc_series
#         if self.gdppc_series.name is None:
#             self.gdppc_series.name = 'gdp_per_capita'

#     def fit(self, **fit_kws):
#         """Learn to predict GDP per capita on the dimension-reduced data in the
#         `panel_model`."""
#         exports_columns = self.panel_model.df_dim_reduced.columns
#         gdppc_column = self.gdppc_series.name
#         self.gdppc_predictor.fit(
#             self.df_exports_and_gdppc.loc[:, exports_columns],
#             self.df_exports_and_gdppc.loc[:, gdppc_column], **fit_kws)
#         self._fitted = True
#         return self.gdppc_predictor

#     def compute_iterated_predictions(
#             self, num_time_steps=10, **kws):
#         self.iterated_predictions_df = (self.panel_model.iterated_predictions(
#             num_time_steps=num_time_steps, as_combined_dataframe=True, **kws))

#         self.iterated_predictions_and_history_df = (
#             concat_on_history_earlier_than_first_prediction(
#                 self.panel_model.df_dim_reduced, self.iterated_predictions_df,
#                 drop_earliest_prediction=True))

#     def predict_gdppc(self):
#         predicted_future_gdppc = self._predict_future_gdppc()
#         self._predict_past_gdppc(predicted_future_gdppc)
#         self._have_predicted_gdppc = True

#     def _predict_future_gdppc(self):
#         """Predict future GDP per capita from exports and return the resulting
#         DataFrame.

#         The future predictions and past data are concatenated, converted to a
#         dataframe (with a column `predicted` containing `True` if that row is
#         a prediction and `False` if that row is empirical data), and stored in
#         the attribute `gdppc_predictions_and_empirical_history`.
#         """
#         self._check_fitted_gdppc_predictor()
#         self.predicted_future_gdppc = pd.DataFrame({
#             self.gdppc_series.name: self.gdppc_predictor.predict(
#                 self.iterated_predictions_df),
#             'predicted': True},
#             index=self.iterated_predictions_df.index)

#         gdppc_df = pd.DataFrame(self.gdppc_series.copy())
#         gdppc_df['predicted'] = False
#         self.gdppc_predictions_and_empirical_history = (
#             concat_on_history_earlier_than_first_prediction(
#                 gdppc_df, self.predicted_future_gdppc))

#         return self.predicted_future_gdppc

#     def _predict_past_gdppc(self, predicted_future_gdppc):
#         """Predict past GDP per capita from exports.

#         The future predictions and past predictions are concatenated and stored
#         in the attribute `gdppc_predictions_of_past_and_future`.
#         """
#         self._check_fitted_gdppc_predictor()
#         self.predicted_past_gdppc = pd.DataFrame({
#             self.gdppc_series.name: self.gdppc_predictor.predict(
#                 self.panel_model.df_dim_reduced),
#             'predicted': True},
#             index=self.panel_model.df_dim_reduced.index)
#         self.gdppc_predictions_of_past_and_future = (
#             concat_on_history_earlier_than_first_prediction(
#                 self.predicted_past_gdppc, self.predicted_future_gdppc))

#     def _check_fitted_gdppc_predictor(self):
#         if not self._fitted:
#             raise RuntimeError("The GDPpc predictor has not yet been fiited. "
#                                "First call the `fit` method.")

#     def _check_computed_iterated_predictions(self):
#         if self.iterated_predictions_df is None:
#             raise RuntimeError("Iterated predictions have not been computed. "
#                                "Call `compute_iterated_predictions` first.")
#         return

#     def plot_gdppc_predictions_and_empirical_history(
#             self, exponentiate=False, **kws):
#         self._check_computed_iterated_predictions()
#         plot_gdppc(self.gdppc_predictions_and_empirical_history,
#                    exponentiate=exponentiate, **kws)

#     def plot_gdppc_predictions_of_past_and_future(
#             self, exponentiate=False, **kws):
#         self._check_computed_iterated_predictions()
#         plot_gdppc(self.gdppc_predictions_of_past_and_future,
#                    exponentiate=exponentiate, **kws)

#     def plot_iterated_predictions_and_history(
#             self, columns_to_plot='all', subplots_kws=None, plot_kws=None,
#             map_countries_to_colors=None):
#         subplots_kws = {} if subplots_kws is None else subplots_kws
#         plot_kws = {} if plot_kws is None else plot_kws
#         df = self.iterated_predictions_and_history_df
#         if columns_to_plot == 'all':
#             columns_to_plot = df.columns
#         fig, axes = plt.subplots(len(columns_to_plot), 1, **subplots_kws)
#         axes_flat = axes.flatten()
#         countries = df.index.levels[0]
#         for col, ax in zip(columns_to_plot, axes_flat):
#             for country in countries:
#                 country_df = df.loc[country]
#                 if map_countries_to_colors is not None:
#                     plot_kws['c'] = map_countries_to_colors[country]
#                 country_df.loc[:, 0].plot(
#                     ax=ax, **plot_kws)

#     def sorted_gdppc_predicted(self, year, ascending=False):
#         return (
#             self.gdppc_predictions_and_empirical_history
#             .loc[(slice(None), str(year)), self.gdppc_series.name]
#             .sort_values(ascending=ascending))

#     def countries_that_grow_most(
#             self, start_year, end_year, ascending=False, method='divide',
#             predictions_only=False):
#         if predictions_only:
#             gdppc_series = (self.gdppc_predictions_of_past_and_future
#                             .loc[:, self.gdppc_series.name])
#         else:
#             gdppc_series = (self.gdppc_predictions_and_empirical_history
#                             .loc[:, self.gdppc_series.name])

#         gdppc_start = gdppc_series.loc[:, str(start_year)]
#         gdppc_end = gdppc_series.loc[:, str(end_year)]
#         if method == 'divide':
#             result = gdppc_end.div(gdppc_start)
#         else:
#             result = gdppc_end.subtract(gdppc_start)
#         return result.dropna().sort_values(ascending=ascending)


# def merge_on_multiindex(x, y, sort_index=False):
#     index_levels_x = x.index.names
#     index_levels_y = y.index.names
#     if index_levels_x != index_levels_y:
#         raise ValueError("The levels of the indices of `x` and `y` must have "
#                          "the same names.")
#     merged = (pd.merge(
#         x.reset_index(), y.reset_index(), on=index_levels_x)
#         .set_index(index_levels_x))
#     return merged.sort_index() if sort_index else merged


# def concat_on_history_earlier_than_first_prediction(
#         history, predictions, drop_earliest_prediction=True):
#     """Concatenate two Series (or DataFrames) but take only the rows in history
#     that are earlier than (or the same year as) the earliest prediction for
#     that country, and drop the earliest prediction for each country."""
#     country_to_first_year_predicted = (
#         predictions.reset_index()
#         .groupby('country_code')['year'].min())

#     def keep_prediction_row(country, year):
#         if drop_earliest_prediction:
#             return (country in country_to_first_year_predicted and
#                     year > country_to_first_year_predicted[country])
#         else:
#             return True
#     mask_predictions = np.array([
#         keep_prediction_row(country, year)
#         for (country, year) in predictions.index.values])

#     def keep_historical_row(country, year):
#         return (country in country_to_first_year_predicted and
#                 year <= country_to_first_year_predicted[country])
#     mask_history = np.array([
#         keep_historical_row(country, year)
#         for (country, year) in history.index.values])

#     return pd.concat([
#         predictions[mask_predictions], history[mask_history]
#     ]).sort_index()


# def plot_gdppc(
#         gdppc_df, exponentiate=False, ax=None,
#         kws_predicted=None, kws_empirical=None, subplots_kws=None,
#         map_countries_to_colors=None, fallback_color='.5',
#         label_countries_in_color_dict=False, xytext=(5, 0),
#         connect_last_history_to_first_prediction=True, **kws):
#     """Plot a dataframe of GDP per capita, with potentially different options
#     for the predicted values (rows with column `predicted == True`) than for
#     the empirical values (rows with column `predicted == False`).
#     """
#     kws_predicted = {} if kws_predicted is None else kws_predicted
#     kws_empirical = {} if kws_empirical is None else kws_empirical
#     kws_predicted.update(kws)
#     kws_empirical.update(kws)
#     subplots_kws = {} if subplots_kws is None else subplots_kws
#     gdppc_column = [c for c in gdppc_df.columns if c != 'predicted'][0]
#     fig, ax = create_fig_ax(ax, **subplots_kws)
#     gdppc_df_predicted = gdppc_df[gdppc_df.predicted]
#     gdppc_df_empirical = gdppc_df[~gdppc_df.predicted]

#     def identity_fn(x):
#         return x
#     transformation = (np.exp if exponentiate else identity_fn)

#     if map_countries_to_colors is None:
#         map_countries_to_colors = {}
#         color_cycler = ax._get_lines.prop_cycler
#         for country in gdppc_df.index.levels[0]:
#             map_countries_to_colors[country] = next(color_cycler)['color']

#     for country in gdppc_df.index.levels[0]:
#         color = map_countries_to_colors.get(country, fallback_color)

#         if len(gdppc_df_predicted.loc[country, gdppc_column]) > 0:
#             if connect_last_history_to_first_prediction:
#                 pred_data = pd.concat([
#                     gdppc_df_empirical.loc[country, gdppc_column].iloc[-1:],
#                     gdppc_df_predicted.loc[country, gdppc_column]], axis=0)
#             else:
#                 pred_data = gdppc_df_predicted.loc[country, gdppc_column]
#             pred_data.apply(transformation).plot(
#                 c=color, ax=ax, **kws_predicted)
#         if len(gdppc_df_empirical.loc[country, gdppc_column]):
#             (
#                 gdppc_df_empirical.loc[country, gdppc_column]
#                 .apply(transformation)
#                 .plot(c=color, ax=ax, **kws_empirical))
#         if (label_countries_in_color_dict and
#                 country in map_countries_to_colors):
#             last_val = gdppc_df_predicted.loc[country, gdppc_column].iloc[-1:]
#             x = last_val.index
#             y = last_val.values[0]
#             ax.annotate(xy=(x, y), s=country, ha='center', va='center',
#                         xytext=xytext, textcoords='offset points')


# def predict_gdppc(
#         model_exports, model_gdppc,
#         exports_dataframe, gdppc_indexed_by_country_year,
#         base_years, horizon=5,
#         iterated_one_year_predictions=True, preprocessor=None):
#     """Return predictions and errors of GDP per capita from export baskets.

#     This function is used to make comparison's with the predictive performance
#     of the IMF.

#     Parameters
#     ----------
#     model_exports : estimator for predicting next year's export baskets
#         This model need not be fitted already; it is fit by this function.

#     model_gdppc : estimator for predicting GDP per capita from exports
#         This model need not be fitted already; it is fit by this function.

#     exports_dataframe ; multiindex dataframe of export time-series

#     base_years : list of integers
#         The base years for making predictions `horizon` years later. That is,
#         predict GDP per capita in year `t + horizon` based on data up to year
#         `t` for each year `t` in `base_years`.

#     horizon : int
#         The horizon of the forecast

#     iterated_one_year_predictions : bool, default: True

#     preprocessor : dimension reducor or other transformer, default: None
#         The exports data is transformed by this estimator before predictions
#         are made.

#     Returns
#     -------
#     result : dict
#         A dictionary of many results, with keys:
#             'exports_predictions',
#             'gdppc_predictions',
#             'gdppc_target_test',
#             'annualized_growth_rate_predictions',
#             'annualized_growth_rate_actual',
#             'mse_growth_rate',
#             'rmse_growth_rate',
#             'mae_growth_rate',
#             'mse_gdppc',
#             'mae_gdppc',
#             'mse_exports',
#             'mae_exports',
#             'num_test_samples',
#             'num_train_samples_exports_to_exports',
#             'num_train_samples_exports_to_gdppc',
#     """
#     years_to_predict = [y + horizon for y in base_years]
#     num_lags = (1 if iterated_one_year_predictions else horizon)
#     data_exports, target_exports = (
#         split_data_target.split_panel_into_data_and_target_and_fill_missing(
#             multiindex_to_panel(exports_dataframe), num_lags=num_lags))
#     data_exports = data_exports.loc[:, num_lags]
#     target_exports = target_exports.loc[:, 0]

#     # learn to predict export baskets on this training set
#     slice_training_set = (
#         slice(None),  # every country
#         slice(None, str(min(base_years) - horizon)))
#     exports_data_train = data_exports.loc[slice_training_set, :]
#     exports_target_train = target_exports.loc[slice_training_set, :]

#     slice_test_set = (slice(None), [str(y) for y in years_to_predict])
#     exports_data_test = data_exports.loc[slice_test_set, :]
#     exports_target_test = target_exports.loc[slice_test_set, :]

#     if preprocessor:
#         preprocessor.fit(exports_data_train)
#         exports_data_train = pd.DataFrame(
#             preprocessor.transform(exports_data_train),
#             index=exports_data_train.index)
#         exports_target_train = pd.DataFrame(
#             preprocessor.transform(exports_target_train),
#             index=exports_target_train.index)
#         exports_data_test = pd.DataFrame(
#             preprocessor.transform(exports_data_test),
#             index=exports_data_test.index)
#         exports_target_test = pd.DataFrame(
#             preprocessor.transform(exports_target_test),
#             index=exports_target_test.index)

#     model_exports.fit(exports_data_train, exports_target_train)

#     # learn to predict GDPpc PPP from export baskets using this training set
#     gdppc_target_train = gdppc_indexed_by_country_year.loc[
#         exports_target_train.index.values]

#     not_missing_gdppc = pd.notnull(gdppc_target_train)
#     model_gdppc.fit(exports_target_train.loc[not_missing_gdppc],
#                     gdppc_target_train.loc[not_missing_gdppc])

#     # predict future exports
#     exports_predictions = exports_data_test
#     # either we do 1 horizon-length prediction
#     # or horizon many 1-step predictions
#     for __ in range(horizon // num_lags):
#         exports_predictions = model_exports.predict(exports_predictions)
#     mse_exports = mean_squared_error(exports_predictions, exports_target_test)
#     mae_exports = mean_absolute_error(exports_predictions, exports_target_test)

#     gdppc_predictions = model_gdppc.predict(exports_predictions)
#     gdppc_target_test = gdppc_indexed_by_country_year.loc[
#         exports_target_test.index.values]

#     # remove entries for which GDPpc is missing
#     mask_have_gdppc = pd.notnull(gdppc_target_test)
#     gdppc_target_test = gdppc_target_test[mask_have_gdppc]
#     gdppc_predictions = pd.Series(gdppc_predictions[mask_have_gdppc],
#                                   index=gdppc_target_test.index)

#     mse_gdppc = mean_squared_error(gdppc_predictions, gdppc_target_test)
#     mae_gdppc = mean_absolute_error(gdppc_predictions, gdppc_target_test)

#     gdppc_target_train = gdppc_indexed_by_country_year.loc[
#         [(c, y - horizon) for (c, y) in gdppc_target_test.index.values]]

#     annualized_growth_rate_pred = (
#         (gdppc_predictions.values / gdppc_target_train.values)**(1 / horizon) -
#         1) * 100
#     annualized_growth_rate_actual = (
#         (gdppc_target_test.values / gdppc_target_train.values)**(1 / horizon) -
#         1) * 100

#     mse_growth_rate = mean_squared_error(
#         annualized_growth_rate_pred, annualized_growth_rate_actual)
#     rmse_growth_rate = np.sqrt(mse_growth_rate)
#     mae_growth_rate = mean_absolute_error(
#         annualized_growth_rate_pred, annualized_growth_rate_actual)

#     return {
#         'exports_predictions': exports_predictions,
#         'gdppc_predictions': gdppc_predictions,
#         'gdppc_target_test': gdppc_target_test,
#         'annualized_growth_rate_predictions': annualized_growth_rate_pred,
#         'annualized_growth_rate_actual': annualized_growth_rate_actual,
#         'mse_growth_rate': mse_growth_rate,
#         'rmse_growth_rate': rmse_growth_rate,
#         'mae_growth_rate': mae_growth_rate,
#         'mse_gdppc': mse_gdppc,
#         'mae_gdppc': mae_gdppc,
#         'mse_exports': mse_exports,
#         'mae_exports': mae_exports,
#         'num_test_samples': sum(mask_have_gdppc),
#         'num_train_samples_exports_to_exports': exports_data_train.shape[0],
#         'num_train_samples_exports_to_gdppc': sum(not_missing_gdppc),
#         # 'model_exports': model_exports, 'model_gdppc': model_gdppc
#     }


# class ExportsAndGDPpcAutoencoder():
#     """Train an autoencoder that may also learn to predict GDPpc."""
#     def __init__(self, n_reduced_dimensions=2):
#         self.n_reduced_dimensions = n_reduced_dimensions

#     def fit(self, X, gdppc_column='gdp_per_capita_constant2010USD',
#             initialize_with_pca=False,
#             scaler=None, train_split=0.8,
#             epochs=200, batch_size=32,
#             nn_kws=None):
#         nn_kws = {} if nn_kws is None else nn_kws
#         default_kws = dict(
#             lr=.001, loss='mse', weight_l1=0.0, weight_l2=0.0,
#             activity_l1=0.0, activity_l2=0.0,
#             kernel_initializer='glorot_uniform', bias_initializer='zeros',
#             activation='relu', activation_last_layer='linear',
#             optimizer='rmsprop',
#             leaky_relu=None, batch_normalize=True,
#             batch_normalize_at_start=False)
#         for kw in default_kws:
#             if kw not in nn_kws:
#                 nn_kws[kw] = default_kws[kw]

#         if scaler is None:
#             self.scaler = FunctionTransformer(None)
#         else:
#             self.scaler = scaler

#         if gdppc_column not in X.columns:
#             raise ValueError('gdppc_column = {} is not one of the columns '
#                              'of X.'.format(gdppc_column))

#         mtss = MultiTimeSeriesSplit(
#             n_train_test_sets=1,
#             first_split_fraction=train_split,
#             level_of_index_for_time_values='year')
#         ae_train_indices, ae_test_indices = next(mtss.split(X, X))

#         scaler.fit(X.iloc[ae_train_indices])
#         X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

#         export_columns = [c for c in X.columns if c != gdppc_column]
#         input_train = X.iloc[ae_train_indices].loc[:, export_columns]
#         input_test = X.iloc[ae_test_indices].loc[:, export_columns]
#         output_train = X.iloc[ae_train_indices]
#         output_test = X.iloc[ae_test_indices]

#         training_years = [
#             str(y)
#             for y in input_train.index.get_level_values('year').unique()]
#         print('training set: {} indices, years {}'.format(
#             len(ae_train_indices), ', '.join(sorted(training_years))))
#         testing_years = [
#             str(y) for y in input_test.index.get_level_values('year').unique()]
#         print('testing set: {} indices, years {}'.format(
#             len(ae_test_indices), ', '.join(sorted(testing_years))))

#         kernel_initializer_first_layer = None
#         if initialize_with_pca:
#             print('initializing with PCA')
#             pca = PCA(n_components=self.n_reduced_dimensions)
#             pca.fit(input_train)
#             print('mean of columns of pca components:',
#                   pca.components_.T.mean(axis=0))

#             def pca_init(shape, dtype=None):
#                 return pca.components_.T
#             kernel_initializer_first_layer = pca_init

#         self.autoencoder = build_hourglass(
#             n_hidden_layers=1,
#             size_smallest_layer=self.n_reduced_dimensions, ramp_type='linear',
#             n_inputs=input_train.shape[1], n_outputs=output_train.shape[1],
#             size_extra_first_layer=None,
#             kernel_initializer_first_layer=kernel_initializer_first_layer)

#         try:
#             t0 = time.time()
#             self.history = self.autoencoder.fit(
#                 np.array(input_train), np.array(output_train),
#                 validation_data=(input_test.values, output_test.values),
#                 epochs=epochs, batch_size=batch_size,
#                 callbacks=early_stopping_and_reduce_learning_rate)
#             t1 = time.time()
#             print('time:', datetime.timedelta(seconds=(t1 - t0)))
#         except KeyboardInterrupt:
#             pass

#         self.dimension_reducing_model = self.make_dimension_reducing_model()

#     def make_dimension_reducing_model(self):
#         autoencoder = self.autoencoder
#         indx_smallest_layer = np.argmin([
#             l.output_shape[1] for l in autoencoder.layers])
#         if indx_smallest_layer == len(autoencoder.layers) - 1:
#             raise ValueError('Cannot split a model that only reduces dimension'
#                              's. The smallest layer cannot be the last.')
#         smallest_layer = autoencoder.layers[indx_smallest_layer]

#         input_to_predictor_from_reduced_dimensions = keras.models.Input(
#             shape=smallest_layer.output_shape[1:])

#         predictor_tensor = input_to_predictor_from_reduced_dimensions
#         for expanding_layer in autoencoder.layers[indx_smallest_layer + 1:]:
#             predictor_tensor = expanding_layer(predictor_tensor)

#         predictor_from_reduced_dimensions = keras.models.Model(
#             inputs=input_to_predictor_from_reduced_dimensions,
#             outputs=predictor_tensor)
#         dim_reducer = keras.models.Model(
#             inputs=autoencoder.input, outputs=smallest_layer.output)

#         return DimensionReducingKerasRegressor(
#             autoencoder, dim_reducer, predictor_from_reduced_dimensions)

#     def visualize_history(self, ax=None):
#         return visualize_history(self.history, ax=ax)

#     def weight_histograms(self, layer=0, ax=None):
#         fig, ax = create_fig_ax(ax, nrows=1, ncols=self.n_reduced_dimensions)
#         for i in range(self.n_reduced_dimensions):
#             ax[i].hist(self.autoencoder.get_weights()[layer][:, i])

#     def weight_scatter(self, layer=0, ax=None):
#         fig, ax = create_fig_ax(ax)
#         ax.scatter(*self.autoencoder.get_weights()[layer][:, :2].T)

#     def plot_trajectories(self, X, ax=None, dimensions_to_keep=(0, 1),
#                           color_dict=None):
#         fig, ax = create_fig_ax(ax)

#         X_scaled = self.scaler.transform(X)
#         X_scaled = X_scaled[
#             :, :self.dimension_reducing_model.model.input_shape[1]]
#         X_reduced_by_autoencoder = pd.DataFrame(
#             self.dimension_reducing_model.reduce_dimensions(X_scaled))
#         X_reduced_by_autoencoder.index = X.index
#         X_reduced_by_autoencoder.columns = [
#             'dimension_{}'.format(i)
#             for i in range(X_reduced_by_autoencoder.shape[1])]
#         countries = X_reduced_by_autoencoder.index.get_level_values(
#             'country_code')
#         if color_dict is None:
#             color_dict = dict(zip(countries, make_color_cycler()))

#         for country in countries:
#             color = color_dict[country]
#             timeseries = X_reduced_by_autoencoder.loc[country]
#             ax.plot(*timeseries.values[:, dimensions_to_keep].T,
#                     alpha=.6, color=color)
#             if len(timeseries):
#                 ax.text(*timeseries.values[-1, dimensions_to_keep],
#                         country, color=color)

#     def loading_heatmap(self, layer=0, ax=None):
#         vis_dim_red.feature_loading_heatmap(
#             ae.autoencoder.get_weights()[layer].T, X.columns, ax=ax)


def append_gdppc(df):
    gdp_per_capita_nonmissing = (
        load_gdp_per_capita_nonmissing().apply(np.log10))
    gdp_per_capita_nonmissing.name = 'log10_gdp_per_capita_constant2010USD'
    df_copy = df.copy()
    if 'product_code' in df_copy.columns.names:
        df_copy.columns = [
            int(product_code)
            for product_code in df.columns.get_level_values('product_code')]
    result = pd.merge(
        df_copy.reset_index(),
        gdp_per_capita_nonmissing.reset_index(),
        on=['country_code', 'year']).set_index(['country_code', 'year'])
    if hasattr(df, 'name'):
        result.name = df.name
    if hasattr(df, 'filename'):
        result.filename = df.filename
    return result


def load_gdp_per_capita_data(
        start_year, end_year=None, countries='all', force_download=False,
        save_dir=os.path.join(os.pardir, "data", "raw", "income")):
    if end_year is None:
        end_year = start_year

    file_name = (
        "WorldBank_GDPperCapita_{start_year}_to_{end_year}.json".format(
            start_year=start_year, end_year=end_year))
    path = os.path.join(save_dir, file_name)

    if os.path.exists(path) and not force_download:
        logging.info(
            "The data already exists in the path \n\t{path}"
            "\nThat data will be returned. To force a new download of the "
            "data, use `force_download=True`.".format(path=path))
        return pd.read_json(path, convert_dates='year_start').sort_index()
    else:
        gdp_data = download_gdp_per_capita_data(
            start_year, end_year, countries=countries)

        # Write the population data to disk
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print("Created directory {dir}".format(dir=save_dir))
        gdp_data.to_json(path, date_format='iso')
        return gdp_data


def download_gdp_per_capita_data(start_year, end_year, countries='all'):
    gdp_data = wb.download(
        indicator=['NY.GDP.PCAP.KD', 'NY.GDP.PCAP.PP.CD', 'NY.GDP.PCAP.PP.KD'],
        country=countries, start=start_year, end=end_year)
    rename_cols = {
        'NY.GDP.PCAP.KD': 'gdp_per_capita_constant2010USD',
        'NY.GDP.PCAP.PP.CD': 'gdp_per_capita_PPP_current_international_dollar',
        'NY.GDP.PCAP.PP.KD': (
            'gdp_per_capita_PPP_constant_2011_international_dollar')}
    gdp_data = pd.merge(
        gdp_data.reset_index().rename(columns=rename_cols),
        wb.get_countries(), left_on='country', right_on='name', how='inner')
    return gdp_data


def load_gdp_per_capita_nonmissing(start_year=1962, end_year=2016):
    """Load GDP per capita in constant 2010 USD (indicator NY.GDP.PCAP.KD)
    and drop missing rows.

    This indicator (from the World Bank) is the one missing the least data, at
    least compared to GDP PPP (constant 2011 international dollar, indicator
    NY.GDP.PCAP.PP.CD) and GDP PPP (current international dollar,
    indicator NY.GDP.PCAP.PP.CD).
    """
    gdp_per_capita = (
        load_gdp_per_capita_data(
            start_year=start_year, end_year=end_year,
            countries='all', force_download=False)
        .rename(columns={'iso3c': 'country_code'})
        .set_index(['country_code', 'year']).sort_index())

    gdp_per_capita.index.set_levels(
        pd.PeriodIndex(
            pd.to_datetime(gdp_per_capita.index.levels[1], format="%Y"),
            freq='A-DEC'), 1, inplace=True)

    gdp_per_capita_nonmissing = (
        gdp_per_capita.gdp_per_capita_constant2010USD.dropna())
    return gdp_per_capita_nonmissing



# def visualize_gams(
#         gams, feature_labels, target_labels='feature_labels',
#         plot_mean_kws=None, plot_ci_kws=None, fill_kws=None,
#         convert_last_axis_to_log_10=False):
#     plot_mean_kws = {} if plot_mean_kws is None else plot_mean_kws
#     plot_ci_kws = {} if plot_ci_kws is None else plot_ci_kws
#     fill_kws = {} if fill_kws is None else fill_kws

#     # Default parameter values
#     if 'color' not in plot_mean_kws:
#         plot_mean_kws.update(color='k')
#     if 'color' not in plot_ci_kws:
#         plot_ci_kws.update(color='1.0')
#     if 'color' not in fill_kws:
#         fill_kws.update(color='.8')

#     if target_labels == 'feature_labels':
#         target_labels = feature_labels

#     n_features = len(feature_labels)
#     fig, axes = plt.subplots(len(gams), n_features, sharey='row', sharex='col')
#     for i, gam in enumerate(gams):
#         X_grid = generate_X_grid(gam)
#         partial_dependence, confidence_interval = gam.partial_dependence(
#             X_grid, width=.95)

#         if convert_last_axis_to_log_10:
#             X_grid[:, -1] /= np.log(10)
#             partial_dependence /= np.log(10)
#             confidence_interval /= np.log(10)

#         for j in range(n_features):
#             axes[i, j].plot(X_grid[:, j], partial_dependence[:, j],
#                             **plot_mean_kws)
#             axes[i, j].plot(X_grid[:, j], confidence_interval[j],
#                             **plot_ci_kws)
#             axes[i, j].fill_between(X_grid[:, j], *confidence_interval[j].T,
#                                     **fill_kws)
#     for j, (xlabel, ylabel) in enumerate(zip(feature_labels, target_labels)):
#         axes[j, 0].set_ylabel(ylabel)
#         axes[-1, j].set_xlabel(xlabel)


# def visualize_gdppc_from_gam(panel_model):
#     """Visualize GDPpc's partial dependence on the first 2 dimensions."""
#     gam = panel_model.best_model.estimators_[2]
#     X_grid = generate_X_grid(gam, n=50)
#     partial_dependence, __ = gam.partial_dependence(X_grid, width=.95)

#     XY_meshgrid = np.meshgrid(X_grid[:, 0], X_grid[:, 1])
#     pred_grid = (partial_dependence[:, 0].reshape(1, -1) +
#                  partial_dependence[:, 1].reshape(-1, 1))
#     pred_grid = vis_model.mask_arrays_with_convex_hull(
#         [pred_grid], [X_grid[:, 0], X_grid[:, 1]],
#         ConvexHull(panel_model.X.iloc[:, :2]))[0]
#     plt.pcolor(
#         *XY_meshgrid,
#         pred_grid,
#         cmap=shifted_color_map(
#             plt.get_cmap('coolwarm'),
#             data=pred_grid))
#     plt.colorbar(label='change in $\log_{10}$(GDPpc) predicted by PC0 and PC1,'
#                  ' i.e., $f_{20}(PC_0) + f_{21}(PC_1)$')
#     plt.xlabel('score on the first principal component')
#     plt.ylabel('score on the second principal component')
#     plt.title(panel_model.panel.name)
#     plt.show()
