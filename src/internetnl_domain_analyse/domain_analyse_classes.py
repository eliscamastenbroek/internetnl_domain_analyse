import logging
import os
import pickle
import re
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from internetnl_domain_analyse.domain_plots import (make_cdf_plot, make_bar_plot)
from internetnl_domain_analyse.utils import (read_tables_from_sqlite,
                                             get_clean_url,
                                             fill_booleans,
                                             prepare_stat_data_for_write,
                                             get_option_mask,
                                             impose_variable_defaults)
from internetnl_domain_analyse.latex_output import make_latex_overview
from ict_analyser.analyser_tool.utils import (prepare_df_for_statistics,
                                              get_records_select,
                                              rename_all_variables)
from ict_analyser.shared.utils import variable_dict_to_df
from ict_analyser.analyser_tool.sample_statistics import SampleStatistics
from ict_analyser.analyser_tool.variable_properties import VariableProperties

_logger = logging.getLogger(__name__)

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


def make_plot_cache_file_name(cache_directory, file_base, prefix):
    return cache_directory / Path("_".join([prefix, file_base, "cache_for_plot.pkl"]))


class DomainAnalyser(object):
    def __init__(self,
                 scan_data_key=None,
                 cache_file_base="tables_df",
                 cache_directory=None,
                 output_file=None,
                 reset=None,
                 records_filename=None,
                 records_table_names=None,
                 internet_nl_filename=None,
                 breakdown_labels=None,
                 statistics: dict = None,
                 default_scan=None,
                 variables: dict = None,
                 module_info: dict = None,
                 weights=None,
                 url_key="website_url",
                 translations=None,
                 module_key="module",
                 variable_key="variable",
                 sheet_renames=None,
                 n_digits=None,
                 write_dataframe_to_sqlite=False,
                 statistics_to_xls=False,
                 n_bins=100,
                 mode=None,
                 correlations=None,
                 categories=None,
                 ):

        _logger.info(f"Running here {os.getcwd()}")

        if output_file is None:
            self.output_file = "output.sqlite"
        else:
            self.output_file = output_file

        self.scan_data_key = scan_data_key
        self.breakdown_labels = breakdown_labels

        self.correlations = correlations
        self.categories = categories
        self.statistics = statistics
        self.default_scan = default_scan
        self.module_key = module_key
        self.variable_key = variable_key
        self.module_info = module_info
        self.variables = self.variable_dict2df(variables, module_info)
        self.n_digits = n_digits
        self.n_bins = n_bins

        self.sheet_renames = sheet_renames

        self.url_key = url_key
        self.be_id = "be_id"
        self.mi_labels = ["sbi", "gk_sbs", self.be_id]
        self.translations = translations

        self.categories_coefficient_df = None
        self.correlation_coefficient_df = None

        if records_filename is None:
            self.records_filename = Path(cache_directory) / Path("records_cache.sqlite")
        else:
            self.records_filename = records_filename

        if records_table_names is None:
            self.records_tables_names = ["records_df_20_2", "info_records_df_20_2"]
        else:
            self.records_tables_names = records_table_names

        if internet_nl_filename is not None:
            self.internet_nl_filename = internet_nl_filename
        else:
            self.internet_nl_filename = Path("internet_nl.sqlite")

        self.cache_directory = cache_directory
        cache_file_base = Path("_".join([cache_file_base, scan_data_key]) + ".pkl")
        self.cache_file = Path(cache_directory) / cache_file_base
        self.cate_outfile = None
        self.cate_pkl_file = None
        self.corr_outfile = None
        self.corr_pkl_file = None
        self.score_outfile = None
        self.score_pkl_file = None
        if self.correlations is not None:
            plot_info = self.correlations["plots"]
        else:
            plot_info = None
        try:
            self.cate_outfile = self.cache_directory / Path(
                self.categories["categories_output_file"])
        except TypeError:
            _logger.debug("categories not defined")
        else:
            self.cate_pkl_file = self.cate_outfile.with_suffix(".pkl")
        try:
            self.corr_outfile = self.cache_directory / Path(
                plot_info["correlation"]["output_file"])
        except TypeError:
            _logger.debug("correlations not defined")
        else:
            self.corr_pkl_file = self.corr_outfile.with_suffix(".pkl")
            self.score_outfile = self.cache_directory / Path(
                plot_info["scores_per_interval"]["output_file"])
            self.score_pkl_file = self.score_outfile.with_suffix(".pkl")

        if reset is None:
            self.reset = None
        else:
            self.reset = int(reset)
        self.weight_key = weights

        self.dataframe = None
        self.score_df = None
        self.all_stats_per_format = dict()
        self.all_hist_per_format = dict()

        self.all_plots = None

        have_cache = self.check_if_cache_exist(mode)

        if (self.reset is not None and self.reset <= 1) or not have_cache:
            # de microdata alleen lezen als we geen pickle files van de statistische output hebben
            # als we alleen plaatjes willen maken is het sneller om de uitgerekende tabellen van
            # cache te lezen
            self.read_data()

        if write_dataframe_to_sqlite:
            if self.dataframe is not None:
                self.write_data()
                sys.exit(0)
            else:
                msg = "The write_data_frame option only works if you force to read the micro " \
                      "data with --reset 0 or --reset 1"
                raise ValueError(msg)

        if mode in ("all", "statistics"):
            self.calculate_statistics()
            if statistics_to_xls or reset is not None:
                self.write_statistics()
        if mode in ("all", "correlations"):
            self.calculate_correlations_and_scores()
        if mode in ("all", "categories"):
            self.calculate_categories()

    def check_if_cache_exist(self, mode: str):

        cache_exists = True
        if mode in ("all", "statistics"):
            cache_exists = self.cache_file.exists()
        if mode in ("all", "correlations"):
            cache_exists = cache_exists and self.corr_pkl_file.exists()
            cache_exists = cache_exists and self.score_pkl_file.exists()
        if mode in ("all", "categories"):
            cache_exists = cache_exists and self.cate_pkl_file.exists()

        return cache_exists

    def variable_dict2df(self, variables, module_info: dict = None) -> pd.DataFrame:
        """
        Converteer de directory met variable info naar een data frame
        Args:
            variables:  dict met variable info
            module_info: dict met module informatie

        Returns:
            dataframe
        """
        var_df = pd.DataFrame.from_dict(variables).unstack().dropna()
        var_df = var_df.reset_index()
        var_df = var_df.rename(columns={"level_0": self.module_key, "level_1": self.variable_key,
                                        0: "properties"})
        var_df.set_index(self.variable_key, drop=True, inplace=True)

        var_df = impose_variable_defaults(var_df, module_info=module_info,
                                          module_key=self.module_key)
        return var_df

    def write_statistics(self):
        _logger.info("Writing statistics")
        connection = sqlite3.connect(self.output_file)

        excel_file = Path(self.output_file).with_suffix(".xlsx")
        sheets = list()
        cnt = 0
        with pd.ExcelWriter(str(excel_file), engine="openpyxl") as writer:
            _logger.info(f"Start writing standard output to {excel_file}")

            for file_base, all_stats in self.all_stats_per_format.items():

                stat_df = prepare_stat_data_for_write(file_base=file_base,
                                                      all_stats=all_stats,
                                                      variables=self.variables,
                                                      variable_key=self.variable_key,
                                                      module_key=self.module_key,
                                                      breakdown_labels=self.breakdown_labels,
                                                      n_digits=self.n_digits,
                                                      connection=connection
                                                      )

                cache_file = make_plot_cache_file_name(cache_directory=self.cache_directory,
                                                       prefix=self.scan_data_key,
                                                       file_base=file_base)
                _logger.info(f"Writing cache for stat {cache_file}")
                with open(cache_file, "wb") as stream:
                    pickle.dump(stat_df, stream)

                sheet_name = file_base
                if self.sheet_renames is not None:
                    for rename_key, sheet_rename in self.sheet_renames.items():
                        pat = sheet_rename["pattern"]
                        rep = sheet_rename["replace"]
                        sheet_name = re.sub(pat, rep, sheet_name)
                if len(sheet_name) > 32:
                    sheet_name = sheet_name[:32]
                if sheet_name in sheets:
                    sheet_name = sheet_name[:30] + "{:02d}".format(cnt)
                cnt += 1
                sheets.append(sheets)
                stat_df.to_excel(writer, sheet_name)

    def calculate_statistics_one_breakdown(self, group_by):

        index_names = group_by + [self.be_id]
        dataframe = prepare_df_for_statistics(self.dataframe, index_names=index_names,
                                              units_key="units")

        all_stats = dict()
        all_hist = dict()

        for var_key, var_prop in self.variables.iterrows():
            _logger.debug(f"{var_key}")
            var_prop_klass = VariableProperties(variables=self.variables, column=var_key)

            column = var_key
            column_list = list([var_key])
            var_module = var_prop["module"]
            module = self.module_info[var_module]
            if not module.get("include", True):
                continue

            var_type = var_prop["type"]
            var_filter = var_prop["filter"]
            var_weight_key = var_prop["gewicht"]
            schaal_factor_key = "_".join(["ratio", var_weight_key])
            units_schaal_factor_key = "_".join(["ratio", "units"])
            weight_cols = set(
                list([var_weight_key, schaal_factor_key, units_schaal_factor_key]))
            df_weights = dataframe.loc[:, list(weight_cols)]

            try:
                data, column_list = get_records_select(dataframe=dataframe,
                                                       variables=self.variables,
                                                       var_type=var_type,
                                                       column=column,
                                                       column_list=column_list,
                                                       output_format="statline",
                                                       var_filter=var_filter
                                                       )
            except KeyError:
                _logger.warning(f"Failed to get selection of {column}. Skipping")
                continue

            stats = SampleStatistics(group_keys=group_by,
                                     records_df_selection=data,
                                     weights_df=df_weights,
                                     column_list=column_list,
                                     var_type=var_type,
                                     var_weight_key=var_weight_key,
                                     scaling_factor_key=schaal_factor_key,
                                     units_scaling_factor_key=units_schaal_factor_key)

            _logger.debug(f"Storing {stats.records_weighted_mean_agg}")
            all_stats[var_key] = stats.records_weighted_mean_agg
            all_hist[var_key] = dict()
            try:
                for grp_key, df in data.groupby(level=0, axis=0):
                    ww = df_weights.loc[grp_key, "ratio_units"].to_numpy()
                    dd = df.loc[grp_key, var_key].to_numpy()
                    try:
                        all_hist[var_key][grp_key] = np.histogram(dd, weights=ww,
                                                                  density=False,
                                                                  bins=self.n_bins,
                                                                  range=(0, 100))
                    except ValueError as err:
                        _logger.warning("Fails for dicts. Skip for now")
                        all_hist[var_key][grp_key] = None
                    else:
                        _logger.debug(f"Success with {var_key}")
            except KeyError:
                pass

        return all_stats, all_hist

    def get_correct_categories_count(self):
        """ bekijk per record hoeveel categorieen goed zijn en geef terug als dataframe """

        col_sel = list()

        for cat_key, cat_prop in self.categories["index_categories"].items():
            variable = cat_prop["variable"]
            col_sel.append(variable)

        _logger.debug(f"make selection\n{col_sel}")
        data_df: pd.DataFrame = self.dataframe[col_sel]

        # alleen 1 wordt als succes beschouwd
        data_df = data_df == 1

        count = data_df.sum(axis=1)
        count = count.rename("count")

        return data_df, count

    def calculate_categories(self):
        if self.cate_pkl_file.exists() and self.reset is None:
            _logger.info(f"Cache {self.cate_pkl_file} and already exist. "
                         f"Skip calculation cateogires and go to plot")
            return
        if self.dataframe is None:
            msg = "For correlations you need the microdata. Run with --reset 1"
            raise ValueError(msg)

        _logger.info("Calculating cateogires")

        score_df = self.dataframe["percentage"].copy()
        score_df = score_df.rename("score")
        weights = self.dataframe[self.weight_key].copy()
        data_df, count = self.get_correct_categories_count()

        tot = pd.concat([score_df, count], axis=1)

        conditional_scores = list()
        sum_per_number_of_cat = list()

        total_sum = 0
        mask_tot: pd.Series = None
        for number_of_cat in range(0, 5):
            mask = tot['count'] == number_of_cat
            tot_cond = tot.loc[mask, "score"]
            sel_df = data_df[mask]
            sum_per_number_of_cat.append(sel_df.sum(axis=0))
            ww = weights[mask].to_numpy()
            if mask_tot is None:
                mask_tot = mask
            else:
                mask_tot = mask_tot | mask
            hist, bin_edge = np.histogram(tot_cond.to_numpy(),
                                          weights=ww,
                                          density=False,
                                          range=(0, 100),
                                          bins=self.n_bins)
            hist_sum = hist.sum()
            total_sum += hist_sum
            conditional_scores.append(hist)

        sum_per_number_of_cat_df = pd.DataFrame.from_records(sum_per_number_of_cat)
        bin_width = bin_edge[1] - bin_edge[0]

        conditional_scores_df = pd.DataFrame().from_records(conditional_scores)
        conditional_scores_df.index = conditional_scores_df.index.rename("n_categories")
        conditional_scores_df = conditional_scores_df.T
        conditional_scores_df.index = bin_edge[:-1]
        conditional_scores_df /= (total_sum * bin_width)

        check_sum = conditional_scores_df.sum().sum() * bin_width

        _logger.debug(f"sum {check_sum}")

        _logger.info(f"Writing to {self.cate_pkl_file}")
        conditional_scores_df.to_pickle(self.cate_pkl_file)

        sum_file = self.cate_pkl_file.parent / Path(self.cate_pkl_file.stem + "_sum.pkl")
        _logger.info(f"Writing to {sum_file}")
        sum_per_number_of_cat_df.to_pickle(sum_file)

    def calculate_correlations_and_scores(self):

        if self.corr_pkl_file.exists() and self.score_pkl_file.exists() and self.reset is None:
            _logger.info(f"Cache {self.corr_pkl_file} and {self.score_pkl_file} already exist. "
                         f"Skip calculation and go to plot")
            return

        if self.dataframe is None:
            msg = "For correlations you need the microdata. Run with --reset 1"
            raise ValueError(msg)

        data_df_count, count = self.get_correct_categories_count()

        index_columns = self.correlations["index_correlations"]

        _logger.info("Calculating correlations")
        col_sel = list(index_columns.keys())

        _logger.debug(f"make selection\n{col_sel}")
        data_df: pd.DataFrame = self.dataframe[col_sel]

        # alleen 1 wordt als succes beschouwd
        data_df = data_df == 1

        # verkrijg de categorieën van variabele met hoge correlatie
        categories = dict()
        for col_name, categorie in index_columns.items():
            try:
                categories[categorie].append(col_name)
            except KeyError:
                categories[categorie] = [col_name]

        # bereken de score per category en vergelijk met de internet.nl-score
        self.score_df = self.dataframe[["percentage"]].copy() / 100
        self.score_df.rename(columns={"percentage": "score"}, inplace=True)
        for categorie, columns in categories.items():
            selection = data_df[columns]
            max_score = len(columns)
            self.score_df[categorie] = selection.sum(axis=1) / max_score

        self.score_df = pd.concat([self.score_df, count], axis=1)

        desc = data_df.describe()
        _logger.debug(f"making descr\n{desc}")
        # reken correlatie twee keer uit
        corr = data_df.corr()
        ordered_index = corr.sum().sort_values(ascending=False).index
        data_df = data_df[ordered_index]
        corr = data_df.corr()
        self.correlation_coefficient_df = corr

        _logger.info(f"Schrijf naar {self.corr_outfile}")
        with sqlite3.connect(str(self.corr_outfile)) as connection:
            corr.to_sql(name="correlations", con=connection, if_exists="replace")
        _logger.info(f"Schrijf naar {self.corr_pkl_file}")
        corr.to_pickle(self.corr_pkl_file.as_posix())

        _logger.info(f"Schrijf naar {self.score_pkl_file}")
        self.score_df.to_pickle(self.score_pkl_file.as_posix())
        _logger.debug(f"making corrected\n{corr}")

    def calculate_statistics(self):
        _logger.info("Calculating statistics")

        self.all_stats_per_format = dict()
        self.all_hist_per_format = dict()

        for file_base, props in self.statistics.items():
            scan_data = props.get("scan_data", self.default_scan)
            if scan_data != self.scan_data_key:
                _logger.debug(f"SKipping {scan_data} for {self.scan_data_key}")
                continue

            if not props.get('do_it', True):
                _logger.debug(f"SKipping breakdown {file_base} for {self.scan_data_key}")
                continue

            _logger.info(f"Processing {file_base}")

            file_name = Path("_".join([file_base, self.scan_data_key]) + ".pkl")
            cache_file = self.cache_directory / file_name

            if cache_file.exists() and self.reset is None:
                _logger.info(f"Reading stats from cache {cache_file}")
                with open(str(cache_file), "rb") as stream:
                    all_stats, all_hist = pickle.load(stream)
            else:
                group_by = list(props["groupby"].values())
                all_stats, all_hist = self.calculate_statistics_one_breakdown(group_by=group_by)
                _logger.info(f"Writing stats to cache {cache_file}")
                with open(str(cache_file), "wb") as stream:
                    pickle.dump([all_stats, all_hist], stream)

            stat_df = pd.concat(list(all_stats.values()), axis=1, sort=False)
            self.all_stats_per_format[file_base] = stat_df
            self.all_hist_per_format[file_base] = all_hist
            _logger.debug("Done with statistics")

    def write_data(self):
        """ write the combined data frame to sqlite lite """

        count_per_lower_col = Counter([col.lower() for col in self.dataframe.columns])
        for col_lower, multiplicity in count_per_lower_col.items():
            if multiplicity > 1:
                for col in self.dataframe.columns:
                    if col.lower() == col_lower:
                        _logger.info(f"Dropping duplicated column {col}")
                        self.dataframe.drop([col], axis=1, inplace=True)
                        break

        output_file_name = self.cache_file.with_suffix(".sqlite")
        _logger.info(f"Writing dataframe to {output_file_name}")
        with sqlite3.connect(str(output_file_name)) as connection:
            self.dataframe.to_sql(name="dataframe", con=connection, if_exists="replace")

    def read_data(self):
        if not self.cache_file.exists() or self.reset == 0:

            index_name = self.be_id
            _logger.info(f"Reading table data from {self.records_filename}")
            records = read_tables_from_sqlite(self.records_filename, self.records_tables_names,
                                              index_name)

            table_names = ["report", "scoring", "status", "results"]
            index_name = "index"
            _logger.info(f"Reading tables {table_names} from {self.internet_nl_filename}")
            tables = read_tables_from_sqlite(self.internet_nl_filename, table_names, index_name)
            _logger.info(f"Done")
            tables.reset_index(inplace=True)
            tables.rename(columns=dict(index=self.url_key), inplace=True)

            if self.translations is not None:
                tables = fill_booleans(tables, self.translations, variables=self.variables)

            rename_all_variables(tables, self.variables)

            for column in tables:
                try:
                    var_props = self.variables.loc[column, :]
                except KeyError:
                    _logger.debug("Column {} not yet defined. Skipping")
                    continue

                var_type = var_props.get("type")
                var_translate = var_props.get("translateopts")

                if var_translate is not None:
                    # op deze manier kunnen we de vertaling {Nee: 0, Ja: 1} op de column waardes los
                    # laten, zodat we alle Nee met 0 en Ja met 1 vervangen
                    trans = yaml.load(str(var_translate), Loader=yaml.Loader)
                    if set(trans.keys()).intersection(set(tables[column].unique())):
                        _logger.debug(f"Convert for {column} trans keys {trans}")
                        tables[column] = tables[column].map(trans)
                    else:
                        _logger.debug(f"No Convert for {column} trans keys {trans}")

                if var_type == "dict":
                    tables[column] = tables[column].astype('category')
                elif var_type in ("bool", "percentage", "float"):
                    tables[column] = tables[column].astype('float64')

            # hier gaan we de url name opschonen. sla eerst de oorsponkelijke url op
            original_url = "_".join([self.url_key, "original"])
            records[original_url] = records[self.url_key]
            tables[original_url] = tables[self.url_key]

            records[self.url_key] = [get_clean_url(url) for url in records[self.url_key]]
            records.dropna(subset=[self.url_key], axis=0, inplace=True)
            records.reset_index(inplace=True)
            tables[self.url_key] = [get_clean_url(url) for url in tables[self.url_key]]

            duplicated = tables[self.url_key].duplicated(keep='first')
            tables = tables[~duplicated]
            tables.dropna(subset=[self.url_key], axis=0, inplace=True)
            tables.dropna(how='all', axis=1, inplace=True)

            # doe een left join omdat meerdere be's dezelfde url kunnen hebben. Dit is sowieso
            # het geval voor holdings. Dan moeten we de score van hodlings ook meerdere keren
            # meenemen
            self.dataframe = pd.merge(left=records, right=tables, on=self.url_key, how='left')
            self.dataframe.dropna(subset=[self.weight_key], axis='index', how='any', inplace=True)
            try:
                has_url = self.dataframe['url'].notnull()
            except KeyError as err:
                _logger.warning(err)
            else:
                self.dataframe = self.dataframe[has_url]
            mask = self.dataframe[self.be_id].duplicated()
            self.dataframe = self.dataframe[~mask]
            self.dataframe.set_index(self.be_id, inplace=True, drop=True)
            _logger.info(f"Writing {self.dataframe.index.size} records to "
                         f"cache {self.cache_file.absolute()}")
            with open(str(self.cache_file), "wb") as stream:
                self.dataframe.to_pickle(stream)
            _logger.debug(self.dataframe.info())

        else:
            _logger.debug(f"Reading tables from cache {self.cache_file}")
            with open(str(self.cache_file), "rb") as stream:
                self.dataframe = pd.read_pickle(stream)
            _logger.info(f"Read {self.dataframe.index.size} records from "
                         f"cache {self.cache_file.absolute()}")


class DomainPlotter(object):
    def __init__(self, scan_data,
                 default_scan=None,
                 plot_info=None,
                 show_plots=False,
                 barh=False,
                 image_directory=None,
                 cache_directory=None,
                 image_type=".pdf",
                 max_plots=None,
                 tex_prepend_path=None,
                 statistics=None,
                 variables=None,
                 cdf_plot=False,
                 bar_plot=False,
                 cor_plot=False,
                 cumulative=False,
                 show_title=False,
                 breakdown_labels=None,
                 translations: dict = None,
                 export_highcharts=False,
                 highcharts_directory=None,
                 correlations=None,
                 tex_horizontal_shift=None,
                 bovenschrift=True,
                 ):

        self.scan_data = scan_data
        self.default_scan = default_scan
        self.plot_info = plot_info
        self.show_plots = show_plots
        self.barh = barh
        self.max_plots = max_plots
        self.tex_prepend_path = tex_prepend_path
        self.cache_directory = cache_directory
        self.statistics = statistics
        self.variables = variables
        self.bar_plot = bar_plot
        self.cdf_plot = cdf_plot
        self.cumulative = cumulative
        self.show_title = show_title
        self.translations = translations
        self.correlations = correlations
        self.export_highcharts = export_highcharts
        if highcharts_directory is None:
            self.highcharts_directory = Path(".")
        else:
            self.highcharts_directory = Path(highcharts_directory)

        self.image_type = image_type
        self.image_directory = image_directory
        self.breakdown_labels = breakdown_labels
        self.all_plots = dict()
        self.all_shifts = dict()

        self.image_files = Path("image_files.pkl")
        self.cache_image_file_list = self.cache_directory / self.image_files
        self.make_plots()
        with open(self.cache_image_file_list, "wb") as stream:
            pickle.dump(self.all_plots, stream)

        if self.all_plots is None:
            with open(self.cache_image_file_list, "rb") as stream:
                self.all_plots = pickle.load(stream)

        _logger.debug(f"making latex with bovenschrift={bovenschrift}")
        make_latex_overview(all_plots=self.all_plots,
                            variables=self.variables,
                            image_directory=self.image_directory, image_files=self.image_files,
                            tex_prepend_path=self.tex_prepend_path,
                            tex_horizontal_shift=tex_horizontal_shift,
                            all_shifts=self.all_shifts,
                            bovenschrift=bovenschrift
                            )

    #
    def get_plot_cache(self, scan_data_key, plot_key):
        cache_file = make_plot_cache_file_name(cache_directory=self.cache_directory,
                                               prefix=scan_data_key,
                                               file_base=plot_key)
        _logger.debug(f"Reading {cache_file}")
        try:
            with open(cache_file, "rb") as stream:
                stats_df = pickle.load(stream)
        except FileNotFoundError as err:
            _logger.warning(err)
            _logger.warning("Run script with option '--statistics_to_xls'  first")
            stats_df = None
        return stats_df

    def make_plots(self):
        _logger.info("Making the plot")

        self.all_plots = dict()

        for plot_key, plot_prop in self.plot_info.items():
            if not plot_prop.get("do_it", True):
                _logger.debug(f"Skipping plot {plot_key}")
                continue

            stat_prop = self.statistics[plot_key]
            scan_data_key = stat_prop.get("scan_data", self.default_scan)

            scan_data_analyses = self.scan_data[scan_data_key]["analyses"]
            variables = scan_data_analyses.variables
            module_info = scan_data_analyses.module_info

            stats_df = self.get_plot_cache(scan_data_key=scan_data_key, plot_key=plot_key)

            highcharts_title = plot_prop.get("title")
            export_svg_cdf = False
            export_svg_bar = False
            export_highcharts_cdf = self.export_highcharts
            export_highcharts_bar = self.export_highcharts
            highcharts_directory_cdf = None
            highcharts_directory_bar = None
            cdf_variables = {}
            if self.cdf_plot:
                plot_cdf = plot_prop.get("cdf_plot")
                if isinstance(plot_cdf, dict):
                    cdf_variables = plot_cdf.get("variables", {})
            tex_horizontal_shift = None
            if self.bar_plot:
                plot_bar = plot_prop.get("bar_plot")
                highcharts_directory_bar = self.highcharts_directory
                if isinstance(plot_bar, dict):
                    if hc_sub_dir := plot_bar.get("highcharts_output_directory"):
                        highcharts_directory_bar = highcharts_directory_bar / Path(hc_sub_dir)
                    export_svg_bar = plot_bar.get("export_svg", False)
                    export_hc_bar = plot_bar.get("export_highcharts")
                    tex_horizontal_shift = plot_bar.get("tex_horizontal_shift")
                    plot_bar = plot_bar.get("apply", True)
                    if export_hc_bar is not None:
                        export_highcharts_cdf = export_hc_bar
            else:
                plot_bar = False

            y_max_pdf_plot = plot_prop.get("y_max_pdf_plot", 10)
            y_spacing_pdf_plot = plot_prop.get("y_spacing_pdf_plot", 5)
            y_max_bar_plot = plot_prop.get("y_max_bar_plot")
            y_spacing_bar_plot = plot_prop.get("y_spacing_bar_plot")

            box_margin = plot_prop.get("box_margin")

            sort_values = plot_prop.get("sort_values", False)
            subplot_adjust = plot_prop.get("subplot_adjust")
            reference_lines = plot_prop.get("reference_lines")
            if reference_lines is not None:
                for ref_key, ref_prop in reference_lines.items():
                    stat_prop = self.statistics[ref_key]
                    scan_data_key = stat_prop.get("scan_data", self.default_scan)
                    ref_stat = self.get_plot_cache(scan_data_key=scan_data_key, plot_key=plot_key)
                    reference_lines[ref_key]["data"] = ref_stat

            label = plot_prop.get("label", plot_key)
            figsize = plot_prop.get("figsize")

            if plot_prop.get("use_breakdown_keys", False):
                breakdown = self.breakdown_labels[plot_key]
                renames = {v: k for k, v in breakdown.items()}
                stats_df.rename(columns=renames, inplace=True)

            _logger.info(f"Plotting {plot_key}")

            plot_count = 0
            stop_plotting = False
            for module_name, module_df in stats_df.groupby(level=0, sort=False):
                do_this_module = True
                for mod_key, mod_prop in module_info.items():
                    if mod_prop.get('label') == module_name and not mod_prop.get('include', True):
                        do_this_module = False
                if not do_this_module:
                    continue

                _logger.info(f"Module {module_name}")
                if stop_plotting:
                    break
                for question_name, question_df in module_df.groupby(level=1, sort=False):
                    _logger.debug(f"Question {question_name}")

                    original_name = re.sub(r"_\d\.0$", "", question_df["variable"].values[0])
                    question_type = variables.loc[original_name, "type"]

                    hc_info = HighchartsInfo(variables_df=variables,
                                             var_name=original_name,
                                             breakdown_name=plot_key)
                    export_highcharts = export_highcharts_bar
                    if cdf_prop := cdf_variables.get(original_name):
                        highcharts_directory_cdf = self.highcharts_directory
                        if hc_sub_dir := cdf_prop.get("highcharts_output_directory"):
                            highcharts_directory_cdf = highcharts_directory_cdf / Path(hc_sub_dir)
                        export_svg_cdf = cdf_prop.get("export_svg", False)
                        export_hc_cdf = cdf_prop.get("export_highcharts")
                        plot_cdf = cdf_prop.get("apply", True)
                        if export_hc_cdf is not None:
                            export_highcharts_cdf = export_hc_cdf
                    else:
                        plot_cdf = False
                    if hc_info.directory is not None:
                        # we overschrijven hier de subdir die onder de statistiek opgegeven is
                        highcharts_directory = self.highcharts_directory / hc_info.directory
                    else:
                        if plot_bar:
                            highcharts_directory = highcharts_directory_bar
                        else:
                            highcharts_directory = highcharts_directory_cdf
                    if hc_info.label is not None:
                        title = hc_info.label
                    else:
                        title = highcharts_title

                    title = re.sub("\s{2,}", " ", title)

                    if hc_info.y_max is not None:
                        y_max = hc_info.y_max
                    else:
                        y_max = y_max_bar_plot

                    if hc_info.y_spacing is not None:
                        y_spacing = hc_info.y_spacing
                    else:
                        y_spacing = y_spacing_bar_plot

                    if original_name not in self.all_plots.keys():
                        _logger.debug(f"Initialize dict for {original_name}")
                        self.all_plots[original_name] = dict()
                        self.all_shifts[original_name] = dict()

                    mask = get_option_mask(question_df=question_df, variables=variables,
                                           question_type=question_type)

                    plot_df = question_df.loc[(module_name, question_name, mask)].copy()

                    if self.translations is not None:
                        plot_df.rename(columns=self.translations, inplace=True)

                    xoff = 0
                    yoff = 0

                    if reference_lines is not None:
                        for ref_key, ref_prop in reference_lines.items():
                            ref_stat_df = reference_lines[ref_key]["data"]
                            ref_quest_df = None
                            for ref_quest_name, ref_quest_df in ref_stat_df.groupby(level=1):
                                if ref_quest_name == question_name:
                                    break
                            if ref_quest_df is not None:
                                mask2 = get_option_mask(question_df=ref_quest_df,
                                                        variables=variables,
                                                        question_type=question_type)
                                ref_df = ref_quest_df.loc[
                                    (module_name, question_name, mask2)].copy()
                                reference_lines[ref_key]["plot_df"] = ref_df

                    _logger.info(f"Plot nr {plot_count}")
                    if plot_bar:
                        image_file = make_bar_plot(plot_df=plot_df,
                                                   plot_key=plot_key,
                                                   module_name=module_name,
                                                   question_name=question_name,
                                                   image_directory=self.image_directory,
                                                   show_plots=self.show_plots,
                                                   figsize=figsize,
                                                   image_type=self.image_type,
                                                   reference_lines=reference_lines,
                                                   xoff=xoff, yoff=yoff,
                                                   show_title=self.show_title,
                                                   barh=self.barh,
                                                   subplot_adjust=subplot_adjust,
                                                   box_margin=box_margin,
                                                   sort_values=sort_values,
                                                   y_max_bar_plot=y_max,
                                                   y_spacing_bar_plot=y_spacing,
                                                   translations=self.translations,
                                                   export_highcharts=export_highcharts,
                                                   export_svg=export_svg_bar,
                                                   highcharts_directory=highcharts_directory,
                                                   title=title
                                                   )

                        _logger.debug(f"Store [{original_name}][{label}] : {image_file}")
                        self.all_plots[original_name][label] = image_file
                        self.all_shifts[original_name][label] = tex_horizontal_shift

                    if plot_cdf:
                        hist_info = scan_data_analyses.all_hist_per_format[plot_key][original_name]

                        if hist_info is not None:
                            for grp_key, hist in hist_info.items():
                                im_file_2 = make_cdf_plot(hist=hist,
                                                          plot_key=plot_key,
                                                          grp_key=grp_key,
                                                          module_name=module_name,
                                                          question_name=question_name,
                                                          image_file_base=original_name,
                                                          image_directory=self.image_directory,
                                                          show_plots=self.show_plots,
                                                          figsize=figsize,
                                                          image_type=self.image_type,
                                                          reference_lines=reference_lines,
                                                          cummulative=self.cumulative,
                                                          xoff=xoff, yoff=yoff,
                                                          y_max=y_max_pdf_plot,
                                                          y_spacing=y_spacing_pdf_plot,
                                                          translations=self.translations,
                                                          export_highcharts=export_highcharts_cdf,
                                                          export_svg=export_svg_cdf,
                                                          highcharts_directory=highcharts_directory_cdf,
                                                          title=title
                                                          )
                        if self.show_plots:
                            plt.show()

                    plot_count += 1
                    if self.max_plots is not None and plot_count == self.max_plots:
                        _logger.info(f"Maximum number of plot ({self.max_plots}) reached")
                        stop_plotting = True
                        break


class HighchartsInfo:
    def __init__(self, variables_df, var_name, breakdown_name):
        self.variables_df = variables_df
        self.var_name = var_name
        self.breakdown_name = breakdown_name

        self.label = None
        self.directory = None
        self.y_max = None
        self.y_spacing = None

        self.get_highcharts_info()

    def get_highcharts_info(self):
        """ in de variables dataframe  kunnen we ook uitdrukkelijk de highcharts directory en highcharts
        label opgeven per variabele. Zoek dat hier op """
        label = None
        directory = None
        try:
            var_prop = self.variables_df.loc[self.var_name]
        except KeyError:
            _logger.debug(f"could not find variable {self.var_name} in variables datafrrame")
        else:
            info_per_breakdown = var_prop["info_per_breakdown"]
            if info_per_breakdown is not None:
                try:
                    info = info_per_breakdown[self.breakdown_name]
                except KeyError:
                    _logger.debug(f"variable {self.var_name} does not have a breakdown defined")
                else:
                    self.directory = info.get("highcharts_directory")
                    if self.directory is not None:
                        self.directory = Path(self.directory)
                    self.label = info.get("highcharts_label")
                    self.y_max = info.get("y_max")
                    self.y_spacing = info.get("y_spacing")
