import codecs
import logging
import os
import pickle
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import yaml

from weighted_sample_statistics import WeightedSampleStatistics, VariableProperties
from weighted_sample_statistics import (
    get_records_select,
    rename_all_variables,
    prepare_df_for_statistics,
)
from internetnl_domain_analyse.domain_plots import (
    make_cdf_plot,
    make_bar_plot,
    make_bar_plot_stacked,
)
from internetnl_domain_analyse.latex_output import make_latex_overview
from internetnl_domain_analyse.utils import (
    read_tables_from_sqlite,
    get_all_clean_urls,
    dump_data_frame_as_sqlite,
    add_derived_variables,
    fill_booleans,
    prepare_stat_data_for_write,
    get_option_mask,
    impose_variable_defaults,
    add_missing_groups,
    clean_all_suffix,
    get_windows_or_linux_value,
)

_logger = logging.getLogger(__name__)

tld_logger = logging.getLogger("tldextract")

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


def make_plot_cache_file_name(cache_directory, file_base, prefix):
    return cache_directory / Path("_".join([prefix, file_base, "cache_for_plot.pkl"]))


class ImageFileInfo:
    def __init__(
        self, scan_data_key, cache_file_name_base="image_info", cache_directory="cache"
    ):
        self.scan_data_key = scan_data_key
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(exist_ok=True)
        cache_file_name = Path(
            "_".join([cache_file_name_base, scan_data_key])
        ).with_suffix(".yml")
        self.cache_file_name = self.cache_directory / cache_file_name

        self.data = None

    def add_entry(
        self,
        plot_key,
        plot_info,
        image_key,
        sub_image_label,
        file_name,
        tex_right_shift=None,
        section=None,
    ):
        """add a new entry"""

        if image_key not in self.data.keys():
            self.data[image_key] = dict()
        self.data[image_key][plot_key] = dict(
            file_name=file_name,
            tex_right_shift=tex_right_shift,
            sub_image_label=sub_image_label,
        )
        if section:
            self.data[image_key][plot_key]["section"] = section

        # in order to get the key order in the dict the same as in the input file, alter the order
        if len(self.data[image_key].keys()) > 1:
            tmp_data = self.data[image_key].copy()
            self.data[image_key] = dict()
            for plot_key in plot_info.keys():
                try:
                    tex_prop = tmp_data[plot_key]
                except KeyError as err:
                    # this entry is not in the plot. No problem. skip it
                    pass
                else:
                    self.data[image_key][plot_key] = tex_prop

    def fix_order(self, variables):
        tmp_info = self.data.copy()
        self.data = dict()
        for var_name in variables.index:
            try:
                var_entry = tmp_info[var_name]
            except KeyError as err:
                _logger.debug(f"no entry for {var_name}. No problem, skipping")
            else:
                self.data[var_name] = var_entry

    def read_cache(self):
        """Lees de cache"""
        if self.cache_file_name.exists():
            with codecs.open(
                self.cache_file_name.as_posix(), "r", encoding="UTF-8"
            ) as stream:
                self.data = yaml.load(stream=stream, Loader=yaml.Loader)
        else:
            self.data = dict()

    def write_cache(self):
        """Schrijf de cache"""
        with codecs.open(
            self.cache_file_name.as_posix(), "w", encoding="UTF-8"
        ) as stream:
            yaml.dump(data=self.data, stream=stream, Dumper=yaml.Dumper)


class RecordsCacheInfo:
    def __init__(self, records_cache_data: dict, year: str, stat_directory: str = None):
        """
        Store the properties of the cache file in a class
        """
        self.records_cache_data = records_cache_data
        self.stat_directory = stat_directory

        self.year_digits = f"{year}"

        self.cache_dir = None
        self.file_name = None
        self.table_names = None

        self.get_cache_file_name()
        self.get_cache_table_names()

    def get_cache_file_name(self):
        """
        Retrieve the cache file name from the dictionary. If environment variables are given,
        base the directory on the environment name. Names are given like RECORDS_CACHE_DIR_20,
        RECORDS_CACHE_DIR_21, for 2020, 2021 resp.
        """

        records_environment_variable = "_".join(["RECORDS_CACHE_DIR", self.year_digits])
        records_cache_dir_name = os.getenv(records_environment_variable)

        if records_cache_dir_name is None:
            records_cache_dir_name = self.records_cache_data.get(
                "records_cache_directory", "."
            )

        records_cache_dir_name = records_cache_dir_name.replace(
            "{{ stat_directory }}", self.stat_directory
        )

        self.cache_dir = Path(records_cache_dir_name)

        records_file_basename = Path(self.records_cache_data["records_cache_file"])

        self.file_name = self.cache_dir / records_file_basename

    def get_cache_table_names(self):
        """
        Get the table names of the cache files.
        """
        self.table_names = self.records_cache_data.get("records_table_names")
        if self.table_names is None:
            self.table_names = [
                f"records_df_{self.year_digits}_2",
                f"info_records_df_{self.year_digits}",
            ]


class DomainAnalyser:
    def __init__(
        self,
        scan_data_key=None,
        cache_file_base="tables_df",
        cache_directory_base_name=None,
        tld_extract_cache_directory=None,
        output_file=None,
        reset=None,
        records_cache_info: RecordsCacheInfo = None,
        internet_nl_filename=None,
        breakdown_labels=None,
        statistics: dict = None,
        default_scan=None,
        variables: dict = None,
        module_info: dict = None,
        weights=None,
        url_key="website_url",
        suffix_key="suffix",
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
        dump_cache_as_sqlite=False,
    ):

        _logger.info(f"Running here {os.getcwd()}")

        self.records_cache_info = records_cache_info

        if output_file is None:
            self.output_file = Path("output.sqlite")
        else:
            self.output_file = Path(output_file)
        self.output_directory = Path("output")
        self.output_directory.mkdir(exist_ok=True)
        self.output_file = self.output_directory / self.output_file

        outfile_suff = self.output_file.suffixes
        outfile_base = self.output_file.with_suffix("").with_suffix("").as_posix()
        outfile_year = Path(
            "_".join([outfile_base, scan_data_key, self.records_cache_info.year_digits])
        )
        self.output_file = outfile_year.with_suffix(".".join(outfile_suff))
        self.dump_cache_as_sqlite = dump_cache_as_sqlite

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
        self.suffix_key = suffix_key
        self.be_id = "be_id"
        self.mi_labels = ["sbi", "gk_sbs", self.be_id]
        self.translations = translations

        self.categories_coefficient_df = None
        self.correlation_coefficient_df = None

        if internet_nl_filename is not None:
            self.internet_nl_filename = internet_nl_filename
        else:
            self.internet_nl_filename = Path("internet_nl.sqlite")

        self.cache_directory = Path(
            "_".join([cache_directory_base_name, self.records_cache_info.year_digits])
        )
        self.cache_directory.mkdir(exist_ok=True)
        if tld_extract_cache_directory is None:
            self.tld_extract_cache_directory = "tld_cache"
        else:
            self.tld_extract_cache_directory = tld_extract_cache_directory
        cache_file_base = Path(
            "_".join(
                [cache_file_base, self.records_cache_info.year_digits, scan_data_key]
            )
            + ".pkl"
        )
        self.cache_file = self.cache_directory / cache_file_base
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
                self.categories["categories_output_file"]
            )
        except TypeError:
            _logger.debug("categories not defined")
        else:
            self.cate_pkl_file = self.cate_outfile.with_suffix(".pkl")
        try:
            self.corr_outfile = self.cache_directory / Path(
                plot_info["correlation"]["output_file"]
            )
        except TypeError:
            _logger.debug("correlations not defined")
        else:
            self.corr_pkl_file = self.corr_outfile.with_suffix(".pkl")
            self.score_outfile = self.cache_directory / Path(
                plot_info["scores_per_interval"]["output_file"]
            )
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
                msg = (
                    "The write_data_frame option only works if you force to read the micro "
                    "data with --reset 0 or --reset 1"
                )
                _logger.warning(msg)

        if mode in ("all", "statistics"):
            self.calculate_statistics()
            if statistics_to_xls or reset is not None or not have_cache:
                self.write_statistics()
        if mode in ("all", "correlations") and self.dataframe is not None:
            self.calculate_correlations_and_scores()
        if mode in ("all", "categories") and self.dataframe is not None:
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

    def variable_dict2df(self, variables, module_info: dict = None) -> DataFrame:
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
        var_df = var_df.rename(
            columns={
                "level_0": self.module_key,
                "level_1": self.variable_key,
                0: "properties",
            }
        )
        var_df.set_index(self.variable_key, drop=True, inplace=True)

        var_df = impose_variable_defaults(
            var_df, module_info=module_info, module_key=self.module_key
        )
        return var_df

    def write_statistics(self):
        _logger.info(f"Writing statistics {self.output_file}")
        connection = sqlite3.connect(self.output_file)

        excel_file = Path(self.output_file).with_suffix(".xlsx")
        sheets = list()
        cnt = 0
        with pd.ExcelWriter(str(excel_file), engine="openpyxl") as writer:
            _logger.info(f"Start writing standard output to {excel_file}")

            for file_base, all_stats in self.all_stats_per_format.items():

                stat_df = prepare_stat_data_for_write(
                    file_base=file_base,
                    all_stats=all_stats,
                    variables=self.variables,
                    variable_key=self.variable_key,
                    module_key=self.module_key,
                    breakdown_labels=self.breakdown_labels,
                    n_digits=self.n_digits,
                    connection=connection,
                )

                cache_file = make_plot_cache_file_name(
                    cache_directory=self.cache_directory,
                    prefix=self.scan_data_key,
                    file_base=file_base,
                )
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
                stat_df.to_excel(excel_writer=writer, sheet_name=sheet_name)

    def calculate_statistics_one_breakdown(self, group_by):

        index_names = group_by + [self.be_id]
        try:
            dataframe = prepare_df_for_statistics(
                self.dataframe, index_names=index_names, units_key="units"
            )
        except KeyError:
            _logger.info(f"Breakdown on {index_names} does not exist")
            return None, None

        all_stats = dict()
        all_hist = dict()

        for var_key, var_prop in self.variables.iterrows():
            _logger.debug(f"{var_key}")
            var_prop_klass = VariableProperties(
                variables=self.variables, column=var_key
            )

            column = var_key
            column_list = list([var_key])
            var_module = var_prop["module"]
            try:
                module = self.module_info[var_module]
            except KeyError as err:
                _logger.warning(err)
                continue
            if not module.get("include", True):
                continue

            var_type = var_prop["type"]
            var_filter = var_prop["filter"]
            var_weight_key = var_prop["gewicht"]
            schaal_factor_key = "_".join(["ratio", var_weight_key])
            units_schaal_factor_key = "_".join(["ratio", "units"])
            weight_cols = set(
                list([var_weight_key, schaal_factor_key, units_schaal_factor_key])
            )
            df_weights = dataframe.loc[:, list(weight_cols)]

            try:
                data, column_list = get_records_select(
                    dataframe=dataframe,
                    variables=self.variables,
                    var_type=var_type,
                    column=column,
                    column_list=column_list,
                    output_format="statline",
                    var_filter=var_filter,
                )
            except KeyError:
                _logger.info(f"Failed to get selection of {column}. Skipping")
                continue

            if data is None:
                _logger.info(f"Could not get data selection for {var_key}. Skipping")
                continue

            if data.index.size < df_weights.index.size:
                _logger.info(
                    f"we filtered data, reducing from {df_weights.index.size} to {data.index.size}"
                )
                df_weights = df_weights.reindex(data.index)

            stats = WeightedSampleStatistics(
                group_keys=group_by,
                records_df_selection=data,
                weights_df=df_weights,
                column_list=column_list,
                var_type=var_type,
                var_weight_key=var_weight_key,
                scaling_factor_key=schaal_factor_key,
                units_scaling_factor_key=units_schaal_factor_key,
                report_numbers=var_prop_klass.report_number,
            )

            stats.calculate()

            if (
                not np.isnan(var_prop_klass.report_number)
                and var_prop_klass.report_number
            ):
                all_stats[column] = stats.records_sum
            else:
                all_stats[column] = stats.records_weighted_mean_agg

            # voeg hier het histogram van de data toe
            all_hist[var_key] = calculate_histogram_per_breakdown(
                data, var_key=var_key, df_weights=df_weights, n_bins=self.n_bins
            )
        return all_stats, all_hist

    def get_correct_categories_count(self):
        """Bekijk per record hoeveel categorieën goed zijn en geef terug als dataframe"""

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
            _logger.info(
                f"Cache {self.cate_pkl_file} and already exist. "
                f"Skip calculation categories and go to plot"
            )
            return
        if self.dataframe is None:
            msg = "For correlations you need the microdata. Run with --reset 1"
            raise ValueError(msg)

        _logger.info("Calculating categories")

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
            mask = tot["count"] == number_of_cat
            tot_cond = tot.loc[mask, "score"]
            sel_df = data_df[mask]
            sum_per_number_of_cat.append(sel_df.sum(axis=0))
            ww = weights[mask].to_numpy()
            if mask_tot is None:
                mask_tot = mask
            else:
                mask_tot = mask_tot | mask
            hist, bin_edge = np.histogram(
                tot_cond.to_numpy(),
                weights=ww,
                density=False,
                range=(0, 100),
                bins=self.n_bins,
            )
            hist_sum = hist.sum()
            total_sum += hist_sum
            conditional_scores.append(hist)

        sum_per_number_of_cat_df = pd.DataFrame.from_records(sum_per_number_of_cat)
        bin_width = bin_edge[1] - bin_edge[0]

        conditional_scores_df = pd.DataFrame().from_records(conditional_scores)
        conditional_scores_df.index = conditional_scores_df.index.rename("n_categories")
        conditional_scores_df = conditional_scores_df.T
        conditional_scores_df.index = bin_edge[:-1]
        conditional_scores_df /= total_sum * bin_width

        check_sum = conditional_scores_df.sum().sum() * bin_width

        _logger.debug(f"sum {check_sum}")

        _logger.info(f"Writing to {self.cate_pkl_file}")
        conditional_scores_df.to_pickle(self.cate_pkl_file)

        sum_file = self.cate_pkl_file.parent / Path(
            self.cate_pkl_file.stem + "_sum.pkl"
        )
        _logger.info(f"Writing to {sum_file}")
        sum_per_number_of_cat_df.to_pickle(sum_file)

    def calculate_correlations_and_scores(self):

        if (
            self.corr_pkl_file.exists()
            and self.score_pkl_file.exists()
            and self.reset is None
        ):
            _logger.info(
                f"Cache {self.corr_pkl_file} and {self.score_pkl_file} already exist. "
                f"Skip calculation and go to plot"
            )
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
        missing_groups = None

        for file_base, props in self.statistics.items():
            scan_data = props.get("scan_data", self.scan_data_key)
            if scan_data != self.scan_data_key:
                _logger.debug(f"SKipping {scan_data} for {self.scan_data_key}")
                continue

            if not props.get("do_it", True):
                _logger.debug(
                    f"SKipping breakdown {file_base} for {self.scan_data_key}"
                )
                continue

            _logger.info(f"Processing {file_base}")

            file_name = Path("_".join([file_base, self.scan_data_key]) + ".pkl")
            cache_file = self.cache_directory / file_name

            group_by = list(props["groupby"].values())
            group_by_original = None
            if (
                group_by_if_not_exist := props.get("groupby_if_not_exist")
            ) and self.dataframe is not None:
                have_missing_groups = False
                for group in group_by:
                    if group not in self.dataframe.columns:
                        have_missing_groups = True
                if have_missing_groups:
                    group_by_original = group_by
                    group_by = list(group_by_if_not_exist.values())
                    missing_groups = props.get("missing_groups")

            combination: list = props.get("combination")

            if combination is None:
                if cache_file.exists() and self.reset is None:
                    _logger.info(f"Reading stats from cache {cache_file}")
                    with open(str(cache_file), "rb") as stream:
                        stat_df, all_hist = pickle.load(stream)
                elif self.dataframe is not None:
                    _logger.info("Calculating statistics from micro data")
                    all_stats, all_hist = self.calculate_statistics_one_breakdown(
                        group_by=group_by
                    )
                    if group_by_original is not None:
                        all_stats = add_missing_groups(
                            all_stats, group_by, group_by_original, missing_groups
                        )

                    if all_stats is None:
                        _logger.info(
                            f"Could not calculate statistics for breakdown {group_by}. Skip"
                        )
                        continue

                    # maak er een pandas data frame van
                    stat_df = pd.concat(list(all_stats.values()), axis=1, sort=False)
                    is_nan = stat_df.index == "nan"
                    stat_df = stat_df.loc[~is_nan]

                    _logger.info(f"Writing stats to cache {cache_file}")
                    with open(str(cache_file), "wb") as stream:
                        pickle.dump([stat_df, all_hist], stream)
                else:
                    _logger.info(f"Statistics not available for {group_by}. Skipping")
                    continue

            else:
                stats = list()
                for file_com in combination:
                    try:
                        prev_stats = self.all_stats_per_format[file_com]
                    except KeyError:
                        raise KeyError(
                            f"Trying to add to combination {file_com}, but does not exists"
                        )
                    else:
                        stats.append(prev_stats)
                stat_df = pd.concat(stats, axis=0, sort=False)

            self.all_stats_per_format[file_base] = stat_df
            self.all_hist_per_format[file_base] = all_hist
            _logger.debug("Done with statistics")

    def write_data(self):
        """write the combined data frame to sqlite lite"""

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
            _logger.info(f"Reading table data from {self.records_cache_info.file_name}")
            records = read_tables_from_sqlite(
                self.records_cache_info.file_name,
                self.records_cache_info.table_names,
                index_name,
            )

            table_names = ["report", "scoring", "status", "results"]
            index_name = "index"
            _logger.info(
                f"Reading tables {table_names} from {self.internet_nl_filename}"
            )
            tables = read_tables_from_sqlite(
                self.internet_nl_filename, table_names, index_name
            )
            _logger.info(f"Done")
            tables.reset_index(inplace=True)
            tables.rename(columns=dict(index=self.url_key), inplace=True)

            # na are always set to 0
            # some urls did not give any results, drop the empty lines. note that
            # the first column is left out as that is the url name
            tables.dropna(axis=0, how="all", subset=tables.columns[1:], inplace=True)

            # split the tables in numerical and non-numerical part and fill with either NA (for
            # numerical) or with 'nan' for strings.
            original_columns = tables.columns
            tables_num = tables.select_dtypes(include="number")
            number_columns = tables_num.columns
            non_number_columns = [
                col for col in original_columns if col not in number_columns
            ]
            tables_non_num = tables[non_number_columns]

            # fill with NA for numerical values and 'nan' for non-numerical values
            tables_num = tables_num.fillna(pd.NA)
            tables_non_num = tables_non_num.fillna("nan")

            # put back to one data frame again and set order equal to orignal dataframe
            tables = pd.concat([tables_num, tables_non_num], axis=1)
            tables = tables[original_columns]

            self.translations["nans"] = dict(nan=0)

            if self.translations is not None:
                tables = fill_booleans(
                    tables,
                    translations=self.translations.copy(),
                    variables=self.variables,
                )

            rename_all_variables(tables, self.variables)

            for column in tables:
                try:
                    var_props = self.variables.loc[column, :]
                except KeyError:
                    _logger.debug(f"Column {column} not defined in settings. Skipping")
                    continue

                var_type = var_props.get("type")
                var_translate = var_props.get("translateopts")

                if var_translate is not None:
                    # op deze manier kunnen we de vertaling {Nee: 0, Ja: 1} op de column waardes los
                    # laten, zodat we alle Nee met 0 en Ja met 1 vervangen
                    trans = yaml.load(str(var_translate), Loader=yaml.Loader)
                    for nan_string in ("na", "nan", "NaN"):
                        if nan_string in trans.keys():
                            # we have added an 'na' option for the translations. Take care of it
                            is_na = tables[column].isna()
                            if is_na.any():
                                # should not happen anymore because of the dropna above
                                _logger.info(
                                    f"Filling {nan_string} with na in {column}"
                                )
                                tables[column] = tables[column].fillna(nan_string)

                    unique_values = set(tables[column].unique())
                    vals_to_translate = set(trans.keys()).intersection(unique_values)
                    missing_values = unique_values.difference(set(trans.keys()))
                    if vals_to_translate:
                        if missing_values:
                            _logger.warning(
                                f"Column {column} misses the translations for "
                                f"{missing_values}. Please update your settings"
                            )
                        _logger.debug(f"Convert for {column} trans keys {trans}")
                        tables[column] = tables[column].map(trans)
                    else:
                        _logger.debug(f"No Convert for {column} trans keys {trans}")

                if var_type == "dict":
                    tables[column] = tables[column].astype("category")
                elif var_type in ("bool", "percentage", "float"):
                    tables[column] = tables[column].astype("float64")

            # Hier gaan we de url name opschonen. Sla eerst de oorspronkelijke url op
            original_url = "_".join([self.url_key, "original"])
            records = pd.concat(
                [records, records[self.url_key].rename(original_url)], axis=1
            )
            tables = pd.concat(
                [tables, tables[self.url_key].rename(original_url)], axis=1
            )

            _logger.info("Start cleaning urls...")

            if _logger.getEffectiveLevel() > logging.DEBUG:
                show_progress = True
            else:
                show_progress = False
            clean_url_cache = self.cache_directory / Path("clean_url_cache.pkl")
            if clean_url_cache.exists():
                _logger.info(f"Reading clean urls from cache {clean_url_cache}")
                with open(clean_url_cache, "rb") as stream:
                    all_clean_urls, all_suffix = pickle.load(stream)
            else:
                all_clean_urls, all_suffix = get_all_clean_urls(
                    urls=records[self.url_key],
                    show_progress=show_progress,
                    cache_directory=self.tld_extract_cache_directory,
                )
                _logger.info(f"Writing clean urls to cache {clean_url_cache}")
                with open(clean_url_cache, "wb") as stream:
                    pickle.dump([all_clean_urls, all_suffix], stream)
            _logger.info("Done!")
            records[self.url_key] = all_clean_urls
            suffix_df = pd.DataFrame(
                index=records.index, data=all_suffix, columns=[self.suffix_key]
            )
            suffix_df_org = suffix_df.rename(
                columns={self.suffix_key: self.suffix_key + "_org"}
            )
            records = pd.concat([records, suffix_df, suffix_df_org], axis=1)
            records.dropna(subset=[self.url_key], axis=0, inplace=True)
            records.reset_index(inplace=True)

            duplicated = tables[self.url_key].duplicated(keep="first")
            tables = tables[~duplicated]
            tables.dropna(subset=[self.url_key], axis=0, inplace=True)
            tables.dropna(how="all", axis=1, inplace=True)

            # hier voegen we nog afgeleide kolommen to
            tables = add_derived_variables(tables, self.variables)

            # Doe een left join omdat meerdere be's dezelfde url kunnen hebben. Dit is sowieso
            # het geval voor holdings. Dan moeten we de score van holdings ook meerdere keren
            # meenemen
            self.dataframe = pd.merge(
                left=records, right=tables, on=self.url_key, how="left"
            )
            self.dataframe.dropna(
                subset=[self.weight_key], axis="index", how="any", inplace=True
            )
            try:
                has_url = self.dataframe["url"].notnull()
            except KeyError as err:
                _logger.warning(err)
            else:
                self.dataframe = self.dataframe[has_url]
            mask = self.dataframe[self.be_id].duplicated()
            self.dataframe = self.dataframe[~mask]
            self.dataframe.set_index(self.be_id, inplace=True, drop=True)
            self.dataframe = clean_all_suffix(
                dataframe=self.dataframe,
                suffix_key=self.suffix_key,
                variables=self.variables,
            )
            _logger.info(
                f"Writing {self.dataframe.index.size} records to "
                f"cache {self.cache_file.absolute()}"
            )
            with open(str(self.cache_file), "wb") as stream:
                self.dataframe.to_pickle(stream)

        else:
            _logger.debug(f"Reading tables from cache {self.cache_file}")
            with open(str(self.cache_file), "rb") as stream:
                self.dataframe = pd.read_pickle(stream)
            _logger.info(
                f"Read {self.dataframe.index.size} records from "
                f"cache {self.cache_file.absolute()}"
            )

        if self.dump_cache_as_sqlite:
            sqlite_cache = self.cache_file.with_suffix(".sqlite")
            dump_data_frame_as_sqlite(dataframe=self.dataframe, file_name=sqlite_cache)


class DomainPlotter:
    def __init__(
        self,
        scan_data,
        scan_data_key=None,
        default_scan=None,
        plot_info=None,
        show_plots=False,
        barh=False,
        image_directory=None,
        cache_directory=None,
        image_type="pdf",
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
        variables_to_plot=None,
        exclude_variables=None,
        force_plots=False,
        latex_files=False,
        years_to_add_to_plot_legend=None,
        module_info=None,
        english=False,
    ):

        self.english = english
        self.scan_data = scan_data
        self.scan_data_key = scan_data_key
        self.default_scan = default_scan
        self.plot_info = plot_info
        self.show_plots = show_plots
        self.barh = barh
        self.max_plots = max_plots
        self.tex_prepend_path = tex_prepend_path
        self.cache_directory = Path(cache_directory)
        self.statistics = statistics
        self.variables = variables
        self.bar_plot = bar_plot
        self.cdf_plot = cdf_plot
        self.cumulative = cumulative
        self.show_title = show_title
        self.translations = translations
        self.correlations = correlations
        self.export_highcharts = export_highcharts
        self.force_plots = force_plots
        self.years_to_add_to_plot_legend = years_to_add_to_plot_legend
        if highcharts_directory is None:
            self.highcharts_directory = Path(".")
        else:
            self.highcharts_directory = Path(highcharts_directory)
        self.variables_to_plot = variables_to_plot
        self.exclude_variables = exclude_variables

        self.image_type = image_type
        self.image_directory = image_directory
        self.breakdown_labels = breakdown_labels

        self.image_info = ImageFileInfo(
            scan_data_key=scan_data_key, cache_directory=self.cache_directory
        )
        self.image_info.read_cache()

        self.make_plots()

        self.image_info.fix_order(self.variables)
        self.image_info.write_cache()

        if latex_files:
            _logger.debug(f"making latex with bovenschrift={bovenschrift}")
            make_latex_overview(
                image_info=self.image_info,
                variables=self.variables,
                image_directory=self.image_directory,
                cache_directory=self.cache_directory,
                image_files=Path("image_files"),
                tex_prepend_path=self.tex_prepend_path,
                tex_horizontal_shift=tex_horizontal_shift,
                bovenschrift=bovenschrift,
                module_info=module_info,
            )

    #
    def get_plot_cache(self, scan_data_key, plot_key, year):
        last_two_digits = f"{year}"[-2:]
        cache_directory = "_".join([self.cache_directory.as_posix(), last_two_digits])
        cache_file = make_plot_cache_file_name(
            cache_directory=Path(cache_directory),
            prefix=scan_data_key,
            file_base=plot_key,
        )
        _logger.debug(f"Reading {cache_file}")
        try:
            with open(cache_file, "rb") as stream:
                stats_df_per_year = pickle.load(stream)
        except FileNotFoundError as err:
            if self.scan_data[scan_data_key][year].get("data_file") is None:
                _logger.debug("We are skipping this year as the data is not available.")
            else:
                # we missen de pkl file terwijl we wel een data file hebben. Genereer de foutmelding
                _logger.warning(err)
                _logger.warning("Run script with option '--statistics_to_xls'  first")
            stats_df_per_year = None
        return stats_df_per_year

    def make_plots(self):
        _logger.info("Making the plot")

        for plot_key, plot_prop in self.plot_info.items():
            if not plot_prop.get("do_it", True):
                _logger.debug(f"Skipping plot {plot_key}")
                continue

            _logger.debug(f"Plotting plot {plot_key}")
            label = plot_prop.get("label", plot_key)
            figsize = plot_prop.get("figsize")
            highcharts_height = plot_prop.get("highcharts_height")

            stat_prop = self.statistics[plot_key]
            scan_data_key = stat_prop.get("scan_data", self.scan_data_key)

            scan_data_per_year = self.scan_data[scan_data_key]
            last_year = list(scan_data_per_year.keys())[-1]
            scan_data_analyses = scan_data_per_year[last_year]["analyses"]
            variables = scan_data_analyses.variables
            try:
                report_number_empty = variables["report_number"].isna()
            except KeyError:
                report_number_empty = True
                variables["report_number"] = False
            else:
                variables.loc[report_number_empty, "report_number"] = False

            module_info = scan_data_analyses.module_info

            stats_df_per_year = {}
            last_year = None
            df_index_names = None
            for year in scan_data_per_year.keys():
                df = self.get_plot_cache(
                    scan_data_key=scan_data_key, plot_key=plot_key, year=year
                )
                if df is not None:
                    stats_df_per_year[year] = df
                    last_year = year
                    df_index_names = list(df.index.names)

            if not self.english:
                jaar_level_name = "Jaar"
            else:
                jaar_level_name = "Year"
            index_names = [jaar_level_name] + df_index_names
            new_index_names = df_index_names + [jaar_level_name]
            module_level_name = new_index_names[0]
            question_level_name = new_index_names[1]
            stats_df = pd.concat(stats_df_per_year, names=index_names)
            # zet module vraag optie jaar als volgorde.
            stats_df = stats_df.reorder_levels(new_index_names)

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
                    cdf_variables = plot_cdf["variables"][scan_data_key]
                    highcharts_title = cdf_variables.get("title")
                export_svg_cdf = False
                if plot_cdf:
                    if cdf_fig_size := plot_cdf.get("figsize"):
                        figsize = cdf_fig_size
            tex_horizontal_shift = None
            if self.bar_plot:
                plot_bar = plot_prop.get("bar_plot")
                highcharts_directory_bar = self.highcharts_directory
                if isinstance(plot_bar, dict):
                    if hc_sub_dir := plot_bar.get("highcharts_output_directory"):
                        highcharts_directory_bar = highcharts_directory_bar / Path(
                            hc_sub_dir
                        )
                    export_svg_bar = plot_bar.get("export_svg", False)
                    export_hc_bar = plot_bar.get("export_highcharts")
                    tex_horizontal_shift = get_windows_or_linux_value(
                        plot_bar.get("tex_horizontal_shift")
                    )
                    plot_bar = plot_bar.get("apply", True)
                    if export_hc_bar is not None:
                        export_highcharts_cdf = export_hc_bar
            else:
                plot_bar = False

            y_max_pdf_plot = plot_prop.get("y_max_pdf_plot", 10)
            y_spacing_pdf_plot = plot_prop.get("y_spacing_pdf_plot", 5)
            y_max_bar_plot = plot_prop.get("y_max_bar_plot")
            legend_position = plot_prop.get("legend_position")
            legend_max_columns = plot_prop.get("legend_max_columns")
            y_spacing_bar_plot = plot_prop.get("y_spacing_bar_plot")
            bar_width = plot_prop.get("bar_width")

            box_margin = plot_prop.get("box_margin")

            sort_values = plot_prop.get("sort_values", False)
            subplot_adjust = plot_prop.get("subplot_adjust")
            reference_lines = plot_prop.get("reference_lines")
            if reference_lines is not None:
                for ref_key, ref_prop in reference_lines.items():
                    stat_prop = self.statistics[ref_key]
                    scan_data_key = stat_prop.get("scan_data", self.default_scan)
                    ref_stat = self.get_plot_cache(
                        scan_data_key=scan_data_key, plot_key=plot_key, year=last_year
                    )
                    reference_lines[ref_key]["data"] = ref_stat

            if plot_prop.get("use_breakdown_keys", False):
                breakdown = self.breakdown_labels[plot_key]
                renames = {v: k for k, v in breakdown.items()}
                stats_df.rename(columns=renames, inplace=True)

            _logger.info(f"Plotting {plot_key}")

            plot_count = 0
            stop_plotting = False
            if stats_df is not None:

                for module_name, module_df in stats_df.groupby(
                    level=module_level_name, sort=False
                ):
                    do_this_module = True
                    for mod_key, mod_prop in module_info.items():
                        if mod_prop.get("label") == module_name and not mod_prop.get(
                            "include", True
                        ):
                            do_this_module = False
                    if not do_this_module:
                        continue

                    _logger.info(f"Module {module_name}")
                    if stop_plotting:
                        break
                    for question_name, question_df in module_df.groupby(
                        level=question_level_name, sort=False
                    ):
                        _logger.debug(f"Question {question_name}")

                        # voorlaatste kolom bevat de variabele namen
                        variable_name_key = question_df.index.names[-2]
                        plot_variable = question_df.index.get_level_values(
                            variable_name_key
                        ).values[0]
                        original_name = re.sub(r"_\d\.0$", "", plot_variable)
                        question_type = variables.loc[original_name, "type"]
                        unit = variables.loc[original_name, "unit"]
                        keep_options = variables.loc[original_name, "keep_options"]
                        section = variables.loc[original_name, "section"]
                        question_df_clean = question_df.droplevel(variable_name_key)

                        # variables_to_plot wordt als een list van een list in een tuple meegegeven
                        # dus ([[variable1], [variables2]). Haal eerst level 0 eruit om te tuple
                        # te verwijderen. Als variable_to_plot niet gegeven is dan is deze waarde
                        # None, en slaan we het over. Als hij wel gegeven is dan zetten we de list
                        # van lists om in een platte list
                        if self.variables_to_plot is not None:
                            var_to_plot_clean = [
                                vv[0] for vv in self.variables_to_plot if vv is not None
                            ]
                            if original_name not in var_to_plot_clean:
                                _logger.debug(
                                    f"{original_name} not in variables to plot {self.variables_to_plot}. "
                                    f"Skipping..."
                                )
                                continue

                        if self.exclude_variables is not None:
                            exclude_vars_clean = [
                                vv[0] for vv in self.exclude_variables if vv is not None
                            ]
                            if original_name in exclude_vars_clean:
                                _logger.debug(
                                    f"{original_name} in exclude variables {self.exclude_variables}. "
                                    f"Skipping..."
                                )
                                continue

                        plot_info = PlotInfo(
                            variables_df=variables,
                            var_name=original_name,
                            breakdown_name=plot_key,
                        )
                        export_highcharts = export_highcharts_bar
                        if cdf_prop := cdf_variables.get(original_name):
                            highcharts_directory_cdf = self.highcharts_directory
                            if highcharts_info_per_year := cdf_prop.get(
                                "highcharts_info_per_year"
                            ):
                                for (
                                    hc_year_key,
                                    hc_year_prop,
                                ) in highcharts_info_per_year.items():
                                    hc_dir = highcharts_directory_cdf / Path(
                                        hc_year_prop["highcharts_directory"]
                                    )
                                    hc_lab = hc_year_prop.get("highcharts_label")
                                    highcharts_info_per_year[hc_year_key] = dict(
                                        highcharts_directory=hc_dir,
                                        highcharts_label=hc_lab,
                                    )
                            export_svg_cdf = cdf_prop.get("export_svg", False)
                            export_hc_cdf = cdf_prop.get("export_highcharts")
                            plot_cdf = cdf_prop.get("apply", True)
                            if export_hc_cdf is not None:
                                export_highcharts_cdf = export_hc_cdf
                        else:
                            plot_cdf = False
                            highcharts_info_per_year = None
                        if plot_info.directory is not None:
                            # we overschrijven hier de subdir die onder de statistiek opgegeven is
                            highcharts_directory = (
                                self.highcharts_directory / plot_info.directory
                            )
                        else:
                            if plot_bar:
                                highcharts_directory = highcharts_directory_bar
                            else:
                                highcharts_directory = highcharts_directory_cdf
                        if plot_info.label is not None:
                            title = plot_info.label
                        else:
                            title = highcharts_title

                        if title is not None:
                            title = re.sub("\s{2,}", " ", title)

                        if plot_info.y_max is not None:
                            y_max = plot_info.y_max
                        else:
                            y_max = y_max_bar_plot

                        if plot_info.legend_position is not None:
                            legend_pos = plot_info.legend_position
                        else:
                            legend_pos = legend_position

                        if plot_info.y_spacing is not None:
                            y_spacing = plot_info.y_spacing
                        else:
                            y_spacing = y_spacing_bar_plot

                        if plot_info.bar_width is not None:
                            bar_width = plot_info.bar_width
                        else:
                            bar_width = bar_width

                        if keep_options:
                            # als keep options gegeven is dan houden we alle opties
                            valide_opties = variables.loc[
                                original_name, "options"
                            ].values()
                            mask = get_option_mask(
                                question_df=question_df_clean,
                                variables=variables,
                                question_type=question_type,
                                valid_options=valide_opties,
                            )
                            plot_df = module_df.loc[
                                (module_name, question_name, mask)
                            ].copy()

                        else:
                            # neem de default die we als true bestempelen
                            valide_opties = None
                            mask = get_option_mask(
                                question_df=question_df_clean,
                                variables=variables,
                                question_type=question_type,
                                valid_options=valide_opties,
                            )

                            plot_df = question_df_clean.loc[
                                (module_name, question_name, mask)
                            ].copy()

                        # dit is niet meer nodig omdat de kleuren toch gelijk blijven
                        # plot_df = add_missing_years(plot_df,
                        #                            years_to_plot=self.years_to_add_to_plot_legend,
                        #                            jaar_level_name=jaar_level_name,
                        #                            column=original_name)

                        if variables.loc[original_name, "report_number"]:
                            normalize_data = True
                        else:
                            normalize_data = False

                        if self.translations is not None:
                            plot_df.rename(columns=self.translations, inplace=True)

                        xoff = 0
                        yoff = 0

                        if reference_lines is not None:
                            for ref_key, ref_prop in reference_lines.items():
                                ref_stat_df = reference_lines[ref_key]["data"]
                                ref_quest_df = None
                                for ref_quest_name, ref_quest_df in ref_stat_df.groupby(
                                    level=question_level_name
                                ):
                                    if ref_quest_name == question_name:
                                        break
                                if ref_quest_df is not None:
                                    mask2 = get_option_mask(
                                        question_df=ref_quest_df,
                                        variables=variables,
                                        question_type=question_type,
                                    )
                                    try:
                                        ref_df = ref_quest_df.loc[
                                            (module_name, question_name, mask2)
                                        ].copy()
                                    except KeyError as err:
                                        _logger.warning(err)
                                    else:
                                        reference_lines[ref_key]["plot_df"] = ref_df

                        _logger.info(f"Plot nr {plot_count}")
                        if plot_bar:
                            if keep_options:
                                for year in scan_data_per_year.keys():
                                    this_year_df = plot_df.loc[
                                        slice(None),
                                        slice(None),
                                        slice(None),
                                        slice(None),
                                        year,
                                    ]
                                    image_file = make_bar_plot_stacked(
                                        year=year,
                                        plot_df=this_year_df,
                                        plot_key=plot_key,
                                        plot_variable=plot_variable,
                                        scan_data_key=scan_data_key,
                                        module_name=module_name,
                                        question_name=question_name,
                                        image_directory=self.image_directory,
                                        show_plots=self.show_plots,
                                        figsize=figsize,
                                        image_type=self.image_type,
                                        reference_lines=reference_lines,
                                        xoff=xoff,
                                        yoff=yoff,
                                        show_title=self.show_title,
                                        barh=self.barh,
                                        subplot_adjust=subplot_adjust,
                                        box_margin=box_margin,
                                        sort_values=sort_values,
                                        y_max_bar_plot=y_max,
                                        y_spacing_bar_plot=y_spacing,
                                        translations=self.variables.loc[
                                            original_name, "options"
                                        ],
                                        export_highcharts=export_highcharts,
                                        export_svg=export_svg_bar,
                                        highcharts_directory=highcharts_directory,
                                        title=title,
                                        legend_position=legend_pos,
                                        normalize_data=normalize_data,
                                        force_plot=self.force_plots,
                                        enable_highcharts_legend=plot_info.enable_highcharts_legend,
                                        unit=unit,
                                        english=self.english,
                                    )
                            else:
                                image_file = make_bar_plot(
                                    plot_df=plot_df,
                                    plot_key=plot_key,
                                    plot_variable=plot_variable,
                                    scan_data_key=scan_data_key,
                                    module_name=module_name,
                                    question_name=question_name,
                                    image_directory=self.image_directory,
                                    show_plots=self.show_plots,
                                    figsize=figsize,
                                    highcharts_height=highcharts_height,
                                    image_type=self.image_type,
                                    reference_lines=reference_lines,
                                    xoff=xoff,
                                    yoff=yoff,
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
                                    title=title,
                                    legend_position=legend_pos,
                                    legend_max_columns=legend_max_columns,
                                    normalize_data=normalize_data,
                                    force_plot=self.force_plots,
                                    enable_highcharts_legend=plot_info.enable_highcharts_legend,
                                    unit=unit,
                                    english=self.english,
                                    bar_width=bar_width,
                                )

                                _logger.debug(
                                    f"Store [{original_name}][{label}] : {image_file}"
                                )
                                self.image_info.add_entry(
                                    plot_key=plot_key,
                                    plot_info=self.plot_info,
                                    image_key=original_name,
                                    section=section,
                                    file_name=image_file,
                                    sub_image_label=label,
                                    tex_right_shift=tex_horizontal_shift,
                                )

                        if plot_cdf:
                            for year in scan_data_per_year.keys():
                                try:
                                    scan_data_analyses_year = scan_data_per_year[year][
                                        "analyses"
                                    ]
                                except KeyError:
                                    _logger.info(
                                        f"Year {year} does not have data. Skipping"
                                    )
                                    continue
                                hist_info = scan_data_analyses_year.all_hist_per_format[
                                    plot_key
                                ][original_name]
                                highcharts_info = highcharts_info_per_year[year]

                                if hist_info is not None and isinstance(
                                    hist_info, dict
                                ):
                                    for grp_key, hist in hist_info.items():
                                        if hist is None:
                                            _logger.warning(
                                                f"Hist {grp_key} does not have a histogram. Skipping"
                                            )
                                            continue
                                        im_file_2 = make_cdf_plot(
                                            hist=hist,
                                            plot_key=plot_key,
                                            scan_data_key=scan_data_key,
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
                                            xoff=xoff,
                                            yoff=yoff,
                                            y_max=y_max_pdf_plot,
                                            y_spacing=y_spacing_pdf_plot,
                                            translations=self.translations,
                                            export_highcharts=export_highcharts_cdf,
                                            export_svg=export_svg_cdf,
                                            highcharts_info=highcharts_info,
                                            title=title,
                                            year=year,
                                        )
                            if self.show_plots:
                                plt.show()

                        plot_count += 1
                        if self.max_plots is not None and plot_count == self.max_plots:
                            _logger.info(
                                f"Maximum number of plot ({self.max_plots}) reached"
                            )
                            stop_plotting = True
                            break


class PlotInfo:
    def __init__(self, variables_df, var_name, breakdown_name):
        self.variables_df = variables_df
        self.var_name = var_name
        self.breakdown_name = breakdown_name

        self.label = None
        self.directory = None
        self.y_max = None
        self.y_spacing = None
        self.legend_position = None
        self.enable_highcharts_legend = True
        self.bar_width = None

        self.get_plot_info()

    def get_plot_info(self):
        """In de variables dataframe kunnen we ook uitdrukkelijk de highcharts directory en highcharts
        label opgeven per variabele. Zoek dat hier op"""
        label = None
        directory = None
        try:
            var_prop = self.variables_df.loc[self.var_name]
        except KeyError:
            _logger.debug(
                f"could not find variable {self.var_name} in variables dataframe"
            )
        else:
            info_per_breakdown = var_prop["info_per_breakdown"]
            if info_per_breakdown is not None:
                try:
                    info = info_per_breakdown[self.breakdown_name]
                except KeyError:
                    _logger.debug(
                        f"variable {self.var_name} does not have a breakdown defined"
                    )
                else:
                    self.directory = info.get("highcharts_directory")
                    if self.directory is not None:
                        self.directory = Path(self.directory)
                    self.label = info.get("highcharts_label")
                    self.y_max = info.get("y_max")
                    self.y_spacing = info.get("y_spacing")
                    self.bar_width = info.get("bar_width")
                    self.legend_position = get_windows_or_linux_value(
                        info.get("legend_position")
                    )
                    self.enable_highcharts_legend = info.get(
                        "enable_highcharts_legend", True
                    )


def add_missing_years(plot_df, years_to_plot=None, jaar_level_name="Jaar", column=None):
    """
    Voeg missende jaren toe

    Args:
        plot_df: pd.DataFrame
            DataFrame om te plotetn
        years_to_plot: list
            De jaren die we willen plotten
        jaar_level_name: str
            De naam van de level= van de jaren
        column: str
            Naam van de column voor de foutmelding

    Returns:
        pd.DataFrame
    """

    years_in_plot = plot_df.index.get_level_values(jaar_level_name)
    missing_years = set(years_to_plot).difference(years_in_plot)
    if missing_years:
        index_names = plot_df.index.names
        df = plot_df.reset_index().set_index(jaar_level_name)
        try:
            df = df.reindex(years_to_plot)
        except ValueError as err:
            _logger.warning(f"{err}. Check  {column}")
        for column_name in index_names:
            if column_name == jaar_level_name:
                continue
            df[column_name] = df[column_name].pad()
        plot_df = df.reset_index().set_index(index_names, drop=True)

    return plot_df


def calculate_histogram_per_breakdown(
    data: DataFrame, var_key: str, df_weights: Series, n_bins: int = 100
) -> dict:
    """
    Bereken per breakdown van de data het histogram die hoort bij var_key

    Parameters
    ----------
    data: DataFrame
        De data met breakdown op de index
    var_key: str
        De naam van de kolom waarvoor we de histogram gaan berekenen
    df_weights: Series
        De weegfactoren die we voor de histogram gebruiken
    n_bins: int
        Aan binnen in het histogram

    Returns
    -------
    dict:
        De histogrammen per breakdown


    """

    histogram_per_breakdown = dict()

    for grp_key, df in data.groupby(level=0):
        # initieer histogram voor deze breakdown met None
        histogram_per_breakdown[grp_key] = None

        try:
            ww = df_weights.loc[grp_key, "ratio_units"].to_numpy()
        except KeyError:
            _logger.debug("Could not get weight factors. Skip for now")
            continue
        try:
            dd = df.loc[grp_key, var_key].to_numpy()
        except KeyError:
            _logger.debug(f"Could not get data belonging to {var_key}. Skip for now")
            continue

        try:
            histogram = np.histogram(
                dd,
                weights=ww,
                density=False,
                bins=n_bins,
                range=(0, 100),
            )
        except ValueError as err:
            _logger.debug("Fails for dicts. Skip for now")
        else:
            _logger.debug(f"Success with {var_key}")
            histogram_per_breakdown[grp_key] = histogram

    return histogram_per_breakdown
