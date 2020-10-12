from tldextract import tldextract
import yaml
import logging
import os
import pickle
import sqlite3
from pathlib import Path

import pandas as pd

from utils import read_tables_from_sqlite, get_domain, fill_booleans

from ict_analyser.utils import (SampleStatistics,
                                prepare_df_for_statistics,
                                get_records_select, rename_all_variables,
                                impose_variable_defaults, VariableProperties)

_logger = logging.getLogger(__name__)


class DomainAnalyser(object):
    def __init__(self,
                 cache_file="tables_df.pkl",
                 cache_directory=None,
                 output_file=None,
                 reset=None,
                 records_filename="records_cache.sqlite",
                 internet_nl_filename="internet_nl.sqlite",
                 statistics: dict = None,
                 gk_data: dict = None,
                 variables: dict = None,
                 weights=None,
                 url_key="website_url",
                 translations=None,
                 module_key="module",
                 variable_key="variable"
                 ):

        _logger.info(f"Runing here {os.getcwd()}")

        if output_file is None:
            self.output_file = "output.sqlite"
        else:
            self.output_file = output_file

        self.statistics = statistics
        self.gk_data = gk_data
        self.module_key = module_key
        self.variable_key = variable_key
        self.variables = self.variable_dict2fd(variables)

        self.url_key = url_key
        self.be_id = "be_id"
        self.mi_labels = ["sbi", "gk_sbs", self.be_id]
        self.translations = translations

        self.records_filename = records_filename
        self.internet_nl_filename = internet_nl_filename

        self.cache_directory = cache_directory
        self.cache_file = self.cache_directory / Path(cache_file)
        if reset is None:
            self.reset = None
        else:
            self.reset = int(reset)
        self.weight_key = weights

        self.dataframe = None
        self.all_stats_per_format = dict()

        self.read_data()

        self.calculate_statistics()
        self.write_statistics()

    def variable_dict2fd(self, variables) -> pd.DataFrame:
        """
        Converteer de directory met variable info naar een data frame
        Args:
            variables:  dict met variable info

        Returns:
            dataframe
        """
        var_df = pd.DataFrame.from_dict(variables).unstack().dropna()
        var_df = var_df.reset_index()
        var_df = var_df.rename(columns={"level_0": self.module_key, "level_1": self.variable_key,
                                        0: "properties"})
        var_df.set_index(self.variable_key, drop=True, inplace=True)

        var_df = impose_variable_defaults(var_df)
        return var_df

    def write_statistics(self):
        _logger.info("Writing statistics")
        connection = sqlite3.connect(self.output_file)

        for file_base, all_stats in self.all_stats_per_format.items():
            data = pd.DataFrame.from_dict(all_stats)
            data.to_sql(name=file_base, con=connection, if_exists="replace")

    def calculate_statistics_one_breakdown(self, group_by):

        index_names = group_by + [self.be_id]
        dataframe = prepare_df_for_statistics(self.dataframe, index_names=index_names,
                                              units_key="units")

        all_stats = dict()

        for var_key, var_prop in self.variables.iterrows():
            _logger.info(f"{var_key}")
            var_prop_klass = VariableProperties(variables=self.variables, column=var_key)

            column = var_key
            column_list = list([var_key])

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
                                                       var_filter=var_filter,
                                                       var_gewicht_key=var_weight_key,
                                                       schaal_factor_key=schaal_factor_key)
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

        return all_stats

    def calculate_statistics(self):
        _logger.info("Calculating statistics")

        self.all_stats_per_format = dict()

        for file_base, props in self.statistics.items():
            _logger.info(f"Processing {file_base}")

            cache_file = self.cache_directory / Path(file_base + ".pkl")

            if cache_file.exists() and self.reset is None:
                _logger.info(f"Reading stats from cache {cache_file}")
                with open(str(cache_file), "rb") as stream:
                    all_stats = pickle.load(stream)
            else:
                group_by = list(props["groupby"].values())
                all_stats = self.calculate_statistics_one_breakdown(group_by=group_by)
                _logger.info(f"Writing stats to cache {cache_file}")
                with open(str(cache_file), "wb") as stream:
                    pickle.dump(all_stats, stream)

            stat_df = pd.concat(list(all_stats.values()), axis=1, sort=False)
            self.all_stats_per_format[file_base] = stat_df
            _logger.info("Done with statistics")

    def read_data(self):
        if not self.cache_file.exists() or self.reset == 0:

            table_names = ["records_df_2", "info_records_df"]
            index_name = self.be_id
            records = read_tables_from_sqlite(self.records_filename, table_names, index_name)

            records[self.url_key] = [get_domain(url) for url in records[self.url_key]]
            records.reset_index(inplace=True)

            table_names = ["report", "scoring", "status", "results"]
            index_name = "index"
            tables = read_tables_from_sqlite(self.internet_nl_filename, table_names, index_name)
            tables.reset_index(inplace=True)
            tables.rename(columns=dict(index=self.url_key), inplace=True)

            tables[self.url_key] = [get_domain(url) for url in tables[self.url_key]]

            if self.translations is not None:
                tables = fill_booleans(tables, self.translations)

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

            self.dataframe = pd.merge(left=records, right=tables, on=self.url_key)
            self.dataframe.dropna(subset=[self.weight_key], axis='index', how='any', inplace=True)
            mask = self.dataframe[self.be_id].duplicated()
            self.dataframe = self.dataframe[~mask]
            self.dataframe.set_index(self.be_id)
            _logger.info(f"Writing results to cache {self.cache_file}")
            with open(str(self.cache_file), "wb") as stream:
                pickle.dump(self.dataframe, stream)

        else:
            _logger.info(f"Reading tables from cache {self.cache_file}")
            with open(str(self.cache_file), "rb") as stream:
                self.dataframe = pickle.load(stream)
