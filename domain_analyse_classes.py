from tldextract import tldextract
import logging
import os
import pickle
from pathlib import Path

import pandas as pd

from utils import read_tables_from_sqlite, get_domain

from ict_analyser.utils import SampleStatistics, prepare_df_for_statistics, get_records_select

_logger = logging.getLogger(__name__)


class DomainAnalyser(object):
    def __init__(self,
                 cache_file="tables_df.pkl",
                 reset=False,
                 records_filename="records_cache.sqlite",
                 internet_nl_filename="internet_nl.sqlite",
                 statistics: dict = None,
                 gk_data: dict = None,
                 variables: dict = None,
                 weights=None,
                 url_key="website_url"
                 ):

        _logger.info(f"Runing here {os.getcwd()}")

        self.statistics = statistics
        self.gk_data = gk_data
        self.variables = variables

        self.url_key = url_key
        self.be_id = "be_id"
        self.mi_labels = ["sbi", "gk_sbs", self.be_id]

        self.records_filename = records_filename
        self.internet_nl_filename = internet_nl_filename

        self.cache_file = Path(cache_file)
        self.reset = reset
        self.weight_key = weights

        self.dataframe = None

        self.read_data()

        self.calculate_statistics()

    def calculate_statistics(self):
        _logger.info("Calculating statistics")

        for file_base, props in self.statistics.items():
            _logger.info(f"Processing {file_base}")

            group_by = list(props["groupby"].values())

            group_by.append(self.be_id)
            dataframe = prepare_df_for_statistics(self.dataframe, index_names=group_by,
                                                  units_key="units")

            all_stats = dict()

            for var_key, var_prop in self.variables.items():
                _logger.info(f"{var_key}")

                column = var_key
                column_list = list([var_key])

                var_type = var_prop.get("type", "bool")
                var_weight_key = var_prop.get("gewicht", "units")
                schaal_factor_key = "_".join(["ratio", var_weight_key])
                units_schaal_factor_key = "_".join(["ratio", "units"])
                weight_cols = set(
                    list([var_weight_key, schaal_factor_key, units_schaal_factor_key]))
                df_weights = dataframe.loc[:, list(weight_cols)]

                data, column_list = get_records_select(dataframe=dataframe, variables=None,
                                                       var_type=var_type, column=column,
                                                       column_list=column_list,
                                                       output_format="statline", var_filter=None,
                                                       var_gewicht_key=var_weight_key,
                                                       schaal_factor_key=schaal_factor_key)

                stats = SampleStatistics(group_keys=group_by,
                                         records_df_selection=data,
                                         weights_df=df_weights,
                                         column_list=column_list,
                                         var_type=var_type,
                                         var_weight_key=var_weight_key,
                                         scaling_factor_key=schaal_factor_key,
                                         units_scaling_factor_key=units_schaal_factor_key)

                _logger.debug(f"Storing {stats.records_weighted_mean_agg}")
                all_stats[column] = stats.records_weighted_mean_agg

            _logger.info("Done with statsistics")

    def read_data(self):
        if not self.cache_file.exists() or self.reset:

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
            self.dataframe = pd.merge(left=records, right=tables, on=self.url_key)
            self.dataframe.dropna(subset=[self.weight_key], axis='index', how='any', inplace=True)
            _logger.info(f"Writing results to cache {self.cache_file}")
            with open(str(self.cache_file), "wb") as stream:
                pickle.dump(self.dataframe, stream)

        else:
            _logger.info(f"Reading tables from cache {self.cache_file}")
            with open(str(self.cache_file), "rb") as stream:
                self.dataframe = pickle.load(stream)
