from tldextract import tldextract
import logging
import os
import pickle
from pathlib import Path

import pandas as pd

from utils import read_tables_from_sqlite, get_domain

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

            weights_df = self.dataframe[[self.weight_key]]
            data_grp = self.dataframe.groupby(group_by)
            weight_grp = weights_df.groupby(group_by)

            weight_sum = weight_grp.transform('sum')
            weight_prime = weights_df.div(weight_sum, axis='index')

            for var_key, var_prop in self.variables.items():
                _logger.info(f"{var_key}")

                values = data[var_key]
                values_scaled = values.mul(weight_prime, axis='index')
                var_mean = values_scaled.sum()
                _logger.info(var_mean)

    def read_data(self):
        if not self.cache_file.exists() or self.reset:

            table_names = ["records_df_2", "info_records_df"]
            index_name = "be_id"
            records = read_tables_from_sqlite(self.records_filename, table_names, index_name)

            records[self.url_key] = [get_domain(url) for url in records[self.url_key]]

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

