import pickle
import os
import logging
from pathlib import Path

from utils import read_tables_from_sqlite

_logger = logging.getLogger(__name__)


class DomainAnalyser(object):
    def __init__(self,
                 settings,
                 cache_file="tables_df.pkl", reset=False,
                 records_filename="records_cache.sqlite",
                 internet_nl_filename="internet_nl.sqlite"):

        _logger.info(f"Runing here {os.getcwd()}")
        self.settings = settings

        self.records_filename = records_filename
        self.internet_nl_filename = internet_nl_filename

        self.cache_file = Path(cache_file)
        self.reset = reset

        self.dataframe = None

        self.read_data()

    def read_data(self):
        if not self.cache_file.exists() or self.reset:

            table_names = ["records_df_2", "info_records_df"]
            index_name = "be_id"
            records = read_tables_from_sqlite(self.records_filename, table_names, index_name)

            table_names = ["report", "scoring", "status", "results"]
            index_name = "index"
            tables = read_tables_from_sqlite(self.internet_nl_filename, table_names, index_name)
            tables.reset_index(inplace=True)
            url_key = "website_url"
            tables.rename(columns=dict(index=url_key), inplace=True)

            self.dataframe = pd.merge(left=records, right=tables, on=url_key)
            _logger.info(f"Writing results to cache {self.cache_file}")
            with open(str(self.cache_file), "wb") as stream:
                pickle.dump(self.dataframe, stream)

        else:
            _logger.info(f"Reading tables from cache {self.cache_file}")
            with open(str(self.cache_file), "rb") as stream:
                self.dataframe = pickle.load(stream)
