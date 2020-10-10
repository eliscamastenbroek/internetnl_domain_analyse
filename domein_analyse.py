import codecs
import yaml
import argparse
import logging
import pickle
import sqlite3
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(format='%(asctime)s [%(lineno)4s] - %(levelname)-8s : %(message)s',
                    level=logging.DEBUG)
_logger = logging.getLogger()


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Analyse the domains")
    parser.add_argument("settings_filename", help="Settings file")
    parser.add_argument("--verbose", dest="loglevel", help="set loglevel to INFO",
                        action="store_const", const=logging.INFO, default=logging.INFO)
    parser.add_argument("--debug", dest="loglevel", help="set loglevel to DEBUG"
                        , action="store_const", const=logging.DEBUG)
    parser.add_argument("--working_directory", help="Directory relative to what we work")

    parsed_arguments = parser.parse_args(args)

    return parsed_arguments


def read_tables_from_sqlite(filename: str, table_names, index_name):
    if isinstance(table_names, str):
        table_names = [table_names]

    _logger.info(f"Reading from {filename}")
    connection = sqlite3.connect(filename)
    tables = list()
    for table_name in table_names:
        df = pd.read_sql(f"select * from {table_name}", con=connection, index_col=index_name)
        tables.append(df)
    if len(tables) > 1:
        tables_df = pd.concat(tables, axis=1)
    else:
        tables_df = tables[0]

    connection.close()

    return tables_df


class DomainAnalyser(object):
    def __init__(self,
                 settings,
                 cache_file="tables_df.pkl", reset=False,
                 records_filename="records_cache.sqlite",
                 internet_nl_filename="internet_nl.sqlite"):

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


def main(argv):
    args = parse_args(argv)

    _logger.setLevel(args.loglevel)

    _logger.info("Reading settings file {}".format(args.settings_filename))
    with codecs.open(args.settings_filename, "r", encoding="UTF-8") as stream:
        settings = yaml.load(stream=stream, Loader=yaml.Loader)

    if args.working_directory is None:
        working_directory = Path(".")
    else:
        working_directory = Path(args.working_directory)


    DomainAnalyser(settings=settings)


if __name__ == "__main__":
    main(sys.argv[1:])
