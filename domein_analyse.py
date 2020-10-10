import pickle
import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s [%(lineno)4s] - %(levelname)-8s : %(message)s',
                    level=logging.DEBUG)
_logger = logging.getLogger()

cache_file = Path("tables_df.pkl")
reset = True


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


def main():
    if not cache_file.exists() or reset:

        filename = "records_cache.sqlite"
        table_names = ["records_df_2", "info_records_df"]
        index_name = "be_id"
        records = read_tables_from_sqlite(filename, table_names, index_name)

        filename = "internet_nl.sqlite"
        table_names = ["report", "scoring", "status", "results"]
        index_name = "index"
        tables = read_tables_from_sqlite(filename, table_names, index_name)
        tables.reset_index(inplace=True)
        tables.rename(columns=dict(index="website_url"), inplace=True)

        result = records.join(tables, on="website_url", how="left")
        _logger.info(f"Writing results to cache {cache_file}")
        with open(str(cache_file), "wb") as stream:
            pickle.dump(result, stream)

    else:
        _logger.info(f"Reading tables from cache {cache_file}")
        with open(str(cache_file), "rb") as stream:
            result     = pickle.load(stream)

    result.head()


if __name__ == "__main__":
    main()
