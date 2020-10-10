import logging
import pandas as pd
import sqlite3

_logger = logging.getLogger(__name__)


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
