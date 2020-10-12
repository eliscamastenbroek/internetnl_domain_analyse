import numpy as np
from tldextract import tldextract
import logging
import pandas as pd
import sqlite3

_logger = logging.getLogger(__name__)


def read_tables_from_sqlite(filename: str, table_names, index_name) -> pd.DataFrame:
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


def get_domain(url):
    try:
        tld = tldextract.extract(url)
    except TypeError:
        domain = None
    else:
        domain = tld.domain.lower()
    return domain


def fill_booleans(tables, translations):
    for col in tables.columns:
        unique_values = tables[col].unique()
        if len(unique_values) <= 3:
            for trans_key, trans_prop in translations.items():
                bool_keys = set(trans_prop.keys())
                intersection = bool_keys.intersection(unique_values)
                if intersection:
                    nan_val = set(unique_values).difference(bool_keys)
                    if nan_val:
                        trans_prop[list(nan_val)[0]] = np.nan
                    for key, val in trans_prop.items():
                        mask = tables[col] == key
                        if any(mask):
                            tables.loc[mask, col] = float(val)
    return tables
