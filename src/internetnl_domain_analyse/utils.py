import logging
import yaml
import sqlite3
import ssl
import sys
from pathlib import Path

import pandas as pd
import requests
import tldextract
from tqdm import tqdm

from ict_analyser.analyser_tool.utils import (reorganise_stat_df)

_logger = logging.getLogger(__name__)
tld_logger = logging.getLogger("tldextract")
tld_logger.setLevel(logging.WARNING)


def read_tables_from_sqlite(filename: Path, table_names, index_name) -> pd.DataFrame:
    if isinstance(table_names, str):
        table_names = [table_names]

    if not filename.exists():
        _logger.warning("Records file not found. Make sure you set the environment variable "
                        "RECORDS_CACHE_DIR or pass it via the command line argument "
                        "--records_cache_dir")
        raise FileNotFoundError(f"Records file not found {filename.absolute()}")

    _logger.info(f"Reading from {filename}")
    connection = sqlite3.connect(filename.as_posix())
    tables = list()
    for table_name in table_names:
        _logger.debug(f"Reading table {table_name}")
        df = pd.read_sql(f"select * from {table_name}", con=connection, index_col=index_name)
        tables.append(df)
    _logger.debug(f"Done")
    if len(tables) > 1:
        tables_df = pd.concat(tables, axis=1)
    else:
        tables_df = tables[0]

    _logger.debug(f"Closing database")

    connection.close()

    _logger.debug(f"Done reading")
    return tables_df


def get_clean_url(url, cache_dir=None):
    clean_url = url
    suffix = None
    if cache_dir is not None:
        extract = tldextract.TLDExtract(cache_dir=cache_dir)
    else:
        extract = tldextract.tldextract.extract
    try:
        url = url.strip()
    except AttributeError:
        pass
    else:
        try:
            tld = extract(url)
        except TypeError:
            _logger.debug(f"Type error occurred for {url}")
        except ssl.SSLEOFError as ssl_err:
            _logger.debug(f"SSLEOF error occurred for {url}")
        except requests.exceptions.SSLError as req_err:
            _logger.debug(f"SSLError error occurred for {url}")
        else:
            if tld.subdomain == "" and tld.domain == "" and tld.suffix == "":
                clean_url = None
            elif tld.subdomain == "" and tld.suffix == "":
                clean_url = None
            elif tld.subdomain == "" and tld.domain == "":
                clean_url = None
            elif tld.domain == "" and tld.suffix == "":
                clean_url = None
            elif tld.subdomain == "":
                clean_url = ".".join([tld.domain, tld.suffix])
            elif tld.suffix == "":
                clean_url = ".".join([tld.subdomain, tld.domain])
            elif tld.domain == "":
                clean_url = ".".join([tld.subdomain, tld.suffix])
            else:
                clean_url = ".".join([tld.subdomain, tld.domain, tld.suffix])
            if clean_url is not None:
                if " " in clean_url:
                    _logger.debug(f"{clean_url} cannot be real url with space. skipping")
                    clean_url = None
                else:
                    # We hebben een url gevonden. Maak hem met kleine letters en sla de suffix op
                    clean_url = clean_url.lower()
                    suffix = tld.suffix.lower()

    return clean_url, suffix


def fill_booleans(tables, translations, variables):
    for col in tables.columns:
        convert_to_bool = True
        try:
            var_type = variables.loc[col, "type"]
        except KeyError:
            pass
        else:
            if var_type == "dict":
                convert_to_bool = False

        if convert_to_bool:
            unique_values = tables[col].unique()
            if len(unique_values) <= 3:
                for trans_key, trans_prop in translations.items():
                    bool_keys = set(trans_prop.keys())
                    intersection = bool_keys.intersection(unique_values)
                    if intersection:
                        nan_val = set(unique_values).difference(bool_keys)
                        # if nan_val:
                        #    trans_prop[list(nan_val)[0]] = np.nan
                        for key, val in trans_prop.items():
                            mask = tables[col] == key
                            if any(mask):
                                tables.loc[mask, col] = float(val)
    return tables


def prepare_stat_data_for_write(all_stats, file_base, variables, module_key, variable_key,
                                breakdown_labels=None, n_digits=3, connection=None):
    data = pd.DataFrame.from_dict(all_stats)
    if connection is not None:
        data.to_sql(name=file_base, con=connection, if_exists="replace")

    stat_df = reorganise_stat_df(records_stats=data,
                                 variables=variables,
                                 use_original_names=True,
                                 module_key=module_key,
                                 variable_key=variable_key,
                                 n_digits=n_digits,
                                 sort_index=False)
    index_names = list(stat_df.index.names)
    new_index_names = index_names + [stat_df.columns[0]]
    stat_df = stat_df.reset_index().set_index(new_index_names, drop=True)
    if breakdown_labels is not None:
        try:
            labels = breakdown_labels[file_base]
        except KeyError:
            _logger.info(f"No breakdown labels for {file_base}")
        else:
            stat_df.rename(columns=labels, inplace=True)

    return stat_df


def get_option_mask(question_df, variables, question_type, valid_options=None):
    """  get the mask to filter the positive options from a question """
    mask_total = None
    if valid_options is None:
        options = ("Passed", "Yes", "Good")
    else:
        options = valid_options
    if question_type == "dict":
        for optie in options:
            mask = question_df.index.get_level_values(2) == optie
            if mask_total is None:
                mask_total = mask
            else:
                mask_total = mask | mask_total
    else:
        mask_total = pd.Series(True, index=question_df.index)

    return mask_total


def impose_variable_defaults(variables,
                             module_info: dict = None,
                             module_key: str = None):
    """
    Impose default values to  the variables data frame

    Parameters
    ----------
    variables: pd.DataFrame
        Dataframe with the initial variables
    module_info: pd.DataFrame
        Dataframe with information per module
    module_key: str
        Key of the module in the dataframe

    Returns
    -------
    pd.DataFrame
        Filled dataframe

    """
    variables["type"] = "bool"
    variables["section"] = ""
    variables["fixed"] = False
    variables["original_name"] = variables.index.values
    variables["label"] = ""
    variables["question"] = ""
    variables["module_label"] = ""
    variables["module_include"] = True
    # if the check  flag is true , it indicates this is a check question which can be discarded
    # in the final output
    variables["check"] = False
    variables["optional"] = False
    variables["no_impute"] = False
    variables["info_per_breakdown"] = None

    variables["gewicht"] = "units"
    variables["keep_options"] = False
    # variables["filter"] = ""

    # als toevallig de eerste key: value in de options een dict is dan kan je geen from_dict
    # gebruiken. Daarom voegen we nu een dummy string to, die halen we dadelijk weer weg
    dummy = "dummy"
    options = {dummy: dummy}
    filter_dummy = {dummy: dummy}
    translate = {dummy: dummy}
    for var_key, var_row in variables.iterrows():
        var_prop = var_row["properties"]
        # loop over the given column names as try to read the value from 'properties' field
        # in the variables dataframe. This properties field is the dict we have read from
        # the settings file which may contain the same key  with a value. If this value was
        # defined for the current variable, copy it to the associate column in the data frame
        # such that we can access it more easily
        for name in (
                "type", "fixed", "original_name", "question", "label", "check", "optional",
                "gewicht", "no_impute", "info_per_breakdown", "report_number", "section",
                "keep_options"):
            try:
                variables.loc[var_key, name] = var_prop[name]
            except ValueError:
                # de info_per_breakdown is een dictionary die we met 'at' moeten imposen
                variables.at[var_key, name] = var_prop.get(name)
            except KeyError:
                pass
        # separately get the options field as that contains a dict and therefore can not be
        # imposed as a single value to a row. Instead, collect them and append as a single col
        try:
            var_options = var_prop["options"]
        except KeyError:
            options[var_key] = None
        else:
            options[var_key] = var_options

        try:
            var_filter = var_prop["filter"]
        except KeyError:
            filter_dummy[var_key] = None
        else:
            filter_dummy[var_key] = var_filter

        try:
            var_translate = var_prop["translate"]
        except KeyError:
            translate[var_key] = None
        else:
            translate[var_key] = var_translate

        # add the module label  to this dataframe as well
        if module_info is not None:
            module_name = var_row[module_key]
            try:
                module_label = module_info[module_name]["label"]
            except KeyError:
                _logger.warning("failed to get the label from {}".format(module_name))
            else:
                variables.loc[var_key, "module_label"] = module_label
            try:
                module_include = module_info[module_name]["include"]
            except KeyError:
                _logger.warning("failed to get the include flag from {}".format(module_name))
            else:
                variables.loc[var_key, "module_include"] = module_include

    # create data frame of one columns from the option dictionaries.
    opt_df = pd.DataFrame.from_dict(options, orient="index").rename(columns={0: "options"})
    opt_df = opt_df[opt_df.index != dummy]

    filter_df = pd.DataFrame.from_dict(filter_dummy, orient="index").rename(columns={0: "filter"})
    filter_df = filter_df[filter_df.index != dummy]

    trans_df = pd.DataFrame.from_dict(translate, orient="index").rename(
        columns={0: "translateopts"})
    trans_df = trans_df[trans_df.index != dummy]

    # drop the original column with properties
    variables.drop(["properties"],
                   inplace=True,
                   axis=1)

    # merge the options column with the rest of the columns
    variables = pd.concat([variables, opt_df],
                          axis=1)
    variables = pd.concat([variables, filter_df],
                          axis=1)
    variables = pd.concat([variables, trans_df],
                          axis=1)

    # check if we have a dict data type that does not has a option field set. In that case
    # raise a warning: all the dict type need the options defined
    is_dict = variables["type"] == "dict"
    if (is_dict & variables["options"].isnull()).sum() > 0:
        raise ValueError("Found a dict with no options defined")

    _logger.debug("Done")
    return variables


def add_missing_groups(all_stats, group_by, group_by_original, missing_groups):
    new_stats = {}
    for indicator, data_df in all_stats.items():
        for gb_new, gb_org in zip(group_by, group_by_original):
            data_df.index = data_df.index.rename(gb_org)
        if missing_groups is not None:
            df_extra = pd.DataFrame(index=missing_groups,
                                    columns=data_df.columns)
            data_df = pd.concat([df_extra, data_df])
            new_stats[indicator] = data_df
    return new_stats


def clean_all_suffix(dataframe, suffix_key, variables):
    """
    Hier gaan we de suffixen selecteren die we gedefinieerd hebben.

    Args:
        dataframe: dataframe met tabellen, waaronder een kolom met website extensies
        suffix_key: de naam van de kollom met website extensies
        variables: dataframe met variable informatie. Moet minimaal een variabele
        gelijk aan de suffix_key hebben waarin de categorieen gedefinieerd zijn
    Returns:
        dataframe

    """

    if suffix_key in variables.index:
        translateopts = variables.loc[suffix_key,  "translateopts"]
        categories = dataframe[suffix_key].astype("category")
        # we nemen aan dat de laatste category in de definitie 'rest' is
        categorie_names = list(translateopts.keys())[:-1]
        rest_category = list(translateopts.keys())[-1]
        categories = categories.cat.set_categories(categorie_names)
        categories = categories.cat.add_categories(rest_category)
        categories.fillna(rest_category, inplace=True)

        # dit is nodig om van de strings 'com', 'nl' etc getallen 1, 2, 3 etc te maken.
        categories = categories.astype(str)
        trans = yaml.load(str(translateopts), Loader=yaml.Loader)
        categories = categories.map(trans)

        # kopieer terug naar data frame als category type
        dataframe[suffix_key] = categories.astype("category")

    else:
        _logger.info("Could not find suffix info to translate")

    return dataframe



def get_all_clean_urls(urls, show_progress=False, cache_directory=None):
    if show_progress:
        progress_bar = tqdm(total=urls.size, file=sys.stdout, position=0,
                            ncols=100,
                            leave=True,
                            colour="GREEN")
    else:
        progress_bar = None

    all_clean_urls = list()
    all_suffix = list()

    for url in urls:
        clean_url, suffix = get_clean_url(url, cache_dir=cache_directory)
        _logger.debug(f"Converted {url} to {clean_url}")
        all_clean_urls.append(clean_url)
        all_suffix.append(suffix)
        if progress_bar:
            if clean_url is not None:
                progress_bar.set_description("{:5s} - {:30s}".format("URL", clean_url))
            else:
                progress_bar.set_description("{:5s} - {:30s}".format("URL", "None"))
            progress_bar.update()
    return all_clean_urls, all_suffix
