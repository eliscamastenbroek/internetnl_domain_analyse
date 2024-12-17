import logging
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yaml
from internetnl_scan.utils import get_clean_url
from tqdm import tqdm

_logger = logging.getLogger(__name__)
tld_logger = logging.getLogger("tldextract")
tld_logger.setLevel(logging.WARNING)


def reorganise_stat_df(
    records_stats,
    variables,
    variable_key,
    use_original_names=False,
    n_digits=3,
    sort_index=True,
    module_key="module",
    vraag_key="vraag",
    optie_key="optie",
):
    """
    We have a statistics data frame, but not yet all the information of the variables

    Parameters
    ----------
    use_original_names: bool
        Use the original name of the variable
    """

    _logger.debug("Reorganising stat")
    # at the beginning, sbi and gk as multindex index, variabel/choice as mi-columns
    try:
        mask = records_stats.index.get_level_values(1) != ""
    except IndexError:
        mask = records_stats.index.get_level_values(0) != ""
    stat_df = records_stats.loc[mask].copy()
    # with unstack, the gk is put as an extra level to the columns. sbi is now a normal  index
    if len(stat_df.index.names) > 1:
        stat_df = stat_df.unstack()

    # transposing and resetting index puts the variables and choice + gk at the index,
    # reset index create columns out of them. The sbi codes are now at the columns
    temp_df = stat_df.transpose()
    stat_df = temp_df.reset_index()
    # stat_df = stat_df.T.reset_index()
    # onderstaande bij 1 index
    stat_df.rename(columns={"index": "variable"}, inplace=True)
    # onderstaand alleen bij unstack
    stat_df.rename(columns={"level_0": "variable"}, inplace=True)

    try:
        stat_df.drop([""], inplace=True, axis=1)
    except KeyError:
        pass

    # add new columns with the module type so we can reorganise the questions into modules
    stat_df[module_key] = "Unknown"
    stat_df["module_include"] = True
    stat_df[optie_key] = ""
    stat_df[vraag_key] = ""
    stat_df["check"] = False
    stat_df["od_key"] = None

    for var_name in stat_df[variable_key].unique():
        _logger.debug("var varname {}".format(var_name))
        # copy the module key from the variables to the statistics data frame
        # get the mask to identify all variable in the stat_df equal to the current var_name
        # note that var_name in stat_df is not unique as each varname occurs multiple time
        # for other gk code, sbi groups etc. However, it is unique in the variabel dataframe

        mask = stat_df[variable_key] == var_name

        # tijdelijke oplossing categorien
        # import re
        # we hebben de naam nodig zonder nummertje erachter. De naam is nodig om gegevens
        # uit de yaml file te halen\
        match = re.search(r"_(\d)[\.0]*$", var_name)
        if bool(match):
            choice = int(match.group(1))
            var_name_clean = re.sub(r"_\d[\.0]*$", "", var_name)
        else:
            choice = None
            var_name_clean = var_name
        # var_name_clean = re.sub("\_x$", "", var_name_clean)
        try:
            module_key_key = variables.loc[var_name_clean, module_key]
        except KeyError:
            pass
        else:
            stat_df.loc[mask, module_key] = module_key_key

        try:
            module_label = variables.loc[var_name_clean, "module_label"]
        except KeyError:
            pass
        else:
            if module_label is not None:
                stat_df.loc[mask, module_key] = module_label

        if use_original_names:
            try:
                label = variables.loc[var_name_clean, "original_name"]
            except KeyError:
                label = var_name
            else:
                if label in ("", None):
                    label = var_name
        else:
            try:
                label = variables.loc[var_name_clean, "label"]
            except KeyError:
                label = var_name
            else:
                if label in ("", None):
                    label = var_name

        stat_df.loc[mask, "vraag"] = label

        try:
            module_include = variables.loc[var_name_clean, "module_include"]
        except KeyError:
            pass
        else:
            stat_df.loc[mask, "module_include"] = module_include

        try:
            check_vraag = variables.loc[var_name_clean, "check"]
        except KeyError:
            pass
        else:
            stat_df.loc[mask, "check"] = check_vraag

        try:
            options_dict = variables.loc[var_name_clean, "options"]
        except KeyError:
            pass
        else:
            if options_dict is not None and choice is not None:
                try:
                    option_label = options_dict[choice]
                except KeyError:
                    _logger.warning(f"Invalid option {choice} for {var_name}")
                else:
                    stat_df.loc[mask, "optie"] = option_label

    # select only the module with the include flag to true
    stat_df = stat_df[stat_df["module_include"]]
    stat_df = stat_df[stat_df[module_key] != "Unknown"]
    stat_df.drop(["module_include", "check", "od_key"], axis=1, inplace=True)

    if n_digits is not None:
        stat_df = stat_df.round(decimals=n_digits)

    index_variables = [module_key, vraag_key, optie_key]
    if sort_index:
        stat_df.sort_values([module_key, vraag_key, variable_key], axis=0, inplace=True)
    stat_df.set_index(index_variables, inplace=True, drop=True)
    return stat_df


# noinspection SqlDialectInspection
def read_tables_from_sqlite(filename: Path, table_names, index_name) -> pd.DataFrame:
    if isinstance(table_names, str):
        table_names = [table_names]

    if not filename.exists():
        _logger.warning(
            "Records file not found. Make sure you set the environment variable "
            "RECORDS_CACHE_DIR_<yearkey> for the file and RECORDS_TABLE_RECS_<yearkey> and "
            "RECORDS_TABLE_INFO<yearkey> for the tables or pass it via the command line argument "
            "--records_cache_dir"
        )
        raise FileNotFoundError(f"Records file not found {filename.absolute()}")

    _logger.info(f"Reading from {filename}")
    connection = sqlite3.connect(filename.as_posix())
    tables = list()
    for table_name in table_names:
        _logger.debug(f"Reading table {table_name}")
        df = pd.read_sql(
            f"select * from {table_name}", con=connection, index_col=index_name
        )
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


def add_derived_variables(tables, variables):
    """
    Add the variables we defined in the settings files which do not exist yet, but are defined with an eval statement

    Args:
        tables: pd.DataFrame
            original table of variables
        variables: pd.DataFrame
            properties of variables

    Returns:
        pd.DataFame

    """
    undefined_variables = variables.index.difference(tables.columns)

    for variable_name in undefined_variables:
        _logger.debug(f"deriving properties for {variable_name}")

        var_props = variables.loc[variable_name, :]
        eval_statement = var_props.get("eval")
        if eval_statement is None:
            _logger.debug(
                f"Column {variable_name} does not exist but settings do not provide an eval statement.\n"
                f"Is ok, not all years have all properties defined. Only if an eval statement"
                f"is defined, we are going to add a new columns"
            )
            continue

        _logger.info(f"creating new column {variable_name} as {eval_statement}")
        tables[variable_name] = tables.eval(eval_statement)

    return tables


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


def prepare_stat_data_for_write(
    all_stats,
    file_base,
    variables,
    module_key,
    variable_key,
    breakdown_labels=None,
    n_digits=3,
    connection=None,
):
    data = pd.DataFrame.from_dict(all_stats)
    if connection is not None:
        data.to_sql(name=file_base, con=connection, if_exists="replace")

    stat_df = reorganise_stat_df(
        records_stats=data,
        variables=variables,
        use_original_names=True,
        module_key=module_key,
        variable_key=variable_key,
        n_digits=n_digits,
        sort_index=False,
    )
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
    """get the mask to filter the positive options from a question"""
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


def impose_variable_defaults(
    variables, module_info: dict = None, module_key: str = None
):
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
    variables["eval"] = None
    variables["unit"] = None

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
            "type",
            "fixed",
            "original_name",
            "question",
            "label",
            "check",
            "optional",
            "gewicht",
            "no_impute",
            "info_per_breakdown",
            "report_number",
            "section",
            "keep_options",
            "eval",
            "unit",
        ):
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
                _logger.warning(
                    "failed to get the include flag from {}".format(module_name)
                )
            else:
                variables.loc[var_key, "module_include"] = module_include

    # create data frame of one columns from the option dictionaries.
    opt_df = pd.DataFrame.from_dict(options, orient="index").rename(
        columns={0: "options"}
    )
    opt_df = opt_df[opt_df.index != dummy]

    filter_df = pd.DataFrame.from_dict(filter_dummy, orient="index").rename(
        columns={0: "filter"}
    )
    filter_df = filter_df[filter_df.index != dummy]

    trans_df = pd.DataFrame.from_dict(translate, orient="index").rename(
        columns={0: "translateopts"}
    )
    trans_df = trans_df[trans_df.index != dummy]

    # drop the original column with properties
    variables.drop(["properties"], inplace=True, axis=1)

    # merge the options column with the rest of the columns
    variables = pd.concat([variables, opt_df], axis=1)
    variables = pd.concat([variables, filter_df], axis=1)
    variables = pd.concat([variables, trans_df], axis=1)

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
            df_extra = pd.DataFrame(index=missing_groups, columns=data_df.columns)
            df_extra = df_extra.astype(data_df.dtypes)
            data_df = pd.concat([df_extra, data_df])
            new_stats[indicator] = data_df
    return new_stats


def clean_all_suffix(dataframe, suffix_key, variables):
    """
    Hier gaan we de suffixen selecteren die we gedefinieerd hebben.

    Args:
        dataframe: dataframe met tabellen, waaronder een kolom met website extensies
        suffix_key: de naam van de kolom met website extensies
        variables: dataframe met variable informatie. Moet minimaal een variabele
        gelijk aan de suffix_key hebben waarin de categorieën gedefinieerd zijn
    Returns:
        dataframe

    """

    if suffix_key in variables.index:
        translateopts = variables.loc[suffix_key, "translateopts"]
        categories = dataframe[suffix_key].astype("category")
        # we nemen aan dat de laatste category in de definitie 'rest' is
        categorie_names = list(translateopts.keys())[:-1]
        rest_category = list(translateopts.keys())[-1]
        # we gaan hier categorieen toevoegen op basis van het lijst dat we
        # in de settings file gegeven hebben (.nl, .com, .eu). De laatste beschouwen
        # we als rest categorie waartoe we alle dat niet bij de hoofdcategorieen
        # hoort gaan indelen.
        categories = categories.cat.set_categories(categorie_names)
        # alle velden die nu nog geen categorie hebben worden nu aangeduid
        # met de rest categorie
        categories = categories.cat.add_categories(rest_category)

        # de rest categorieen worden nu op 'na' gezet
        categories.fillna(rest_category, inplace=True)

        # dit is nodig om van de strings 'com', 'nl' etc getallen 1, 2, 3 etc te maken.
        categories = categories.astype(str)
        # vertaal de strings per categorie 'com', 'nl' etc naar de digits
        trans = yaml.load(str(translateopts), Loader=yaml.Loader)
        categories = categories.map(trans)

        # kopieer terug naar data frame als category type
        dataframe[suffix_key] = categories.astype("category")

    else:
        _logger.info("Could not find suffix info to translate")

    return dataframe


def get_all_clean_urls(urls, show_progress=False, cache_directory=None):
    if show_progress:
        progress_bar = tqdm(
            total=urls.size,
            file=sys.stdout,
            position=0,
            ncols=100,
            leave=True,
            colour="GREEN",
        )
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


def get_windows_or_linux_value(value):
    """Pas de waarde aan als deze in een dict gegeven is met een windows en linux veld"""
    if isinstance(value, dict):
        if "win" in sys.platform:
            new_value = value["windows"]
        else:
            new_value = value["linux"]
    else:
        new_value = value

    return new_value


def dump_data_frame_as_sqlite(dataframe, file_name):
    """Dump data als sqlite, maar zorg dat je duplicates eruit haalt"""

    # maak lower van de columnnamen want sqlite is case insensitive
    is_duplicated_column = dataframe.columns.str.lower().duplicated()
    duplicated_columns = dataframe.columns[is_duplicated_column]
    _logger.debug(f"Dropping duplicated columns {duplicated_columns}")
    clean_df = dataframe.drop(columns=duplicated_columns)

    _logger.info(f"Writing cache as sqlite {file_name}")
    with sqlite3.connect(file_name) as connection:
        clean_df.to_sql(name="table", con=connection, if_exists="replace")
