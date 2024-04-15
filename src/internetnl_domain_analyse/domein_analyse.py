import argparse
import logging
import codecs
import logging
from datetime import datetime
import os
import sys
from pathlib import Path

import yaml

from internetnl_domain_analyse import __version__
from internetnl_domain_analyse.domain_analyse_classes import (
    DomainAnalyser,
    DomainPlotter,
    RecordCacheInfo,
)
from internetnl_domain_analyse.domain_plots import (
    make_heatmap,
    make_conditional_score_plot,
    make_conditional_pdf_plot,
    make_verdeling_per_aantal_categorie,
)
from internetnl_domain_analyse.utils import get_windows_or_linux_value

logging.basicConfig(
    format="%(asctime)s %(filename)25s[%(lineno)4s] - %(levelname)-8s : %(message)s",
    level=logging.WARNING,
)
_logger = logging.getLogger()
_log_hc = logging.getLogger("cbsplotlib")
_log_hc.setLevel(_logger.getEffectiveLevel())

MODES = {"statistics", "correlations", "categories", "all"}
IMAGE_TYPES = {"pdf", "png"}


def parse_args():
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Analyse the domains")
    parser.add_argument("settings_filename", help="Settings file")
    parser.add_argument(
        "--version",
        action="version",
        version="{file} version: {ver}".format(
            file=os.path.basename(__file__), ver=__version__
        ),
    )
    parser.add_argument(
        "--quiet",
        dest="loglevel",
        help="set loglevel to WARNING",
        action="store_const",
        const=logging.WARNING,
        default=logging.INFO,
    )
    parser.add_argument(
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
        default=logging.INFO,
    )
    parser.add_argument(
        "--debug",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    parser.add_argument("--records_cache_dir", help="Directory of the records cache")
    parser.add_argument("--records_filename", help="Name of the records cache")
    parser.add_argument("--output_filename", help="Name of the output")
    parser.add_argument(
        "--reset", choices={"0", "1"}, default=None, help="Reset the cached data"
    )
    parser.add_argument(
        "--statistics_to_xls",
        help="Write the statistics ot an excel file",
        action="store_true",
    )
    parser.add_argument(
        "--write_dataframe_to_sqlite",
        action="store_true",
        help="Store combined data frame to sqlite and quit",
    )
    parser.add_argument(
        "--show_plots", action="store_true", help="Show each plot before continuing"
    )
    parser.add_argument(
        "--max_plots",
        action="store",
        type=int,
        help="Maximum number of plots. If not given, plot all",
    )
    parser.add_argument("--show_title", action="store_true", help="Show title in plot")
    parser.add_argument("--cumulative", action="store_true", help="Plot pdf cmmulitve")
    parser.add_argument(
        "--not_cumulative",
        action="store_false",
        dest="cumulative",
        help="Do not plot pdf cumulitve",
    )
    parser.add_argument(
        "--plot_all", action="store_true", help="Plot alles", default=False
    )
    parser.add_argument(
        "--cdf_plot", action="store_true", help="Plot de cdf function", default=False
    )
    parser.add_argument(
        "--bar_plot", action="store_true", help="Plot het staafdiagram", default=False
    )
    parser.add_argument(
        "--cate_plot",
        action="store_true",
        help="Plot de heatmap of the categories",
        default=False,
    )
    parser.add_argument(
        "--verdeling_plot",
        action="store_true",
        help="Plot de verdeling per categorie",
        default=False,
    )
    parser.add_argument(
        "--cor_plot", action="store_true", help="Plot de heatmap", default=False
    )
    parser.add_argument(
        "--score_plot",
        action="store_true",
        help="Plot de conditionele score",
        default=False,
    )
    parser.add_argument(
        "--export_highcharts",
        help="Export each image to a highcharts file",
        action="store_true",
    )
    parser.add_argument(
        "--force_plots",
        help="Force making plot, even if it already exists",
        action="store_true",
    )
    parser.add_argument(
        "--latex_files",
        help="Write the latex files containing all de plots",
        action="store_true",
    )
    parser.add_argument(
        "--highcharts_output_directory",
        help="Directory waar alle highcharts naar toe geschreven wordt",
    )
    parser.add_argument(
        "--mode", choices=MODES, default="statistics", help="Type  analyse die we doen"
    )
    parser.add_argument(
        "--bovenschrift",
        action="store_true",
        help="De latex overview file krijgt de captions boven de figuren",
        default=True,
    )
    parser.add_argument(
        "--onderschrift",
        action="store_false",
        dest="bovenschrift",
        help="De latex overview file krijgt de captions boven de figuren",
    )
    parser.add_argument("--scan_type", help="Type van de scan die we willen processen.")
    parser.add_argument(
        "--image_type", choices=IMAGE_TYPES, default="pdf", help="Type van de plaatjes"
    )
    parser.add_argument(
        "--variable_to_plot",
        action="append",
        nargs="*",
        default=None,
        help="Maak alleen het plaatje van deze variabele. "
        "Als niet gegeven dan worden alle variabelen geplot",
    )
    parser.add_argument(
        "--exclude_variable",
        action="append",
        nargs="*",
        default=None,
        help="Sla deze variabele over in het geval alle variabelen geplot worden.",
    )
    parser.add_argument(
        "--statistics",
        action="append",
        nargs="*",
        default=None,
        help="Bereken alleen de breakdowns die gegeven zijn en negeer de "
        "settingsfile flags."
        "Als 'all' in required keys dan doe je alle statistieken",
    )
    parser.add_argument(
        "--plots",
        action="append",
        nargs="*",
        default=None,
        help="Plot alleen de breakdowns die gegeven zijn en negeer de settingsfile "
        "flags."
        "Als 'all' gegeven wordt dan maak je gewoon alle plots",
    )
    parser.add_argument(
        "--tld_extract_cache_directory",
        help="Naam van de directory als je het"
        "script naar cache wilt laten lezen"
        "en schrijven",
    )
    parser.add_argument(
        "--dump_cache_as_sqlite",
        help="Dump de cache files als sqlite zodat je ze in kan zien",
        action="store_true",
    )
    parser.add_argument(
        "--english",
        help="Gebruik Engelse vertaling voor labels van plaatjes",
        action="store_true",
    )
    parsed_arguments = parser.parse_args()

    return parsed_arguments


def set_do_it_vlaggen(required_keys, chapter_info, recursive=False):
    """
    Van een hoofdstukje uit je settings file, druk de do_it vlaggen op

    Args:
        required_keys: list
            List van de items waarvan je de do_it vlag wilt opdrukken
        chapter_info: de dictionary waarvan je de vlaggen zet.
        recursive: bool
            Als dit een recursieve call is, dan willen we de waardes die niet in de lijst zitten
            niet op False zetten

    Returns: dict
        De nieuwe dictionary.
    """
    for key, properties in chapter_info.items():
        if key in required_keys or "all" in required_keys:
            properties["do_it"] = True
            combination: list = properties.get("combination")
            if combination is not None:
                # als de breakdown een combination is dan moet we de onderliggende breakdowns
                # allemaal activeren
                chapter_info = set_do_it_vlaggen(
                    required_keys=combination, chapter_info=chapter_info, recursive=True
                )
        elif not recursive:
            properties["do_it"] = False
    return chapter_info


def main():
    args = parse_args()
    print("-" * 100)
    exe = Path(sys.argv[0]).stem
    now = datetime.now()
    print(
        f"Starting {exe} ({__version__}) at {now} with arguments:\n"
        + " ".join(sys.argv[1:])
    )
    print("-" * 100)

    _logger.setLevel(args.loglevel)

    _logger.info("Reading settings file {}".format(args.settings_filename))
    with codecs.open(args.settings_filename, "r", encoding="UTF-8") as stream:
        settings = yaml.load(stream=stream, Loader=yaml.Loader)

    general_settings = settings["general"]
    cache_directory_base_name = general_settings.get("cache_directory", ".")

    if args.highcharts_output_directory is not None:
        highcharts_directory = args.highcharts_output_directory
    else:
        highcharts_directory = general_settings.get("highcharts_output_directory")

    label_translations = general_settings.get("translations")

    image_directory = Path(general_settings.get("image_directory", "."))
    tex_prepend_path = Path(general_settings.get("tex_prepend_path", "."))
    tex_horizontal_shift = get_windows_or_linux_value(
        general_settings.get("tex_horizontal_shift", "-1.15cm")
    )

    scan_data = general_settings["scan_data"]
    default_scan = general_settings["default_scan"]
    stat_directory = general_settings.get("stat_directory")

    sheet_renames = general_settings["sheet_renames"]
    n_digits = general_settings["n_digits"]
    n_bins = general_settings["n_bins"]
    barh = general_settings.get("barh", False)
    cumulative = general_settings.get("cumulative", False)
    if args.cumulative is not None:
        cumulative = args.cumulative
    show_title = general_settings.get("show_title", False)
    if args.show_title is not None:
        show_title = args.show_title

    bar_plot = args.bar_plot or args.plot_all
    cdf_plot = args.cdf_plot or args.plot_all
    cor_plot = args.cor_plot or args.plot_all
    cate_plot = args.cate_plot or args.plot_all
    verdeling_plot = args.verdeling_plot or args.plot_all
    score_plot = args.score_plot or args.plot_all

    categories = settings.get("categories")
    correlations = settings.get("correlations")
    statistics = settings["statistics"]
    translations = settings["translations"]
    breakdown_labels = settings["breakdown_labels"]
    variables_per_type = settings["variables"]
    module_info = settings["module_info"]
    weights = settings["weight"]
    plot_info = settings["plots"]

    if args.statistics:
        required_keys = [stat[0] for stat in args.statistics]
        statistics = set_do_it_vlaggen(
            required_keys=required_keys, chapter_info=statistics
        )
    if args.plots:
        required_keys = [plot[0] for plot in args.plots]
        plot_info = set_do_it_vlaggen(
            required_keys=required_keys, chapter_info=plot_info
        )

    if args.output_filename is None:
        output_file = general_settings.get("output", "internet_nl_stats")
    else:
        output_file = args.output_filename

    image_directory.mkdir(exist_ok=True)
    var_df = None
    try:
        records_cache_data_per_year = general_settings["records_cache_data"]
    except KeyError as err:
        _logger.warning(err)
        raise KeyError("records cache data needs to be given per year!")

    _logger.info(f"Running domain analyser in {os.getcwd()}")
    for key_scan_type, scan_prop_per_year in scan_data.items():
        if args.scan_type is not None and key_scan_type != args.scan_type:
            _logger.info(f"Scan type {key_scan_type} not requested. Skipping")
            continue

        try:
            variables = variables_per_type[key_scan_type]
        except KeyError as err:
            msg = "Since version in 2022, we need to give the variables per scan type"
            _logger.warning(msg)
            raise KeyError(msg)

        years_to_add_to_plot_legend = list()
        for scan_year, scan_prop in scan_prop_per_year.items():

            if not scan_prop.get("do_it", True):
                continue
            filename = scan_prop["data_file"]
            years_to_add_to_plot_legend.append(scan_year)
            if filename is None:
                _logger.info(
                    f"File name for year {scan_year} was None. Skip it but add it to"
                    f"the plot legend"
                )
                continue
            internet_nl_filename = Path(filename)
            records_cache_data = records_cache_data_per_year[scan_year]
            records_cache_info = RecordCacheInfo(
                records_cache_data=records_cache_data,
                year_key=scan_year,
                stat_directory=stat_directory,
            )

            _logger.info(f"Start analyse {scan_year}: {internet_nl_filename}")
            domain_analyses = DomainAnalyser(
                scan_data_key=key_scan_type,
                records_cache_info=records_cache_info,
                internet_nl_filename=internet_nl_filename,
                reset=args.reset,
                output_file=output_file,
                cache_directory_base_name=cache_directory_base_name,
                tld_extract_cache_directory=args.tld_extract_cache_directory,
                statistics=statistics,
                default_scan=default_scan,
                variables=variables,
                module_info=module_info,
                weights=weights,
                translations=translations,
                breakdown_labels=breakdown_labels,
                sheet_renames=sheet_renames,
                n_digits=n_digits,
                write_dataframe_to_sqlite=args.write_dataframe_to_sqlite,
                statistics_to_xls=args.statistics_to_xls,
                n_bins=n_bins,
                mode=args.mode,
                correlations=correlations,
                categories=categories,
                dump_cache_as_sqlite=args.dump_cache_as_sqlite,
            )
            scan_prop["analyses"] = domain_analyses

            if var_df is None:
                var_df = domain_analyses.variables

        if cor_plot and correlations is not None:
            make_heatmap(
                correlations=correlations,
                image_directory=image_directory,
                highcharts_directory=highcharts_directory,
                show_plots=args.show_plots,
                cache_directory=cache_directory_base_name,
            )
        if cate_plot and categories is not None:
            make_conditional_pdf_plot(
                categories=categories,
                image_directory=image_directory,
                highcharts_directory=highcharts_directory,
                show_plots=args.show_plots,
                cache_directory=cache_directory_base_name,
            )
        if verdeling_plot and categories is not None:
            make_verdeling_per_aantal_categorie(
                categories=categories,
                image_directory=image_directory,
                highcharts_directory=highcharts_directory,
                show_plots=args.show_plots,
                cache_directory=cache_directory_base_name,
                export_highcharts=args.export_highcharts,
            )
        if score_plot and correlations is not None:
            make_conditional_score_plot(
                correlations=correlations,
                image_directory=image_directory,
                highcharts_directory=highcharts_directory,
                show_plots=args.show_plots,
                cache_directory=cache_directory_base_name,
            )

        if bar_plot or cdf_plot:
            DomainPlotter(
                scan_data=scan_data,
                scan_data_key=key_scan_type,
                default_scan=default_scan,
                plot_info=plot_info,
                barh=barh,
                show_plots=args.show_plots,
                max_plots=args.max_plots,
                statistics=statistics,
                breakdown_labels=breakdown_labels,
                image_directory=image_directory,
                variables=var_df,
                tex_prepend_path=tex_prepend_path,
                cumulative=cumulative,
                show_title=show_title,
                cdf_plot=cdf_plot,
                bar_plot=bar_plot,
                cache_directory=cache_directory_base_name,
                translations=label_translations,
                export_highcharts=args.export_highcharts,
                highcharts_directory=highcharts_directory,
                tex_horizontal_shift=tex_horizontal_shift,
                bovenschrift=args.bovenschrift,
                image_type=args.image_type,
                variables_to_plot=args.variable_to_plot,
                exclude_variables=args.exclude_variable,
                force_plots=args.force_plots,
                latex_files=args.latex_files,
                years_to_add_to_plot_legend=years_to_add_to_plot_legend,
                module_info=module_info,
                english=args.english,
            )


if __name__ == "__main__":
    main()
