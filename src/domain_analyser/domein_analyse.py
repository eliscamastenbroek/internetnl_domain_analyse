import argparse
import codecs
import logging
import os
from pathlib import Path

import yaml
from domain_analyser import __version__
from domain_analyser.domain_analyse_classes import (DomainAnalyser, DomainPlotter)

logging.basicConfig(format='%(asctime)s [%(lineno)4s] - %(levelname)-8s : %(message)s',
                    level=logging.DEBUG)
_logger = logging.getLogger()


def parse_args():
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Analyse the domains")
    parser.add_argument("settings_filename", help="Settings file")
    parser.add_argument("--version", action="version",
                        version="{file} version: {ver}".format(file=os.path.basename(__file__),
                                                               ver=__version__))
    parser.add_argument("--verbose", dest="loglevel", help="set loglevel to INFO",
                        action="store_const", const=logging.INFO, default=logging.INFO)
    parser.add_argument("--debug", dest="loglevel", help="set loglevel to DEBUG"
                        , action="store_const", const=logging.DEBUG)
    parser.add_argument("--records_cache_dir", help="Directory of the records cache")
    parser.add_argument("--records_filename", help="Name of the records cache")
    parser.add_argument("--working_directory", help="Directory relative to what we work")
    parser.add_argument("--output_filename", help="Name of the output")
    parser.add_argument("--reset", choices={"0", "1"}, default=None, help="Reset the cached data")
    parser.add_argument("--statistics_to_xls", help="Write the statistics ot an excel file",
                        action="store_true")
    parser.add_argument("--write_dataframe_to_sqlite", action="store_true",
                        help="Store combined data frame to sqlite and quit")
    parser.add_argument("--show_plots", action="store_true",
                        help="Show each plot before continuing")
    parser.add_argument("--max_plots", action="store", type=int,
                        help="Maximum number of plots. If not given, plot all")
    parser.add_argument("--image_type", default=".pdf", choices={".pdf", ".png", ".jpg"},
                        help="Type of the images")
    parser.add_argument("--show_title", action="store_true", help="Show title in plot")
    parser.add_argument("--cummulative", action="store_true", help="Plot pdf cummulitve")
    parser.add_argument("--not_cummulative", action="store_false", dest="cummulative",
                        help="Do not plot pdf cummulitve")
    parser.add_argument("--plot_all", action="store_true", help="Plot alles", default=False)
    parser.add_argument("--cdf_plot", action="store_true", help="Plot de cdf function",
                        default=False)
    parser.add_argument("--bar_plot", action="store_true", help="Plot het staafdiagram",
                        default=False)

    parsed_arguments = parser.parse_args()

    return parsed_arguments


def main():
    args = parse_args()

    _logger.setLevel(args.loglevel)

    _logger.info("Reading settings file {}".format(args.settings_filename))
    with codecs.open(args.settings_filename, "r", encoding="UTF-8") as stream:
        settings = yaml.load(stream=stream, Loader=yaml.Loader)

    general_settings = settings["general"]
    cache_directory = Path(general_settings.get("cache_directory", "."))

    image_directory = Path(general_settings.get("image_directory", "."))
    tex_prepend_path = Path(general_settings.get("tex_prepend_path", "."))

    scan_data = general_settings["scan_data"]
    default_scan = general_settings["default_scan"]

    sheet_renames = general_settings["sheet_renames"]
    n_digits = general_settings["n_digits"]
    n_bins = general_settings["n_bins"]
    barh = general_settings["barh"]
    cummulative = general_settings.get("cummulative", False)
    if args.cummulative is not None:
        cummulative = args.cummulative
    show_title = general_settings.get("show_title", False)
    if args.show_title is not None:
        show_title = args.show_title

    bar_plot = args.bar_plot or args.plot_all
    cdf_plot = args.cdf_plot or args.plot_all

    statistics = settings["statistics"]
    translations = settings["translations"]
    breakdown_labels = settings["breakdown_labels"]
    variables = settings["variables"]
    module_info = settings["module_info"]
    weights = settings["weight"]
    plot_info = settings["plots"]

    if args.records_cache_dir is not None:
        records_cache_dir = args.records_cache_dir
    elif os.getenv("RECORDS_CACHE_DIR") is not None:
        records_cache_dir = os.getenv("RECORDS_CACHE_DIR")
    else:
        records_cache_dir = Path(".")
    if args.records_filename is not None:
        records_filename = Path(args.records_filename)
    elif general_settings.get("records_cache_file") is not None:
        records_filename = Path(general_settings["records_cache_file"])
    else:
        records_filename = Path("records_cache.sqlite")

    records_filename = records_cache_dir / records_filename

    if args.output_filename is None:
        output_file = general_settings.get("output", "internet_nl_stats")
    else:
        output_file = args.output_filename

    if args.working_directory is None:
        wd = general_settings.get("working_directory", ".")
        if wd is None:
            wd = "."
        working_directory = Path(wd)

    else:
        working_directory = Path(args.working_directory)

    with path.Path(str(working_directory)):
        cache_directory.mkdir(exist_ok=True)
        image_directory.mkdir(exist_ok=True)
        _logger.info(f"Running domain analyser in {os.getcwd()}")
        for key, scan_prop in scan_data.items():
            if not scan_prop.get("do_it", True):
                continue
            internet_nl_filename = Path(scan_prop["data_file"])
            _logger.info(f"Start analyse {key}: {internet_nl_filename}")
            domain_analyses = DomainAnalyser(
                scan_data_key=key,
                records_filename=records_filename,
                internet_nl_filename=internet_nl_filename,
                reset=args.reset,
                output_file=output_file,
                cache_directory=cache_directory,
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
                n_bins=n_bins
            )
            scan_prop["analyses"] = domain_analyses

        if bar_plot or cdf_plot:
            DomainPlotter(
                scan_data=scan_data,
                default_scan=default_scan,
                plot_info=plot_info,
                barh=barh,
                show_plots=args.show_plots,
                max_plots=args.max_plots,
                statistics=statistics,
                breakdown_labels=breakdown_labels,
                image_directory=image_directory,
                tex_prepend_path=tex_prepend_path,
                cummulative=cummulative,
                show_title=show_title,
                cdf_plot=cdf_plot,
                bar_plot=bar_plot,
                cache_directory=cache_directory,
            )


if __name__ == "__main__":
    main()
