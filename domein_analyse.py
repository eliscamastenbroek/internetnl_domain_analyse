import argparse
import codecs
import logging
import os
import sys
from pathlib import Path

import path
import yaml

from domain_analyse_classes import DomainAnalyser

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
    parser.add_argument("--output_filename", help="Name of the output")
    parser.add_argument("--reset", choices={"0", "1"}, default=None, help="Reset the cached data")
    parser.add_argument("--plot", help="Make the plots of the statistics", action="store_true")
    parser.add_argument("--statistics_to_xls", help="Write the statistics ot an excel file",
                        action="store_true")
    parser.add_argument("--write_dataframe_to_sqlite", action="store_true",
                        help="Store combined data frame to sqlite and quit")
    parser.add_argument("--show_plots", action="store_true",
                        help="Show each plot before continuing")
    parser.add_argument("--image_type", default=".pdf", choices={".pdf", ".png", ".jpg"},
                        help="Type of the images")

    parsed_arguments = parser.parse_args(args)

    return parsed_arguments


def main(argv):
    args = parse_args(argv)

    _logger.setLevel(args.loglevel)

    _logger.info("Reading settings file {}".format(args.settings_filename))
    with codecs.open(args.settings_filename, "r", encoding="UTF-8") as stream:
        settings = yaml.load(stream=stream, Loader=yaml.Loader)

    general_settings = settings["general"]
    cache_directory = Path(general_settings.get("cache_directory", "."))

    image_directory = Path(general_settings.get("image_directory", "."))

    sheet_renames = general_settings["sheet_renames"]
    n_digits = general_settings["n_digits"]

    statistics = settings["statistics"]
    translations = settings["translations"]
    breakdown_labels = settings["breakdown_labels"]
    variables = settings["variables"]
    module_info = settings["module_info"]
    weights = settings["weight"]
    plot_info = settings["plots"]

    if args.output_filename is None:
        output_file = general_settings.get("output", "internet_nl_stats")
    else:
        output_file = args.output_filename

    if args.working_directory is None:
        working_directory = Path(general_settings.get("working_directory", "."))
    else:
        working_directory = Path(args.working_directory)

    with path.Path(str(working_directory)):
        cache_directory.mkdir(exist_ok=True)
        image_directory.mkdir(exist_ok=True)
        _logger.info(f"Running domain analyser in {os.getcwd()}")
        DomainAnalyser(
            reset=args.reset,
            output_file=output_file,
            cache_directory=cache_directory,
            image_directory=image_directory,
            statistics=statistics,
            variables=variables,
            module_info=module_info,
            weights=weights,
            translations=translations,
            breakdown_labels=breakdown_labels,
            sheet_renames=sheet_renames,
            n_digits=n_digits,
            write_dataframe_to_sqlite=args.write_dataframe_to_sqlite,
            statistics_to_xls=args.statistics_to_xls,
            plot_statistics=args.plot,
            plot_info=plot_info,
            show_plots=args.show_plots
        )


if __name__ == "__main__":
    main(sys.argv[1:])
