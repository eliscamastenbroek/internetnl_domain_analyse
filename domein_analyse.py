import os
import argparse
import codecs
import logging
import sys
import path
from pathlib import Path

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

    parsed_arguments = parser.parse_args(args)

    return parsed_arguments


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

    with path.Path(str(working_directory)):
        _logger.info(f"Running domain analyser in {os.getcwd()}")
        DomainAnalyser(settings=settings)


if __name__ == "__main__":
    main(sys.argv[1:])
