"""
Generic script to install the executable
Usage:

python install_exe.py

This script runs the following steps

1.: python install.py bdist_wheel

Generates a wheel file in the dest directory

2.: pip install internetnl_domain_analyse --no-index --find-links <name of the wheel file> --prefix <location> -U

Installs the APP as a library

3.: python install.py install --prefix <location>

Installs the internetnl_domain_analyse executable

See python install_exe.py --help for the options
"""
from pathlib import Path
import re
import sys
from subprocess import Popen, PIPE
import logging
import argparse

TOOLS_DIR = "\\\\cbsp.nl\\Productie\\secundair\\DecentraleTools\\Output\\CBS_Python"

parser = argparse.ArgumentParser("Install the python executable of the internetnl_domain_analyse")
parser.add_argument("--debug", help="Give debug info", dest="log_level", default=logging.INFO, const=logging.DEBUG,
                    action="store_const")
parser.add_argument("--app_name", help="Name of the application", default="internetnl_domain_analyse")
parser.add_argument("--system_install", help="If true, install in the system directory",
                    action="store_true")
parser.add_argument("--destination", help="Destination where the app is installed", default=TOOLS_DIR)
parser.add_argument("--python_version", help="Version of python to install for")
parser.add_argument("--update", help="App date a previous installed version", default="-U",
                    const="-U",
                    action="store_const")
parser.add_argument("--no_update", help="Do not update a previous installed version", const="", 
                    dest="update",
                    action="store_const")
parser.add_argument("--test", help="Do not run, only show generated commands", action="store_true")

args = parser.parse_args()

logging.basicConfig(level=args.log_level, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

python_exe = Path(sys.executable)
pip_exe = python_exe.parent / Path("Scripts") / Path("pip.exe")

assert sys.version_info.major > 2

if args.python_version is None:
    minor = sys.version_info.minor
    python_version = f"Python3.{minor}"
else:
    python_version = args.python_version

destination = Path(args.destination) / Path(python_version)

if not destination.exists():
    raise FileNotFoundError(f"Could not find destination location {destination}")

p = Popen([str(python_exe), "setup.py", "bdist_wheel"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate()
logger.debug("output:\n{}".format(output))
logger.debug("err:\n{}".format(err))
lines = output.decode("utf-8").split("\n")
wheel = None
for line in lines:
    logger.info(line)
    match = re.search("creating '(.*?)'", line)
    if match:
        wheel = match.group(1)

if wheel is not None:
    logging.debug(f"Found {wheel}")

    dist = Path(wheel)
    if not dist.exists():
        raise FileNotFoundError(f"Could not find dist file {dist}")

    pip = [str(pip_exe), "install", args.app_name, "--no-index", "--find-links", str(wheel),
           args.update]
    python = [str(python_exe), "setup.py", "install"]

    if not args.system_install:
        pip += ["--prefix", str(destination)]
        python += ["--prefix", str(destination)]

    logger.info(" ".join(pip))
    if not args.test:
        p2 = Popen(pip, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out2, err2 = p2.communicate()
        logger.debug("output:\n{}".format(out2))
        logger.debug("err:\n{}".format(err2))

    logger.info(" ".join(python))
    if not args.test:
        p3 = Popen(python, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out3, err3 = p3.communicate()
        logger.debug("output:\n{}".format(out3))
        logger.debug("err:\n{}".format(err3))

else:
    logging.warning(f"Could not find the wheel: {wheel}")
