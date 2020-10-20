import logging
from pylatex import Document, Figure, SubFigure, NoEscape, Command
from pylatex.base_classes import Environment, CommandBase, Arguments
from pylatex.package import Package

_logger = logging.getLogger(__name__)


class ExampleEnvironment(Environment):
    """
    A class representing a custom LaTeX environment.

    This class represents a custom LaTeX environment named
    ``exampleEnvironment``.
    """

    _latex_name = 'exampleEnvironment'
    packages = [Package('mdframed')]


class SubFloat(CommandBase):
    """
    A class representing a custom LaTeX command.

    This class represents a custom LaTeX command named
    ``exampleCommand``.
    """

    _latex_name = 'subfloat'


def make_latex_overview(all_plots, variables, image_directory, image_files, horizontal_shift="-2cm"):
    """
    Maak latex ouput file met alle plaatjes
    Args:
        all_plots: dict met eerste nivear variabele name, dan labels voor SBI, GK, dan file namen 
        variables: dict met variabele eigenschappen
        image_directory: str
        image_files: 
        horizontal_shift: verschuiving naar links 
    """
    doc = Document(default_filepath=image_directory)

    for original_name, images in all_plots.items():
        _logger.debug(f"Adding {original_name}")
        caption = variables.loc[original_name, "label"]
        with doc.create(Figure(position="htb")) as plots:
            add_new_line = True
            for label, image_name in images.items():
                _logger.debug(f"Adding {image_name}")
                ref_label = Command("label", "fig:" + label.lower().replace(" ", "_"))
                lab = Command("footnotesize", Arguments(label, ref_label))
                include_graphics = Command("includegraphics", NoEscape(image_name))
                hspace = Command("hspace", Arguments(NoEscape(horizontal_shift), include_graphics))
                sub_plot = SubFloat(
                    options=[lab],
                    arguments=Arguments(hspace))
                plots.append(sub_plot)
                if add_new_line:
                    plots.append(Command("newline"))
                    add_new_line = False
            plots.add_caption(caption)

    file_name = image_directory / image_files.with_suffix(".tex")
    _logger.info(f"Writing tex file list to {file_name}")
    doc.generate_tex(filepath=file_name.as_posix())
