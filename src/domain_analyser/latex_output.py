import pandas as pd
import logging

from ict_analyser.shared.utils import variable_dict_to_df
from pylatex import Document, Figure, NoEscape, Command
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


def make_latex_overview(all_plots, variables, image_directory, image_files,
                        tex_horizontal_shift="-2cm", tex_prepend_path=None,
                        all_shifts=None):
    """
    Maak latex ouput file met alle plaatjes
    Args:
        all_plots: dict met eerste nivear variabele name, dan labels voor SBI, GK, dan file namen 
        variables: dict met variabele eigenschappen
        image_directory: str
        tex_prepend_path: str
        image_files:
        tex_horizontal_shift: verschuiving naar links
    """
    if tex_prepend_path is None:
        full_image_directory = image_directory
    else:
        full_image_directory = tex_prepend_path / image_directory

    doc = Document(default_filepath=full_image_directory)

    for original_name, images in all_plots.items():
        _logger.debug(f"Adding {original_name}")
        caption = variables.loc[original_name, "label"]
        with doc.create(Figure(position="htb")) as plots:
            add_new_line = True
            for label, image_name in images.items():
                if tex_prepend_path is None:
                    full_image_name = image_name
                else:
                    full_image_name = tex_prepend_path / image_name
                horizontal_shift = tex_horizontal_shift
                if shift_props := all_shifts.get(original_name):
                    if hz := shift_props.get(label):
                        horizontal_shift = hz
                _logger.debug(f"Adding {full_image_name}")
                ref = "_".join([original_name, label.lower().replace(" ", "_")])
                ref_sublabel = Command("label", NoEscape("fig:" + ref))
                lab = Command("footnotesize", Arguments(label, ref_sublabel))
                include_graphics = Command("includegraphics", NoEscape(full_image_name))
                if horizontal_shift is not None:
                    hspace = Command("hspace", Arguments(NoEscape(horizontal_shift), include_graphics))
                sub_plot = SubFloat(
                    options=[lab],
                    arguments=Arguments(hspace))
                plots.append(sub_plot)
                if add_new_line:
                    plots.append(Command("newline"))
                    add_new_line = False
            plots.add_caption(caption)
            ref_label = Command("label", NoEscape("fig:" + original_name))
            plots.append(ref_label)

    file_name = image_directory / image_files.with_suffix("")
    _logger.info(f"Writing tex file list to {file_name}.tex")
    doc.generate_tex(filepath=file_name.as_posix())
