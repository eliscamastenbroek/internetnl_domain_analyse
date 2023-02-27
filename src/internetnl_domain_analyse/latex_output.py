import logging
from pathlib import Path

from pylatex import Document, Figure, NoEscape, Command, SubFigure
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


def make_latex_overview(
        image_info=None,
        variables=None,
        image_directory=None,
        image_files=None,
        tex_horizontal_shift="-2cm",
        tex_prepend_path=None,
        bovenschrift=False):
    """
    Maak latex ouput file met alle plaatjes
    Args:
        image_info: object: ImageInfo
        variables: dict met variabele eigenschappen
        image_directory: str
        image_files: obj:Path
        tex_prepend_path: str
        tex_horizontal_shift: verschuiving naar links
        bovenschrift: boolean
            Voeg caption bovenaan figuren
    """
    if tex_prepend_path is None:
        full_image_directory = image_directory
    else:
        full_image_directory = tex_prepend_path / image_directory

    doc_per_module = dict()

    for original_name, image_prop in image_info.data.items():
        _logger.debug(f"Adding {original_name}")
        caption = variables.loc[original_name, "label"]
        module = variables.loc[original_name, "module"]
        try:
            doc = doc_per_module[module]
        except KeyError:
            doc = Document(default_filepath=full_image_directory)
            doc_per_module[module] = doc
        with doc.create(Figure(position="htb")) as plots:
            add_new_line = True
            if bovenschrift:
                # hiermee worden caption boven toegevoegd, maar ik wil hier vanaf gaan wijken.
                plots.add_caption(caption)
                ref_label = Command("label", NoEscape("fig:" + original_name))
                plots.append(ref_label)
            for sub_image_key, sub_image_prop in image_prop.items():
                sub_image_label = sub_image_prop.get("sub_image_label", "Empty")
                image_name = sub_image_prop["file_name"]
                horizontal_shift = sub_image_prop.get("tex_right_shift", tex_horizontal_shift)
                with doc.create(SubFigure(width=NoEscape(r'\linewidth'))) as sub_plot:
                    if tex_prepend_path is None:
                        full_image_name = Path(image_name)
                    else:
                        if image_name is not None:
                            full_image_name = tex_prepend_path / image_name
                        else:
                            full_image_name = Path("bla")
                    _logger.debug(f"Adding {full_image_name}")
                    ref = "_".join([original_name, sub_image_label.lower().replace(" ", "_")])
                    ref_sublabel = Command("label", NoEscape("fig:" + ref))
                    lab = Command("footnotesize", Arguments(sub_image_label))
                    include_graphics = Command("includegraphics",
                                               NoEscape(full_image_name.with_suffix("").as_posix()))
                    if horizontal_shift is not None:
                        include_graphics = Command("hspace", Arguments(NoEscape(horizontal_shift),
                                                                       include_graphics))
                    # sub_plot = SubFloat(
                    #    options=[lab],
                    #    arguments=Arguments(hspace))
                    if not bovenschrift:
                        sub_plot.append(include_graphics)
                    sub_plot.add_caption(lab)
                    sub_plot.append(ref_sublabel)
                    if bovenschrift:
                        sub_plot.append(include_graphics)
                if add_new_line:
                    plots.append(Command("newline"))
                    add_new_line = False
            if not bovenschrift:
                # voeg een onderschrift toe
                plots.add_caption(caption)
                ref_label = Command("label", NoEscape("fig:" + original_name))
                plots.append(ref_label)

    for module, doc in doc_per_module.items():
        file_name = Path("_".join([image_info.scan_data_key,
                                   image_files.with_suffix("").as_posix(), module]))
        _logger.info(f"Writing tex file list to {file_name}.tex")
        doc.generate_tex(filepath=file_name.as_posix())
        file_name = file_name.with_suffix(".tex")
        new_lines = list()
        start = False
        with open(file_name.as_posix(), "r") as stream:
            for line in stream.readlines():
                if "figure" in line:
                    start = True
                if "end{document}" in line:
                    start = False
                if start:
                    new_lines.append(line)
        with open(file_name.as_posix(), "w") as stream:
            stream.writelines(new_lines)
