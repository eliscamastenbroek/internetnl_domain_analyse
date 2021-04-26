import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as trn

import seaborn as sns
from cbs_utils.plotting import CBSPlotSettings

_logger = logging.getLogger(__name__)
sns.set_style('whitegrid')


def make_bar_plot(plot_df, plot_key, module_name, question_name, image_directory, show_plots=False,
                  figsize=None, image_type=".pdf", reference_lines=None, xoff=0.02, yoff=0.02):
    """ create a bar plot from the question 'plot_df'"""
    figure_properties = CBSPlotSettings()

    if figsize is None:
        figsize = figure_properties.fig_size

    _logger.debug(f"Figsize: {figsize}")

    names = plot_df.index.names
    plot_df.reset_index(inplace=True)
    plot_title = " - ".join([module_name, question_name])
    result = plot_df.loc[0, names[2]]
    if result == "":
        result = "True"
    plot_title += f": {result}"
    plot_df[plot_title] = None
    plot_df.drop(names + ["variable"], axis=1, inplace=True)
    plot_df.set_index(plot_title, inplace=True)
    plot_df = plot_df.T

    fig, axis = plt.subplots(figsize=figsize)

    line_iter = axis._get_lines
    trans = trn.blended_transform_factory(axis.transAxes, axis.transData)

    try:
        plot_df.plot(kind="bar", ax=axis, rot=0, legend=None)
    except IndexError as err:
        _logger.warning(err)
        _logger.warning(f"skip {plot_title}")
        pass
    else:

        yticks = axis.get_yticks()
        min_y = yticks[0]
        max_y = yticks[-1]
        y_range = (max_y - min_y)
        axis.set_ylim((min_y, max_y))

        axis.set_title(plot_title)
        axis.set_xlabel("")
        axis.set_ylabel("% enterprises")
        axis.xaxis.grid(False)
        sns.despine(ax=axis, left=True)
        axis.tick_params(which="both", bottom=False)

        if reference_lines is not None:
            color = line_iter.get_next_color()
            for ref_key, ref_line in reference_lines.items():
                ref_label = ref_line["label"]
                ref_plot_df = ref_line["plot_df"]
                value = ref_plot_df.values[0][1]
                color = line_iter.get_next_color()
                axis.axhline(y=value, color=color, linestyle='-.')
                axis.text(xoff, value + yoff * y_range, ref_label, color=color, transform=trans)

        image_name = re.sub("\s", "_", plot_title.replace(" - ", "_"))
        image_name = re.sub(":_.*$", "", image_name)
        image_file = image_directory / Path("_".join([plot_key, image_name + image_type]))
        image_file_name = image_file.as_posix()
        _logger.info(f"Saving plot {image_file_name}")
        fig.savefig(image_file)

        if show_plots:
            plt.show()

        _logger.debug("Done")

    plt.close()

    return image_file_name
