import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as trn
import numpy as np
import pandas as pd
import seaborn as sns
from cbsplotlib import LOGGER_BASE_NAME
from cbsplotlib.settings import CBSPlotSettings
from cbsplotlib.utils import add_axis_label_background
from cbsplotlib.highcharts import CBSHighChart

_logger = logging.getLogger(__name__)
cbsplotlib_logger = logging.getLogger(LOGGER_BASE_NAME)
cbsplotlib_logger.setLevel(_logger.getEffectiveLevel())
sns.set_style('whitegrid')


def make_cdf_plot(hist,
                  grp_key,
                  plot_key,
                  module_name=None,
                  question_name=None,
                  image_directory=None,
                  show_plots=False,
                  figsize=None,
                  image_type=None,
                  cummulative=False,
                  reference_lines=None,
                  xoff=None,
                  yoff=None,
                  y_max=None,
                  y_spacing=None,
                  translations=None,
                  export_highcharts=None,
                  highcharts_directory: str = None,
                  title: str = None
                  ):
    figure_properties = CBSPlotSettings()

    if figsize is None:
        figsize = figure_properties.fig_size

    counts = hist[0]
    sum_pdf = counts.sum()
    _logger.info(f"Plot pdf gebaseerd op {sum_pdf} bedrijven (door gewichten)")
    pdf = 100 * counts / sum_pdf
    bins = hist[1]
    fig, axis = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(bottom=0.25, top=0.92, right=0.98)
    axis.tick_params(which="both", bottom=True)

    cdf = pdf.cumsum()
    delta_bin = np.diff(bins)[0]

    if cummulative:
        fnc = cdf
        fnc_str = "cdf"
    else:
        fnc = pdf
        fnc_str = "pdf"

    axis.bar(bins[:-1], fnc, width=delta_bin, edgecolor=None, linewidth=0)

    start, end = axis.get_ylim()
    if y_max is not None:
        end = y_max
    if cummulative:
        axis.yaxis.set_ticks(np.arange(start, end, 25))
    elif y_spacing is not None:
        axis.yaxis.set_ticks(np.arange(start, end + 1, y_spacing))

    if y_max is not None:
        axis.set_ylim((0, y_max))

    stats = dict()
    stats["mean"] = (pdf * bins[:-1]).sum()
    for ii, percentile in enumerate([0, 25, 50, 75, 100]):
        below = cdf < percentile
        if below.all():
            index = cdf.size - 1
        else:
            index = np.argmax(np.diff(cdf < percentile))
        if cummulative:
            pval = fnc[index]
        else:
            if y_max is None:
                pval = end
            else:
                pval = y_max
        value = (index + 1) * delta_bin
        stats[f"p{percentile}"] = value
        _logger.info(f"Adding line {percentile}: {value} {pval}")
        if 0 < percentile < 100:
            axis.vlines(value, 0, pval, color="cbs:appelgroen")
            axis.text(value, 1.02 * pval, f"Q{ii}", color="cbs:appelgroen",
                      ha="center")
    stats_df = pd.DataFrame.from_dict(stats, orient="index", columns=["value"])
    stats_df.index.rename("Stat", inplace=True)

    # this triggers the drawing, otherwise we can not retrieve the xtick labels
    fig.canvas.draw()
    fig.canvas.set_window_title(f"{grp_key}  {plot_key}")

    if cummulative:
        y_label = "Cumulatief % bedrijven"
    else:
        y_label = "% bedrijven"

    if translations is not None:
        for key_in, label_out in translations.items():
            if label_out is not None and key_in in y_label:
                _logger.debug(f"Replacing {key_in} -> {label_out}")
                y_label = y_label.replace(key_in, label_out)
            if label_out is not None and key_in in module_name:
                _logger.debug(f"Replacing {key_in} -> {label_out}")
                module_name = module_name.replace(key_in, label_out)

    axis.set_ylabel(y_label, rotation="horizontal", horizontalalignment="left")
    axis.yaxis.set_label_coords(-0.04, 1.05)
    axis.xaxis.grid(False)
    axis.set_xlabel(module_name, horizontalalignment="right")
    axis.xaxis.set_label_coords(0.95, -0.15)
    sns.despine(ax=axis, left=True)

    labels = [_.get_text() for _ in axis.get_xticklabels()]
    axis.set_xticklabels(labels, ha='center')

    add_axis_label_background(fig=fig, axes=axis, loc="south")

    if title is not None:
        plot_title = title
    else:
        plot_title = " - ".join([fnc_str, module_name, question_name, plot_key, grp_key])
    image_name = re.sub("\s", "_", plot_title.replace(" - ", "_"))
    image_name = re.sub(":_.*$", "", image_name)
    image_file = image_directory / Path("_".join([plot_key, image_name + image_type]))
    image_file_name = image_file.as_posix()
    _logger.info(f"Saving plot {image_file_name}")
    fig.savefig(image_file)

    stat_file = image_file.with_suffix(".out").as_posix()
    _logger.info(f"Saving stats to {stat_file}")
    stats_df.to_csv(stat_file)
    if export_highcharts:
        hc_df = pd.DataFrame(index=bins[:-1], data=fnc, columns=[fnc_str])
        hc_df = hc_df.reindex(hc_df.index[::-1])
        hc_df.index = hc_df.index.rename(module_name)
        CBSHighChart(
            data=hc_df,
            chart_type="column",
            output_directory=highcharts_directory,
            output_file_name=image_file.stem,
            ylabel=y_label,
            title=plot_title,
            enable_legend=False,
        )

    if show_plots:
        plt.show()

    _logger.debug("Done")

    plt.close()

    return image_file_name


def make_bar_plot(plot_df, plot_key, module_name, question_name, image_directory, show_plots=False,
                  figsize=None, image_type=".pdf", reference_lines=None, xoff=0.02, yoff=0.02,
                  show_title=False, barh=False, subplot_adjust=None, sort_values=False,
                  y_max_bar_plot=None, y_spacing_bar_plot=None, translations=None,
                  export_highcharts=False, highcharts_directory=None, title=None
                  ):
    """ create a bar plot from the question 'plot_df'"""
    figure_properties = CBSPlotSettings()

    if figsize is None:
        figsize = figure_properties.fig_size

    _logger.debug(f"Figsize: {figsize}")

    names = plot_df.index.names
    plot_df.reset_index(inplace=True)
    if title is not None:
        plot_title = title
    else:
        plot_title = " - ".join([module_name, question_name])
        result = plot_df.loc[0, names[2]]
        if result == "":
            result = "True"
        plot_title += f": {result}"
    values_column = "Values"
    plot_df.index.rename(values_column, inplace=True)
    plot_df[plot_title] = None
    plot_df.drop(names + ["variable"], axis=1, inplace=True)
    plot_df.set_index(plot_title, inplace=True)
    plot_df.index = range(plot_df.index.size)
    plot_df = plot_df.T
    plot_df.rename(columns={0: values_column}, inplace=True)

    if sort_values:
        plot_df.sort_values(by=[values_column], inplace=True, ascending=True)

    fig, axis = plt.subplots(figsize=figsize)
    if subplot_adjust is None:
        subplot_adjust = dict()
    bottom = subplot_adjust.get("bottom", 0.15)
    left = subplot_adjust.get("left", 0.45)
    top = subplot_adjust.get("top", 0.95)
    right = subplot_adjust.get("right", 0.95)
    fig.subplots_adjust(bottom=bottom, left=left, top=top, right=right)

    line_iter = axis._get_lines
    trans = trn.blended_transform_factory(axis.transAxes, axis.transData)

    x_label = None
    y_label = None

    if not barh:

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

            if show_title:
                axis.set_title(plot_title)
            axis.set_xlabel("")
            if re.search("score", plot_title, re.IGNORECASE):
                y_label = "Score %"
            else:
                y_label = "% bedrijven"

            if translations is not None:
                for key_in, label_out in translations.items():
                    if label_out is not None and key_in in y_label:
                        _logger.debug(f"Replacing {key_in} -> {label_out}")
                        y_label = y_label.replace(key_in, label_out)

            axis.set_ylabel(y_label, rotation="horizontal", horizontalalignment="left")
            axis.yaxis.set_label_coords(-0.04, 1.05)
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

    else:
        try:
            plot_df.plot(kind="barh", ax=axis, rot=0, legend=None)
        except IndexError as err:
            _logger.warning(err)
            _logger.warning(f"skip {plot_title}")
            pass
        else:

            xticks = axis.get_xticks()
            min_x = xticks[0]
            max_x = xticks[-1]
            x_range = (max_x - min_x)
            if y_max_bar_plot is not None:
                axis.set_xlim((0, y_max_bar_plot))
            else:
                axis.set_xlim((min_x, max_x + 1))
            start, end = axis.get_xlim()
            if y_spacing_bar_plot is not None:
                axis.xaxis.set_ticks(np.arange(start, end + 1, y_spacing_bar_plot))

            if show_title:
                axis.set_title(plot_title)
            axis.set_ylabel("")
            if re.search("score", plot_title, re.IGNORECASE):
                x_label = "Score %"
            else:
                x_label = "% bedrijven"

            if translations is not None:
                for key_in, label_out in translations.items():
                    if label_out is not None and key_in in x_label:
                        logger.debug(f"Replacing {key_in} -> {label_out}")
                        x_label = x_label.replace(key_in, label_out)

            axis.set_xlabel(x_label, rotation="horizontal", horizontalalignment="right")
            axis.xaxis.set_label_coords(1.01, -0.12)
            axis.yaxis.grid(False)
            sns.despine(ax=axis, bottom=True)
            axis.tick_params(which="both", left=False)

            add_axis_label_background(fig=fig, axes=axis, loc="east", radius_corner_in_mm=1,
                                      margin=0.1)

            if reference_lines is not None:
                color = line_iter.get_next_color()
                for ref_key, ref_line in reference_lines.items():
                    ref_label = ref_line["label"]
                    ref_plot_df = ref_line["plot_df"]
                    value = ref_plot_df.values[0][1]
                    color = line_iter.get_next_color()
                    axis.axhline(y=value, color=color, linestyle='-.')
                    axis.text(xoff, value + yoff * x_range, ref_label, color=color, transform=trans)

    image_name = re.sub("\s", "_", plot_title.replace(" - ", "_"))
    image_name = re.sub(":_.*$", "", image_name)
    image_file = image_directory / Path("_".join([plot_key, image_name + image_type]))
    image_file_name = image_file.as_posix()
    _logger.info(f"Saving plot {image_file_name}")
    fig.savefig(image_file)

    if export_highcharts:
        if barh:
            hc_ylabel = x_label
        else:
            hc_ylabel = y_label
        _logger.info(f"Saving plot to highcharts")
        plot_df = plot_df.reindex(plot_df.index[::-1])
        CBSHighChart(
            data=plot_df,
            chart_type="bar",
            output_directory=highcharts_directory,
            output_file_name=image_file.stem,
            ylabel=hc_ylabel,
            title=plot_title,
            enable_legend=False
        )


    if show_plots:
        plt.show()

    plt.close()

    return image_file_name
