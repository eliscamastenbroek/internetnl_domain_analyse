import logging
import re
from pathlib import Path

import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import matplotlib.transforms as trn
import numpy as np
import pandas as pd
import seaborn as sns
from cbsplotlib.colors import CBS_COLORS_RBG
from cbsplotlib.highcharts import CBSHighChart
from cbsplotlib.settings import CBSPlotSettings
from cbsplotlib.utils import add_axis_label_background

from internetnl_domain_analyse.utils import get_windows_or_linux_value

_logger = logging.getLogger(__name__)
cbsplotlib_logger = logging.getLogger("cbsplotlib")
cbsplotlib_logger.setLevel(_logger.getEffectiveLevel())
sns.set_style("whitegrid")


class AxisLabel:
    """
    class om de eigenschappen van een as label op te slaan
    """

    def __init__(self, label_properties, text_default=None, positie_default=None):
        self.label_properties = label_properties

        self.text = text_default
        self.positie = positie_default

        self.set_properties()
        if self.text is None:
            raise ValueError(
                "Geen text gezet. Geef label eigenschappen of default waarden mee"
            )
        if self.positie is None:
            raise ValueError(
                "Geen positie gezet. Geef label eigenschappen of default waarden mee"
            )

    def set_properties(self):
        if self.label_properties is not None:
            self.text = self.label_properties["text"]
            if position := self.label_properties.get("positie"):
                self.positie = position


def make_cdf_plot(
    hist,
    grp_key,
    plot_key,
    scan_data_key,
    module_name=None,
    question_name=None,
    image_directory=None,
    show_plots=False,
    figsize=None,
    image_type=None,
    image_file_base=None,
    cummulative=False,
    reference_lines=None,
    xoff=None,
    yoff=None,
    y_max=None,
    y_spacing=None,
    translations=None,
    export_highcharts=None,
    export_svg=False,
    highcharts_info: dict = None,
    title: str = None,
    year: int = None,
    english=False,
):
    figure_properties = CBSPlotSettings()

    if figsize is None:
        figsize = figure_properties.fig_size

    counts = hist[0]
    sum_pdf = counts.sum()
    _logger.info(f"Plot pdf gebaseerd op {sum_pdf} bedrijven (door gewichten)")
    bins = hist[1]
    delta_bin = np.diff(bins)[0]
    pdf = 100 * counts / sum_pdf / delta_bin
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.subplots_adjust(bottom=0.25, top=0.92, right=0.98)
    axis.tick_params(which="both", bottom=True)

    cdf = pdf.cumsum() * delta_bin

    if cummulative:
        fnc = cdf
        fnc_str = "cdf"
    else:
        fnc = pdf
        fnc_str = "pdf"

    xgrid = bins[:-1] + delta_bin / 2

    axis.set_xlim((0, 100))
    axis.bar(xgrid, fnc, width=delta_bin, edgecolor=None, linewidth=0)

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
            axis.text(value, 1.02 * pval, f"Q{ii}", color="cbs:appelgroen", ha="center")
    stats_df = pd.DataFrame.from_dict(stats, orient="index", columns=["value"])
    stats_df.index.rename("Stat", inplace=True)

    # this triggers the drawing, otherwise we can not retrieve the xtick labels
    fig.canvas.draw()
    windows_title = f"{grp_key}  {plot_key}"
    try:
        fig.canvas.set_window_title(windows_title)
    except AttributeError as err:
        fig.canvas.manager.set_window_title(windows_title)

    if cummulative:
        if not english:
            y_label = "Cumulatief % van bedrijven met website"
        else:
            y_label = "Cumulative % of companies with met website"

    else:
        if not english:
            y_label = "% van bedrijven met website"
        else:
            y_label = "% of companies with website"

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
    axis.xaxis.set_ticks(axis.get_xticks())
    axis.set_xticklabels(labels, ha="center")

    add_axis_label_background(fig=fig, axes=axis, loc="south")

    plot_title = " - ".join([fnc_str, module_name, question_name, plot_key, grp_key])

    image_name_suffix = "_".join([fnc_str, image_file_base, str(year)])
    image_name = "_".join([scan_data_key, plot_key, image_name_suffix])
    image_name_with_ext = ".".join([image_name, image_type])
    image_file = image_directory / Path(image_name_with_ext)

    fig.savefig(image_file)

    stat_file = image_file.with_suffix(".out").as_posix()
    _logger.info(f"Saving stats to {stat_file}")
    stats_df.to_csv(stat_file)

    highcharts_directory = Path(highcharts_info.get("highcharts_directory"))
    highcharts_directory.mkdir(exist_ok=True, parents=True)
    highcharts_label = highcharts_info.get("highcharts_label")

    if export_svg:
        svg_image_file = highcharts_directory / Path(
            "_".join([plot_key, image_name + ".svg"])
        )
        _logger.info(f"Saving plot to {svg_image_file}")
        fig.savefig(svg_image_file)

    if export_highcharts:

        # voor highcharts de titel setten
        if highcharts_label is not None:
            plot_title = highcharts_label
        hc_df = pd.DataFrame(index=bins[:-1], data=fnc, columns=[fnc_str])
        hc_df.index = hc_df.index.rename(module_name)
        CBSHighChart(
            data=hc_df,
            chart_type="column",
            output_directory=highcharts_directory.as_posix(),
            output_file_name=image_file.stem,
            ylabel=y_label,
            title=plot_title,
            enable_legend=False,
        )

    if show_plots:
        plt.show()

    _logger.debug("Done")

    plt.close()

    return image_name_with_ext


def make_bar_plot_horizontal(
    plot_df,
    fig,
    axis,
    margin,
    plot_title,
    show_title,
    translations,
    reference_lines,
    line_iter,
    xoff,
    yoff,
    trans,
    y_spacing_bar_plot,
    y_max_bar_plot,
    legend_position,
    legend_max_columns,
    unit=None,
    english=False,
    bar_width=None,
):
    x_range = None

    if bar_width is not None:
        kwargs = dict(width=bar_width)
    else:
        kwargs = {}

    try:
        plot_df.plot(kind="barh", ax=axis, rot=0, legend=None, **kwargs)
    except IndexError as err:
        _logger.warning(err)
        _logger.warning(f"skip {plot_title}")
        pass
    else:

        # put the high
        axis.invert_yaxis()

        xticks = axis.get_xticks()
        min_x = xticks[0]
        max_x = xticks[-1]
        x_range = max_x - min_x
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
        if unit is None:
            if not english:
                x_label = "% van bedrijven met website"
            else:
                x_label = "% of companies with website"
        else:
            x_label = unit

        if translations is not None:
            for key_in, label_out in translations.items():
                if label_out is not None and key_in in x_label:
                    _logger.debug(f"Replacing {key_in} -> {label_out}")
                    x_label = x_label.replace(key_in, label_out)

        axis.set_xlabel(x_label, rotation="horizontal", horizontalalignment="right")
        axis.xaxis.set_label_coords(1.01, -0.12)
        axis.yaxis.grid(False)
        sns.despine(ax=axis, bottom=True)
        axis.tick_params(which="both", left=False)

        add_axis_label_background(
            fig=fig, axes=axis, loc="east", radius_corner_in_mm=1, margin=margin
        )

        number_of_columns = plot_df.columns.values.size
        if legend_max_columns is not None and number_of_columns > legend_max_columns:
            number_of_columns = legend_max_columns
        if legend_position is None:
            legend_bbox_to_anchor = (0.02, 0.00)
        else:
            legend_bbox_to_anchor = legend_position

        legend_bbox_to_anchor = get_windows_or_linux_value(legend_bbox_to_anchor)
        axis.legend(
            loc="lower left",
            frameon=False,
            ncol=number_of_columns,
            bbox_to_anchor=legend_bbox_to_anchor,
            bbox_transform=fig.transFigure,
        )

    if reference_lines is not None:
        color = line_iter.get_next_color()
        for ref_key, ref_line in reference_lines.items():
            ref_label = ref_line["label"]
            ref_plot_df = ref_line["plot_df"]
            value = ref_plot_df.values[0][1]
            color = line_iter.get_next_color()
            axis.axhline(y=value, color=color, linestyle="-.")
            axis.text(
                xoff, value + yoff * x_range, ref_label, color=color, transform=trans
            )


def make_bar_plot_vertical(
    plot_df,
    axis,
    plot_title,
    show_title,
    translations,
    reference_lines,
    line_iter,
    xoff,
    yoff,
    trans,
    unit=None,
    english=False,
):
    y_label = ""
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
        y_range = max_y - min_y
        axis.set_ylim((min_y, max_y))

        if show_title:
            axis.set_title(plot_title)
        axis.set_xlabel("")
        if unit is None:
            if not english:
                x_label = "% van bedrijven met website"
            else:
                x_label = "% of companies with website"
        else:
            x_label = unit

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
                axis.axhline(y=value, color=color, linestyle="-.")
                axis.text(
                    xoff,
                    value + yoff * y_range,
                    ref_label,
                    color=color,
                    transform=trans,
                )


def make_bar_plot(
    plot_df,
    plot_key,
    plot_variable,
    scan_data_key,
    module_name,
    question_name,
    image_directory,
    show_plots=False,
    figsize=None,
    highcharts_height=None,
    image_type="pdf",
    reference_lines=None,
    xoff=0.02,
    yoff=0.02,
    show_title=False,
    barh=False,
    subplot_adjust=None,
    sort_values=False,
    y_max_bar_plot=None,
    y_spacing_bar_plot=None,
    translations=None,
    legend_position=None,
    legend_max_columns=None,
    box_margin=None,
    export_svg=False,
    export_highcharts=False,
    highcharts_directory=None,
    title=None,
    normalize_data=False,
    force_plot=False,
    enable_highcharts_legend=True,
    unit=None,
    english=False,
    bar_width=None,
):
    image_name = re.sub("_\d(\.\d){0,1}$", "", plot_variable)
    image_file = image_directory / Path(
        "_".join([scan_data_key, plot_key, ".".join([image_name, image_type])])
    )
    image_file_name = image_file.as_posix()
    if image_file.exists() and not force_plot:
        _logger.info(f"File {image_file_name} already exists. Skipping plot")
        return image_file_name
    """ create a bar plot from the question 'plot_df'"""
    figure_properties = CBSPlotSettings()

    if figsize is None:
        figsize = figure_properties.fig_size

    _logger.debug(f"Figsize: {figsize}")

    names = plot_df.index.names
    plot_title = " - ".join([module_name, question_name])
    plot_df = plot_df.droplevel(names[:3]).T

    # inverteer de volgorde
    plot_df = plot_df[plot_df.columns[::-1]]
    plot_df = plot_df.reindex(plot_df.index[::-1])

    if normalize_data:
        _logger.info("Normalize data")
        plot_df = 100 * plot_df / plot_df.sum(axis=0)

    fig, axis = plt.subplots(figsize=figsize)
    if subplot_adjust is None:
        s_adjust = dict()
    else:
        s_adjust = subplot_adjust
    bottom = s_adjust.get("bottom", 0.15)
    left = s_adjust.get("left", 0.45)
    top = s_adjust.get("top", 0.95)
    right = s_adjust.get("right", 0.95)
    fig.subplots_adjust(bottom=bottom, left=left, top=top, right=right)

    if box_margin is None:
        margin = 0.1
    else:
        margin = box_margin

    line_iter = axis._get_lines
    trans = trn.blended_transform_factory(axis.transAxes, axis.transData)

    x_label = None
    y_label = None
    y_lim = None

    if not barh:
        make_bar_plot_vertical(
            plot_df=plot_df,
            axis=axis,
            plot_title=plot_title,
            show_title=show_title,
            translations=translations,
            reference_lines=reference_lines,
            line_iter=line_iter,
            xoff=xoff,
            yoff=yoff,
            trans=trans,
            unit=unit,
            english=english,
        )

    else:
        make_bar_plot_horizontal(
            plot_df=plot_df,
            fig=fig,
            axis=axis,
            margin=margin,
            plot_title=plot_title,
            show_title=show_title,
            translations=translations,
            reference_lines=reference_lines,
            line_iter=line_iter,
            xoff=xoff,
            yoff=yoff,
            trans=trans,
            y_spacing_bar_plot=y_spacing_bar_plot,
            y_max_bar_plot=y_max_bar_plot,
            legend_position=legend_position,
            legend_max_columns=legend_max_columns,
            unit=unit,
            english=english,
            bar_width=bar_width,
        )

    _logger.info(f"Saving plot {image_file_name}")
    fig.savefig(image_file)

    if highcharts_directory is not None:
        highcharts_directory.mkdir(exist_ok=True, parents=True)
    if export_svg:
        # met export highcharts gaan we ook een svg exporten
        svg_image_file = highcharts_directory / Path(
            "_".join([plot_key, image_name + ".svg"])
        )
        _logger.info(f"Saving plot {svg_image_file}")
        fig.savefig(svg_image_file)

    if export_highcharts:
        if y_max_bar_plot is not None:
            y_lim = (0, y_max_bar_plot)
        else:
            y_lim = None

        if title is not None:
            plot_title = title
        if barh:
            hc_ylabel = x_label
        else:
            hc_ylabel = y_label
        hc_file = "/".join([highcharts_directory.as_posix(), image_file.stem]) + ".json"
        _logger.info(f"Saving highcharts plot to: {hc_file}")
        CBSHighChart(
            data=plot_df,
            chart_type="bar",
            output_directory=highcharts_directory.as_posix(),
            output_file_name=image_file.stem,
            ylabel=hc_ylabel,
            y_lim=y_lim,
            y_tick_interval=y_spacing_bar_plot,
            title=plot_title,
            enable_legend=enable_highcharts_legend,
            chart_height=highcharts_height,
        )

    if show_plots:
        plt.show()

    plt.close()

    return image_file_name


def make_bar_plot_stacked(
    year,
    plot_df,
    plot_key,
    plot_variable,
    scan_data_key,
    module_name,
    question_name,
    image_directory,
    show_plots=False,
    figsize=None,
    image_type="pdf",
    reference_lines=None,
    xoff=0.02,
    yoff=0.02,
    show_title=False,
    barh=False,
    subplot_adjust=None,
    sort_values=False,
    y_max_bar_plot=None,
    y_spacing_bar_plot=None,
    translations=None,
    legend_position=None,
    box_margin=None,
    export_svg=False,
    export_highcharts=False,
    highcharts_directory=None,
    title=None,
    normalize_data=False,
    force_plot=False,
    enable_highcharts_legend=True,
    unit=None,
    english=False,
):
    image_name = re.sub("_\d(\.\d){0,1}$", "", plot_variable)
    image_name_suffix = "_".join([image_name, str(year)])
    image_name = "_".join([scan_data_key, plot_key, image_name_suffix])
    image_name_with_ext = ".".join([image_name, image_type])
    image_file = image_directory / Path(image_name_with_ext)

    image_file_name = image_file.as_posix()
    if image_file.exists() and not force_plot:
        _logger.info(f"File {image_file_name} already exists. Skipping plot")
        return image_file_name
    """ create a bar plot from the question 'plot_df'"""
    figure_properties = CBSPlotSettings()

    if figsize is None:
        figsize = figure_properties.fig_size

    _logger.debug(f"Figsize: {figsize}")

    names = plot_df.index.names
    plot_title = " - ".join([module_name, question_name])
    plot_df = plot_df.droplevel(names[:3]).T

    # inverteer de volgorde
    # plot_df = plot_df[plot_df.columns[::-1]]
    plot_df = plot_df.reindex(plot_df.index[::-1])

    plot_df.dropna(how="all", axis=0, inplace=True)

    if normalize_data:
        _logger.info("Normalize data")
        plot_df = 100 * plot_df / plot_df.sum(axis=0)

    fig, axis = plt.subplots(figsize=figsize)
    if subplot_adjust is None:
        s_adjust = dict()
    else:
        s_adjust = subplot_adjust
    bottom = s_adjust.get("bottom", 0.15)
    left = s_adjust.get("left", 0.45)
    top = s_adjust.get("top", 0.95)
    right = s_adjust.get("right", 0.95)
    fig.subplots_adjust(bottom=bottom, left=left, top=top, right=right)

    if box_margin is None:
        margin = 0.1
    else:
        margin = box_margin

    line_iter = axis._get_lines
    trans = trn.blended_transform_factory(axis.transAxes, axis.transData)

    x_label = None
    y_label = None
    y_lim = None
    x_range = None

    renames = dict()
    for nr, name in translations.items():
        col = re.sub("_\d(\.\d){0,1}$", f"_{nr}.0", plot_variable)
        renames[col] = name

    _logger.debug(f"Translate with {renames}")

    plot_df.rename(columns=renames, inplace=True)

    try:
        plot_df.plot(kind="barh", ax=axis, rot=0, legend=None, stacked=True)
    except IndexError as err:
        _logger.warning(err)
        _logger.warning(f"skip {plot_title}")
        pass
    else:

        # put the high
        axis.invert_yaxis()

        xticks = axis.get_xticks()
        min_x = xticks[0]
        max_x = xticks[-1]
        x_range = max_x - min_x
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
        if unit is None:
            if not english:
                x_label = "% van bedrijven met website"
            else:
                x_label = "% of companies with website"
        else:
            x_label = unit

        axis.set_xlabel(x_label, rotation="horizontal", horizontalalignment="right")
        axis.xaxis.set_label_coords(1.01, -0.12)
        axis.yaxis.grid(False)
        sns.despine(ax=axis, bottom=True)
        axis.tick_params(which="both", left=False)

        add_axis_label_background(
            fig=fig, axes=axis, loc="east", radius_corner_in_mm=1, margin=margin
        )

        number_of_columns = plot_df.columns.values.size
        if legend_position is None:
            legend_bbox_to_anchor = (0.02, 0.00)
        else:
            legend_bbox_to_anchor = legend_position
        legend_bbox_to_anchor = get_windows_or_linux_value(legend_bbox_to_anchor)
        axis.legend(
            loc="lower left",
            frameon=False,
            ncol=number_of_columns,
            bbox_to_anchor=legend_bbox_to_anchor,
            bbox_transform=fig.transFigure,
        )

    if reference_lines is not None:
        color = line_iter.get_next_color()
        for ref_key, ref_line in reference_lines.items():
            ref_label = ref_line["label"]
            ref_plot_df = ref_line["plot_df"]
            value = ref_plot_df.values[0][1]
            color = line_iter.get_next_color()
            axis.axhline(y=value, color=color, linestyle="-.")
            axis.text(
                xoff, value + yoff * x_range, ref_label, color=color, transform=trans
            )

    _logger.info(f"Saving plot {image_file_name}")
    fig.savefig(image_file)

    if highcharts_directory is not None:
        highcharts_directory.mkdir(exist_ok=True, parents=True)
    if export_svg:
        # met export highcharts gaan we ook een svg exporten
        svg_image_file = highcharts_directory / Path(
            "_".join([plot_key, image_name + ".svg"])
        )
        _logger.info(f"Saving plot {svg_image_file}")
        fig.savefig(svg_image_file)

    if export_highcharts:
        if y_max_bar_plot is not None:
            y_lim = (0, y_max_bar_plot)
        else:
            y_lim = None

        if title is not None:
            plot_title = title
        if barh:
            hc_ylabel = x_label
        else:
            hc_ylabel = y_label
        hc_file = "/".join([highcharts_directory.as_posix(), image_file.stem]) + ".json"
        _logger.info(f"Saving highcharts plot to: {hc_file}")
        CBSHighChart(
            data=plot_df,
            chart_type="bar_stacked",
            output_directory=highcharts_directory.as_posix(),
            output_file_name=image_file.stem,
            ylabel=hc_ylabel,
            y_lim=y_lim,
            y_tick_interval=y_spacing_bar_plot,
            title=plot_title,
            enable_legend=enable_highcharts_legend,
        )

    if show_plots:
        plt.show()

    plt.close()

    return image_file_name


def make_conditional_score_plot(
    correlations,
    image_directory,
    show_plots=False,
    figsize=None,
    image_type=".pdf",
    export_svg=False,
    export_highcharts=False,
    highcharts_directory=None,
    title=None,
    cache_directory=None,
    english=False,
):
    plot_info = correlations["plots"]

    index_labels = correlations["index_labels"]
    categories = correlations["index_categories"]
    score_intervallen = correlations["score_intervallen"]

    for plot_key, plot_prop in plot_info.items():

        # we maken hier alleen de score plots
        if plot_key not in (
            "scores_per_interval",
            "scores_per_number_correct",
        ) or not plot_prop.get("do_it", True):
            continue

        outfile = Path(plot_prop["output_file"])
        if cache_directory is not None:
            outfile = Path(cache_directory) / outfile
        in_file = outfile.with_suffix(".pkl")

        if highcharts_directory is None:
            hc_dir = Path(".")
        else:
            hc_dir = Path(highcharts_directory)

        if hc_sub_dir := plot_prop.get("highcharts_output_directory"):
            hc_dir = hc_dir / Path(hc_sub_dir)

        if hc_label := plot_prop.get("highcharts_label"):
            label = hc_label
        else:
            label = "Leeg"

        _logger.info(f"Reading scores from {in_file}")
        scores = pd.read_pickle(in_file.with_suffix(".pkl"))

        if plot_key == "scores_per_interval":
            x_label = AxisLabel(
                plot_prop.get("x_label"),
                text_default="Eindscoreniveau",
                positie_default=(0.98, -0.15),
            )
            y_label = AxisLabel(
                plot_prop.get("y_label"),
                text_default="Subgroepscore [%]",
                positie_default=(-0.065, 1.07),
            )

            im_file_base = "_".join([outfile.stem, "per_score_interval"])
            im_file = image_directory / Path(im_file_base).with_suffix(".pdf")
            plot_score_per_interval(
                scores=scores,
                score_intervallen=score_intervallen,
                index_labels=index_labels,
                categories=categories,
                highcharts_directory=hc_dir,
                im_file=im_file,
                show_plots=show_plots,
                plot_title=label,
                x_label=x_label,
                y_label=y_label,
                english=english,
            )
        elif plot_key == "scores_per_number_correct":
            x_label = AxisLabel(
                plot_prop.get("x_label"),
                text_default="Aantal geslaagde categorieën",
                positie_default=(0.98, -0.15),
            )
            y_label = AxisLabel(
                plot_prop.get("y_label"),
                text_default="Subgroepscore [%]",
                positie_default=(-0.065, 1.07),
            )

            im_file_base = "_".join([outfile.stem, "per_count_interval"])
            im_file = image_directory / Path(im_file_base).with_suffix(".pdf")
            plot_score_per_count(
                scores=scores,
                categories=categories,
                highcharts_directory=hc_dir,
                im_file=im_file,
                show_plots=show_plots,
                plot_title=label,
                x_label=x_label,
                y_label=y_label,
                english=english,
            )


def plot_score_per_count(
    scores,
    categories,
    highcharts_directory,
    im_file,
    show_plots,
    plot_title,
    x_label,
    y_label,
    english=False,
):
    _logger.info("Plot score per count")
    # add a new columns with the interval label belonging to the gk code bin. Note that we
    # merge all the grootte klass below 40 to a group smaller than 10

    score_per_category = dict()
    for categorie_key, category_df in scores.groupby("count"):
        _logger.debug(f"Plotting {categorie_key}")
        df = category_df[list(categories.keys())]
        score_per_category[categorie_key] = df.mean()

    score_per_category_df = pd.DataFrame(score_per_category).T * 100
    score_per_category_df = score_per_category_df.round(1)

    settings = CBSPlotSettings(color_palette="koelextended")
    fig, axis = plt.subplots()
    fig.subplots_adjust(bottom=0.3, top=0.90)
    score_per_category_df.plot.bar(
        ax=axis, rot=0, stacked=False, edgecolor="white", linewidth=1.5
    )
    yticks = axis.get_yticks()
    # axis.set_ylim((yticks[0], yticks[-1]))
    axis.set_ylim((0, 100))

    axis.set_xlabel(x_label.text, rotation="horizontal", horizontalalignment="right")
    axis.xaxis.set_label_coords(*x_label.positie)

    axis.set_ylabel(y_label.text, rotation="horizontal", horizontalalignment="left")
    axis.yaxis.set_label_coords(*y_label.positie)
    axis.xaxis.grid(False)
    sns.despine(ax=axis, left=True)

    # niet meer volgens de richtlijnen
    # add_values_to_bars(axis=axis, color="w")

    sns.despine(ax=axis, left=True)

    axis.tick_params(which="both", bottom=False)

    add_axis_label_background(fig=fig, axes=axis, loc="south")

    ncol = (score_per_category_df.columns.size - 1) // 2 + 1

    legend = axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.105, -0.00),
        frameon=False,
        bbox_transform=fig.transFigure,
        ncol=ncol,
    )

    _logger.info(f"Writing score plot to {im_file}")
    fig.savefig(im_file.as_posix())

    highcharts_directory.mkdir(exist_ok=True, parents=True)

    CBSHighChart(
        data=score_per_category_df,
        chart_type="column_grouped",
        output_directory=highcharts_directory.as_posix(),
        output_file_name=im_file.stem,
        y_lim=(0, 100),
        ylabel=y_label.text,
        title=plot_title,
        enable_legend=True,
    )

    if show_plots:
        plt.show()

    _logger.debug("Klaar")


def plot_score_per_interval(
    scores,
    score_intervallen,
    index_labels,
    categories,
    highcharts_directory,
    im_file,
    show_plots,
    plot_title,
    x_label,
    y_label,
    english=False,
):
    score_labels = list(score_intervallen.keys())
    score_bins = list([s / 100 for s in score_intervallen.values()]) + [1.01]
    # add a new columns with the interval label belonging to the gk code bin. Note that we
    # merge all the grootte klass below 40 to a group smaller than 10
    scores["score_category"] = pd.cut(
        scores["score"],
        bins=score_bins,
        labels=score_labels,
        right=True,
        include_lowest=True,
    )

    score_per_category = dict()
    for categorie_key, category_df in scores.groupby("score_category"):
        _logger.debug(f"Plotting {categorie_key}")
        df = category_df[list(categories.keys())]
        category_label = index_labels[categorie_key]
        score_per_category[category_label] = df.mean()

    score_per_category_df = pd.DataFrame(score_per_category).T * 100
    score_per_category_df = score_per_category_df.round(1)

    settings = CBSPlotSettings(color_palette="koelextended")
    fig, axis = plt.subplots()
    fig.subplots_adjust(bottom=0.3, top=0.90)
    score_per_category_df.plot.bar(
        ax=axis, rot=0, stacked=False, edgecolor="white", linewidth=1.5
    )
    yticks = axis.get_yticks()
    # axis.set_ylim((yticks[0], yticks[-1]))
    axis.set_ylim((0, 100))

    axis.set_xlabel(x_label.text, rotation="horizontal", horizontalalignment="right")
    axis.xaxis.set_label_coords(*x_label.positie)

    axis.set_ylabel(y_label.text, rotation="horizontal", horizontalalignment="left")
    axis.yaxis.set_label_coords(*y_label.positie)
    axis.xaxis.grid(False)
    sns.despine(ax=axis, left=True)

    # niet meer volgens de richtlijnen
    # add_values_to_bars(axis=axis, color="w")

    sns.despine(ax=axis, left=True)

    axis.tick_params(which="both", bottom=False)

    add_axis_label_background(fig=fig, axes=axis, loc="south")

    ncol = (score_per_category_df.columns.size - 1) // 2 + 1

    legend = axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.105, -0.00),
        frameon=False,
        bbox_transform=fig.transFigure,
        ncol=ncol,
    )

    _logger.info(f"Writing score plot to {im_file}")
    fig.savefig(im_file.as_posix())

    highcharts_directory.mkdir(exist_ok=True, parents=True)

    CBSHighChart(
        data=score_per_category_df,
        chart_type="column_grouped",
        output_directory=highcharts_directory.as_posix(),
        output_file_name=im_file.stem,
        y_lim=(0, 100),
        ylabel=y_label.text,
        title=plot_title,
        enable_legend=True,
    )

    if show_plots:
        plt.show()

    _logger.debug("Klaar")


# fig, axis = plt.subplots(figsize=(10, 10))
# cbar_ax = fig.add_axes([.91, .315, .02, .62])
# cmap = sns.color_palette("deep", 10)


def make_heatmap(
    correlations,
    image_directory,
    show_plots=False,
    figsize=None,
    image_type=".pdf",
    export_svg=False,
    export_highcharts=False,
    highcharts_directory=None,
    title=None,
    cache_directory=None,
    english=False,
):
    plot_properties = correlations["plots"]["correlation"]
    outfile = Path(plot_properties["output_file"])
    if cache_directory is not None:
        outfile = Path(cache_directory) / outfile

    in_file = outfile.with_suffix(".pkl")

    if highcharts_directory is None:
        hc_dir = Path(".")
    else:
        hc_dir = Path(highcharts_directory)

    if hc_sub_dir := plot_properties.get("highcharts_output_directory"):
        hc_dir = highcharts_directory / Path(hc_sub_dir)

    _logger.info(f"Reading correlation from {in_file}")
    corr = pd.read_pickle(in_file.with_suffix(".pkl"))

    categories = correlations["index_categories"]
    corr_index = correlations["index_correlations"]
    corr = corr.reindex(list(corr_index.keys()))
    corr = corr[list(corr_index.keys())]

    sns.set(font_scale=0.8)
    # cmap is now a list of colors
    cmap = mpc.ListedColormap(
        sns.cubehelix_palette(start=2.8, rot=0.1, light=0.9, n_colors=12)
    )

    # Create two appropriately sized subplots
    # grid_kws = {'width_ratios': (0.9, 0.03), 'wspace': 0.18}
    # fig, (axis, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(8.3, 8.3))

    im_file = image_directory / Path(outfile.stem).with_suffix(".pdf")
    fig, axis = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0.28, bottom=0.27, top=0.98, right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.315, 0.02, 0.62])
    # cmap = sns.color_palette("deep", 10)

    sns.heatmap(
        corr,
        square=True,
        ax=axis,
        cbar_ax=cbar_ax,
        cmap=cmap,
        vmin=-0.2,
        vmax=1.0,
        cbar_kws={"orientation": "vertical", "label": r"Correlatiecoëfficiënt $\rho$"},
    )
    xlabels = axis.get_xticklabels()
    ylabels = axis.get_yticklabels()
    for xlbl, ylbl in zip(xlabels, ylabels):
        tekst = xlbl.get_text()
        categorie = corr_index[tekst]
        categorie_properties = categories[categorie]
        kleur = categorie_properties["color"]
        RGB = CBS_COLORS_RBG.get(kleur, [0, 0, 0])
        rgb = [_ / 255 for _ in RGB]
        tekst_clean = tekst.replace("_verdict", "").replace("tests_", "")
        xlbl.set_text(tekst_clean)
        xlbl.set_color(rgb)
        ylbl.set_text(tekst_clean)
        ylbl.set_color(rgb)

    axis.set_xticklabels(xlabels, rotation=90, ha="right")
    axis.set_yticklabels(ylabels, rotation=0, ha="right")

    plt.legend(loc="upper left", prop={"size": 10})

    _logger.info(f"Writing heatmap to {im_file}")
    fig.savefig(im_file.as_posix())

    hc_dir.mkdir(exist_ok=True, parents=True)

    hc_out = hc_dir / Path(im_file.stem + ".svg")

    _logger.info(f"Writing heatmap to {hc_out}")
    fig.savefig(hc_out.as_posix())

    if show_plots:
        plt.show()


def make_conditional_pdf_plot(
    categories,
    image_directory,
    show_plots=False,
    export_highcharts=False,
    highcharts_directory=None,
    cache_directory=None,
    english=False,
):
    outfile = Path(categories["categories_output_file"])
    if cache_directory is not None:
        outfile = Path(cache_directory) / outfile

    image_key = "pdf_per_category"
    plot_settings = categories["plot_settings"]["pdf_per_category"]
    y_max = plot_settings.get("y_max_pdf_plot")
    y_spacing = plot_settings.get("y_spacing_pdf_plot")
    export_svg = plot_settings.get("export_svg")

    in_file = outfile.with_suffix(".pkl")

    if highcharts_directory is None:
        hc_dir = Path(".")
    else:
        hc_dir = Path(highcharts_directory)

    if hc_sub_dir := plot_settings.get("highcharts_output_directory"):
        hc_dir = hc_dir / Path(hc_sub_dir)

    hc_dir.mkdir(exist_ok=True, parents=True)

    _logger.info(f"Reading correlation from {in_file}")
    conditional_scores_df = pd.read_pickle(in_file.with_suffix(".pkl"))

    im_file = image_directory / Path(outfile.stem).with_suffix(".pdf")

    figure_properties = CBSPlotSettings()

    fig, axis = plt.subplots()
    axis.tick_params(which="both", bottom=True)
    delta_bin = np.diff(conditional_scores_df.index)[0]

    fig.subplots_adjust(bottom=0.25, top=0.92, right=0.98)
    axis.tick_params(which="both", bottom=True)

    conditional_scores_df.index = conditional_scores_df.index + delta_bin / 2

    for col_name in conditional_scores_df.columns:
        pdf = 100 * conditional_scores_df[col_name].to_numpy()
        axis.bar(conditional_scores_df.index, pdf, width=delta_bin, label=col_name)

    xtics = np.linspace(0, 100, endpoint=True, num=6)
    _logger.debug(xtics)
    _logger.debug(conditional_scores_df.index)
    axis.xaxis.set_ticks(xtics)
    axis.set_xlim((-5, 105))

    start, end = axis.get_ylim()
    if y_max is not None:
        end = y_max
    if y_spacing is not None:
        axis.yaxis.set_ticks(np.arange(start, end + 1, y_spacing))

    if y_max is not None:
        axis.set_ylim((0, y_max))

    # this triggers the drawing, otherwise we can not retrieve the xtick labels
    fig.canvas.draw()

    #    y_label = '% van bedrijven met website'
    y_label = "% of companies with website"

    axis.set_ylabel(y_label, rotation="horizontal", horizontalalignment="left")
    axis.yaxis.set_label_coords(-0.04, 1.05)
    axis.xaxis.grid(False)
    axis.set_xlabel("Eindscore", horizontalalignment="right")
    axis.xaxis.set_label_coords(0.98, -0.12)
    sns.despine(ax=axis, left=True)

    labels = [_.get_text() for _ in axis.get_xticklabels()]
    axis.set_xticklabels(labels, ha="center")

    add_axis_label_background(fig=fig, axes=axis, loc="south", margin=0.10)

    if not english:
        plot_title = "Aantal geslaagde categorieën"
        hc_plot_title = "Verdeling scores per categorie"
    else:
        plot_title = "Number of succeeded categories "
        hc_plot_title = "Distribution of scores per category"

    legend = axis.legend(
        loc="lower left",
        title=plot_title,
        prop={"size": 10},
        bbox_to_anchor=(0.2, 0.02),
        frameon=False,
        bbox_transform=fig.transFigure,
        ncol=5,
    )

    legend._legend_box.align = "left"
    for patch in legend.get_patches():
        patch.set_linewidth(0)

    _logger.info(f"Saving plot {im_file}")
    fig.savefig(im_file)

    if export_svg:
        svg_image_file = hc_dir / Path(im_file.stem).with_suffix(".svg")
        _logger.info(f"Saving plot to {svg_image_file}")
        svg_image_file.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(svg_image_file.as_posix())

    if export_highcharts:
        # voor highcharts de titel setten
        CBSHighChart(
            data=conditional_scores_df,
            chart_type="column",
            output_directory=hc_dir.as_posix(),
            output_file_name=im_file.stem,
            ylabel=y_label,
            title=hc_plot_title,
            enable_legend=False,
        )

    if show_plots:
        plt.show()

    _logger.debug("Done")

    plt.close()


def make_verdeling_per_aantal_categorie(
    categories,
    image_directory,
    show_plots=False,
    export_highcharts=False,
    highcharts_directory=None,
    cache_directory=None,
    english=False,
):
    outfile = Path(categories["categories_output_file"])
    if cache_directory is not None:
        outfile = Path(cache_directory) / outfile

    image_key = "verdeling_per_categorie"
    plot_settings = categories["plot_settings"][image_key]
    y_max = plot_settings.get("y_max_pdf_plot")
    y_spacing = plot_settings.get("y_spacing_pdf_plot")
    export_svg = plot_settings.get("export_svg")

    index_categories = categories["index_categories"]
    renames = dict()
    for index_key, index_prop in index_categories.items():
        variable_name = index_prop["variable"]
        renames[variable_name] = index_key

    in_file = outfile.with_suffix(".pkl")
    sum_file = in_file.parent / Path(in_file.stem + "_sum.pkl")
    _logger.info(f"Reading from {sum_file}")
    sum_per_number_of_cat_df = pd.read_pickle(sum_file)
    sum_per_number_of_cat_df.rename(columns=renames, inplace=True)
    # zet de volgorde gelijk aan de settings file
    sum_per_number_of_cat_df = sum_per_number_of_cat_df[list(index_categories.keys())]

    sum_per_number_of_cat_df = sum_per_number_of_cat_df.T
    sum_per_number_of_cat_df.drop(0, axis=1, inplace=True)

    sum_of_all_categories = sum_per_number_of_cat_df.sum()

    percentage_per_number_of_cat = (
        100 * sum_per_number_of_cat_df / sum_of_all_categories
    )

    x_label = "undefined x"
    y_label = "undefined y"
    if hc_title := plot_settings.get("highcharts_label"):
        title = hc_title
        hc_plot_title = "undefined"
        l_label = "undefined"
    else:
        if not english:
            x_label = "Aantal geslaagde categorieën"
            y_label = "Aandeel per categorie [%]"
            l_label = "Categorie"
            hc_plot_title = "Verdeling scores per categorie"
        else:
            x_label = "Number of succeeded categories"
            y_label = "Part per category [%]"
            l_label = "Category"
            hc_plot_title = "Distribution of scores per category"

    if highcharts_directory is None:
        hc_dir = Path(".")
    else:
        hc_dir = Path(highcharts_directory)

    if hc_sub_dir := plot_settings.get("highcharts_output_directory"):
        hc_dir = hc_dir / Path(hc_sub_dir)

    hc_dir.mkdir(exist_ok=True, parents=True)

    im_file = image_directory / Path("_".join([outfile.stem, image_key])).with_suffix(
        ".pdf"
    )

    figure_properties = CBSPlotSettings()

    fig, axis = plt.subplots()
    axis.tick_params(which="both", bottom=True)
    fig.subplots_adjust(bottom=0.25, top=0.92, right=0.98)

    percentage_per_number_of_cat.T.plot.bar(stacked=True, ax=axis)

    axis.set_ylim((0, 101))

    axis.set_xlabel(x_label, horizontalalignment="right")

    axis.set_ylabel(y_label, rotation="horizontal", horizontalalignment="left")
    axis.yaxis.set_label_coords(-0.06, 1.05)
    axis.xaxis.grid(False)
    axis.xaxis.set_label_coords(0.98, -0.1)
    xlabels = axis.get_xticklabels()
    axis.set_xticklabels(xlabels, rotation=0, ha="right")
    sns.despine(ax=axis, left=True)

    legend = axis.legend(
        loc="lower left",
        title=l_label,
        bbox_to_anchor=(0.2, 0.03),
        frameon=False,
        bbox_transform=fig.transFigure,
        ncol=5,
    )

    legend._legend_box.align = "left"
    for patch in legend.get_patches():
        patch.set_linewidth(0)

    axis.tick_params(which="both", bottom=False)
    add_axis_label_background(fig=fig, axes=axis, loc="south", margin=0.02)

    _logger.info(f"Saving plot {im_file}")
    fig.savefig(im_file)

    if export_svg:
        svg_image_file = hc_dir / Path(im_file.stem).with_suffix(".svg")
        _logger.info(f"Saving plot to {svg_image_file}")
        fig.savefig(svg_image_file)

    if export_highcharts:
        # voor highcharts de titel setten
        CBSHighChart(
            data=percentage_per_number_of_cat.T,
            chart_type="column_stacked_percentage",
            output_directory=hc_dir.as_posix(),
            output_file_name=im_file.stem,
            y_lim=(0, 100),
            title=hc_plot_title,
            xlabel=x_label,
            ylabel=y_label,
            enable_legend=True,
        )

    if show_plots:
        plt.show()

    _logger.debug("Done")

    plt.close()
