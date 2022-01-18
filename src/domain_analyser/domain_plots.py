import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as trn
import matplotlib.colors as mpc
import numpy as np
import pandas as pd
import seaborn as sns
from cbsplotlib import LOGGER_BASE_NAME
from cbsplotlib.settings import CBSPlotSettings
from cbsplotlib.utils import add_axis_label_background
from cbsplotlib.highcharts import CBSHighChart
from cbsplotlib.colors import CBS_COLORS_RBG

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
                  export_svg=False,
                  highcharts_directory: Path = None,
                  title: str = None
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
    fig, axis = plt.subplots(nrows=1, ncols=1)
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
    highcharts_directory.mkdir(exist_ok=True, parents=True)

    if export_svg:
        svg_image_file = highcharts_directory / Path("_".join([plot_key, image_name + ".svg"]))
        _logger.info(f"Saving plot to {svg_image_file}")
        fig.savefig(svg_image_file)

    if export_highcharts:

        # voor highcharts de titel setten
        if title is not None:
            plot_title = title
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

    return image_file_name


def make_bar_plot(plot_df, plot_key, module_name, question_name, image_directory, show_plots=False,
                  figsize=None, image_type=".pdf", reference_lines=None, xoff=0.02, yoff=0.02,
                  show_title=False, barh=False, subplot_adjust=None, sort_values=False,
                  y_max_bar_plot=None, y_spacing_bar_plot=None, translations=None,
                  export_svg=False,
                  export_highcharts=False,
                  highcharts_directory=None,
                  title=None
                  ):
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
    values_column = "Values"
    plot_df.index.rename(values_column, inplace=True)
    plot_df[plot_title] = None
    plot_variable = plot_df["variable"].to_numpy()[0]
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
                        _logger.debug(f"Replacing {key_in} -> {label_out}")
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

    # image_name = re.sub("\s", "_", plot_title.replace(" - ", "_"))
    # image_name = re.sub(":_.*$", "", image_name)
    image_name = re.sub("_\d(\.\d){0,1}$", "", plot_variable)
    image_file = image_directory / Path("_".join([plot_key, image_name + image_type]))
    image_file_name = image_file.as_posix()
    _logger.info(f"Saving plot {image_file_name}")
    fig.savefig(image_file)

    if highcharts_directory is not None:
        highcharts_directory.mkdir(exist_ok=True, parents=True)
    if export_svg:
        # met export highcharts gaan we ook een svg exporten
        svg_image_file = highcharts_directory / Path("_".join([plot_key, image_name + ".svg"]))
        _logger.info(f"Saving plot {svg_image_file}")
        fig.savefig(svg_image_file)

    if export_highcharts:

        if title is not None:
            plot_title = title
        if barh:
            hc_ylabel = x_label
        else:
            hc_ylabel = y_label
        _logger.info(f"Saving plot to highcharts")
        plot_df = plot_df.reindex(plot_df.index[::-1])
        CBSHighChart(
            data=plot_df,
            chart_type="bar",
            output_directory=highcharts_directory.as_posix(),
            output_file_name=image_file.stem,
            ylabel=hc_ylabel,
            title=plot_title,
            enable_legend=False
        )

    if show_plots:
        plt.show()

    plt.close()

    return image_file_name


def make_conditional_score_plot(correlations,
                                image_directory,
                                show_plots=False,
                                figsize=None, image_type=".pdf",
                                export_svg=False,
                                export_highcharts=False,
                                highcharts_directory=None,
                                title=None,
                                cache_directory=None
                                ):
    plot_info = correlations["plots"]

    index_labels = correlations["index_labels"]
    categories = correlations["index_categories"]
    score_intervallen = correlations["score_intervallen"]

    for plot_key, plot_prop in plot_info.items():

        # we maken hier alleen de score plots
        if plot_key not in ("scores_per_interval", "scores_per_number_correct") or \
                not plot_prop.get("do_it", True):
            continue

        outfile = Path(plot_prop["output_file"])
        if cache_directory is not None:
            outfile = Path(cache_directory) / outfile
        in_file = outfile.with_suffix(".pkl")

        if highcharts_directory is None:
            highcharts_directory = Path(".")

        if hc_sub_dir := plot_prop.get("highcharts_output_directory"):
            highcharts_directory = highcharts_directory / Path(hc_sub_dir)

        _logger.info(f"Reading scores from {in_file}")
        scores = pd.read_pickle(in_file.with_suffix(".pkl"))

        if plot_key == "scores_per_interval":
            im_file_base = "_".join([outfile.stem, "per_score_interval"])
            im_file = image_directory / Path(im_file_base).with_suffix(".pdf")
            plot_score_per_interval(scores=scores,
                                    score_intervallen=score_intervallen,
                                    index_labels=index_labels,
                                    categories=categories,
                                    highcharts_directory=highcharts_directory,
                                    im_file=im_file,
                                    show_plots=show_plots)
        elif plot_key == "scores_per_number_correct":
            im_file_base = "_".join([outfile.stem, "per_count_interval"])
            im_file = image_directory / Path(im_file_base).with_suffix(".pdf")
            plot_score_per_count(scores=scores,
                                 categories=categories,
                                 highcharts_directory=highcharts_directory,
                                 im_file=im_file,
                                 show_plots=show_plots)


def plot_score_per_count(scores, categories, highcharts_directory, im_file, show_plots):
    _logger.info("Plot score per count")
    # add a new columns with the interval label belonging to the gk code bin. Note that we
    # merge all the grootte klass below 40 to a group smaller than 10

    score_per_category = dict()
    for categorie_key, category_df in scores.groupby("count"):
        _logger.debug(f"Plotting {categorie_key}")
        df = category_df[list(categories.keys())]
        score_per_category[categorie_key] = df.mean()

    score_per_category_df = pd.DataFrame(score_per_category).T * 100

    plot_title = "Score per count"
    y_label = "Score"

    settings = CBSPlotSettings(color_palette="koelextended")
    fig, axis = plt.subplots()
    fig.subplots_adjust(bottom=0.3, top=0.92)
    score_per_category_df.plot.bar(ax=axis, rot=0, stacked=False, edgecolor="white", linewidth=1.5)
    yticks = axis.get_yticks()
    # axis.set_ylim((yticks[0], yticks[-1]))
    axis.set_ylim((0, 100))

    axis.set_xlabel("# categorieën goed", rotation="horizontal", horizontalalignment="right")
    axis.xaxis.set_label_coords(0.98, -0.15)

    axis.set_ylabel("Genormaliseerde score per categorie [%]",
                    rotation="horizontal", horizontalalignment="left")
    axis.yaxis.set_label_coords(-0.065, 1.07)
    axis.xaxis.grid(False)
    sns.despine(ax=axis, left=True)

    # niet meer volgens de richtlijnen
    # add_values_to_bars(axis=axis, color="w")

    sns.despine(ax=axis, left=True)

    axis.tick_params(which="both", bottom=False)

    add_axis_label_background(fig=fig, axes=axis, loc="south")

    ncol = (score_per_category_df.columns.size - 1) // 2 + 1

    legend = axis.legend(loc="lower left",
                         bbox_to_anchor=(0.105, -0.00), frameon=False,
                         bbox_transform=fig.transFigure, ncol=ncol)

    _logger.info(f"Writing score plot to {im_file}")
    fig.savefig(im_file.as_posix())

    highcharts_directory.mkdir(exist_ok=True, parents=True)

    CBSHighChart(
        data=score_per_category_df,
        chart_type="column_grouped_stacked",
        output_directory=highcharts_directory.as_posix(),
        output_file_name=im_file.stem,
        ylabel=y_label,
        title=plot_title,
        enable_legend=False,
    )

    if show_plots:
        plt.show()

    _logger.debug("Klaar")


def plot_score_per_interval(scores, score_intervallen, index_labels, categories,
                            highcharts_directory, im_file, show_plots):
    score_labels = list(score_intervallen.keys())
    score_bins = list([s / 100 for s in score_intervallen.values()]) + [1.01]
    # add a new columns with the interval label belonging to the gk code bin. Note that we
    # merge all the grootte klass below 40 to a group smaller than 10
    scores["score_category"] = pd.cut(scores["score"],
                                      bins=score_bins,
                                      labels=score_labels,
                                      right=True,
                                      include_lowest=True)

    score_per_category = dict()
    for categorie_key, category_df in scores.groupby("score_category"):
        _logger.debug(f"Plotting {categorie_key}")
        df = category_df[list(categories.keys())]
        category_label = index_labels[categorie_key]
        score_per_category[category_label] = df.mean()

    score_per_category_df = pd.DataFrame(score_per_category).T * 100

    plot_title = "Score per categorie"
    y_label = "Score"

    settings = CBSPlotSettings(color_palette="koelextended")
    fig, axis = plt.subplots()
    fig.subplots_adjust(bottom=0.3, top=0.92)
    score_per_category_df.plot.bar(ax=axis, rot=0, stacked=False, edgecolor="white", linewidth=1.5)
    yticks = axis.get_yticks()
    # axis.set_ylim((yticks[0], yticks[-1]))
    axis.set_ylim((0, 100))

    axis.set_xlabel("Scorecategorie", rotation="horizontal", horizontalalignment="right")
    axis.xaxis.set_label_coords(0.98, -0.15)

    axis.set_ylabel("Genormaliseerde score per categorie [%]",
                    rotation="horizontal", horizontalalignment="left")
    axis.yaxis.set_label_coords(-0.065, 1.07)
    axis.xaxis.grid(False)
    sns.despine(ax=axis, left=True)

    # niet meer volgens de richtlijnen
    # add_values_to_bars(axis=axis, color="w")

    sns.despine(ax=axis, left=True)

    axis.tick_params(which="both", bottom=False)

    add_axis_label_background(fig=fig, axes=axis, loc="south")

    ncol = (score_per_category_df.columns.size - 1) // 2 + 1

    legend = axis.legend(loc="lower left",
                         bbox_to_anchor=(0.105, -0.00), frameon=False,
                         bbox_transform=fig.transFigure, ncol=ncol)

    _logger.info(f"Writing score plot to {im_file}")
    fig.savefig(im_file.as_posix())

    highcharts_directory.mkdir(exist_ok=True, parents=True)

    CBSHighChart(
        data=score_per_category_df,
        chart_type="column_grouped_stacked",
        output_directory=highcharts_directory.as_posix(),
        output_file_name=im_file.stem,
        ylabel=y_label,
        title=plot_title,
        enable_legend=False,
    )

    if show_plots:
        plt.show()

    _logger.debug("Klaar")


# fig, axis = plt.subplots(figsize=(10, 10))
# cbar_ax = fig.add_axes([.91, .315, .02, .62])
# cmap = sns.color_palette("deep", 10)


def make_heatmap(correlations, image_directory,
                 show_plots=False,
                 figsize=None, image_type=".pdf",
                 export_svg=False,
                 export_highcharts=False,
                 highcharts_directory=None,
                 title=None,
                 cache_directory=None
                 ):
    plot_properties = correlations["plots"]["correlation"]
    outfile = Path(plot_properties["output_file"])
    if cache_directory is not None:
        outfile = Path(cache_directory) / outfile

    in_file = outfile.with_suffix(".pkl")

    if highcharts_directory is None:
        highcharts_directory = Path(".")

    if hc_sub_dir := plot_properties.get("highcharts_output_directory"):
        highcharts_directory = highcharts_directory / Path(hc_sub_dir)

    _logger.info(f"Reading correlation from {in_file}")
    corr = pd.read_pickle(in_file.with_suffix(".pkl"))

    categories = correlations["index_categories"]
    corr_index = correlations["index_correlations"]
    corr = corr.reindex(list(corr_index.keys()))
    corr = corr[list(corr_index.keys())]

    sns.set(font_scale=0.8)
    # cmap is now a list of colors
    cmap = mpc.ListedColormap(sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, n_colors=12))

    # Create two appropriately sized subplots
    # grid_kws = {'width_ratios': (0.9, 0.03), 'wspace': 0.18}
    # fig, (axis, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(8.3, 8.3))

    im_file = image_directory / Path(outfile.stem).with_suffix(".pdf")
    fig, axis = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=.28, bottom=.27, top=0.98, right=0.9)
    cbar_ax = fig.add_axes([.91, .315, .02, .62])
    # cmap = sns.color_palette("deep", 10)

    sns.heatmap(corr, square=True, ax=axis, cbar_ax=cbar_ax, cmap=cmap,
                vmin=-0.2, vmax=1.0,
                cbar_kws={
                    'orientation': 'vertical',
                    'label': r'Correlatiecoëfficiënt $\rho$'}
                )
    xlabels = axis.get_xticklabels()
    ylabels = axis.get_yticklabels()
    for xlbl, ylbl in zip(xlabels, ylabels):
        tekst = xlbl.get_text()
        categorie = corr_index[tekst]
        categorie_properties = categories[categorie]
        kleur = categorie_properties['color']
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

    highcharts_directory.mkdir(exist_ok=True, parents=True)

    hc_out = highcharts_directory / Path(im_file.stem + ".svg")

    _logger.info(f"Writing heatmap to {hc_out}")
    fig.savefig(hc_out.as_posix())

    if show_plots:
        plt.show()


def make_conditional_pdf_plot(categories, image_directory,
                              show_plots=False,
                              export_highcharts=False,
                              highcharts_directory=None,
                              cache_directory=None
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
        highcharts_directory = Path(".")

    if hc_sub_dir := plot_settings.get("highcharts_output_directory"):
        highcharts_directory = highcharts_directory / Path(hc_sub_dir)

    highcharts_directory.mkdir(exist_ok=True, parents=True)

    _logger.info(f"Reading correlation from {in_file}")
    conditional_scores_df = pd.read_pickle(in_file.with_suffix(".pkl"))

    im_file = image_directory / Path("_".join([outfile.stem, image_key])).with_suffix(".pdf")
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
    # , edgecolor=None, linewidth=0)
    # conditional_scores_df[0].plot.bar(ax=axis, stacked=True, width=delta_bin / 2)
    # edgecolor=None, linewidth=0)

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

    y_label = '% bedrijven'

    axis.set_ylabel(y_label, rotation="horizontal", horizontalalignment="left")
    axis.yaxis.set_label_coords(-0.04, 1.05)
    axis.xaxis.grid(False)
    axis.set_xlabel("Totaal score", horizontalalignment="right")
    axis.xaxis.set_label_coords(0.98, -0.12)
    sns.despine(ax=axis, left=True)

    labels = [_.get_text() for _ in axis.get_xticklabels()]
    axis.set_xticklabels(labels, ha='center')

    add_axis_label_background(fig=fig, axes=axis, loc="south", margin=0.10)

    legend = axis.legend(loc="lower left",
                         title="# categorieën goed",
                         prop={"size": 10},
                         bbox_to_anchor=(0.2, 0.02), frameon=False,
                         bbox_transform=fig.transFigure, ncol=5)

    legend._legend_box.align = "left"
    for patch in legend.get_patches():
        patch.set_linewidth(0)

    _logger.info(f"Saving plot {im_file}")
    fig.savefig(im_file)

    if export_svg:
        svg_image_file = highcharts_directory / Path(im_file.with_suffix(".svg").stem)
        _logger.info(f"Saving plot to {svg_image_file}")
        fig.savefig(svg_image_file)

    if export_highcharts:
        # voor highcharts de titel setten
        CBSHighChart(
            data=conditional_scores_df,
            chart_type="column",
            output_directory=highcharts_directory.as_posix(),
            output_file_name=im_file.stem,
            ylabel=y_label,
            title="Verdeling scores per categorie",
            enable_legend=False,
        )

    if show_plots:
        plt.show()

    _logger.debug("Done")

    plt.close()


def make_verdeling_per_aantal_categorie(categories, image_directory,
                                        show_plots=False,
                                        export_highcharts=False,
                                        highcharts_directory=None,
                                        cache_directory=None
                                        ):
    outfile = Path(categories["categories_output_file"])
    if cache_directory is not None:
        outfile = Path(cache_directory) / outfile

    image_key = "verdeling_per_category"
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

    percentage_per_number_of_cat = 100 * sum_per_number_of_cat_df / sum_of_all_categories

    if highcharts_directory is None:
        highcharts_directory = Path(".")
    else:
        highcharts_directory = Path(highcharts_directory)

    if hc_sub_dir := plot_settings.get("highcharts_output_directory"):
        highcharts_directory = highcharts_directory / Path(hc_sub_dir)

    highcharts_directory.mkdir(exist_ok=True, parents=True)

    im_file = image_directory / Path("_".join([outfile.stem, image_key])).with_suffix(".pdf")

    figure_properties = CBSPlotSettings()

    fig, axis = plt.subplots()
    axis.tick_params(which="both", bottom=True)
    fig.subplots_adjust(bottom=0.25, top=0.92, right=0.98)

    percentage_per_number_of_cat.T.plot.bar(stacked=True, ax=axis)

    axis.set_ylim((0, 101))

    axis.set_xlabel("Aantal goede categorieën", horizontalalignment="right")
    y_label = "% bedrijven"

    axis.set_ylabel(y_label, rotation="horizontal", horizontalalignment="left")
    axis.yaxis.set_label_coords(-0.06, 1.05)
    axis.xaxis.grid(False)
    axis.xaxis.set_label_coords(0.98, -0.1)
    xlabels = axis.get_xticklabels()
    axis.set_xticklabels(xlabels, rotation=0, ha="right")
    sns.despine(ax=axis, left=True)

    legend = axis.legend(loc="lower left",
                         title="Categorie",
                         bbox_to_anchor=(0.2, 0.03), frameon=False,
                         bbox_transform=fig.transFigure, ncol=5)

    legend._legend_box.align = "left"
    for patch in legend.get_patches():
        patch.set_linewidth(0)

    axis.tick_params(which="both", bottom=False)
    add_axis_label_background(fig=fig, axes=axis, loc="south", margin=0.02)

    _logger.info(f"Saving plot {im_file}")
    fig.savefig(im_file)

    if export_svg:
        svg_image_file = highcharts_directory / Path(im_file.with_suffix(".svg").stem)
        _logger.info(f"Saving plot to {svg_image_file}")
        fig.savefig(svg_image_file)

    if export_highcharts:
        # voor highcharts de titel setten
        CBSHighChart(
            data=sum_per_number_of_cat_df,
            chart_type="column",
            output_directory=highcharts_directory.as_posix(),
            output_file_name=im_file.stem,
            title="Verdeling scores per categorie",
            enable_legend=False,
        )

    if show_plots:
        plt.show()

    _logger.debug("Done")

    plt.close()
