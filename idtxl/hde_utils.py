"""Utils for the history dependence estimators

All but the visualization function are adapted from:
    [1]: L. Rudelt, D. G. Marx, M. Wibral, V. Priesemann: Embedding
        optimization reveals long-lasting history dependence in
        neural spiking activity, 2021, PLOS Computational Biology, 17(6)

    [2]: https://github.com/Priesemann-Group/hdestimator

implemented in idtxl by Michael Lindner, Göttingen 2021

"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime


def get_CI_bounds(
    R,
    bs_Rs,
    bootstrap_CI_use_sd=True,
    bootstrap_CI_percentile_lo=2.5,
    bootstrap_CI_percentile_hi=97.5,
):
    """
    Given bootstrap replications bs_Rs of the estimate for R,
    obtain the lower and upper bound of a 95% confidence
    interval based on the standard deviation; or an arbitrary
    confidence interval based on percentiles of the replications.
    """

    if bootstrap_CI_use_sd:
        sigma_R = np.std(bs_Rs)
        CI_lo = R - 2 * sigma_R
        CI_hi = R + 2 * sigma_R
    else:
        CI_lo = np.percentile(bs_Rs, bootstrap_CI_percentile_lo)
        CI_hi = np.percentile(bs_Rs, bootstrap_CI_percentile_hi)
    return CI_lo, CI_hi


def add_up_dicts(dicts):
    return sum((Counter(dict(d)) for d in dicts), Counter())


def get_min_key_for_max_value(d):
    """
    For a dictionary d, get the key for the largest value.
    If largest value is attained several times, get the
    smallest respective key.
    """

    sorted_keys = sorted([key for key in d.keys()])
    values = [d[key] for key in sorted_keys]
    max_value = max(values)
    for key, value in zip(sorted_keys, values):
        if value == max_value:
            return key


def get_max_R_T(max_Rs):
    """
    Get R and T for which R is maximised.

    If R is maximised at several T, get the
    smallest respective T.
    """

    max_R_T = get_min_key_for_max_value(max_Rs)
    max_R = max_Rs[max_R_T]
    return max_R, max_R_T


def get_shannon_entropy(probabilities):
    """
    Get the entropy of a random variable based on the probabilities
    of its outcomes.
    """

    return -sum((p * np.log(p) for p in probabilities if not p == 0))


def get_H_spiking(symbol_counts):
    """
    Get the (unconditional) entropy of a spike train, based
    on its outcomes, as stored in the symbol_counts dictionary.

    For each symbol, determine what the response was (spike/ no spike),
    and obtain the spiking probability. From that compute the entropy.
    """

    number_of_spikes = 0
    number_of_symbols = 0
    for symbol, counts in symbol_counts.items():
        number_of_symbols += counts
        if symbol % 2 == 1:
            number_of_spikes += counts

    p_spike = number_of_spikes / number_of_symbols
    return get_shannon_entropy([p_spike, 1 - p_spike])


def get_binned_neuron_activity(
    spike_times, bin_size, relative_to_median_activity=False
):
    """
    Get an array of 0s and 1s representing the spike train.
    """

    number_of_bins = int(spike_times[-1] / bin_size) + 1
    binned_neuron_activity = np.zeros(number_of_bins, dtype=int)
    if relative_to_median_activity:
        for spike_time in spike_times:
            binned_neuron_activity[int(spike_time / bin_size)] += 1
        median_activity = np.median(binned_neuron_activity)
        spike_indices = np.where(binned_neuron_activity > median_activity)
        binned_neuron_activity = np.zeros(number_of_bins, dtype=int)
        binned_neuron_activity[spike_indices] = 1
    else:
        binned_neuron_activity[
            [int(spike_time / bin_size) for spike_time in spike_times]
        ] = 1
    return binned_neuron_activity


def get_binned_firing_rate(spike_times, bin_size):
    """
    Get the firing rate of a spike train, as obtained after binning the activity.
    """

    binned_neuron_activity = get_binned_neuron_activity(spike_times, bin_size)
    number_of_bins = int(spike_times[-1] / bin_size) + 1
    return sum(binned_neuron_activity) / (number_of_bins * bin_size)


def remove_key(d, key):
    """
    Remove an entry from a dictionary .
    """

    r = d.copy()
    del r[key]
    return r


def get_past_symbol_counts(symbol_counts, merge=True):
    """
    Get symbol_counts for the symbols excluding the response.
    """

    past_symbol_counts = [Counter(), Counter()]
    for symbol in symbol_counts:
        response = int(symbol % 2)
        past_symbol = symbol // 2
        past_symbol_counts[response][past_symbol] = symbol_counts[symbol]

    if merge:
        merged_past_symbol_counts = Counter()
        for response in [0, 1]:
            for symbol in past_symbol_counts[response]:
                merged_past_symbol_counts[symbol] += past_symbol_counts[response][
                    symbol
                ]
        return merged_past_symbol_counts
    else:
        return past_symbol_counts


def make_plot_pretty(ax, ticklabelsize):
    ax.tick_params(axis="x", top=False)
    ax.tick_params(axis="y", right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(direction="out", width=3, length=10, labelsize=ticklabelsize)
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.8", linestyle="-", linewidth=1)


def format_x_label(x, pos):
    return "{:.0f}".format(x)


def format_x_label_in_ms(x, pos):
    return "{:.0f}".format(x * 1000)


def format_y_label(y, pos):
    if y > 0 and y <= 0.02:
        ret = "{:.3f}".format(y)
        return ret[1:]
    return "{:.2f}".format(y)


def hde_visualize_results(results, process, filename=None):
    """
    This method provides a plot or an output .svg image for the given process containing:
        - optimized values for the process
        - graph for the history dependence
        - graph for auto mutual information (if calculated)

    implemented in idtxl by Michael Lindner, Göttingen 2021

    Args:
        results : ResultsSingleProcessRudelt instance
            Results from optimization_Rudelt
        process : int
            index of optimized process
        filename : String
            path and filename where the result image should be store.
            If filename is not set or None, the result image is shown

    Returns:
        returns a plot or an .svg image with the content mentioned above

    """
    settings = results.settings
    res = results.get_single_process(process)

    titlesize = 24
    ticklabelsize = 18
    xlabelsize = 22
    ylabelsize = 22
    legendsize = 18
    date = datetime.today().strftime("%Y-%m-%d")
    linewidth = 3.0
    markersize = 9.0

    plt.figure(figsize=(18, 18))
    maintitle = (
        "Results of Process "
        + str(res.Process)
        + " using estimation method "
        + settings["estimation_method"]
    )
    plt.suptitle(maintitle, fontsize=titlesize, fontweight="bold")
    # plot table
    plt.subplot(221)
    row_labels = [
        "$\mathit{T}_{D}$ [s]",
        "$\\tau_{R}$ [s]",
        "$\mathit{R}_{tot}$",
        "$\mathit{R}_{tot}$ CI",
        "opt. $\mathit{d}$",
        "opt. $\mathit{k}$",
        "opt. $\\tau_{1}$ [s]",
        "firing rate [Hz]",
        "recording length [s]",
        "H spiking",
    ]
    if res.R_tot_CI[0] is None:
        r_tot_ci_lo = "nan"
        r_tot_ci_hi = "nan"
    else:
        r_tot_ci_lo = str(round(res.R_tot_CI[0], 3))
        r_tot_ci_hi = str(round(res.R_tot_CI[1], 3))

    table_vals = [
        [str(round(res.T_D, 3))],
        [str(round(res.tau_R, 3))],
        [str(round(res.R_tot, 3))],
        [r_tot_ci_lo + ", " + r_tot_ci_hi],
        [str(round(res.opt_number_of_bins_d, 3))],
        [str(round(res.opt_scaling_k, 3))],
        [str(round(res.opt_first_bin_size, 3))],
        [str(round(res.firing_rate, 1))],
        [str(round(res.recording_length, 1))],
        [str(round(res.H_spiking, 3))],
    ]
    the_table = plt.table(
        cellText=table_vals,
        colWidths=[0.1] * 3,
        rowLabels=row_labels,
        loc="center",
        edges="open",
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(20)
    the_table.scale(4, 4)
    plt.axis("off")

    # plot date
    axd = plt.subplot(325)
    plt.text(
        0,
        0,
        date,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axd.transAxes,
        fontsize=xlabelsize,
    )
    plt.axis("off")

    # plot history dependence
    ax1 = plt.subplot(322)
    ax1.set_title(
        "History Dependence", fontdict={"fontsize": titlesize, "fontweight": "medium"}
    )
    y = np.zeros(len(settings["embedding_past_range_set"]))
    count = 0
    for key in settings["embedding_past_range_set"]:
        try:
            y[count] = res.max_R[key]
        except:
            a = 1
        count += 1

    x = settings["embedding_past_range_set"]
    ax1.plot(x, y, "b-", linewidth=linewidth)
    ax1.set_xscale("log")

    ax1.axvline(x=res.tau_R, color="k", ls="--", linewidth=linewidth, label=r"$\tau_R$")
    # ax1.axvline(x=res.T_D, color='k', ls=':', linewidth=linewidth,
    #             label=r"$T_D$")

    legend1 = ax1.legend(loc="lower right", fontsize=legendsize)
    frame = legend1.get_frame()
    frame.set_facecolor("#d0d0d0")
    frame.set_edgecolor("#d0d0d0")

    ax1.set_xlabel("past range T [ms]", fontsize=xlabelsize)
    ax1.set_ylabel("History Dependence R", fontsize=ylabelsize)
    make_plot_pretty(ax1, ticklabelsize)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_x_label_in_ms))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_y_label))

    # plot auto MI (if calculated)
    ax2 = plt.subplot(326)

    ax2.set_title(
        "Auto Mutual Information",
        fontdict={"fontsize": titlesize, "fontweight": "medium"},
    )
    if "auto_MI" in res:
        # set line and marker style
        lin_style = ["-", "--", "-.", ":"]
        mrk_style = ["v", "^", "o", "*", "h", "s"]

        leg2 = []
        c1 = 0
        c2 = 0
        for key in res.auto_MI.keys():
            y2 = res.auto_MI[key]
            leg2.append(key)
            x2 = np.linspace(0, settings["auto_MI_max_delay"], len(res.auto_MI[key]))
            ax2.plot(
                x2,
                y2,
                "b" + lin_style[c1 % len(lin_style)] + mrk_style[c2 % len(mrk_style)],
                linewidth=linewidth,
                markersize=markersize,
                label=str(int(float(key) * 1000)),
            )
            c1 += 1
            c2 += 1

        ax2.set_xscale("log")
        ax2.set_ylabel("normalized Auto MI", fontsize=xlabelsize)
        ax2.set_xlabel("time t [ms]", fontsize=ylabelsize)
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_x_label_in_ms))
        make_plot_pretty(ax2, ticklabelsize)
        legend2 = ax2.legend(
            loc="upper right", title="bin size [ms]", fontsize=legendsize
        )
        legend2.get_title().set_fontsize(legendsize)
        frame = legend2.get_frame()
        frame.set_facecolor("#d0d0d0")
        frame.set_edgecolor("#d0d0d0")
    else:
        plt.text(
            0.5,
            0.5,
            "was not calculated",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax2.transAxes,
            fontsize=titlesize,
        )
        plt.axis("off")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, format="svg")
