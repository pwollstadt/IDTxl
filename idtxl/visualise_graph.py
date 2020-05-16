"""Plot results of network inference."""
import numpy as np
import matplotlib.pyplot as plt
from . import idtxl_io as io
from . import idtxl_exceptions as ex
try:
    import networkx as nx
except ImportError as err:
    ex.package_missing(
        err,
        ('networkx is not available on this system. Install it from '
         'https://pypi.python.org/pypi/networkx/2.0 to export and plot IDTxl '
         'results in this format.'))


def plot_network(results, weights, fdr=True):
    """Plot network of multivariate TE between processes.

    Plot graph of the network of (multivariate) interactions between processes
    (e.g., multivariate TE). The function uses  the networkx class for directed
    graphs (DiGraph) internally. Plots a network and adjacency matrix.

    Args:
        results : ResultsNetworkInference() instance
            output of an network inference algorithm
        weights : str
            for single network inference, it can either be

                - 'max_te_lag': the weights represent the source -> target
                   lag corresponding to the maximum tranfer entropy value
                   (see documentation for method get_target_delays for details)
                - 'max_p_lag': the weights represent the source -> target
                   lag corresponding to the maximum p-value
                   (see documentation for method get_target_delays for details)
                - 'vars_count': the weights represent the number of
                   statistically-significant source -> target lags
                - 'binary': return unweighted adjacency matrix with binary
                   entries

                   - 1 = significant information transfer;
                   - 0 = no significant information transfer.

            for network comparison, it can either be

                - 'union': all links in the union network, i.e., all
                  links that were tested for a difference
                - 'comparison': True for links with a significant difference in
                   inferred effective connectivity (default)
                - 'pvalue': absolute differences in inferred effective
                   connectivity for significant links
                - 'diff_abs': absolute difference

        fdr : bool [optional]
            print FDR-corrected results (default=True)

    Returns:
        DiGraph
            instance of a directed graph class from the networkx package
        Figure
            figure handle, Figure object from the matplotlib package
    """
    adj_matrix = results.get_adjacency_matrix(weights=weights, fdr=fdr)
    graph = io.export_networkx_graph(adj_matrix, weights)

    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(121)  # plot graph
    _plot_graph(graph, ax1, weights)
    plt.subplot(122)  # plot adjacency matrix
    _plot_adj_matrix(
        results.get_adjacency_matrix(weights, fdr),
        cbar_label=weights)
    return graph, fig


def plot_selected_vars(results, target, sign_sources=True,
                       display_edge_labels=False, fdr=True):
    """Plot network of a target process and single variables.

    Plot graph of the network of (multivariate) interactions between source
    variables and the target. The function uses the networkx class for directed
    graphs (DiGraph) internally. Plots a network and reduced adjacency matrix.

    Args:
        results : ResultsNetworkInference() instance
            output of an network inference algorithm
        target : int
            index of target process
        sign_sources : bool [optional]
            plot sources with significant information contribution only
            (default=True)
        display_edge_labels : bool [optional]
            display TE value on edge lables (default=False)
        fdr : bool [optional]
            print FDR-corrected results (default=True)

    Returns:
        DiGraph
            instance of a directed graph class from the networkx package
        Figure
            figure handle, Figure object from the matplotlib package
    """
    graph = io.export_networkx_source_graph(results, target, sign_sources, fdr)
    # Replace time index of current value to be consistent with lag-notation
    # in plot.
    current_value = (results._single_target[target].current_value[0], 0)
    max_lag = max(results.settings.max_lag_sources,
                  results.settings.max_lag_target)

    # Adjust color and position of nodes (variables).
    pos = nx.spring_layout(graph)
    color = ['lavender' for c in range(graph.number_of_nodes())]
    for (ind, n) in enumerate(graph.nodes):

        # Adjust posistions of nodes.
        if n == current_value:
            pos[n] = np.array([max_lag, 0])
        elif n[0] == current_value[0]:  # target history
            pos[n] = np.array([max_lag - n[1], 0])
        elif n[0] < current_value[0]:  # sources with proc. number < target
            pos[n] = np.array([max_lag - n[1], n[0] + 1])
        else:  # sources with proc. number > target
            pos[n] = np.array([max_lag - n[1], n[0]])

        # Adjust color of nodes.
        if n in results._single_target[target].selected_vars_sources:
            color[ind] = 'cadetblue'
        elif n in results._single_target[target].selected_vars_target:
            color[ind] = 'tomato'
        elif n == current_value:
            color[ind] = 'red'

    fig = plt.figure()
    nx.draw(graph, pos=pos, with_labels=True, font_weight='bold',
            node_size=900, alpha=0.7, node_shape='s', node_color=color)
    # Optionally display edge labels showing the TE value
    if display_edge_labels:
        edge_labels = nx.get_edge_attributes(graph, 'te')
        # Change format to only display 2 decimals
        for key, value in edge_labels.items():
            edge_labels[key] = '{0:.2g}'.format(value)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                     font_size=10)  # font_weight='bold'

    plt.plot([-0.5, max_lag + 0.5], [0.5, 0.5],
             linestyle='--', linewidth=1, color='0.5')
    return graph, fig


def _plot_graph(graph, axis, weights=None, display_edge_labels=True):
    """Plot graph using networkx."""
    pos = nx.circular_layout(graph)
    nx.draw_circular(graph, with_labels=True, node_size=600, alpha=1.0,
                     ax=axis, node_color='Gainsboro', font_size=14,
                     font_weight='bold')
    if display_edge_labels:
        edge_labels = nx.get_edge_attributes(graph, weights)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                     font_size=13)  # font_weight='bold'


def _plot_adj_matrix(adj_matrix, mat_color='gray_r', diverging=False,
                     cbar_label='', cbar_stepsize=1):
    """Plot adjacency matrix."""
    # Plot matrix, set minimum and maximum values to the same value for
    # diverging plots to center colormap at 0, i.e., 0 is plotted in white
    # https://stackoverflow.com/questions/25500541/
    # matplotlib-bwr-colormap-always-centered-on-zero
    if diverging:
        max_val = np.max(abs(adj_matrix._weight_matrix))
        min_val = -max_val
    else:
        max_val = np.max(adj_matrix._weight_matrix)
        min_val = -np.min(adj_matrix._weight_matrix)

    adj_matrix_masked = np.ma.masked_where(
        np.invert(adj_matrix._edge_matrix), adj_matrix._weight_matrix)
    plt.imshow(adj_matrix_masked, cmap=mat_color,
               interpolation='nearest', vmin=min_val, vmax=max_val)

    # Set the colorbar and make colorbar match the image in size using the
    # fraction and pad parameters (see https://stackoverflow.com/a/26720422).
    cbar_ticks = np.arange(0, max_val + 1, cbar_stepsize)
    if cbar_label == 'max_te_lag':
        cbar_label = 'delay [samples]'
    elif cbar_label == 'max_p_lag':
        cbar_label = 'max p-value lag [samples]'
    elif cbar_label == 'vars_count':
        cbar_label = '# of selected vars'
    elif cbar_label == 'binary':
        cbar_label = 'Binary adjacency matrix'
    elif cbar_label == 'p-value':
        # cbar_ticks = np.arange(0, 1.001, 0.1)
        cbar_ticks = np.arange(0, max_val * 1.01, max_val / 5)
    else:
        cbar_ticks = np.arange(min_val, max_val + 0.01 * max_val,
                               cbar_stepsize)
    cbar = plt.colorbar(fraction=0.046, pad=0.04, ticks=cbar_ticks)
    cbar.set_label(cbar_label, rotation=90)

    # Set x- and y-ticks.
    plt.xticks(np.arange(adj_matrix.n_nodes()))
    plt.yticks(np.arange(adj_matrix.n_nodes()))
    ax = plt.gca()
    ax.xaxis.tick_top()
    return cbar


def plot_mute_graph():
    """Plot MuTE example network.

    Network of 5 AR-processes, which is used as an example the paper
    on the MuTE toolbox (Montalto, PLOS ONE, 2014, eq. 14). The
    network consists of five autoregressive (AR) processes with model
    orders 2 and les and the following (non-linear) couplings:

        >>> 0 -> 1, u = 2
        >>> 0 -> 2, u = 3
        >>> 0 -> 3, u = 2 (non-linear)
        >>> 3 -> 4, u = 1
        >>> 4 -> 3, u = 1

    Returns:
        Figure handle
            Figure object from the matplotlib package
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(np.arange(5))
    # graph.add_edges_from([(0, 1), (0, 2), (0, 3), (3, 4), (4, 3)])
    graph.add_weighted_edges_from([(0, 1, 2), (0, 2, 3), (0, 3, 2), (3, 4, 1),
                                   (4, 3, 1)], weight='delay')
    pos = {
        0: np.array([1, 1]),
        1: np.array([0, 2]),
        2: np.array([0, 0]),
        3: np.array([2, 1]),
        4: np.array([3, 1]),
    }
    fig = plt.figure()
    nx.draw(graph, pos=pos, with_labels=True, node_size=900, alpha=1.0,
            node_color='cadetblue', font_weight='bold',
            edge_color=['r', 'k', 'r', 'k', 'k'], hold=True)
    nx.draw_networkx_edge_labels(graph, pos=pos)
    plt.text(2, 0.1, 'non-linear interaction in red')
    # see here for an example on how to plot edge labels:
    # http://stackoverflow.com/questions/10104700/how-to-set-networkx-edge-labels-offset-to-avoid-label-overlap
    return fig


def plot_network_comparison(results):
    """Plot results of network comparison.

    Plot results of network comparison. Produces a figure with five subplots,
    where the first plot shows the network graph of the union network, the
    second plot shows the adjacency matrix of the union network, the third
    plot shows the qualitative results of the comparison of each link, the
    fourth plot shows the absolute differences in CMI per link, and the fifth
    plot shows p-values for each link.

    Args:
        results : ResultsNetworkComparison() instance
            network comparison results

    Returns:
        DiGraph
            instance of a directed graph class from the networkx package
        Figure
            figure handle, Figure object from the matplotlib package
    """
    # Get union graph.
    adj_matrix = results.get_adjacency_matrix(weights='union')
    graph_union = io.export_networkx_graph(adj_matrix, weights='union')

    fig = plt.figure(figsize=(10, 15))
    ax1 = plt.subplot(231)  # plot union graph
    _plot_graph(graph_union, ax1)
    ax = plt.subplot(232)  # plot union graph adjacency matrix
    _plot_adj_matrix(results.get_adjacency_matrix('union'), mat_color='PuBu',
                     cbar_label='link in union', cbar_stepsize=1)
    ax.set_title('union network A and B', y=1.1)

    ax = plt.subplot(234)  # plot comparison adjacency matrix
    if results.settings.tail_comp == 'two':
        cbar_label = 'A != B'
    elif results.settings.tail_comp == 'one':
        cbar_label = 'A > B'
    adj_matrix_comparison = results.get_adjacency_matrix('comparison')
    _plot_adj_matrix(adj_matrix_comparison,
                     mat_color='OrRd', cbar_label=cbar_label, cbar_stepsize=1)
    ax.set_title('Comparison {0}'.format(cbar_label), y=1.1)

    ax = plt.subplot(235)  # plot abs. differences adjacency matrix
    adj_matrix_diff = results.get_adjacency_matrix('diff_abs')
    _plot_adj_matrix(adj_matrix_diff,
                     mat_color='BuGn', cbar_label='norm. CMI diff [a.u.]',
                     cbar_stepsize=0.1)
    ax.set_title('CMI diff abs (A - B)', y=1.1)

    ax = plt.subplot(236)  # plot p-value adjacency matrix
    adj_matrix_pval = results.get_adjacency_matrix('pvalue')
    _plot_adj_matrix(adj_matrix_pval, mat_color='Greys',
                     cbar_label='p-value')
    ax.set_title('p-value [%]', y=1.1)
    return graph_union, fig


def plot_spectral_result(spectral_result, freq_rate):
    """Plot results of spectral transfer entropy analysis at each scale.

    Args:
        spectral_results : ResultsSpectralTE instance
            Results of spectral TE analysis
        freq_rate: int
            samplign rate of the system under analysis

    Returns:
        Figure
            figure handle, Figure object from the matplotlib package
    """
    for target in spectral_result.targets_analysed:
        freq1 = np.round(freq_rate/2, 2)
        freq2 = np.round(freq1/2, 2)
        if spectral_result.settings['spectral_analysis_type'] == 'source':

            for process in range(spectral_result._single_target[target][0]['source_tested']):

                # Make a subplot for each link.
                n_scale = spectral_result.settings['n_scale']
                fig, axs = plt.subplots(
                    n_scale, 1, sharex=True, sharey=True, figsize=(10, 5))
                nbin = int(round(np.size(
                    spectral_result._single_target[target]['source'][0]['te_surrogate'][process])/5))
                axs = axs.ravel()
                te_orig = spectral_result._single_target[target][0]['te_full_orig'][process]
                for i in range(0, n_scale):

                    median_s = np.median(
                        spectral_result._single_target[target]['source'][i]['te_surrogate'][process])

                    axs[i].hist(spectral_result._single_target[target][i]['te_surrogate'][process],
                                bins=nbin, color='white', edgecolor='black')

                    axs[i].plot([te_orig, te_orig], [0, nbin],
                                'black', linestyle='dashed')
                    axs[i].plot([median_s, median_s], [0, nbin],
                                'red', linestyle='dashed')
                    axs[0].text(.5, .9,
                                'Scale {0} (Freq {1}-{2}Hz)'.format(
                                    i, freq1, freq2),
                                horizontalalignment='center',
                                transform=axs[0].transAxes)
                    axs[0].set_title('Source Shuffled:'.format(target), y=1.3)

                    freq1 = freq2
                    freq2 = np.round(freq1/2, 2)

                axs[i].set_xlabel("Transfer Entropy")
                fig.subplots_adjust(hspace=0.7)
                plt.show()

        elif spectral_result.settings['spectral_analysis_type'] == 'target':

            for process in range(0, spectral_result._single_target[target][0]['source_tested']):

                n_scale = spectral_result.settings['n_scale']
                fig, axs = plt.subplots(
                    n_scale, 1, sharex=True, sharey=True, figsize=(10, 5))
                nbin = int(round(np.size(
                    spectral_result._single_target[target]['target'][0]['te_surrogate'][process])/5))
                axs = axs.ravel()
                te_orig = spectral_result._single_target[target][0]['te_full_orig'][process]
                for i in range(n_scale):
                    median_t = np.median(
                        spectral_result._single_target[target]['target'][i]['te_surrogate'][process])
                    axs[i].hist(spectral_result._single_target[target][i]['te_surrogate'][process],
                                bins=nbin, color='white', edgecolor='black')

                    axs[i].plot([te_orig, te_orig], [0, nbin],
                                'black', linestyle='dashed')
                    axs[i].plot([median_t, median_t], [0, nbin],
                                'red', linestyle='dashed')
                    axs[i].text(.5, .9, 'Scale {0} (Freq {1}-{2}Hz)'.format(
                                    i, freq1, freq2),
                                horizontalalignment='center',
                                transform=axs[i].transAxes)
                    axs[0].set_title('Target Shuffled:'.format(target), y=1.3)

                    freq1 = freq2
                    freq2 = np.round(freq1/2, 2)

                axs[i].set_xlabel("Transfer Entropy")

                fig.subplots_adjust(hspace=0.7)
                plt.show()

        else:
            # Make one figure for each source tested (rigth column source
            # destroyed-left column target destroyed.
            for process in range(0, len(spectral_result._single_target[target]['source'][0]['source_tested'])):

                process_tested = spectral_result._single_target[target]['source'][0]['source_tested']
                n_scale = spectral_result.settings['n_scale']
                fig, axarr = plt.subplots(
                    n_scale, 2, sharex=True, sharey=True, figsize=(8, 5))
                nbin = int(round(np.size(
                    spectral_result._single_target[target]['source'][0]['te_surrogate'][process])/5))

                te_orig = spectral_result._single_target[target]['source'][0]['te_full_orig'][0]

                for i in range(0, n_scale):
                    median_s = np.median(
                        spectral_result._single_target[target]['source'][i]['te_surrogate'][process])
                    median_t = np.median(
                        spectral_result._single_target[target]['target'][i]['te_surrogate'][process])
                    axarr[i, 0].hist(spectral_result._single_target[target]['source'][i]['te_surrogate'][process],
                                     bins=nbin,
                                     color='white', edgecolor='black')

                    axarr[i, 0].plot([te_orig, te_orig], [
                                     0, nbin], 'black', linestyle='dashed')
                    axarr[i, 0].plot([median_s, median_s], [
                                     0, nbin], 'red', linestyle='dashed')
                    axarr[i, 0].text(.5, 1.1,
                                     'Scale {0} (Freq {1}-{2}Hz)'.format(
                                         i, freq1, freq2),
                                     horizontalalignment='center',
                                     transform=axarr[i, 0].transAxes)
                    axarr[0, 0].set_title(
                        'Source shuffled:'.format(process_tested), y=1.3)

                    axarr[i, 0].get_yaxis().set_visible(False)
                    axarr[i, 1].hist(spectral_result._single_target[target]['target'][i]['te_surrogate'][process],
                                     bins=nbin, color='white',
                                     edgecolor='black')
                    axarr[i, 1].plot([te_orig, te_orig], [
                                     0, nbin], 'black', linestyle='dashed')
                    axarr[i, 1].plot([median_t, median_t], [
                                     0, nbin], 'red', linestyle='dashed')
                    axarr[i, 1].get_yaxis().set_visible(False)
                    axarr[i, 1].text(.5, 1.1,
                                     'Scale {0} (Freq {1}-{2}Hz)'.format(
                                         i, freq1, freq2),
                                     horizontalalignment='center',
                                     transform=axarr[i, 1].transAxes)

                    axarr[0, 1].set_title(
                        'Target shuffled:'.format(target), y=1.3)
                    freq1 = freq2
                    freq2 = np.round(freq1/2, 2)

                axarr[i, 0].set_xlabel("Transfer Entropy")
                axarr[i, 1].set_xlabel("Transfer Entropy")
                fig.subplots_adjust(hspace=0.7)

                plt.show()
    return fig


def plot_SOSO_result(SOSO_result, target, source_scale, target_scale):
    """Plot results of SOSO spectral transfer entropy analysis using.

    Plot results of the 'swap-out swap-out' (SOSO) algorithm for testing for
    direct information transfer from source to receiver frequencies.

    References:

    - Pinzuti, E., Wollstadt, P., Gutknecht, A., Tüscher, O., Wibral, M.
      (2020). Measuring spectrally-resolved information transfer for sender-
      and receiver- specific frequencies.
      https://www.biorxiv.org/content/10.1101/2020.02.08.939744v1

    Args:
        spectral_results : ResultsSpectralTE instance
            Results of spectral TE analysis
        freq_rate: int
            samplign rate of the system under analysis

    Returns:
        Figure
            figure handle, Figure object from the matplotlib package
    """
    fig, axarr = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(4, 4))
    nbin = int(round(np.size(
        SOSO_result._single_target[target]['source'][source_scale]['delta_surrogate'])/5))

    axarr.hist(
        SOSO_result._single_target[target]['source'][source_scale]['delta_surrogate'],
        bins=nbin, color='royalblue', edgecolor='black')
    delta = SOSO_result._single_target[target]['source'][source_scale]['delta']
    median_s = np.median(
        SOSO_result._single_target[target]['source'][source_scale]['delta_surrogate'])
    axarr.plot([delta, delta], [0, nbin], 'black')
    axarr.plot([median_s, median_s], [0, nbin], 'red', linestyle='dashed')
    axarr.set_xlabel("Distance delta")
    axarr.set_title('Source scale'.format(source_scale) +
                    '-Target scale'.format(target_scale))
    plt.show()
    return fig
