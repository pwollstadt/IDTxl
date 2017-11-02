"""Export and plot results as networkx objects."""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def plot_network(results, fdr=False):
    """Plot network of multivariate TE between processes.

    Plot graph of the network of (multivariate) interactions between processes
    (e.g., multivariate TE). The function uses  the networkx class for directed
    graphs (DiGraph) internally. Plots a network and adjacency matrix.

    Args:
        results : ResultsNetworkInference() instance
            output of an network inference algorithm
        fdr : bool [optional]
            print FDR-corrected results (default=False)

    Returns:
        instance of a directed graph class from the networkx package (DiGraph)
    """
    graph = results.export_networkx_graph(fdr)
    print(graph.node)

    plt.figure(figsize=(10, 5))
    # Plot graph.
    ax1 = plt.subplot(121)
    nx.draw_circular(graph, with_labels=True, node_size=600, alpha=1.0, ax=ax1,
                     node_color='Gainsboro', hold=True, font_size=14,
                     font_weight='bold')
    pos = nx.circular_layout(graph)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    print(edge_labels)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                 font_size=13)  # font_weight='bold'

    # Plot adjacency matrix.
    plt.subplot(122)
    if fdr:
        _plot_adj_matrix(results.fdr_correction.adjacency_matrix)
    else:
        _plot_adj_matrix(results.adjacency_matrix)
    plt.show()
    return graph


def plot_selected_vars(results, target, sign_sources=True, fdr=False):
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
            add only sources significant information contribution
            (default=True)
        fdr : bool [optional]
            print FDR-corrected results (default=False)

    Returns:
        instance of a directed graph class from the networkx
        package (DiGraph)
    """
    graph = results.export_networkx_source_graph(target, sign_sources, fdr)
    current_value = results.single_target[target].current_value
    max_lag = max(results.settings.max_lag_sources,
                  results.settings.max_lag_target)

    # Adjust color and position of nodes (variables).
    pos = nx.spring_layout(graph)
    color = ['lavender' for c in range(graph.number_of_nodes())]
    for (ind, n) in enumerate(graph.node):

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
        if n in results.single_target[target].selected_vars_sources:
            color[ind] = 'cadetblue'
        elif n in results.single_target[target].selected_vars_target:
            color[ind] = 'tomato'
        elif n == current_value:
            color[ind] = 'red'

    plt.figure()
    nx.draw(graph, pos=pos, with_labels=True, font_weight='bold',
            node_size=900, alpha=0.7, node_shape='s', node_color=color,
            hold=True)
    plt.plot([-0.5, max_lag + 0.5], [0.5, 0.5],
             linestyle='--', linewidth=1, color='0.5')
    plt.show()
    return graph


def _plot_adj_matrix(adj_matrix, mat_color='gray_r', diverging=False,
                     cbar_label='delay', cbar_stepsize=1):
    """Plot adjacency matrix."""
    # Plot matrix, set minimum and maximum values to the same value for
    # diverging plots to center colormap at 0, i.e., 0 is plotted in white
    # https://stackoverflow.com/questions/25500541/
    # matplotlib-bwr-colormap-always-centered-on-zero
    if diverging:
        max_val = np.max(abs(adj_matrix))
        min_val = -max_val
    else:
        max_val = np.max(adj_matrix)
        min_val = -np.min(adj_matrix)
    plt.imshow(adj_matrix, cmap=mat_color, interpolation='nearest',
               vmin=min_val, vmax=max_val)

    # Set the colorbar and make colorbar match the image in size using the
    # fraction and pad parameters (see https://stackoverflow.com/a/26720422).
    if cbar_label == 'delay':
        cbar_label = 'delay [samples]'
        cbar_ticks = np.arange(0, max_val + 1, cbar_stepsize)
    else:
        cbar_ticks = np.arange(min_val, max_val + 0.01 * max_val,
                               cbar_stepsize)
    cbar = plt.colorbar(fraction=0.046, pad=0.04, ticks=cbar_ticks)
    cbar.set_label(cbar_label, rotation=90)

    # Set x- and y-ticks.
    plt.xticks(np.arange(adj_matrix.shape[1]))
    plt.yticks(np.arange(adj_matrix.shape[0]))
    ax = plt.gca()
    ax.xaxis.tick_top()
    return cbar


def plot_mute_graph():
    """Plot MuTE example network.

    Network of 5 AR-processes, which is used as an example the paper
    on the MuTE toolbox (Montalto, PLOS ONE, 2014, eq. 14). The
    network consists of five autoregressive (AR) processes with model
    orders 2 and les and the following (non-linear) couplings:
        0 -> 1, u = 2
        0 -> 2, u = 3
        0 -> 3, u = 2 (non-linear)
        3 -> 4, u = 1
        4 -> 3, u = 1
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
    plt.figure()
    nx.draw(graph, pos=pos, with_labels=True, node_size=900, alpha=1.0,
            node_color='cadetblue', font_weight='bold',
            edge_color=['r', 'k', 'r', 'k', 'k'], hold=True)
    nx.draw_networkx_edge_labels(graph, pos=pos)
    plt.text(2, 0.1, 'non-linear interaction in red')
    plt.show()
    # see here for an example on how to plot edge labels:
    # http://stackoverflow.com/questions/10104700/how-to-set-networkx-edge-labels-offset-to-avoid-label-overlap


def plot_network_comparison(results, mask_sign=True, show=True):
    """Plot results of network comparison."""
    union = results['union_network']

    targets = results['union_network']['targets']
    n_nodes = max(targets) + 1
    union_network = np.zeros((n_nodes, n_nodes), dtype=int)
    adj_matrix_te_diff = np.zeros((n_nodes, n_nodes))
    adj_matrix_pval = np.zeros((n_nodes, n_nodes))
    adj_matrix_comp = np.zeros((n_nodes, n_nodes))

    for t in targets:
        all_vars_sources = np.array([x[0] for x in
                                     union[t]['selected_vars_sources']])
        all_vars_lags = np.array([x[1] for x in
                                  union[t]['selected_vars_sources']])

        for s in np.unique(all_vars_sources):
            union_network[s, t] = 1

        pval = results['pval'][t]
        sign = results['sign'][t]
        te_diff = results['cmi_diff_abs'][t]
        comp = results['a>b'][t].astype(int)
        comp[comp == 0] = -1
        if mask_sign:
            all_vars_sources = all_vars_sources[sign]
            all_vars_lags = all_vars_lags[sign]
            pval = pval[sign]
            te_diff = te_diff[sign]
            comp = comp[sign]

        sources = np.unique(all_vars_sources)

        for s in sources:
            # For now, if there are multiple variables in the past of a source,
            # report the one with the maximum TE difference
            te_diff_abs = abs(te_diff)
            max_te = max(te_diff_abs[all_vars_sources == s])
            idx = np.where(te_diff_abs == max_te)[0]
            adj_matrix_te_diff[s, t] = te_diff[idx]
            adj_matrix_pval[s, t] = pval[idx]
            adj_matrix_comp[s, t] = comp[idx]

    adj_matrix_te_diff_sign = adj_matrix_te_diff / np.max(adj_matrix_te_diff)
    adj_matrix_te_diff_sign *= adj_matrix_comp

    f = plt.figure(figsize=(10, 10))
    ax = plt.subplot(221)
    _plot_adj_matrix(adj_matrix_te_diff_sign, mat_color='seismic',
                     diverging=True, cbar_label='norm. CMI diff [a.u.]',
                     cbar_stepsize=0.1)
    ax.set_title('CMI raw diff cond. A - B', y=1.1)
    ax = plt.subplot(222)
    _plot_adj_matrix(adj_matrix_pval, cbar_label='p-value [%]',
                     cbar_stepsize=0.005)
    ax.set_title('p-values cond. A vs. B', y=1.1)
    ax = plt.subplot(223)
    _plot_adj_matrix(union_network, mat_color='PuBu',
                     cbar_label='link in union', cbar_stepsize=1)
    ax.set_title('union network A and B', y=1.1)
    # ax = plt.subplot(224)
    # cbar = _plot_adj_matrix(adj_matrix_comp, diverging=True,
    #                         mat_color='bwr', cbar_stepsize=1)
    # cbar.set_ticks([1, -1])
    # cbar.ax.set_yticklabels(['A > B', 'B>=A'])
    # ax.set_title('Comparison mean TE', y=1.1)

    if show:
        plt.show()

    return adj_matrix_te_diff_sign, adj_matrix_pval, f
