"""Export and plot results as networkx objects."""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from .multivariate_te import MultivariateTE

VERBOSE = False


def generate_network_graph(res, n_nodes, fdr=True, find_u='max_te'):
    """Generate graph object for an inferred network.

    Generate a weighted, directed graph object from the network of inferred
    (multivariate) interactions (e.g., multivariate TE), using the networkx
    class for directed graphs (DiGraph). The graph is weighted by the
    reconstructed source-target delays.

    Source-target delays are determined by the lag of the variable in a
    sources' past that has the highest information transfer into the target
    process. There are two ways of idendifying the variable with maximum
    information transfer:

        a) use the variable with the highest absolute TE value (highest
           information transfer)
        b) use the variable with the smallest p-value (highest statistical
           significance)

    Args:
        res : dict
            output of multivariate_te.analyse_network()
        n_nodes : int
            number of nodes in the network
        fdr : bool
            print FDR-corrected results (default=True)
        find_u : str
            use TE value ('max_te') or p-value ('max_p') to determine the
            variable with maximum information transfer into the target in order
            to determine the source-target delay (default='max_te')

    Returns:
        instance of a directed graph class from the networkx
        package (DiGraph)
    """
    adj_matrix = _get_adj_matrix(res, n_nodes, fdr, find_u)
    return nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph())


def generate_source_graph(res, sign_sources=True):
    """Generate graph object for a target process and single variables.

    Generate a graph object from the network of (multivariate)
    interactions (e.g., multivariate TE) between single samples and
    a target sample (current value), using the networkx class for
    directed graphs (DiGraph).

    Args:
        res : dict
            output of multivariate_te.analyse_single_target()
        sign_sources : bool
            add only sources significant information contribution
            (default=True)

    Returns:
        instance of a directed graph class from the networkx
        package (DiGraph)
    """
    try:
        target = (res['current_value'][0], 0)
    except KeyError:
        KeyError('Input res should be result of analyse_single_target() '
                 'method.')
    graph = nx.DiGraph()
    graph.add_node(target)
    # Add either all tested candidates or only significant ones
    # to the graph.
    if sign_sources:
        graph.add_nodes_from(res['selected_vars_full'][:])
    else:
        procs = res['sources_tested']
        samples = np.arange(res['current_value'][1] - res['min_lag_sources'],
                            res['current_value'][1] - res['max_lag_sources'],
                            -res['tau_sources'])
        define_candidates = MultivariateTE._define_candidates
        nodes = define_candidates([], procs, samples)
        graph.add_nodes_from(nodes)

    for v in range(len(res['selected_vars_full'])):
        # Here, one could add additional info in the future, networkx graphs
        # support attributes for graphs, nodes, and edges.
        graph.add_edge(res['selected_vars_full'][v], target)
    if VERBOSE:
        print(graph.node)
        print(graph.edge)
        graph.number_of_edges()
    return graph


def plot_network(res, n_nodes, fdr=True, find_u='max_te'):
    """Plot network of multivariate TE between processes.

    Plot graph of the network of (multivariate) interactions between
    processes (e.g., multivariate TE). The function uses  the
    networkx class for directed graphs (DiGraph) internally.
    Plots a network and adjacency matrix.

    Args:
        res : dict
            output of multivariate_te.analyse_network()
        n_nodes : int
            number of nodes in the network
        fdr : bool
            print FDR-corrected results (default=True)
        find_u : str
            use TE value ('max_te') or p-value ('max_p') to determine the
            variable with maximum information transfer into the target in order
            to determine the source-target delay (default='max_te')

    Returns:
        instance of a directed graph class from the networkx
        package (DiGraph)
    """
    graph = generate_network_graph(res, n_nodes, fdr, find_u)
    adj_matrix = nx.to_numpy_matrix(graph)
    max_u = np.max(adj_matrix)
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
    ax2 = plt.subplot(122)
    plt.imshow(adj_matrix, cmap="gray_r", interpolation="none")
    # Make colorbar match the image in size:
    # https://stackoverflow.com/a/26720422
    cbar = plt.colorbar(fraction=0.046, pad=0.04,
                        ticks=np.arange(0, max_u + 1).astype(int))
    cbar.set_label('delay [samples]', rotation=90)
    plt.xticks(np.arange(adj_matrix.shape[1]))
    plt.yticks(np.arange(adj_matrix.shape[0]))
    ax2.xaxis.tick_top()
    plt.show()
    return graph


def plot_selected_vars(res, sign_sources=True):
    """Plot network of a target process and single variables.

    Plot graph of the network of (multivariate) interactions between
    source variables and the current value. The function uses the
    networkx class for directed graphs (DiGraph) internally.
    Plots a network and reduced adjacency matrix.

    Args:
        res : dict
            output of multivariate_te.analyse_single_target()
        sign_sources : bool
            add only sources significant information contribution
            (default=True)

    Returns:
        instance of a directed graph class from the networkx
        package (DiGraph)
    """
    graph = generate_source_graph(res, sign_sources)
    n = np.array(graph.nodes(),
                 dtype=[('procs', np.int), ('lags', np.int)])
    target = tuple(n[n['lags'] == 0][0])
    max_lag = max(res['max_lag_sources'], res['max_lag_target'])
    ind = 0
    color = ['lavender' for c in range(graph.number_of_nodes())]
    pos = nx.spring_layout(graph)
    for n in graph.node:
        if n == target:  # current value
            pos[n] = np.array([max_lag, 0])
        elif n[0] == target[0]:  # target history
            pos[n] = np.array([max_lag - n[1], 0])
        elif n[0] < target[0]:  # sources with proc. number < target
            pos[n] = np.array([max_lag - n[1], n[0] + 1])
        else:  # sources with proc. number > target
            pos[n] = np.array([max_lag - n[1], n[0]])

        if n in res['selected_vars_sources']:
            color[ind] = 'cadetblue'
        elif n in res['selected_vars_target']:
            color[ind] = 'tomato'
        elif n == target:
            color[ind] = 'red'
        ind += 1

    if VERBOSE:
        print(graph.node)
        print(color)
    plt.figure()
    nx.draw(graph, pos=pos, with_labels=True, font_weight='bold',
            node_size=900, alpha=0.7, node_shape='s', node_color=color,
            hold=True)
    plt.plot([-0.5, max_lag + 0.5], [0.5, 0.5],
             linestyle='--', linewidth=1, color='0.5')
    plt.show()
    return graph


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


def print_res_to_console(data, res, fdr=True, find_u='max_te'):
    """Print results of network inference to console.

    Print results of network inference to console. Output looks like this:

        0 -> 1, u = 2
        0 -> 2, u = 3
        0 -> 3, u = 2
        3 -> 4, u = 1
        4 -> 3, u = 1

    indicating significant information transfer source -> target with a source-
    target delay u. The network can either be plotted from FDR-

    Source-target delays are determined by the lag of the variable in a
    sources' past that has the highest information transfer into the target
    process. There are two ways of idendifying the variable with maximum
    information transfer:

        a) use the variable with the highest absolute TE value (highest
           information transfer)
        b) use the variable with the smallest p-value (highest statistical
           significance)

    Args:
        data : Data() instance
            raw data
        res : dict
            output of network inference algorithm, e.g., MultivariateTE
        fdr : bool
            print FDR-corrected results (default=True)
        find_u : str
            use TE value ('max_te') or p-value ('max_p') to determine the
            variable with maximum information transfer into the target in order
            to determine the source-target delay (default='max_te')

    Returns:
        numpy array
            adjacency matrix describing multivariate TE between all network
            nodes, entries in the matrix denote source-target-delays
    """
    # Generate adjacency matrix from results.
    n_nodes = data.n_processes
    adj_matrix = _get_adj_matrix(res, n_nodes, fdr, find_u)

    # Print link to console.
    link_found = False
    for s in range(n_nodes):
        for t in range(n_nodes):
            if adj_matrix[s, t]:
                print('\t{0} -> {1}, u: {2}'.format(s, t, adj_matrix[s, t]))
                link_found = True
    if not link_found:
        print('No significant links in network.')

    return adj_matrix


def _get_adj_matrix(res, n_nodes, fdr=True, find_u='max_te'):
    """Return adjacency matrix as numpy array.

    Return results of network inference as directed adjacency matrix. Output is
    a 2D numpy-array where non-zero entries indicate a significant link and the
    integer denotes the source-target delay between nodes.

    Source-target delays are determined by the lag of the variable in a
    sources' past that has the highest information transfer into the target
    process. There are two ways of idendifying the variable with maximum
    information transfer:

        a) use the variable with the highest absolute TE value (highest
           information transfer)
        b) use the variable with the smallest p-value (highest statistical
           significance)

    Args:
        res : dict
            output of network inference algorithm, e.g., MultivariateTE
        n_nodes : int
            number of nodes in the network
        fdr : bool
            print FDR-corrected results (default=True)
        find_u : str
            use TE value ('max_te') or p-value ('max_p') to determine the
            variable with maximum information transfer into the target in order
            to determine the source-target delay (default='max_te')

    Returns:
        numpy array
            adjacency matrix describing multivariate TE between all network
            nodes, entries in the matrix denote source-target-delays
    """
    # Check if FDR-corrected or uncorrected results are requested.
    if fdr:
        try:
            r = res['fdr']
        except KeyError:
            raise RuntimeError('No FDR-corrected results found.')
    else:
        r = res.copy()
        try:
            del r['fdr']
        except KeyError:
            pass

    targets = list(r.keys())
    adj_matrix = np.zeros((n_nodes + 1, n_nodes + 1)).astype(int)

    for t in targets:
        all_vars_sources = np.array([x[0] for x in
                                     r[t]['selected_vars_sources']])
        all_vars_lags = np.array([x[1] for x in r[t]['selected_vars_sources']])
        sources = np.unique(all_vars_sources)
        pval = r[t]['selected_sources_pval']
        te = r[t]['selected_sources_te']
        u = np.empty(sources.shape[0])

        for s in sources:
            # Find u as the variable with either the smalles p-value or highest
            # TE-value.
            if find_u == 'max_p':
                u_ind = np.argmin(pval[all_vars_sources == s])
            elif find_u == 'max_te':
                u_ind = np.argmax(te[all_vars_sources == s])
            u = all_vars_lags[all_vars_sources == s][u_ind]

            adj_matrix[s, t] = u

    return adj_matrix
