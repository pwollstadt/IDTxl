"""Export and plot results as networkx objects.
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_network_graph(res):
    """Generate graph object for the full network.

    Generate a graph object from the network of (multivariate)
    interactions (e.g., multivariate TE), using the networkx
    class for directed graphs (DiGraph).

    Args:
        res : dict
            output of multivariate_te.analyse_network()

    Returns:
        instance of a directed graph class from the networkx
        package (DiGraph)
    """
    g = nx.DiGraph()
    g.add_nodes_from(list(res.keys()))
    # Add edges as significant sources.
    for n in res.keys():
        s = np.array(res[n]['selected_vars_sources'],
                     dtype=[('sources', np.int), ('lags', np.int)])
        s = np.unique(s['sources'])
        if s.size > 0:
            print(n)
            print(s)
            edges = [x for x in it.product(s, [n])]
            print(edges)
            g.add_edges_from(edges)
    return g

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
    g = nx.DiGraph()
    target = (res['current_value'][0], 0)
    g.add_node(target)
    # Add either all tested candidates or only significant ones
    # to the graph.
    if sign_sources:
        g.add_nodes_from(res['selected_vars_full'][:])
    else:
        procs = res['sources_tested']
        samples = np.arange(res['current_value'][1] - res['min_lag_sources'],
                            res['current_value'][1] - res['max_lag_sources'],
                            -res['tau_sources'])
        define_candidates = Multivariate_te._define_candidates
        nodes = define_candidates(_, procs, samples)
        g.add_nodes_from(nodes)

    for v in range(len(res['selected_vars_full'])):
        g.add_edge(res['selected_vars_full'][v], target)  # here, I could also add additional info
    print(g.node)
    print(g.edge)
    g.number_of_edges()
    return g

def plot_network(res):
    """Plot network of multivariate TE between processes.

    Plot graph of the network of (multivariate) interactions between
    processes (e.g., multivariate TE). The function uses  the
    networkx class for directed graphs (DiGraph) internally.
    Plots a network and adjacency matrix.

    Args:
        res : dict
            output of multivariate_te.analyse_network()

    Returns:
        instance of a directed graph class from the networkx
        package (DiGraph)
    """
    try:
        res = res['fdr']
    except KeyError:
        print('plotting non-corrected network!')

    g = generate_network_graph(res)
    print(g.node)
    f, (ax1, ax2) = plt.subplots(1,2)
    adj_matrix = nx.to_numpy_matrix(g)
    cmap = sns.light_palette('cadetblue', n_colors=2, as_cmap=True)
    sns.heatmap(adj_matrix, cmap=cmap, cbar=False, ax=ax1,
                square=True, linewidths=1, xticklabels=g.nodes(),
                yticklabels=g.nodes())
    ax1.xaxis.tick_top()
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
    nx.draw_circular(g, with_labels=True, node_size=300, alpha=1.0,
                        ax=ax2, node_color='cadetblue', hold=True,
                        font_weight='bold')
    plt.show()
    return g

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
    g = generate_source_graph(res, sign_sources)
    n = np.array(g.nodes(),
                 dtype=[('procs', np.int), ('lags', np.int)])
    target = tuple(n[n['lags'] == 0][0])
    max_lag = max(res['max_lag_sources'], res['max_lag_target'])
    ind = 0
    color = ['lavender' for c in range(g.number_of_nodes())]
    pos = nx.spring_layout(g)
    for n in g.node:
        if n == target:  # current value
            pos[n] = np.array([max_lag, 0])
        elif n[0] == target[0]:  # target history
            pos[n] = np.array([max_lag - n[1], 0])
        elif n[0] < target[0]:  # sources with proc. number < target
            pos[n] = np.array([max_lag - n[1], n[0] + 1])
        else:  # sources with proc. number > target
            pos[n] = np.array([max_lag - n[1], n[0]])

        if n in res_single['selected_vars_sources']:
            color[ind] = 'cadetblue'
        elif n in res_single['selected_vars_target']:
            color[ind] = 'tomato'
        elif n == target:
            color[ind] = 'red'
        ind += 1

    print(g.node)
    print(color)
    f, (ax1, ax2) = plt.subplots(1,2)
    adj_matrix = nx.to_numpy_matrix(g)
    adj_matrix = adj_matrix[:, g.nodes().index(target)]
    cmap = sns.light_palette('cadetblue', n_colors=2, as_cmap=True)
    sns.heatmap(adj_matrix, cmap=cmap, cbar=False, ax=ax1,
                square=True, linewidths=1, xticklabels=[target],
                yticklabels=g.nodes())
    ax1.xaxis.tick_top()
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
    nx.draw(g, pos=pos, with_labels=True, font_weight='bold',
               node_size=900, alpha=0.7, node_shape='s',
               ax=ax2, node_color=color, hold=True)
    plt.plot([-0.5, max_lag + 0.5], [0.5, 0.5],
             linestyle='--', linewidth=1, color='0.5')
    plt.show()
    return g

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
    g = nx.DiGraph()
    g.add_nodes_from(np.arange(5))
    # g.add_edges_from([(0, 1), (0, 2), (0, 3), (3, 4), (4, 3)])
    g.add_weighted_edges_from([(0, 1, 2), (0, 2, 3),
                               (0, 3, 2), (3, 4, 1), (4, 3, 1)],
                               weight='delay')
    pos = {
        0: np.array([1, 1]),
        1: np.array([0, 2]),
        2: np.array([0, 0]),
        3: np.array([2, 1]),
        4: np.array([3, 1]),
    }
    nx.draw(g, pos=pos, with_labels=True, node_size=900, alpha=1.0,
               node_color='cadetblue', font_weight='bold',
               edge_color=['k', 'k', 'r', 'k', 'k'], hold=True)
    nx.draw_networkx_edge_labels(g, pos=pos)
    plt.text(2, 0.1, 'non-linear interaction in red')
    plt.show()
    # see here for an example on how to plot edge labels:
    # http://stackoverflow.com/questions/10104700/how-to-set-networkx-edge-labels-offset-to-avoid-label-overlap
