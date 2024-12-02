import numpy as np
import scipy as sc
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors as Nb
from scipy.special import digamma
from idtxl.estimator import Estimator

class MixedTEEstimator(Estimator):
    def __init__(self):
        # Initialize the multiprocessing pool with all available CPU cores
        self.pool = mp.Pool(mp.cpu_count())

    def is_parallel(self):
        return False

    def is_analytic_null_estimator(self):
        return False

    @staticmethod
    def norm_combination(x1, x2, dims, n_vecs, marginal_norms):
        # Calculate a combined norm for two vectors x1 and x2
        curr_dims = 0
        marginal_norms_list = np.zeros(n_vecs)
        for i, dimsi in enumerate(dims):
            marginal_norms_list[i] = sum(
                (x1[curr_dims:curr_dims + dimsi] - x2[curr_dims:curr_dims + dimsi]) ** marginal_norms[i]) ** (
                                             1 / marginal_norms[i])
            curr_dims = curr_dims + dimsi
        return np.max(marginal_norms_list)

    @staticmethod
    def disc_entropy_nvars(listofdata, N):
        # Calculate the discrete entropy of multiple variables
        jointdata = np.column_stack([data for data in listofdata])
        joint_counts = np.unique(jointdata, axis=0, return_counts=True)[1]
        return - 1 / N * sum(joint_counts * np.log(joint_counts / N))

    @staticmethod
    def NN_searcher_single_get_n_x(x_past, dims_x, radii, N):
        # Search for the nearest neighbors of points in x_past and calculate the number of neighbors within given radii
        assert N > 0
        if len(x_past) == 0:
            return 0
        if len(x_past) == 1:
            return 0
        try:
            len_past = np.int32(len(x_past[0]) / dims_x)
        except:
            len_past = 1
        neigh_x_past = Nb(radius=np.inf, metric=MixedTEEstimator.norm_combination,
                          metric_params={'dims': dims_x * np.ones(len_past, dtype=np.int32), 'n_vecs': len_past,
                                         'marginal_norms': 2 * np.ones(len_past)})
        n_x_past_i = 0
        if len_past == 1:
            x_past_shaped = x_past.reshape(-1, 1)
            neigh_x_past.fit(x_past_shaped)
            for i in np.arange(len(x_past)):
                curr_dist = radii[i]
                neighbor_info = neigh_x_past.radius_neighbors(X=(x_past[i]).reshape(-1, 1), radius=curr_dist,
                                                              return_distance=False)
                n_x_past_i = n_x_past_i + digamma(len(neighbor_info[0]))
        else:
            x_past_shaped = x_past
            neigh_x_past.fit(x_past_shaped)
            for i in np.arange(N):
                curr_dist = radii[i]
                neighbor_info = neigh_x_past.radius_neighbors(X=np.array([x_past[i]]), radius=curr_dist,
                                                              return_distance=False)
                n_x_past_i = n_x_past_i + digamma(len(neighbor_info[0]))

        return n_x_past_i

    @staticmethod
    def NN_searcher_single_get_radius(x_past, dims_x, k):
        # Find the radius within which there are k nearest neighbors for each point in x_past
        if len(x_past) < k + 1:
            print("Conditional dataset too small")
            return 0
        try:
            len_past = np.int32(len(x_past[0]) / dims_x)
        except:
            len_past = 1
        if len_past == 1:
            x_past_shaped = x_past.reshape(-1, 1)
        else:
            x_past_shaped = x_past

        neigh_x_past = Nb(n_neighbors=k + 1, radius=np.inf, metric=MixedTEEstimator.norm_combination,
                          metric_params={'dims': dims_x * np.ones(len_past, dtype=np.int32), 'n_vecs': len_past,
                                         'marginal_norms': 2 * np.ones(len_past)})
        neigh_x_past.fit(x_past_shaped)
        nn_x_past_dists, nn_x_past_inds = neigh_x_past.kneighbors(x_past_shaped)
        return nn_x_past_dists[:, -1]

    @staticmethod
    def estimateTE_cc(x_past, y_t, y_past, k, N, dims_x, dims_y):
        # Estimate transfer entropy from a continuous past state x_past to a continuous target state y_t
        x_past_transposed = x_past.transpose()
        y_t_transposed = y_t.transpose()
        y_past_transposed = y_past.transpose()
        data_x_past_y_past = np.append(x_past_transposed.ravel(), x_past_transposed.ravel()).reshape(
            (dims_y + dims_x, N)).transpose()
        data_y = np.append(y_past_transposed.ravel(), y_t_transposed.ravel()).reshape((dims_y + 1, N)).transpose()
        data_xy = (np.append(x_past_transposed.ravel(), data_y.ravel()).reshape((dims_x + 1 + dims_y, N))).transpose()
        print("Searching radii...")
        radii = MixedTEEstimator.NN_searcher_single_get_radius(data_xy, dims_x + dims_y + 1, k)
        print("Searching NN...")
        sum_n_y_past = MixedTEEstimator.NN_searcher_single_get_n_x(y_past, dims_y, radii, N)
        sum_n_y = MixedTEEstimator.NN_searcher_single_get_n_x(data_y, dims_y + 1, radii, N)
        sum_n_x_past_y_past = MixedTEEstimator.NN_searcher_single_get_n_x(data_x_past_y_past, dims_x + dims_y, radii, N)

        return digamma(k) + 1 / N * (sum_n_y_past - sum_n_y - sum_n_x_past_y_past)

    @staticmethod
    def estimateTE_cd(x_past, y_t, y_past, k, N, dims_x, dims_y):
        # Estimate transfer entropy from a continuous past state x_past to a discrete target state y_t
        data_y = np.column_stack((y_past, y_t))
        realizations_y_past, count_y_past = np.unique(y_past, axis=0, return_counts=True)
        realizations_y, count_y = np.unique(data_y, axis=0, return_counts=True)
        radii_y = np.zeros(N)
        digamma_nx_past_conditioned_y_past = 0

        for y in realizations_y:
            condition_y = np.all(data_y == y, axis=1)
            x_past_conditioned_y = x_past[condition_y]
            if len(x_past_conditioned_y) >= k:
                radii_conditioned_y = MixedTEEstimator.NN_searcher_single_get_radius(x_past_conditioned_y, dims_x, k)
                radii_y[np.where(condition_y)[0]] = radii_conditioned_y
            else:
                radii_y[np.where(condition_y)[0]] = np.inf * np.ones(sum(condition_y))

        for y_p in realizations_y_past:
            condition_y_p = np.all(data_y == y_p, axis=1)
            x_past_conditioned_y_p = x_past[condition_y_p]
            digamma_nx_past_conditioned_y_past = digamma_nx_past_conditioned_y_past + MixedTEEstimator.NN_searcher_single_get_n_x(
                x_past_conditioned_y_p, dims_x, radii_y[condition_y_p], N)

        return digamma(k) - 1 / N * digamma_nx_past_conditioned_y_past

    @staticmethod
    def estimateTE_dc(x_past, y_t, y_past, k, N, dims_x, dims_y):
        # Estimate transfer entropy from a discrete past state x_past to a continuous target state y_t
        try:
            len_past = np.int32(len(x_past[0]) / dims_x)
        except:
            len_past = 1
        y_t_transposed = y_t.transpose()
        y_past_transposed = y_past.transpose()
        data_y = (
            np.append(y_past_transposed.ravel(), y_t_transposed.ravel()).reshape((len_past + 1) * dims_y, N)).transpose()

        realizations_x_past = np.unique(x_past, axis=0)
        radii_y_past = np.zeros(N)
        radii_y = np.zeros(N)
        for x in realizations_x_past:
            if dims_x == 1:
                x_past_reshape = x_past.reshape(-1, 1)
                condition_x = np.all(x_past_reshape == x.reshape(-1, 1), axis=1)
            else:
                condition_x = np.all(x_past == x, axis=1)
            y_conditioned_x = data_y[condition_x]
            y_past_conditioned_x = y_past[condition_x]
            radii_y_conditioned_x = MixedTEEstimator.NN_searcher_single_get_radius(y_conditioned_x, dims_y, k)
            radii_y[np.where(condition_x)[0]] = radii_y_conditioned_x
            radii_y_past_conditioned_x = MixedTEEstimator.NN_searcher_single_get_radius(y_past_conditioned_x, dims_y, k)
            radii_y_past[np.where(condition_x)[0]] = radii_y_past_conditioned_x
        ny_past_i = MixedTEEstimator.NN_searcher_single_get_n_x(y_past, dims_y, radii_y_past, N)
        ny_i = MixedTEEstimator.NN_searcher_single_get_n_x(data_y, dims_y, radii_y, N)

        return 1 / N * (ny_i - ny_past_i)

    @staticmethod
    def estimateTE_dd(x_past, y_t, y_past, N):
        # Estimate transfer entropy from a discrete past state x_past to a discrete target state y_t
        H_x_past_y_past = MixedTEEstimator.disc_entropy_nvars([x_past, y_past], N)
        H_x_past_y_past_y_t = MixedTEEstimator.disc_entropy_nvars([x_past, y_past, y_t], N)
        H_y = MixedTEEstimator.disc_entropy_nvars([y_past, y_t], N)
        H_y_past = MixedTEEstimator.disc_entropy_nvars([y_past], N)
        return H_x_past_y_past + H_y - H_y_past - H_x_past_y_past_y_t

    @staticmethod
    def estimateTE(x_past, y_t, y_past, k, dims_x, dims_y, variable_kinds=None, marginal_norm=2):
        # Wrapper method that calls the appropriate transfer entropy estimation method based on variable types
        N = len(y_t)
        if variable_kinds is None:
            if len(np.unique(x_past, axis=0)) - N == 0:
                if len(np.unique(y_past, axis=0)) - N == 0:
                    variable_kinds = "cc"
                else:
                    variable_kinds = "cd"
            else:
                if len(np.unique(y_past, axis=0)) - N == 0:
                    variable_kinds = "dc"
                else:
                    variable_kinds = "dd"

        if variable_kinds == "cc":
            return MixedTEEstimator.estimateTE_cc(x_past, y_t, y_past, k, N, dims_x, dims_y)
        if variable_kinds == "cd":
            y_t_double = y_t * 1.0
            y_past_double = y_past * 1.0
            return MixedTEEstimator.estimateTE_cd(x_past, y_t_double, y_past_double, k, N, dims_x, dims_y)
        if variable_kinds == "dc":
            x_past_double = x_past * 1.0
            return MixedTEEstimator.estimateTE_dc(x_past_double, y_t, y_past, k, N, dims_x, dims_y)
        if variable_kinds == "dd":
            return MixedTEEstimator.estimateTE_dd(x_past, y_t, y_past, N)

        return None

    @staticmethod
    def discretize1d(x, N_bins):
        # Discretize a one-dimensional continuous array x into N_bins equal-width bins
        return np.digitize(x, np.linspace(np.min(x), np.max(x), N_bins))

    @staticmethod
    def discretize(x, N, bins=None, N_bins=10000, rounding=False):
        # Discretize a multi-dimensional continuous array x
        if rounding:
            return np.round(x)
        if bins is None:
            if x.ndim == 1:
                return np.digitize(x, np.linspace(np.min(x), np.max(x), N_bins))
            else:
                xt = x.transpose()
                for dim in np.arange(x.ndim):
                    mybins = np.linspace(np.min(xt[dim]).round(), np.max(xt[dim]).round(), N_bins)
                    xt[dim] = np.digitize(xt[dim], bins=mybins)
                return xt.transpose()
        else:
            assert x.ndim == bins.ndim
            if x.ndim == 1:
                return np.digitize(x, bins)
            else:
                xt = x.transpose()
                for dim in np.arange(x.ndim):
                    xt[dim] = np.digitize(xt[dim], bins=bins[dim])
                return xt.transpose()

    @staticmethod
    def fuzzify(x, amplitude, bin_size):
        # Add random noise to the data in x to "fuzzify" it
        return x + amplitude * sc.random.normal(loc=0, scale=bin_size, size=x.shape)
