#!/usr/bin/python3
import sys
import random
import time as tm
import numpy as np
import TartuSynergy
from idtxl import estimators_pid
#
# Main function: main()
# =====================
#
# Usage:
#             main([0, fun, noise_level, n_samples])
#        or   main([0, '-v', fun, noise_level, n_samples])
#
# Makes n_samples samples from function `fun' (fun is a string
# identifying the function), adds noise to it, then determines shared
# information and synergetic information.
# If -v is given, write lots and lots of progress output.
#
# Add new functions
# =================
#
# To add a new function, you have to define a class, and add an object
# of it to the dictionary of available functions.  (This dictionary
# maps capitalized names of function to function objects.)
#
# The functions are defined throuh classes.
# The class must containt the following
#
# *  n_u, n_v, n_w --- desired input length
#
# *  fun(u,v,w) -- take three independent random variables
#    u in {0,...,n_u-1}, v in {0,...,n_v-1}, w in {0,...,n_w-1}
#    Return a triple X,Y,Z
#
# *  noise(X) adds random noise to the X-part of X,Y,Z
#
# * true_input_distrib()    true PDF (not sampled)
#   true_result_distrib()   right optimal PDF (when input is not sampled)
#   true_CI()               true synergetic info (when input is not sampled)
#   true_SI()               true shared info  (when input is not sampled)


#^ END OF DOCUMENTATION

the_available_functions = dict()


class Fun_Data_rdn:

    def __init__(self):
        self.n_u = self.n_v = 0
        self.n_w = 2

    def fun(self, y, z, w):
        return w, w, w

    def noise(self, x):
        x = (x + random.randint(0, 1)) % 2
        return x

    def true_input_distrib(self):
        return {(0, 0, 0): .5, (1, 1, 1): .5}

    def true_result_distrib(self):
        return {(0, 0, 0): .5, (1, 1, 1): .5}

    def true_CI(self):
        return 0

    def true_SI(self):
        return 1


the_available_functions["RDN"] = Fun_Data_rdn


class Fun_Data_unq:

    def __init__(self):
        self.n_u = self.n_v = 2
        self.n_w = 0

    def fun(self, y, z, w):
        return (y, z), y, z

    def noise(self, x):
        y, z = x
        y = (y + random.randint(0, 1)) % 2
        z = (z + random.randint(0, 1)) % 2
        return y, z

    def true_input_distrib(self):
        return {((0, 0), 0, 0): .25, ((0, 1), 0, 1): .25, ((1, 0), 1, 0): .25, ((1, 1), 1, 1): .25}

    def true_result_distrib(self):
        return {((0, 0), 0, 0): .25, ((0, 1), 0, 1): .25, ((1, 0), 1, 0): .25, ((1, 1), 1, 1): .25}

    def true_CI(self):
        return 0  # this is a mistake in BROJA'14

    def true_SI(self):
        return 0
the_available_functions["UNQ"] = Fun_Data_unq()


class Fun_Data_xor:

    def __init__(self):
        self.n_u = self.n_v = 2
        self.n_w = 0
        self.name = 'xor'

    def fun(self, y, z, w):
        return (y + z) % 2, y, z

    def noise(self, x):
        x = (x + random.randint(0, 1)) % 2
        return x

    def true_input_distrib(self):
        return {(0, 0, 0): .25, (1, 0, 1): .25, (1, 1, 0): .25, (0, 1, 1): .25}

    def true_result_distrib(self):
        return {(0, 0, 0): .25, (1, 0, 1): .25, (1, 1, 0): .25, (0, 1, 1): .25}

    def true_CI(self):
        return 1

    def true_SI(self):
        return 0
the_available_functions["XOR"] = Fun_Data_xor()


class Fun_Data_and:

    def __init__(self):
        self.n_u = self.n_v = 2
        self.n_w = 0
        self.name = 'and'

    def fun(self, y, z, w):
        return (y * z) % 2, y, z

    def noise(self, x):
        x = (x + random.randint(0, 1)) % 2
        return x

    def true_input_distrib(self):
        return {(0, 0, 0): .25, (0, 0, 1): .25, (0, 1, 0): .25, (1, 1, 1): .25}

    def true_result_distrib(self):
        return {(0, 0, 0): .5, (0, 1, 1): .25, (1, 1, 1): .25}

    def true_CI(self):
        return .5

    def true_SI(self):
        return .311278124459132843017578125  # I hope
the_available_functions["AND"] = Fun_Data_and()


class Fun_Data_rdnxor:

    def __init__(self):
        self.n_u = self.n_v = self.n_w = 2

    def fun(self, y, z, w):
        return ((y + z) % 2, w), (y, w), (z, w)

    def noise(self, x):
        x1, w = x
        x1 = (x1 + random.randint(0, 1)) % 2
        w = (w + random.randint(0, 1)) % 2
        return x1, w

    def true_input_distrib(self):
        return {
            ((0, 0), (0, 0), (0, 0)): .125,
            ((1, 0), (0, 0), (1, 0)): .125,
            ((1, 0), (1, 0), (0, 0)): .125,
            ((0, 0), (1, 0), (1, 0)): .125,
            ((0, 1), (0, 1), (0, 1)): .125,
            ((1, 1), (0, 1), (1, 1)): .125,
            ((1, 1), (1, 1), (0, 1)): .125,
            ((0, 1), (1, 1), (1, 1)): .125
        }

    def true_result_distrib(self):
        return {((0, 0), (0, 0), (0, 0)): 0.0625, ((0, 0), (0, 0), (1, 0)): 0.0625, ((0, 0), (1, 0), (0, 0)): 0.0625, ((0, 0), (1, 0), (1, 0)): 0.0625, ((0, 1), (0, 1), (0, 1)): 0.0625, ((0, 1), (0, 1), (1, 1)): 0.0625, ((0, 1), (1, 1), (0, 1)): 0.0625, ((0, 1), (1, 1), (1, 1)): 0.0625, ((1, 0), (0, 0), (0, 0)): 0.0625, ((1, 0), (0, 0), (1, 0)): 0.0625, ((1, 0), (1, 0), (0, 0)): 0.0625, ((1, 0), (1, 0), (1, 0)): 0.0625, ((1, 1), (0, 1), (0, 1)): 0.0625, ((1, 1), (0, 1), (1, 1)): 0.0625, ((1, 1), (1, 1), (0, 1)): 0.0625, ((1, 1), (1, 1), (1, 1)): 0.0625}

    def true_CI(self):
        return 1

    def true_SI(self):
        return 1
the_available_functions["RDNXOR"] = Fun_Data_rdnxor()


class Fun_Data_rdnunqxor:

    def __init__(self):
        self.n_u = self.n_v = 4
        self.n_w = 2

    def fun(self, y, z, w):
        y1 = y % 2
        y2 = y // 2
        z1 = z % 2
        z2 = z // 2
        return ((y1 + z1) % 2, y2, z2, w), (y, w), (z, w)

    def noise(self, x):
        x1, y2, z2, w = x
        x1 = (x1 + random.randint(0, 1)) % 2
        y2 = (y2 + random.randint(0, 1)) % 2
        z2 = (z2 + random.randint(0, 1)) % 2
        w = (w + random.randint(0, 1)) % 2
        return x1, y2, z2, w

    def true_input_distrib(self):
        return {
            ((0, 0, 0, 0), (0, 0), (0, 0)): 1 / 32,
            ((1, 0, 0, 0), (0, 0), (1, 0)): 1 / 32,
            ((0, 0, 1, 0), (0, 0), (2, 0)): 1 / 32,
            ((1, 0, 1, 0), (0, 0), (3, 0)): 1 / 32,
            ((1, 0, 0, 0), (1, 0), (0, 0)): 1 / 32,
            ((0, 0, 0, 0), (1, 0), (1, 0)): 1 / 32,
            ((1, 0, 1, 0), (1, 0), (2, 0)): 1 / 32,
            ((0, 0, 1, 0), (1, 0), (3, 0)): 1 / 32,
            ((0, 1, 0, 0), (2, 0), (0, 0)): 1 / 32,
            ((1, 1, 0, 0), (2, 0), (1, 0)): 1 / 32,
            ((0, 1, 1, 0), (2, 0), (2, 0)): 1 / 32,
            ((1, 1, 1, 0), (2, 0), (3, 0)): 1 / 32,
            ((1, 1, 0, 0), (3, 0), (0, 0)): 1 / 32,
            ((0, 1, 0, 0), (3, 0), (1, 0)): 1 / 32,
            ((1, 1, 1, 0), (3, 0), (2, 0)): 1 / 32,
            ((0, 1, 1, 0), (3, 0), (3, 0)): 1 / 32,
            ((0, 0, 0, 1), (0, 1), (0, 1)): 1 / 32,
            ((1, 0, 0, 1), (0, 1), (1, 1)): 1 / 32,
            ((0, 0, 1, 1), (0, 1), (2, 1)): 1 / 32,
            ((1, 0, 1, 1), (0, 1), (3, 1)): 1 / 32,
            ((1, 0, 0, 1), (1, 1), (0, 1)): 1 / 32,
            ((0, 0, 0, 1), (1, 1), (1, 1)): 1 / 32,
            ((1, 0, 1, 1), (1, 1), (2, 1)): 1 / 32,
            ((0, 0, 1, 1), (1, 1), (3, 1)): 1 / 32,
            ((0, 1, 0, 1), (2, 1), (0, 1)): 1 / 32,
            ((1, 1, 0, 1), (2, 1), (1, 1)): 1 / 32,
            ((0, 1, 1, 1), (2, 1), (2, 1)): 1 / 32,
            ((1, 1, 1, 1), (2, 1), (3, 1)): 1 / 32,
            ((1, 1, 0, 1), (3, 1), (0, 1)): 1 / 32,
            ((0, 1, 0, 1), (3, 1), (1, 1)): 1 / 32,
            ((1, 1, 1, 1), (3, 1), (2, 1)): 1 / 32,
            ((0, 1, 1, 1), (3, 1), (3, 1)): 1 / 32
        }

    def true_result_distrib(self):
        return {
            ((0, 0, 0, 0), (0, 0), (0, 0)): 1 / 64,
            ((0, 0, 0, 0), (0, 0), (1, 0)): 1 / 64,
            ((0, 0, 0, 0), (1, 0), (0, 0)): 1 / 64,
            ((0, 0, 0, 0), (1, 0), (1, 0)): 1 / 64,
            ((0, 0, 0, 1), (0, 1), (0, 1)): 1 / 64,
            ((0, 0, 0, 1), (0, 1), (1, 1)): 1 / 64,
            ((0, 0, 0, 1), (1, 1), (0, 1)): 1 / 64,
            ((0, 0, 0, 1), (1, 1), (1, 1)): 1 / 64,
            ((0, 0, 1, 0), (0, 0), (2, 0)): 1 / 64,
            ((0, 0, 1, 0), (0, 0), (3, 0)): 1 / 64,
            ((0, 0, 1, 0), (1, 0), (2, 0)): 1 / 64,
            ((0, 0, 1, 0), (1, 0), (3, 0)): 1 / 64,
            ((0, 0, 1, 1), (0, 1), (2, 1)): 1 / 64,
            ((0, 0, 1, 1), (0, 1), (3, 1)): 1 / 64,
            ((0, 0, 1, 1), (1, 1), (2, 1)): 1 / 64,
            ((0, 0, 1, 1), (1, 1), (3, 1)): 1 / 64,
            ((0, 1, 0, 0), (2, 0), (0, 0)): 1 / 64,
            ((0, 1, 0, 0), (2, 0), (1, 0)): 1 / 64,
            ((0, 1, 0, 0), (3, 0), (0, 0)): 1 / 64,
            ((0, 1, 0, 0), (3, 0), (1, 0)): 1 / 64,
            ((0, 1, 0, 1), (2, 1), (0, 1)): 1 / 64,
            ((0, 1, 0, 1), (2, 1), (1, 1)): 1 / 64,
            ((0, 1, 0, 1), (3, 1), (0, 1)): 1 / 64,
            ((0, 1, 0, 1), (3, 1), (1, 1)): 1 / 64,
            ((0, 1, 1, 0), (2, 0), (2, 0)): 1 / 64,
            ((0, 1, 1, 0), (2, 0), (3, 0)): 1 / 64,
            ((0, 1, 1, 0), (3, 0), (2, 0)): 1 / 64,
            ((0, 1, 1, 0), (3, 0), (3, 0)): 1 / 64,
            ((0, 1, 1, 1), (2, 1), (2, 1)): 1 / 64,
            ((0, 1, 1, 1), (2, 1), (3, 1)): 1 / 64,
            ((0, 1, 1, 1), (3, 1), (2, 1)): 1 / 64,
            ((0, 1, 1, 1), (3, 1), (3, 1)): 1 / 64,
            ((1, 0, 0, 0), (0, 0), (0, 0)): 1 / 64,
            ((1, 0, 0, 0), (0, 0), (1, 0)): 1 / 64,
            ((1, 0, 0, 0), (1, 0), (0, 0)): 1 / 64,
            ((1, 0, 0, 0), (1, 0), (1, 0)): 1 / 64,
            ((1, 0, 0, 1), (0, 1), (0, 1)): 1 / 64,
            ((1, 0, 0, 1), (0, 1), (1, 1)): 1 / 64,
            ((1, 0, 0, 1), (1, 1), (0, 1)): 1 / 64,
            ((1, 0, 0, 1), (1, 1), (1, 1)): 1 / 64,
            ((1, 0, 1, 0), (0, 0), (2, 0)): 1 / 64,
            ((1, 0, 1, 0), (0, 0), (3, 0)): 1 / 64,
            ((1, 0, 1, 0), (1, 0), (2, 0)): 1 / 64,
            ((1, 0, 1, 0), (1, 0), (3, 0)): 1 / 64,
            ((1, 0, 1, 1), (0, 1), (2, 1)): 1 / 64,
            ((1, 0, 1, 1), (0, 1), (3, 1)): 1 / 64,
            ((1, 0, 1, 1), (1, 1), (2, 1)): 1 / 64,
            ((1, 0, 1, 1), (1, 1), (3, 1)): 1 / 64,
            ((1, 1, 0, 0), (2, 0), (0, 0)): 1 / 64,
            ((1, 1, 0, 0), (2, 0), (1, 0)): 1 / 64,
            ((1, 1, 0, 0), (3, 0), (0, 0)): 1 / 64,
            ((1, 1, 0, 0), (3, 0), (1, 0)): 1 / 64,
            ((1, 1, 0, 1), (2, 1), (0, 1)): 1 / 64,
            ((1, 1, 0, 1), (2, 1), (1, 1)): 1 / 64,
            ((1, 1, 0, 1), (3, 1), (0, 1)): 1 / 64,
            ((1, 1, 0, 1), (3, 1), (1, 1)): 1 / 64,
            ((1, 1, 1, 0), (2, 0), (2, 0)): 1 / 64,
            ((1, 1, 1, 0), (2, 0), (3, 0)): 1 / 64,
            ((1, 1, 1, 0), (3, 0), (2, 0)): 1 / 64,
            ((1, 1, 1, 0), (3, 0), (3, 0)): 1 / 64,
            ((1, 1, 1, 1), (2, 1), (2, 1)): 1 / 64,
            ((1, 1, 1, 1), (2, 1), (3, 1)): 1 / 64,
            ((1, 1, 1, 1), (3, 1), (2, 1)): 1 / 64,
            ((1, 1, 1, 1), (3, 1), (3, 1)): 1 / 64}

    def true_CI(self):
        return 1

    def true_SI(self):
        return 1
the_available_functions["RDNUNQXOR"] = Fun_Data_rdnunqxor()


class Fun_Data_xorand:

    def __init__(self):
        self.n_u = self.n_v = 2
        self.n_w = 0

    def fun(self, y, z, w):
        return ((y + z) % 2, y * z), y, z

    def noise(self, x):
        x1, x2 = x
        return (x1 + random.randint(0, 1)) % 2, (x2 + random.randint(0, 1)) % 2

    def true_input_distrib(self):
        return {
            ((0, 0), 0, 0): .25,
            ((1, 0), 0, 1): .25,
            ((1, 0), 1, 0): .25,
            ((0, 1), 1, 1): .25,
        }

    def true_result_distrib(self):
        return {
            ((0, 0), 0, 0): 0.25,
            ((0, 1), 1, 1): 0.25,
            ((1, 0), 0, 0): 0.25,
            ((1, 0), 1, 1): 0.25}

    def true_CI(self):
        return 1

    def true_SI(self):
        return 1 / 2
the_available_functions["XORAND"] = Fun_Data_xorand()


def test__solve_time_series(fun_obj, noise_level=.05, L_range=None,
                            verbose=False):
    if L_range is None:
        L_range = [400 * i for i in range(1, 26)]

    y_list = []
    z_list = []
    x_list = []
    counts = dict()
    old_L = 0
    for L in L_range:
        # fetch next few samples
        for i in range(old_L, L):
            if fun_obj.n_u > 0:
                new_y = random.randint(0, fun_obj.n_u - 1)
            else:
                new_y = None
            if fun_obj.n_v > 0:
                new_z = random.randint(0, fun_obj.n_v - 1)
            else:
                new_z = None
            if fun_obj.n_w > 0:
                new_w = random.randint(0, fun_obj.n_w - 1)
            else:
                new_w = None
            x, y, z = fun_obj.fun(new_y, new_z, new_w)
            x_list.append(x)  # this seems to be the target
            y_list.append(y)
            z_list.append(z)

        # add noise
        for i in range(old_L, L):
            noise = 0
            p = random.random()
            if p < noise_level:
                x_list[i] = fun_obj.noise(x_list[i])

        # update counts
        for i in range(old_L, L):
            if (x_list[i], y_list[i], z_list[i]) in counts.keys():
                counts[(x_list[i], y_list[i], z_list[i])] += 1
            else:
                counts[(x_list[i], y_list[i], z_list[i])] = 1

        # make pdf from counts
        pdf = dict()
        for xyz, c in counts.items():
            pdf[xyz] = c / float(L)

        old_L = L
        # END OF creation of pdf

        print("pdf=", TartuSynergy.sorted_pdf(pdf))
        print("test__solve_time_series(): L = ", L)
        print('\n\n============================== Starting estimation with the Tartu estimator ...')
        tic = tm.clock()
        retval = TartuSynergy.solve_PDF(pdf, fun_obj.true_input_distrib(),
                                        fun_obj.true_result_distrib(),
                                        fun_obj.true_CI(),
                                        fun_obj.true_SI(), verbose=verbose)
        toc = tm.clock()
        print('============================== Total elapsed time: {0} seconds'.format(toc - tic))
        optpdf, feas, kkt, CI, SI = retval
        print("CI:", CI, "\nSI:", SI, "\n  sum of marginal eqns violations:",
              feas, "  maximal KKT-system constraint violation:", kkt)

        cfg = {
            'alph_s1': 2,
            'alph_s2': 2,
            'alph_t': 2,
            'max_unsuc_swaps_row_parm': 1000,
            'num_reps': 60,
            'max_iters': 100
        }

        print('\n\n============================== Starting estimation with the Sydney estimator ...')
        y_np = np.array(y_list)
        z_np = np.array(z_list)
        x_np = np.array(x_list)
        tic = tm.clock()
        estimate = estimators_pid.pid_sydney(s1=y_np, s2=z_np, t=x_np, cfg=cfg)
        toc = tm.clock()
        print('============================== Total elapsed time: {0} seconds'.format(toc - tic))

        # print("unq_s1: {0}".format(estimate['unq_s1']))
        # print("unq_s2: {0}".format(estimate['unq_s2']))
        print("CI: syn_s1_s2:{0}".format(estimate['syn_s1_s2']))
        print("SI: shd_s1_s2: {0}".format(estimate['shd_s1_s2']))

        print('\n\nError - CI: {0}, SI: {1}\n\n'.format(
                                                CI - estimate['syn_s1_s2'],
                                                SI - estimate['shd_s1_s2']))

    return retval


def test_speed(fun_obj, noise_level=.05, verbose=False, n_run=1):
    """Test speed of the two PID estimators."""
    num_samples = [1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000]
    # num_samples = [100, 200, 500]
    ci_tartu = []
    ci_sydney = []
    si_tartu = []
    si_sydney = []
    t_sydney = []
    t_tartu = []
    for r in range(len(num_samples)):

        L_range = [num_samples[r]]
        print('Testing estimators on {0} samples.'.format(L_range[0]))

        y_list = []
        z_list = []
        x_list = []
        counts = dict()
        old_L = 0
        for L in L_range:
            # fetch next few samples
            for i in range(old_L, L):
                if fun_obj.n_u > 0:
                    new_y = random.randint(0, fun_obj.n_u - 1)
                else:
                    new_y = None
                if fun_obj.n_v > 0:
                    new_z = random.randint(0, fun_obj.n_v - 1)
                else:
                    new_z = None
                if fun_obj.n_w > 0:
                    new_w = random.randint(0, fun_obj.n_w - 1)
                else:
                    new_w = None
                x, y, z = fun_obj.fun(new_y, new_z, new_w)
                x_list.append(x)  # this seems to be the target
                y_list.append(y)
                z_list.append(z)

            # add noise
            for i in range(old_L, L):
                noise = 0
                p = random.random()
                if p < noise_level:
                    x_list[i] = fun_obj.noise(x_list[i])

            # update counts
            for i in range(old_L, L):
                if (x_list[i], y_list[i], z_list[i]) in counts.keys():
                    counts[(x_list[i], y_list[i], z_list[i])] += 1
                else:
                    counts[(x_list[i], y_list[i], z_list[i])] = 1

            # make pdf from counts
            pdf = dict()
            for xyz, c in counts.items():
                pdf[xyz] = c / float(L)

            old_L = L
            # END OF creation of pdf

            print("pdf=", TartuSynergy.sorted_pdf(pdf))
            print("test__solve_time_series(): L = ", L)
            print('\n\n============================== Starting estimation with the Tartu estimator ...')
            tic = tm.clock()
            retval = TartuSynergy.solve_PDF(pdf, fun_obj.true_input_distrib(),
                                            fun_obj.true_result_distrib(),
                                            fun_obj.true_CI(),
                                            fun_obj.true_SI(), verbose=verbose)
            toc = tm.clock()
            t_tartu.append(toc - tic)
            print('============================== Total elapsed time: {0} seconds'.format(toc - tic))
            optpdf, feas, kkt, CI, SI = retval
            print("CI:", CI, "\nSI:", SI, "\n  sum of marginal eqns violations:",
                  feas, "  maximal KKT-system constraint violation:", kkt)

            cfg = {
                'alph_s1': 2,
                'alph_s2': 2,
                'alph_t': 2,
                'max_unsuc_swaps_row_parm': 1000,
                'num_reps': 60,
                'max_iters': 100
            }

            print('\n\n============================== Starting estimation with the Sydney estimator ...')
            y_np = np.array(y_list)
            z_np = np.array(z_list)
            x_np = np.array(x_list)
            tic = tm.clock()
            estimate = estimators_pid.pid_sydney(s1=y_np, s2=z_np, t=x_np, cfg=cfg)
            toc = tm.clock()
            t_sydney.append(toc - tic)
            print('============================== Total elapsed time: {0} seconds'.format(toc - tic))

            # print("unq_s1: {0}".format(estimate['unq_s1']))
            # print("unq_s2: {0}".format(estimate['unq_s2']))
            print("CI: syn_s1_s2:{0}".format(estimate['syn_s1_s2']))
            print("SI: shd_s1_s2: {0}".format(estimate['shd_s1_s2']))

            print('\n\nError - CI: {0}, SI: {1}\n\n'.format(
                                                CI - estimate['syn_s1_s2'],
                                                SI - estimate['shd_s1_s2']))
            ci_tartu.append(CI)
            ci_sydney.append(estimate['syn_s1_s2'])
            si_tartu.append(SI)
            si_sydney.append(estimate['shd_s1_s2'])
    np.savetxt('/home/patriciaw/repos/IDTxl/dev/tartu_pid/speedtest_results/' +
               fun_obj.name + '_ci_tartu_run_' + str(n_run) + '.csv', ci_tartu, delimiter=',')
    np.savetxt('/home/patriciaw/repos/IDTxl/dev/tartu_pid/speedtest_results/' +
               fun_obj.name + '_ci_sydney_run_' + str(n_run) + '.csv', ci_sydney, delimiter=',')
    np.savetxt('/home/patriciaw/repos/IDTxl/dev/tartu_pid/speedtest_results/' +
               fun_obj.name + '_si_tartu_run_' + str(n_run) + '.csv', si_tartu, delimiter=',')
    np.savetxt('/home/patriciaw/repos/IDTxl/dev/tartu_pid/speedtest_results/' +
               fun_obj.name + '_si_sydney_run_' + str(n_run) + '.csv', si_sydney, delimiter=',')
    np.savetxt('/home/patriciaw/repos/IDTxl/dev/tartu_pid/speedtest_results/' +
               fun_obj.name + '_t_tartu_run_' + str(n_run) + '.csv', t_tartu, delimiter=',')
    np.savetxt('/home/patriciaw/repos/IDTxl/dev/tartu_pid/speedtest_results/' +
               fun_obj.name + '_t_sydney_run_' + str(n_run) + '.csv', t_sydney, delimiter=',')

def print_parameters_usage_msg():
    print("Usage: time_series.py [-v] fun noiselevel #")
    print(" where  fun          is one of: and, xor;")
    print("        noiselevel   is the probability of noise, in [0,1[;")
    print("        #            is number of samples.")
    print("        -v           verbose mode")
    print("Or use interactive mode: python3 -i time_series.py,")
    print("and call main().")


def main(sys_argv):
    verbose_mode = False
    speed_test = False
    if len(sys_argv) == 5 and sys_argv[1] == '-v':
        verbose_mode = True
        del sys_argv[1]
    if len(sys_argv) == 5 and sys_argv[1] == '-s':
        speed_test = True
        del sys_argv[1]
        n_run = int(sys_argv[3])
    if len(sys_argv) <= 1:
        print_parameters_usage_msg()
    elif len(sys_argv) != 4:
        print(len(sys_argv))
        print(sys_argv)
        print("There's something wrong with the parameters!")
        print_parameters_usage_msg()
    else:
        fun_S = (str(sys_argv[1])).upper()
        n_u = n_v = n_w = None
        noise_p = float(sys_argv[2])
        numo_samples = int(sys_argv[3])
        fun_obj = None
        true_input = None
        true_result = None
        true_CI = None
        true_SI = None
        valid_parameters = True
        if fun_S in the_available_functions.keys():
            fun_obj = the_available_functions[fun_S]
        else:
            print("I don't recognize the function ", fun_S)
            print("Available functions:")
            print(the_available_functions.keys())
            valid_parameters = False

        if noise_p < 0 or noise_p >= 1:
            print("Something is wrong with the noise probability ", noise_p)
            print_parameters_usage_msg()
            valid_parameters = False
        if numo_samples < 1 or numo_samples > 1.e20:
            print("Something is wrong with the number of samples", numo_samples)
            print_parameters_usage_msg()
            valid_parameters = False
        if valid_parameters:
            #            solvers.options['show_progress'] = False
            if speed_test:
                test_speed(fun_obj, noise_p, verbose_mode, n_run)
            else:
                retval = test__solve_time_series(
                    fun_obj, noise_p, [numo_samples], verbose=verbose_mode)
                return retval
#^ main()

main(sys.argv)
