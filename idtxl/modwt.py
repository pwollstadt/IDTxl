"""Provide maximum overlap wavelets transform.

Uses the modwt implementation from the WMTSA toolbox.

References:

- https://atmos.uw.edu/~wmtsa/
- Percival, D.B. & Walden, A.T. (2000). Wavelet Methods for Time Series
  Analysis. Cambridge: Cambridge University Press.
"""
import ctypes
import numpy as np
from pkg_resources import resource_filename

lib = ctypes.CDLL(resource_filename(__name__, 'libmodwtj.so'))


def modwt_C(data, wavelet_name, levels):
    [ht, gt, filter_len] = wavelet_filter(wavelet_name)

    N = int(data.shape[0])
    if len(data.shape) > 1:
        repl = int(data.shape[1])
    else:
        repl = int(1)

    coeff_length = int(filter_len)  # filter length
    type_int = ctypes.c_int
    type_double_pointer = ctypes.POINTER(ctypes.c_double)

    lib.modwtj.argtypes = [
        type_double_pointer,
        type_int,
        type_int,
        type_int,
        type_double_pointer,
        type_double_pointer,
        type_double_pointer,
        type_double_pointer
    ]

    lib.modwtj.restype = None

    coff_ht = ht.ctypes.data_as(type_double_pointer)
    coeff_gt = gt.ctypes.data_as(type_double_pointer)
    cN = type_int(N)
    ccoegg = type_int(coeff_length)

    Woutj = np.zeros((N, levels, repl))
    Voutj = np.zeros((N, 1, repl))
    for rp in range(0, repl):

        if repl > 1:
            Vin = (data[:, rp]).astype(np.double)
        else:
            Vin = (data).astype(np.double)

        pointer_data = Vin.ctypes.data_as(type_double_pointer)

        for j in range(1, levels+1):
            cj = type_int(j)
            Wout = (np.random.rand(N, 1)).astype(np.double)
            Vout = (np.random.rand(N, 1)).astype(np.double)
            pointer_Vout = Vout.ctypes.data_as(type_double_pointer)
            pointer_Wout = Wout.ctypes.data_as(type_double_pointer)

            lib.modwtj(pointer_data, cN, cj, ccoegg, coff_ht,
                       coeff_gt, pointer_Wout, pointer_Vout)

            Woutj[:, j-1, rp] = Wout.reshape(N)
            Vin = Vout.astype(np.double)
            pointer_data = Vin.ctypes.data_as(type_double_pointer)

        Voutj[:, 0, rp] = Vout.reshape(N)
    return Woutj, Voutj


def imodwt_c(WJt, VJt, wavelet_name, levels):
    [ht, gt, filter_len] = wavelet_filter(wavelet_name)
    N = (WJt.shape[0])
    repl = int(WJt.shape[2])
    coeff_length = int(filter_len)  # filter length
    type_int = ctypes.POINTER(ctypes.c_int)
    type_double_pointer = ctypes.POINTER(ctypes.c_double)

    lib.imodwtj.argtypes = [
        type_double_pointer,
        type_double_pointer,
        type_int,
        type_int,
        type_int,
        type_double_pointer,
        type_double_pointer,
        type_double_pointer
    ]

    lib.imodwtj.restype = None

    coff_ht = ht.ctypes.data_as(type_double_pointer)
    coeff_gt = gt.ctypes.data_as(type_double_pointer)

    X = np.zeros((N, repl))
    for rp in range(0, repl):
        Vin = (VJt[:, 0, rp]).astype(np.double)

        pointer_V = Vin.ctypes.data_as(type_double_pointer)

        for j in range(levels, 0, -1):
            Vout = (np.random.rand(N, 1)).astype(np.double)
            pointer_Vout = Vout.ctypes.data_as(type_double_pointer)

            Wjt_data = (WJt[:, j-1, rp]).astype(np.double)
            pointerWJt = Wjt_data.ctypes.data_as(type_double_pointer)

            NN = ctypes.c_int(N)
            jj = ctypes.c_int(j)
            ff = ctypes.c_int(coeff_length)

            lib.imodwtj(pointerWJt, pointer_V, NN, jj, ff,
                        coff_ht, coeff_gt, pointer_Vout)

            Vin = Vout.astype(np.double)
            pointer_V = Vin.ctypes.data_as(type_double_pointer)
        X[:, rp] = Vout.reshape(N)
    return X


def wavelet_filter(wavelet_name):
    if wavelet_name == 'la16':
        gh_filter = np.asarray(
            [-0.00239172925599978, -0.000383345447999936, 0.0224118115209982,
             0.00537930587549959, -0.101324327642992, -0.0433268077029965,
             0.340372673594973, 0.549553315268456, 0.257699335186979,
             -0.0367312543804971, -0.0192467606314985, 0.0347452329554972,
             0.00269319437699982, -0.0105728432639991, -0.000214197149999955,
             0.00133639669649986])
        ht_filter = np.asarray(
            [0.00133639669649986, 0.000214197149999955, -0.0105728432639991,
             -0.00269319437699982, 0.0347452329554972, 0.0192467606314985,
             -0.0367312543804971, -0.257699335186979, 0.549553315268456,
             -0.340372673594973, -0.0433268077029965, 0.101324327642992,
             0.00537930587549959, -0.0224118115209982, -0.000383345447999936,
             0.00239172925599978])
        filter_length = 16

    elif wavelet_name == 'la8':
        gh_filter = np.asarray(
            [-0.0535744507089887, -0.0209554825624955, 0.351869534327926,
             0.568329121703880, 0.210617267101956, -0.0701588120894852,
             -0.00891235072099814, 0.0227851729479951])
        ht_filter = np.asarray(
            [0.0227851729479951, 0.00891235072099814, -0.0701588120894852,
             -0.210617267101956, 0.568329121703880, -0.351869534327926,
             -0.0209554825624955, 0.0535744507089887])
        filter_length = 8

    elif wavelet_name == 'd8':
        gh_filter = np.asarray(
            [0.162901714024621, 0.505472857542726, 0.446100069127636,
             -0.0197875131176976, -0.132253583683686, 0.0218081502369510,
             0.0232518005353442, -0.00749349466513341])

        ht_filter = np.asarray(
            [-0.00749349466513341, -0.0232518005353442, 0.0218081502369510,
             0.132253583683686, -0.0197875131176976, -0.446100069127636,
             0.505472857542726, -0.162901714024621])
        filter_length = 8

    elif wavelet_name == 'd4':
        gh_filter = np.asarray(
            [0.341506350946110, 0.591506350946110, 0.158493649053890,
             -0.0915063509461096])
        ht_filter = np.asarray(
            [-0.0915063509461096, -0.158493649053890, 0.591506350946110,
             -0.341506350946110])
        filter_length = 4

    elif wavelet_name == 'bl14':
        gh_filter = np.asarray(
            [0.0120154192834842, 0.0172133762994439, -0.0649080035533744,
             -0.0641312898189170, 0.3602184608985549, 0.7819215932965554,
             0.4836109156937821, -0.0568044768822707, -0.1010109208664125,
             0.0447423494687405, 0.0204642075778225, -0.0181266051311065,
             -0.0032832978473081, 0.0022918339541009])
        ht_filter = np.asarray(
            [0.00229183395410090, 0.00328329784730810, -0.0181266051311065,
             -0.0204642075778225, 0.0447423494687405, 0.101010920866413,
             -0.0568044768822707, -0.483610915693782, 0.781921593296555,
             -0.360218460898555, -0.0641312898189170, 0.0649080035533744,
             0.0172133762994439, -0.0120154192834842])
        filter_length = 14

    return ht_filter, gh_filter, filter_length
