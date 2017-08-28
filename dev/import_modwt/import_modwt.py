"""Import modwt into python."""
import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary('/home/patriciaw/repos/IDTxl/dev/import_modwt/'
                              'libmodwtj.so')

n = 20
lib.test_c.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.test_c.restype = None
p = (ctypes.c_int * n)()
print(len(p))
lib.test_c(p, len(p))
print(list(p))

# N = int(500)
N = int(20)
Vin = np.arange(N)
print(Vin)
j = 3  # scale
coeff_length = int(16)  # filter length
ht = np.arange(coeff_length).astype(np.double)
gt = np.arange(coeff_length).astype(np.double)
Wout = np.random.rand(N).astype(np.double)
Vout = np.random.rand(N).astype(np.double)
# ht = (ctypes.c_double * coeff_length)()
# gt = (ctypes.c_double * coeff_length)()
# Wout = (ctypes.c_double * N)()  # output
# Vout = (ctypes.c_double * N)()

array_type_a = ctypes.c_double * N
array_type_ht = ctypes.c_double * coeff_length
array_type_gt = ctypes.c_double * coeff_length
array_type_out = ctypes.c_double * N

lib.modwtj.argtypes = [ctypes.POINTER(ctypes.c_double),
                       ctypes.c_int,
                       ctypes.c_int,
                       ctypes.c_int,
                       ctypes.POINTER(ctypes.c_double),
                       ctypes.POINTER(ctypes.c_double),
                       ctypes.POINTER(ctypes.c_double),
                       ctypes.POINTER(ctypes.c_double)]
lib.modwtj.restype = None
lib.modwtj(array_type_a(*Vin),
           ctypes.c_int(N),
           ctypes.c_int(j),
           ctypes.c_int(coeff_length),
           array_type_ht(*ht),
           array_type_ht(*gt),
           array_type_out(*Wout),
           array_type_out(*Vout))
print('output modwtj - Wout: {0}'.format(Wout))

# lib.imodwtj(array_type_out(*Wout),
#             array_type_out(*Vout),
#             ctypes.c_int(N),
#             ctypes.c_int(j),
#             ctypes.c_int(L),
#             array_type_ht(*ht),
#             array_type_gt(*gt),
#             array_type_out(*Vout))
