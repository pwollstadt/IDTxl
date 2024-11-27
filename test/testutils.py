import pytest

#################################### Skip decorators for missing dependencies ####################################

# JIDT skip decorator
_jidt_missing_flag = False
try:
    import jpype
except ImportError as err:
    _jidt_missing_flag = True
jpype_missing = pytest.mark.skipif(
    _jidt_missing_flag,
    reason="Jpype is missing, JIDT estimators are not available")

# OpenCL skip decorator
_opencl_missing_flag = False
try:
    import pyopencl

    # Check if OpenCL is available
    platforms = pyopencl.get_platforms()
    if len(platforms) == 0:
        _opencl_missing_flag = True

except ImportError as err:
    _opencl_missing_flag = True
except pyopencl._cl.LogicError as err:
    _opencl_missing_flag = True

opencl_missing = pytest.mark.skipif(
    _opencl_missing_flag,
    reason="PyOpenCL is missing, OpenCL estimators are not available")

# MPI skip decorator
_mpi_missing_flag = False
try:
    from mpi4py import MPI

    # Check if MPI is available
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    if size < 2:
        _mpi_missing_flag = True

except ImportError as err:
    _mpi_missing_flag = True

mpi_missing = pytest.mark.skipif(
    _mpi_missing_flag,
    reason="MPI is missing, MPI estimators are not available")