"""Provide unit tests for IDTxl's import functions.

Test the import of Fieldtrip's data format into IDTxl's Data class.
Note that ft2idtxl at the moment assumes that fieldtrip data are stored as .mat
files in MATLAB's v7.3 format (which specifies an hdf5 file).abs

Test the import from ordinary Matlab (.mat) files saved in different versions
(v4, v6, v7, v7.3).

Created on Mon Apr 18 10:53:07 2016

@author: wibral
"""
import pytest
import numpy as np
from pkg_resources import resource_filename
from idtxl.idtxl_import import import_fieldtrip, import_matarray


def test_import_fieldtrip():
    file_path = resource_filename(__name__, "data/ABA04_Up_10-140Hz_v7_3.mat")
    dat, label, timestamps, fsample = import_fieldtrip(
        file_name=file_path, ft_struct_name="data", file_version="v7.3"
    )
    assert dat.n_processes == 14, (
        "Wrong number of processes, expected 14, " "found: {0}"
    ).format(dat.n_processes)
    assert dat.n_replications == 135, (
        "Wrong number of replications, expected" " 135, found: {0}"
    ).format(dat.n_replications)
    assert dat.n_samples == 1200, (
        "Wrong number of samples, expected 1200, " "found: {0}"
    ).format(dat.n_samples)

    assert label[0] == "VirtualChannel_3491_pc1", "Wrong channel name for " "label 0."
    assert label[10] == "VirtualChannel_1573_pc2", "Wrong channel name for " "label 10."
    assert label[30] == "VirtualChannel_1804_pc1", "Wrong channel name for " "label 30."
    assert fsample == 600, "Wrong sampling frequency: {0}".format(fsample)
    print(timestamps)  # TODO add assertion for this


def test_import_matarray():
    n_samples = 20  # no. samples in the example data
    n_processes = 2  # no. processes in the example data
    n_replications = 3  # no. replications in the example data

    # Load hdf5, one to three dimensions.
    (dat, label, timestamps, fsample) = import_matarray(
        file_name=resource_filename(__name__, "data/one_dim_v7_3.mat"),
        array_name="a",
        dim_order="s",
        file_version="v7.3",
        normalise=False,
    )
    assert fsample == 1, "Wrong sampling frequency: {0}".format(fsample)
    assert all(timestamps == np.arange(n_samples)), "Wrong time stamps: {0}".format(
        timestamps
    )
    assert label[0] == "channel_000", "Wrong channel label: {0}.".format(label[0])
    assert dat.n_samples == n_samples, "Wrong number of samples: {0}.".format(
        dat.n_samples
    )
    assert dat.n_processes == 1, "Wrong number of processes: {0}.".format(
        dat.n_processes
    )
    assert dat.n_replications == 1, "Wrong number of replications: {0}.".format(
        dat.n_replications
    )

    (dat, label, timestamps, fsample) = import_matarray(
        file_name=resource_filename(__name__, "data/two_dim_v7_3.mat"),
        array_name="b",
        dim_order="sp",
        file_version="v7.3",
        normalise=False,
    )
    assert fsample == 1, "Wrong sampling frequency: {0}".format(fsample)
    assert all(timestamps == np.arange(n_samples)), "Wrong time stamps: {0}".format(
        timestamps
    )
    assert label[0] == "channel_000", "Wrong channel label: {0}.".format(label[0])
    assert label[1] == "channel_001", "Wrong channel label: {0}.".format(label[1])
    assert dat.n_samples == n_samples, "Wrong number of samples: {0}.".format(
        dat.n_samples
    )
    assert dat.n_processes == n_processes, "Wrong number of processes: {0}.".format(
        dat.n_processes
    )
    assert dat.n_replications == 1, "Wrong number of replications: {0}.".format(
        dat.n_replications
    )

    (dat, label, timestamps, fsample) = import_matarray(
        file_name=resource_filename(__name__, "data/three_dim_v7_3.mat"),
        array_name="c",
        dim_order="rsp",
        file_version="v7.3",
        normalise=False,
    )
    assert fsample == 1, "Wrong sampling frequency: {0}".format(fsample)
    assert all(timestamps == np.arange(n_samples)), "Wrong time stamps: {0}".format(
        timestamps
    )
    assert label[0] == "channel_000", "Wrong channel label: {0}.".format(label[0])
    assert dat.n_samples == n_samples, "Wrong number of samples: {0}.".format(
        dat.n_samples
    )
    assert dat.n_processes == n_processes, "Wrong number of processes: {0}.".format(
        dat.n_processes
    )
    assert (
        dat.n_replications == n_replications
    ), "Wrong number of replications: {0}.".format(dat.n_replications)

    # Load matlab versions 4, 6, 7.
    file_path = [
        resource_filename(__name__, "data/two_dim_v4.mat"),
        resource_filename(__name__, "data/two_dim_v6.mat"),
        resource_filename(__name__, "data/two_dim_v7.mat"),
    ]
    file_version = ["v4", "v6", "v7"]
    for i in range(3):
        (dat, label, timestamps, fsample) = import_matarray(
            file_name=file_path[i],
            array_name="b",
            dim_order="ps",
            file_version=file_version[i],
            normalise=False,
        )
        assert dat.n_processes == n_processes, "Wrong number of processes".format(
            dat.n_processes
        )
        assert dat.n_samples == n_samples, "Wrong number of samples".format(
            dat.n_samples
        )
        assert dat.n_replications == 1, "Wrong number of replications".format(
            dat.n_replications
        )

    # Load wrong file name.
    with pytest.raises(FileNotFoundError):
        (dat, label, timestamps, fsample) = import_matarray(
            file_name="test",
            array_name="b",
            dim_order="ps",
            file_version="v6",
            normalise=False,
        )

    # Test wrong variable name.
    with pytest.raises(RuntimeError):
        (dat, label, timestamps, fsample) = import_matarray(
            file_name=resource_filename(__name__, "data/three_dim_v7_3.mat"),
            array_name="test",
            dim_order="rsp",
            file_version="v7.3",
            normalise=False,
        )

    # Test wrong dim order.
    with pytest.raises(RuntimeError):
        (dat, label, timestamps, fsample) = import_matarray(
            file_name=resource_filename(__name__, "data/three_dim_v7_3.mat"),
            array_name="c",
            dim_order="rp",
            file_version="v7.3",
            normalise=False,
        )

    # Test wrong file version
    with pytest.raises(RuntimeError):
        (dat, label, timestamps, fsample) = import_matarray(
            file_name=resource_filename(__name__, "data/three_dim_v7_3.mat"),
            array_name="c",
            dim_order="rp",
            file_version="v4",
            normalise=False,
        )


if __name__ == "__main__":
    # test_import_matarray()
    test_import_fieldtrip()
