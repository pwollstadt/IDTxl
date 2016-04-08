# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:52:24 2016

@author: patricia
"""
import numpy as np
from idtxl import surrogates


def test_create_surrogates():
    pass


def test_swap_blocks():
    """Test block-wise swapping of samples."""
    n = 50
    block_size = 5
    swap_range = 3
    surr = surrogates._swap_blocks(n, block_size, swap_range)
    assert (sum(surr == 0) == block_size), ('Incorrect block size.')


def test_circular_shift():
    """Test circular shifting of samples."""
    n = 50
    max_shift = 10
    surr = surrogates._circular_shift(n, max_shift)
    assert ((n - surr[0]) <= max_shift), 'Actual shift exceeded max_shift.'


def test_swap_local():
    pass


if __name__ == '__main__':
    test_swap_blocks()
    test_circular_shift()
    test_swap_local()
    test_create_surrogates()
