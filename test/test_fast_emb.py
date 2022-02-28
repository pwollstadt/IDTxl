""" Provides test if fast embedding is available.

Fast embedding should be used, otherwise the estimation and optimization will
take a long time. Up to 50x slower than with the fast embedding.

This test should be used after the compilation of hde_fast_embedding.pyx

The compilation of the fast embedding tools via setup_hde_fast_embedding.py:
change to IDTxl/idtxl folder
then type in
>>> python3 setup_hde_fast_embedding.py build_ext --inplace
"""

try:
    import idtxl.hde_fast_embedding as fast_emb
    print("\nFast embedding tools are available.")
except:
    print("Error importing Cython fast embedding module. \n Please try the following command in a linux "
          "terminal in the IDTxl/idtxl folder: \n>> python3 setup_hde_fast_embedding.py build_ext --inplace")