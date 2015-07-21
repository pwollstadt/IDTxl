
"""
Run trentoolxl on example data to check functions.

:author: Patricia Wollstadt
"""
import numpy as np
import trentoolxl.embedding

timeseries = np.arange(21)
dim = 3
tau = 1
print("testing embedding for dim={0}, tau={1}".format(dim,tau))
embedding_1 = trentoolxl.embedding.embed_timeseries(timeseries, dim, tau)
print(embedding_1)

timeseries = np.arange(21)
dim = 3
tau = 2
print("testing embedding for dim={0}, tau={1}".format(dim,tau))
embedding_2 = trentoolxl.embedding.embed_timeseries(timeseries, dim, tau)
print(embedding_2)

embedding_1_correct = np.array([
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
    [8, 9, 10],
    [9, 10, 11],
    [10, 11, 12],
    [11, 12, 13],
    [12, 13, 14],
    [13, 14, 15],
    [14, 15, 16],
    [15, 16, 17],
    [16, 17, 18],
    [17, 18, 19],
    [18, 19, 20]])

embedding_2_correct = np.array([
    [0, 2, 4],
    [1, 3, 5],
    [2, 4, 6],
    [3, 5, 7],
    [4, 6, 8],
    [5, 7, 9],
    [6, 8, 10],
    [7, 9, 11],
    [8, 10, 12],
    [9, 11, 13],
    [10, 12, 14],
    [11, 13, 15],
    [12, 14, 16],
    [13, 15, 17],
    [14, 16, 18],
    [15, 17, 19],
    [16, 18, 20]])

print("Embedding 1 correct: {0}".format((embedding_1_correct==embedding_1).all()))
print("Embedding 2 correct: {0}".format((embedding_2_correct==embedding_2).all()))