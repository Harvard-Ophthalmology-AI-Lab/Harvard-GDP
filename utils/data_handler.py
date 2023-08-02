import numpy as np
import sys, os
import csv
import bz2
from bz2 import BZ2File
from datetime import datetime
import random

def vf_to_matrix(vec):
    mat = np.empty((8,9))
    mat[:] = np.nan

    mat[0, 3:7] = vec[0:4]
    mat[1, 2:8] = vec[4:10]
    mat[2, 1:] = vec[10:18]
    mat[3, :7] = vec[18:25]
    mat[3, 8] = vec[25]
    mat[4, :7] = vec[26:33]
    mat[4, 8] = vec[33]
    mat[5, 1:] = vec[34:42]
    mat[6, 2:8] = vec[42:48]
    mat[7, 3:7] = vec[48:52]

    # mat = np.rot90(mat, k=1).copy()

    return mat

def matrix_to_vf(mat):
    vec = [0.0]*52
    # mat = np.rot90(mat, k=3)

    vec[:4] = mat[0, 3:7]
    vec[4:10] = mat[1, 2:8]
    vec[10:18] = mat[2, 1:]
    vec[18:25] = mat[3, :7] 
    vec[25] = mat[3, 8]
    vec[26:33] = mat[4, :7]
    vec[33] = mat[4, 8] 
    vec[34:42] = mat[5, 1:]
    vec[42:48] = mat[6, 2:8]
    vec[48:52] = mat[7, 3:7]

    return vec
