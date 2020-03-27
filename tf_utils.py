import numpy as np


def print_variables(vars):
    for var in vars:
        print(var.name, var.shape, np.linalg.norm(var.numpy()))
