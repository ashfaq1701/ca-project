import numpy as np


def binary_string_to_binary_array(rule_binary):
    return np.array([int(ch) for ch in list(rule_binary)])


def binary_array_to_number(binary_array):
    binary_string = ''.join(map(str, binary_array))
    return int(binary_string, 2)
