import pickle
import numpy as np


def save_parameter(filename: str, parameter: dict):
    dict_ = {key: param.value for key, param in parameter.items()}
    np.savez_compressed(filename, **dict_)


def load_parameter(filename: str, parameter: dict):
    loaded = np.load(filename)
    for key in parameter:
        np.copyto(parameter[key].value, loaded[key])


def save_object(filename: str, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_object(filename: str, obj):
    with open(filename, "rb") as f:
        return pickle.load(f)
