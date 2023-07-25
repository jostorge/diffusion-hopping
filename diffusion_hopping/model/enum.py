from enum import Enum


class Parametrization(Enum):
    EPS = "eps"
    MEAN = "mean"


class Architecture(Enum):
    EGNN = "egnn"
    GVP = "gvp"


class SamplingMode(Enum):
    DDPM = "ddpm"
    DDIM = "ddim"
