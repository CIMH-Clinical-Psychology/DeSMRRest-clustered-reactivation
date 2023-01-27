# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 08:32:34 2021

This file contains project-wide definitions and settings

@author: Simon Kern
"""
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression


def rescale_meg_transform_outlier(arr):
    """
    same as rescale_meg, but also removes all values that are above [-1, 1]
    and rescales them to smaller values
    """

    arr = rescale_meg(arr)

    arr[arr < -1] *= 1e-2
    arr[arr > 1] *= 1e-2
    return arr


def rescale_meg(arr):
    """
    this tries to statically re-scale the values from Tesla to Nano-Tesla,
    such that most sensor values are between -1 and 1

    If possible, individual scaling is applied to magnetometers and
    gradiometers as both sensor types have a different sensitivity and scaling.

    Basically a histogram normalization between the two sensor types

    gradiometers  = *1e10
    magnetometers = *2e11
    """

    # some sanity check, if these
    if arr.min() < -1e-6 or arr.max() > 1e-6:
        warnings.warn(
            "arr min/max are not in MEG scale, no rescaling applied: {arr.min()} / {arr.max()}"
        )
        raise Exception(
            "arr min/max are not in MEG scale, no rescaling applied: {arr.min()} / {arr.max()}"
        )
    arr = np.array(arr)
    grad_scale = 1e10
    mag_scale = 2e11

    # reshape to 3d to make indexing uniform for all types
    # will be put in its original shape later
    orig_shape = arr.shape
    arr = np.atleast_3d(arr)

    # heuristic to find which dimension is likely the sensor dimension
    for meg_type in [306, 204, 102]:  # mag+grad or grad or mag
        dims = [d for d, size in enumerate(arr.shape) if size % meg_type == 0]
        # how many copies do we have of the sensors?
        stacks = [
            size // meg_type for d, size in enumerate(arr.shape) if size % meg_type == 0
        ]
        if len(dims) > 0:
            break

    if len(dims) != 1:
        warnings.warn(
            f"Several or no matching dimensions found for sensor dimension: {arr.shape}"
            " will simply reshape everything with grad_scale."
        )
        raise Exception(
            f"Several or no matching dimensions found for sensor dimension: {arr.shape}"
            " will simply reshape everything with grad_scale."
        )
        return arr.reshape(*orig_shape) * grad_scale
    sensor_dim = dims[0]
    n_stack = stacks[0]

    if meg_type == 306:
        slicer_grad = [slice(None) for _ in range(3)]
        slicer_grad[sensor_dim] = np.hstack(
            [(i * meg_type) + idx_grad for i in range(n_stack)]
        )
        arr[tuple(slicer_grad)] *= grad_scale
        slicer_mag = [slice(None) for _ in range(3)]
        slicer_mag[sensor_dim] = np.hstack(
            [(i * meg_type) + idx_mag for i in range(n_stack)]
        )
        arr[tuple(slicer_mag)] *= mag_scale

    if meg_type == 204:
        arr *= grad_scale

    if meg_type == 102:
        arr *= mag_scale

    return arr.reshape(*orig_shape)


def get_free_space(path):
    """return the current free space in the cache dir in GB"""
    import shutil

    os.makedirs(path, exist_ok=True)
    total, used, free = shutil.disk_usage(path)
    total //= 1024**3
    used //= 1024**3
    free //= 1024**3
    return free

###############################
#%%userconf
# USER SPECIFIC CONFIGURATION
###############################

data_dir = "/data/Simon/DeSMRRest/upload/"
cache_dir = f"/{data_dir}/cache/"  # used for caching
plot_dir = f"/{data_dir}/plots/"  # plots will be stored here
log_dir = f"/{data_dir}/plots/logs/"  # log files will be created here

results_dir = os.path.expanduser(f"{data_dir}/results/")  # final results here

if data_dir == "":
    raise Exception(f"please set configuration in settings.py")

if not os.path.isdir(data_dir):
    warnings.warn(f"plot_dir does not exist at {plot_dir}, create")
    os.makedirs(plot_dir, exist_ok=True)
if not os.path.isdir(plot_dir):
    warnings.warn(f"plot_dir does not exist at {plot_dir}, create")
    os.makedirs(plot_dir, exist_ok=True)
if not os.path.isdir(log_dir):
    warnings.warn(f"log_dir does not exist at {log_dir}, create")
    os.makedirs(log_dir, exist_ok=True)
if not os.path.isdir(results_dir):
    warnings.warn(f"log_dir does not exist at {log_dir}, create")
    os.makedirs(results_dir, exist_ok=True)

if get_free_space(cache_dir) < 20:
    raise RuntimeError(f"Free space for {cache_dir} is below 20GB. Cannot safely run.")

###############################
#%% SETTINGS and CONSTANTS
###############################

bands_delta = {"delta": (0, 4)}
bands_theta = {"theta": (4, 8)}
bands_alpha = {"alpha": (8, 14)}
bands_beta = {"beta": (15, 30)}
bands_gamma = {"gamma": (30, 45)}

# some default brain band definitions
bands_all = {**bands_delta, **bands_theta, **bands_alpha, **bands_beta, **bands_gamma}
bands_lower = {"lower": (0.5, 20)}
bands_HP = {"only_HP": (0.5, None)}
bands_none = {"none": (None, None)}

# corperate colour palette
zi_palette = [
    "#003e65",
    "#006960",
    "#70305a",
    "#c7361b",
    "#3a98cc",
    "#74ba59",
    "#e8326d",
    "#f7ab64",
    "#85cee4",
    "#bfffd7",
    "#d1bcdc",
    "#fcd8c1",
]

# the sequences with loop included
seq_12 = "ABCDEFGEHIBJAB"

default_predict_function = "predict_proba"  # 'decision_function'

default_seq = seq_12
default_autoreject = True
default_ica_components = 50  # default used by Fungi
default_normalize = rescale_meg_transform_outlier
default_clf_params = {
    "C": 1 / 0.006,
    "max_iter": 1000,
    "penalty": "l1",
    "solver": "liblinear",
}
default_bands = bands_HP

# default classifier to use if non is specified
default_clf = LogisticRegression(**default_clf_params)

caching_enabled = True
timeshift_constant = np.mean(
    [
        1.000559286986059,  # this is the value that we
        1.000559769261213,  # have to multiply the timepoints
        1.0005582875825834,  # of the presentation log files
        1.0005608210420054,  # to get matching positions for the MEG
        1.0005594754801779,  # the numbers on the left
        1.0005585095859724,  # are the mismatched between
        1.0005591506251639,  # individual measurements
        1.0005578477318235,
        1.0005590747786206,
        1.0005578234309724,
        1.0005582714046664,
        1.0005581193610011,
        1.000557486504249,
        1.0005597991661357,
        1.000559275593335,
        1.0005591272826757,
        1.0005586249053116,
        1.0005589597532822,
    ]
)

# this is a lookup table that shows correspondence between
# presentation log file event codes and port codes
event_code_translation = {
    "RS1": 10,
    "RS2": 20,
    "RS1 end": 11,
    "RS2 end": 22,
    "fixation audio": 99,
    "fixation pre audio": 98,
}
event_code_translation.update({f"{x}": x for x in range(256)})

# here some static MEG definitions for Vectorview systems (ELECTRA/NeuroMAG)

idx_grad = np.array(
    [
        1,
        2,
        4,
        5,
        7,
        8,
        10,
        11,
        13,
        14,
        16,
        17,
        19,
        20,
        22,
        23,
        25,
        26,
        28,
        29,
        31,
        32,
        34,
        35,
        37,
        38,
        40,
        41,
        43,
        44,
        46,
        47,
        49,
        50,
        52,
        53,
        55,
        56,
        58,
        59,
        61,
        62,
        64,
        65,
        67,
        68,
        70,
        71,
        73,
        74,
        76,
        77,
        79,
        80,
        82,
        83,
        85,
        86,
        88,
        89,
        91,
        92,
        94,
        95,
        97,
        98,
        100,
        101,
        103,
        104,
        106,
        107,
        109,
        110,
        112,
        113,
        115,
        116,
        118,
        119,
        121,
        122,
        124,
        125,
        127,
        128,
        130,
        131,
        133,
        134,
        136,
        137,
        139,
        140,
        142,
        143,
        145,
        146,
        148,
        149,
        151,
        152,
        154,
        155,
        157,
        158,
        160,
        161,
        163,
        164,
        166,
        167,
        169,
        170,
        172,
        173,
        175,
        176,
        178,
        179,
        181,
        182,
        184,
        185,
        187,
        188,
        190,
        191,
        193,
        194,
        196,
        197,
        199,
        200,
        202,
        203,
        205,
        206,
        208,
        209,
        211,
        212,
        214,
        215,
        217,
        218,
        220,
        221,
        223,
        224,
        226,
        227,
        229,
        230,
        232,
        233,
        235,
        236,
        238,
        239,
        241,
        242,
        244,
        245,
        247,
        248,
        250,
        251,
        253,
        254,
        256,
        257,
        259,
        260,
        262,
        263,
        265,
        266,
        268,
        269,
        271,
        272,
        274,
        275,
        277,
        278,
        280,
        281,
        283,
        284,
        286,
        287,
        289,
        290,
        292,
        293,
        295,
        296,
        298,
        299,
        301,
        302,
        304,
        305,
    ]
)
idx_mag = np.array(
    [
        0,
        3,
        6,
        9,
        12,
        15,
        18,
        21,
        24,
        27,
        30,
        33,
        36,
        39,
        42,
        45,
        48,
        51,
        54,
        57,
        60,
        63,
        66,
        69,
        72,
        75,
        78,
        81,
        84,
        87,
        90,
        93,
        96,
        99,
        102,
        105,
        108,
        111,
        114,
        117,
        120,
        123,
        126,
        129,
        132,
        135,
        138,
        141,
        144,
        147,
        150,
        153,
        156,
        159,
        162,
        165,
        168,
        171,
        174,
        177,
        180,
        183,
        186,
        189,
        192,
        195,
        198,
        201,
        204,
        207,
        210,
        213,
        216,
        219,
        222,
        225,
        228,
        231,
        234,
        237,
        240,
        243,
        246,
        249,
        252,
        255,
        258,
        261,
        264,
        267,
        270,
        273,
        276,
        279,
        282,
        285,
        288,
        291,
        294,
        297,
        300,
        303,
    ]
)
