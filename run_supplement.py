#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:54:53 2023

code to produce the supplementary figures

@author: simon.kern
"""
import json

import compress_pickle
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

import settings
import utils
from load_funcs import load_localizers_seq12, load_testing_img2

clf = utils.LogisticRegressionOvaNegX(C=6, penalty="l1", neg_x_ratio=2)
files = utils.list_files(settings.data_dir, patterns=["*DSMR*"])
subjects = [f"DSMR{subj}" for subj in sorted(set(map(utils.get_id, files)))]
results_dir = settings.results_dir + "/reactivation/"
pkl_dir = settings.cache_dir + "/pkl"

# load data, will be needed for some of the plots
localizer = {}  # localizer training data
testing_img2 = {}  # testing trials after RS2

for subj in tqdm(subjects, desc="Loading data"):
    localizer[subj] = load_localizers_seq12(subj=subj)
    testing_img2[subj] = load_testing_img2(subj=subj, tmin=-0.1, tmax=0.5)

plt.rc("font", size=12)

input("press enter to continue plotting")
#%% S1: rejected trials per participant

json_files = utils.list_files(settings.results_dir + "/log/")
df_rejected = pd.DataFrame()

# loop over all json files, these contain the logs for preprocessing
for file in json_files:
    subj = f"DSMR{utils.get_id(file)}"

    with open(file, "r") as f:
        c = json.load(f)

    # first localizer
    name = "autoreject_epochs event_ids={1, 2, 3, 4, 5, 6, 7, 8, 9, 10} n_events=240"
    perc_bad = sum([[c[f"localizer{i}"][name]["perc_bad"]][0] for i in [1, 2]]) / 2
    df_tmp = pd.DataFrame(
        {"subject": subj, "ratio bad": perc_bad, "type": "localizer"}, index=[0]
    )
    df_rejected = pd.concat([df_rejected, df_tmp])

    # then testing
    name = "autoreject_epochs event_ids={1, 2, 3, 4, 5, 6, 7, 8, 9, 10} n_events=12"
    test = "test" if "test" in c else "testing"  # I sometimes named it differently
    perc_bad = [c[test][name]["perc_bad"]][0]
    df_tmp = pd.DataFrame(
        {"subject": subj, "ratio bad": perc_bad, "type": "test"}, index=[0]
    )
    df_rejected = pd.concat([df_rejected, df_tmp], ignore_index=True)

# plot as bar graph
fig = plt.figure(figsize=[12, 8])
ax = fig.subplots(1, 1)
sns.barplot(data=df_rejected, x="subject", y="ratio bad", hue="type")
ax.tick_params(axis="x", rotation=70)
ax.set_ylim([0, 1])

plt.pause(0.1)
fig.tight_layout()
fig.savefig(results_dir + "/S1 - rejected epochs", bbox_inches="tight")

#%% S2: participant's memory performance and rejection

json_files = utils.list_files(settings.results_dir + "/log/")
df_rejected = pd.DataFrame()

# retrieve memory performance
performance = [utils.get_performance(subj=subj) for subj in subjects]

# retrieve decoding accuracy
accuracies = [utils.get_decoding_accuracy(subj=subj, clf=clf)[0] for subj in subjects]

# plot as bar graph
fig = plt.figure(figsize=[12, 8])
ax1, ax2 = fig.subplots(2, 1)
sns.despine()

df_perf_acc = pd.DataFrame(
    {
        "accuracy": [np.mean(accuracies), 0] + accuracies,
        "performance": [np.mean(performance), 0] + performance,
        "subject": ["Mean", ""] + subjects,
    }
)

palette = ["darkblue", "blue"] + [
    "royalblue" if acc >= 0.3 else "grey" for acc in accuracies
]
sns.barplot(data=df_perf_acc, x="subject", y="accuracy", ax=ax1, palette=palette)
ax1.hlines(0.3, 0, 32, linestyle="--", color="gray")
ax1.tick_params(axis="x", rotation=70)

palette = ["darkred", "blue"] + [
    "indianred" if perf >= 0.5 else "grey" for perf in performance
]
sns.barplot(data=df_perf_acc, x="subject", y="performance", ax=ax2, palette=palette)
ax2.hlines(0.5, 0, 32, linestyle="--", color="gray")
ax2.tick_params(axis="x", rotation=70)

plt.pause(0.1)
fig.tight_layout()
fig.savefig(results_dir + "/S2 - performance and accuracy.png", bbox_inches="tight")
#%% S3: decoding accuracy over time per participant

# load precomputed localizer decoding accuracies
pkl_localizer = pkl_dir + "/1 localizer.pkl.zip"
df_localizer = pd.read_pickle(pkl_localizer)

fig = plt.figure()
axs = fig.subplots(6, 5)
axs = axs.flatten()
for i, (subj, df_subj) in enumerate(tqdm(df_localizer.groupby("subject"), total=30)):
    utils.plot_decoding_accuracy(
        df_subj,
        x="timepoint",
        y="accuracy",
        title=f"{subj}",
        color="tab:blue",
        ax=axs[i],
    )
plt.pause(0.1)
fig.tight_layout()
fig.savefig(results_dir + "/S3 - localizer performance.png", bbox_inches="tight")

#%% S4 histogram: number of learning blocks per participant
from utils import get_performance

n_blocks = [len(get_performance(subj=s, which="learning")) for s in subjects]

fig = plt.figure(figsize=[6, 4])
ax = fig.subplots(1, 1)
sns.histplot(
    data=pd.DataFrame({"subject": subjects, "n blocks": n_blocks}),
    x="n blocks",
    bins=np.arange(1, 7) + 0.5,
)
fig.suptitle("Number of learning blocks per participant")
fig.tight_layout
fig.savefig(results_dir + "/S8 learning blocks.svg")
fig.savefig(results_dir + "/S8 learning blocks.png")


from mne.decoding import SlidingEstimator

#%% S5 sensor map: L1 beta coefficient > 0 visualization per class
import meg_tools

def fit(clf, subj, *args, **kwargs):
    np.random.seed(utils.get_id(subj))
    return clf.fit(*args, **kwargs)


clfs = Parallel(len(localizer) - 1)(
    delayed(fit)(clf, subj, X=localizer[subj][0][:, :, 31], y=localizer[subj][1])
    for subj in localizer
)

# these are the orders of images as they were assigned to the classes
names = [utils.get_image_names(subj) for subj in subjects]
names_base = sorted(names[0])

# we need to sort the rows of the matrix accordingly
sorting = [[names_base.index(x) for x in name] for name in names]
betas = np.array([clf.coef_[idxs, :] for clf, idxs in zip(clfs, sorting)])

fig = plt.figure(figsize=[10, 7])
axs = fig.subplots(3, 4)
axs = axs.flatten()
for i in range(10):
    meg_tools.plot_sensors(
        np.mean(betas > 0, 0)[i], ax=axs[i], mode="size", title=names_base[i]
    )
axs[-1].axis("off")
axs[-2].axis("off")

fig.suptitle("Sensor distribution for different images")
plt.pause(0.1)
fig.tight_layout
fig.savefig(results_dir + "/S6 sensorlocation.svg")
fig.savefig(results_dir + "/S6 sensorlocation.png")

#%% SX confusion matrix: decoder class balance/decodability per class
# did not make it into the paper
from sklearn.metrics import confusion_matrix

pkl_localizer = pkl_dir + "/1 localizer.pkl.zip"
df_localizer = pd.read_pickle(pkl_localizer)
df_localizer = df_localizer[df_localizer.timepoint == 210]

# these are the orders of images as they were assigned to the classes
names = [utils.get_image_names(subj) for subj in subjects]
names_base = sorted(names[0])

# we need to sort the rows of the matrix accordingly
sorting = [[names_base.index(x) for x in name] for name in names]

confmats = []
for (subj, df_subj), labels in zip(df_localizer.groupby("subject"), sorting):
    preds = np.hstack(df_subj.preds)
    truth = np.hstack([np.arange(10)] * len(df_subj))
    confmat = confusion_matrix(truth, preds, labels=labels, normalize="true")
    confmats.append(confmat)

fig = plt.figure(figsize=[6, 5])
ax = fig.subplots(1, 1)
sns.heatmap(np.mean(confmats, 0), ax=ax)
ax.set_xticks(np.arange(10) + 0.5, names_base, rotation=45)
ax.set_yticks(np.arange(10) + 0.5, names_base, rotation=0)
ax.set_xlabel("item name")
ax.set_ylabel("item name")
fig.suptitle("Confusion matrix of decoded items during localizer")
plt.pause(0.1)
fig.tight_layout()
fig.savefig(results_dir + "/S5 confmat.svg")
fig.savefig(results_dir + "/S5 confmat.png")


#%% SX decoding probabilities per class
# did not make it to the final paper
from sklearn.model_selection import StratifiedKFold

ex_per_fold = 1

# these are the orders of images as they were assigned to the classes
items = {subj: utils.get_image_names(subj) for subj in subjects}
items_base = sorted(list(items.values())[0])
# we need to sort the rows of the matrix accordingly
sorting = {subj: [items_base.index(x) for x in item] for subj, item in items.items()}

clf_x = SlidingEstimator(clf)


def train_predict_proba(clf, data_x, data_y, best_tp):
    cv = StratifiedKFold(np.bincount(data_y)[0])
    probas_current = []
    probas_others = []

    def inverse_index(i, n=10):
        idx = np.ones(n, dtype=bool)
        idx[i] = False
        return idx

    for idx_train, idx_test in cv.split(data_x, data_y):
        x_train = data_x[idx_train]
        y_train = data_y[idx_train]
        x_test = data_x[idx_test]
        y_test = data_y[idx_test]

        clf.fit(x_train[:, :, best_tp], y_train)
        proba = np.array(
            [clf.predict_proba(x_test[:, :, t]) for t in range(x_test.shape[-1])]
        )
        proba = np.swapaxes(proba, 0, 1)
        proba_currrent = np.array([p[:, y] for p, y in zip(proba, y_test, strict=True)])
        proba_other = np.mean(
            [p[:, inverse_index(y)] for p, y in zip(proba, y_test)], -1
        )
        probas_current.append(proba_currrent)
        probas_others.append(proba_other)
    probas_current = np.vstack(probas_current)
    probas_others = np.vstack(probas_others)
    return probas_current, probas_others


times = np.arange(-100, 510, 10)

# calculate in parallel
res = Parallel(10)(
    delayed(train_predict_proba)(clf, *localizer[subj], 31) for subj in tqdm(subjects)
)
df_proba = pd.DataFrame()
for i, (probas_current, probas_others) in enumerate(tqdm(res, desc="subject")):
    subj = subjects[i]
    data_x, data_y = localizer[subj]
    assert len(data_x) == len(probas_current)
    df_subj1 = pd.DataFrame(
        {
            "probability": probas_current.ravel(),
            "timepoint": np.hstack([times] * len(probas_current)),
            "image": np.repeat(
                [items_base[sorting[subj][y]] for y in data_y], probas_current.shape[-1]
            ),
            "subject": subj,
            "stimulus": "current",
        }
    )
    df_subj2 = pd.DataFrame(
        {
            "probability": probas_others.ravel(),
            "timepoint": np.hstack([times] * len(probas_others)),
            "image": np.repeat(
                [items_base[sorting[subj][y]] for y in data_y], probas_others.shape[-1]
            ),
            "subject": subj,
            "stimulus": "other",
        }
    )
    df_proba = pd.concat([df_proba, df_subj1, df_subj2])

fig, ax = plt.subplots(1, 1, figsize=[6, 6])
sns.lineplot(
    data=df_proba, x="timepoint", y="probability", hue="stimulus", style="image", ax=ax
)

sns.despine()
ax.grid("on")
ax.set_title("raw class probabilities")
ax.set_xlabel("ms after stimulus onset")
fig.tight_layout()
fig.savefig(results_dir + "/localizer_raw_probabilities.png")
fig.savefig(results_dir + "/localizer_raw_probabilities.svg")
