# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 09:09:39 2022

Script to run the analysis for the paper titled
"Reactivation strength during cued recall is modulated by
graph distance within cognitive maps"



@author: Simon Kern
"""
import os
import random
import time

import compress_pickle
import matplotlib.pyplot as plt
import mne
import networkx
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from joblib import Memory, Parallel, delayed
from mne.stats import permutation_cluster_1samp_test
from scipy import ndimage, stats
from scipy.stats import pearsonr, ttest_rel
from statsmodels.stats.anova import AnovaRM
from tqdm import tqdm

# relative imports from current package
import settings
import utils
from load_funcs import (
    load_localizers_seq12,
    load_neg_x_before_audio_onset,
    load_testing_img2,
    load_testing_img2_fullseq,
)
from tdlm import run_tdlm
from utils import get_best_timepoint, get_performance, load_pkl_pandas

#%% run some preparation and static settings
# here results will be cached by joblib. Set dir in settings
memory = Memory(settings.cache_dir if settings.caching_enabled else None)

# for reproducibility
np.random.seed(0)

# create folders
results_dir = settings.results_dir + "/reactivation/"
pkl_dir = settings.cache_dir + "/pkl"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(pkl_dir, exist_ok=True)

## plotting settings
plt.rc("font", size=14)
meanprops = {
    "marker": "o",
    "markerfacecolor": "black",
    "markeredgecolor": "white",
    "markersize": "10",
}

#%% Settings

# this is the classifier that will be used
clf = utils.LogisticRegressionOvaNegX(C=6, penalty="l1", neg_x_ratio=2)

# list the files
files = utils.list_files(settings.data_dir, patterns=["*DSMR*"])
subjects = [f"DSMR{subj}" for subj in sorted(set(map(utils.get_id, files)))]

min_acc = 0.3  # minimum accuracy of decoders to include subjects
min_perf = 0.5  # minimum memory performance to include subjects
sfreq = 100  # downsample to this frequency. Changing is not supported.
ms_per_point = 10  # ms per sample point
labels = np.arange(1, 11)  # load events with these numbers

bands = settings.bands_HP  # only use HP filter

baseline = None  # no baseline correction of localizer
uid = hex(random.getrandbits(128))[2:10]  # get random UID for saving log files
date = time.strftime("%Y-%m-%d")
times = np.arange(-100, 510, 10)

tdlm_parameters = dict(seq=settings.seq_12,
                        clf=clf,
                        localizer_load_func=load_localizers_seq12,
                        min_acc=min_acc,
                        min_perf=min_perf,
                        maxLag=0.25,
                        alpha_freq=False,
                        neg_x_func=load_neg_x_before_audio_onset,
                        bands=bands,
                        nShuf=1000,
                        nsteps=1,
                        n_jobs=4,
                        plot_corr=False,
                        final_calculation=True,
                        sfreq=sfreq,
                    )

# best_tp will be calculated and replaced later, for debugging it is sometimes
# easier to define it here, so you can run segments without computing everything
best_tp = utils.load_pkl(f"{pkl_dir}/best_tp.pkl.zip", 31)

#%% Data loading

localizer = {}  # localizer training data
testing_img2 = {}  # testing trials, current "cue" image
neg_x = {}  # pre-audio fixation cross neg_x of localizer
seqs = {}  # load sequences of that participant in this dict

for subj in tqdm(subjects, desc="Loading data"):
    localizer[subj] = load_localizers_seq12(subj=subj, sfreq=sfreq, bands=bands)
    testing_img2[subj] = load_testing_img2(
        subj=subj, tmin=-0.1, tmax=0.5, sfreq=sfreq, bands=bands
    )
    neg_x[subj] = load_neg_x_before_audio_onset(subj=subj, sfreq=sfreq, bands=bands)

    seqs[subj] = utils.get_sequences(subj)

# input("Press enter to continue plotting")

#%% FIGURE 3A // Decoding localizer timepoint
# Procedure:
#    For each participant:
#    1. calculate best timepoint
#    2. save best tp in list of all participants
#
# Choose best classifier TP for each participants based on LOSO
#   - use data of all other participant for this participants best_tp
#   - in our case, this yields the same TP for all participants

# preload data that has already been computed
pkl_localizer = pkl_dir + "/1 localizer.pkl.zip"
results_localizer = load_pkl_pandas(
    pkl_localizer, default=pd.DataFrame(columns=["subject"])
)

### Calculation of best timepoint and leave-one-out cross-validation
ex_per_fold = 1
for i, subj in enumerate(tqdm(subjects, desc="subject")):

    # this is quite time consuming, so caching is implemented.
    # you can continue computation at a later stage if you interrupt.
    if subj in results_localizer["subject"].values:
        # skip subjects that are already computed
        print(f"{subj} already computed")
        continue

    data_x, data_y = localizer[subj]
    res = get_best_timepoint(
        data_x, data_y, subj=f"{subj}", n_jobs=1, ex_per_fold=ex_per_fold, clf=clf
    )
    results_localizer = pd.concat([results_localizer, res], ignore_index=True)

    results_localizer.to_pickle(pkl_localizer)  # store intermediate results


max_acc_subj = [
    results_localizer.groupby("subject")
    .get_group(subj)
    .groupby("timepoint")
    .mean(True)["accuracy"]
    .max()
    for subj in subjects
]
# only include participants with decoding accuracy higher than 0.3
included_idx = np.array(max_acc_subj) >= min_acc
included_subj = np.array(subjects)[included_idx]

subj_accuracy_all = (
    results_localizer.groupby(["subject", "timepoint"]).mean(True).reset_index()
)

# compute best decoding time point
best_tp_tmp = [
    subj_accuracy_all.groupby("subject").get_group(subj)["accuracy"].argmax()
    for subj in subjects
]

best_tp_tmp = np.array(best_tp_tmp)
best_acc_tmp = [
    subj_accuracy_all.groupby("subject").get_group(subj)["accuracy"].max()
    for subj in subjects
]
best_acc_tmp = np.array(best_acc_tmp)

# MEAN
best_tp = int(np.round(np.mean(best_tp_tmp[included_idx])))
best_acc = np.mean(best_acc_tmp[included_idx])

# store best timepoint for later retrieval
compress_pickle.dump(best_tp, f"{pkl_dir}/best_tp.pkl.zip")

results_localizer_included = pd.concat(
    [results_localizer.groupby("subject").get_group(subj) for subj in included_subj]
)

subj_acc_included = (
    results_localizer_included.groupby(["subject", "timepoint"])
    .mean(True)
    .reset_index()
)

# start plotting of mean decoding accuracy
fig = plt.figure(figsize=[6, 6])
ax = fig.subplots(1, 1)
sns.despine()
utils.plot_decoding_accuracy(
    results_localizer_included, x="timepoint", y="accuracy", ax=ax, color="tab:blue"
)
ax.set_ylabel("Accuracy")
ax.set_xlabel("ms after stimulus onset")
ax.set_title("Localizer: decoding accuracy")
ax.set_ylim(0, 0.5)

ax.axvspan(
    (best_tp * 10 - 100) - 5, (best_tp * 10 - 100) + 5, alpha=0.3, color="orange"
)
ax.legend(["decoding accuracy", "95% conf.", "chance", "peak"], loc="lower right")

plt.tight_layout()
plt.pause(0.1)
fig.savefig(results_dir + "/localizer.svg", bbox_inches="tight")
fig.savefig(results_dir + "/localizer.png", bbox_inches="tight")

#%% FIGURE 3B // General description of the dataset

# 0.1 - memory performance
df_stats = pd.DataFrame()
df_stats_blocks = pd.DataFrame()

# precomputation, this might take a whi
for subj in tqdm(subjects, desc="cross validating decoding performance"):
    utils.get_decoding_accuracy(subj=subj, clf=clf)

for subj in subjects:
    # retrieve precomputed cross validation results
    acc = utils.get_decoding_accuracy(subj=subj, clf=clf)[0]

    # get participant performance for learning and testing
    val_learn = get_performance(subj=subj, which="learning")
    val_test = get_performance(subj=subj, which="test")
    scatter = (np.random.rand() * -0.5) * 0.02 + 1
    df_tmp = pd.DataFrame(
        {
            "participant": subj,
            "performance": [val_learn[-1] * scatter, val_test * scatter],
            "type": ["learning\n(last block)", "test"],
        }
    )
    df_tmp_blocks = pd.DataFrame(
        {
            "participant": subj,
            "performance": [val_learn[0], val_learn[-1], val_test],
            "block": ["first", "last", "test"],
            "n_blocks": len(val_learn),
            "acc": acc,
        }
    )
    df_stats = pd.concat([df_stats, df_tmp])
    df_stats_blocks = pd.concat([df_stats_blocks, df_tmp_blocks])

# store some results for later usage
df_stats_blocks.to_pickle(pkl_dir + "/df_stats_blocks.pkl")

# start plotting
fig = plt.figure(figsize=[4, 6])
ax = fig.subplots(1, 1)
sns.despine()
df_stats_blocks = df_stats_blocks[df_stats_blocks.acc >= min_acc]
sns.boxplot(
    data=df_stats_blocks,
    x="block",
    y="performance",
    color=sns.color_palette()[0],
    meanprops=meanprops,
    showmeans=True,
    ax=ax,
)

ax.set_title("Memory performance")
sns.despine()
plt.tight_layout()
fig.savefig(results_dir + "/memory_performance.png")
fig.savefig(results_dir + "/memory_performance.svg")


perf_first, perf_last, perf_test = df_stats_blocks.groupby("block")
p = ttest_rel(perf_last[1].performance, perf_test[1].performance)
mean_diff = perf_last[1].performance.mean() - perf_test[1].performance.mean()

print(f"Learning performance first block {perf_first[1].performance.mean():.2f}")
print(f"Learning performance last block {perf_last[1].performance.mean():.2f}")
print(f"Learning increase last->test {p=}")


#%% FIGURE 3C+3D // classifier heatmap transfer across time and to testing

# store precomputed results
pkl_loc = pkl_dir + "/1b heatmap loc.pkl.zip"
pkl_test = pkl_dir + "/1b heatmap test.pkl.zip"

# get decoder accuracy per participants to exclude low decodable participants
accuracies = [utils.get_decoding_accuracy(subj=subj, clf=clf)[0] for subj in subjects]
subj_incl = [subj for subj, acc in zip(subjects, accuracies) if acc > 0.3]

# calculate time by time decoding heatmap from localizer
# basically: How well can a clf trained on t1 predict t2 of the localizer
if os.path.isfile(pkl_loc):
    # either load precomputed results or compute and store
    maps_localizer = compress_pickle.load(pkl_loc)
else:
    # use a parallel pool, as this is computationally quite expensive
    pool = Parallel(len(subj_incl))  # use parallel pool
    maps_localizer = pool(
        delayed(utils.get_decoding_heatmap)(clf, *localizer[subj], n_jobs=2)
        for subj in subj_incl
    )
    compress_pickle.dump(maps_localizer, pkl_loc)


# second calculate testing decoding heatmap. No cross-validation needed.
# I.e. how well can the best localizer classifier classify the current image
# during testing
maps_testing = []
if os.path.isfile(pkl_test):
    maps_testing = compress_pickle.load(pkl_test)
else:
    # use a parallel pool, as this is computationally quite expensive
    pool = Parallel(len(subj_incl))  # use parallel pool
    maps_testing = pool(
        delayed(utils.get_transfer_heatmap)(clf, *localizer[subj], *testing_img2[subj])
        for subj in subj_incl
    )
    compress_pickle.dump(maps_testing, pkl_test)

# zscore maps for statistics
maps_loc_norm = np.array(maps_localizer) - np.mean(maps_localizer)
maps_test_norm = np.array(maps_testing) - np.mean(maps_testing)
maps_loc_norm = maps_loc_norm / maps_loc_norm.std()
maps_test_norm = maps_test_norm / maps_test_norm.std()

# perform cluster permutation testing, basically same as when doing fMRI
t_thresh = stats.distributions.t.ppf(1 - 0.05, df=len(maps_localizer) - 1)
t_clust, clusters1, p_values1, H0 = permutation_cluster_1samp_test(
    maps_loc_norm,
    tail=1,
    n_jobs=None,
    threshold=t_thresh,
    adjacency=None,
    n_permutations=1000,
    out_type="mask",
)
t_clust, clusters2, p_values2, H0 = permutation_cluster_1samp_test(
    maps_test_norm,
    tail=1,
    n_jobs=None,
    threshold=t_thresh,
    adjacency=None,
    n_permutations=1000,
    out_type="mask",
)

# create mask from cluster for all clusters of p<0.05
clusters_sum1 = (np.array(clusters1)[p_values1 < 0.05]).sum(0)
clusters_sum2 = (np.array(clusters2)[p_values2 < 0.05]).sum(0)

# now plot the heatmaps with masking using MNE visualization functions
fig = plt.figure(figsize=[10, 4.5])
axs = fig.subplots(1, 2)
axs = axs.flatten()
x = mne.viz.utils._plot_masked_image(
    axs[0],
    np.mean(maps_localizer, 0),
    times=range(61),
    mask=clusters_sum1,
    cmap="viridis",
    mask_style="contour",
    vmin=0.1,
    vmax=0.4,
)
plt.colorbar(x[0])

x = mne.viz.utils._plot_masked_image(
    axs[1],
    np.mean(maps_testing, 0),
    times=range(61),
    mask=clusters_sum2,
    cmap="viridis",
    mask_style="contour",
    vmin=0.1,
    vmax=0.25,
)
plt.colorbar(x[0])

titles = ["Localizer transfer", "Retrieval transfer"]
for i, ax in enumerate(axs):
    ax.set_title(titles[i])
    times = np.arange(-100, 500, 50)  # t
    ax.set_xticks(np.arange(0, 60, 5), times, rotation=40)
    ax.set_yticks(np.arange(0, 60, 5), times)
    ax.set_xlabel("localizer train time (ms after onset)")
    ax.set_ylabel(
        f'{"retrieval " if i==1 else "localizer "} test time (ms after onset)'
    )

fig.tight_layout()
fig.savefig(results_dir + "/classifier-transfer.svg")
fig.savefig(results_dir + "/classifier-transfer.png")

#%% FIGURE 4 // Sequenceness during Test-Seq2

load_functions = {
    "Test img2 full": load_testing_img2_fullseq,
}

# save and restore precomputed results to this file
pkl_sequenceness = pkl_dir + "/4 sequenceness.pkl.gz"

times = np.arange(-100, 510, 10)

# load precomputed results
if os.path.isfile(pkl_sequenceness):
    res = compress_pickle.load(pkl_sequenceness)
else:
    # if the file doesn't exist, calculate sequenceness using this function.
    res = run_tdlm(
        subjects=subjects,
        load_functions=load_functions,
        title="Test Segment",
        timepoint=times[best_tp] / 1000,
        **tdlm_parameters,
    )
    compress_pickle.dump(res, pkl_sequenceness, pickler_method="dill")

# recreate Figure 4
fig, axs = plt.subplot_mosaic("XXAB\nYYCD", figsize=[14, 6.5])
sns.despine()

palette = sns.color_palette()
ax_sf1 = axs["X"]
ax_sf2 = axs["A"]
ax_sf3 = axs["B"]
ax_sb1 = axs["Y"]
ax_sb2 = axs["C"]
ax_sb3 = axs["D"]

sf2 = res.sf_all["Test img2 full"]
sb2 = res.sb_all["Test img2 full"]

sf2_mean = np.nanmean(sf2[:, 0, :], 1)
sb2_mean = np.nanmean(sb2[:, 0, :], 1)

ax_sf1.hlines(0, 0, 250, color="black", alpha=0.2)
ax_sb1.hlines(0, 0, 250, color="black", alpha=0.2)

plt.figtext(
    0.3,
    0.95,
    "Sequenceness during cued test",
    fontsize=18,
    horizontalalignment="center",
)
plt.figtext(
    0.65, 0.95, "Permutation distribution", fontsize=18, horizontalalignment="center"
)
plt.figtext(
    0.9, 0.95, "Perf. x sequenceness", fontsize=18, horizontalalignment="center"
)

utils.plot_sf_sb(
    sf2,
    sb2,
    which=["fwd"],
    title="\nforward",
    ax=ax_sf1,
    rescale=False,
    plot95=True,
    plotmax=True,
    clear=False,
)
utils.plot_sf_sb(
    sf2,
    sb2,
    which=["bkw"],
    title="backward",
    ax=ax_sb1,
    rescale=False,
    plot95=True,
    plotmax=True,
    clear=False,
)
ax_sf1.legend(
    ["_", "forward sequenceness", "perm. max.", "_", "95%"],
    loc="lower left",
    fontsize=11,
)
ax_sb1.legend(
    ["_", "backward sequenceness", "perm. max.", "_", "95%"],
    loc="lower left",
    fontsize=11,
)

# normalize ylims
ymin = min(*ax_sf1.get_ylim(), *ax_sb1.get_ylim())
ymax = max(*ax_sf1.get_ylim(), *ax_sb1.get_ylim())
ax_sf1.set_ylim([ymin, ymax])
ax_sb1.set_ylim([ymin, ymax])


utils.plot_permutation_distribution(sf2, ax=ax_sf2, title="forward", color=palette[1])
utils.plot_permutation_distribution(sb2, ax=ax_sb2, title="backward", color=palette[2])

#

n_blocks = {
    subj: len(get_performance(subj=subj, which="learning")) for subj in res.subjects
}
perf_test = {subj: get_performance(subj=subj, which="test") for subj in res.subjects}
sf_means2 = dict(zip(res.subjects, sf2_mean, strict=True))
sb_means2 = dict(zip(res.subjects, sb2_mean, strict=True))


ax_sf3.clear()
r, pval1 = utils.plot_correlation(
    sf_means2, values=perf_test, color=palette[1], ax=ax_sf3, absolute=False
)
ax_sf3.set_ylim([None, 1.15])
ax_sf3.set_title("\nforward")
ax_sf3.text(-0.01, 1.1, f"r={r:.2f} p={pval1:.3f}", fontsize=13)


ax_sb3.clear()
r, pval2 = utils.plot_correlation(
    sb_means2, values=perf_test, color=palette[2], ax=ax_sb3, absolute=False
)
ax_sb3.set_ylim([None, 1.15])
ax_sb3.set_title("backward")
ax_sb3.text(-0.01, 1.1, f"r={r:.2f} p={pval2:.3f}", fontsize=13)

fig.tight_layout()
fig.savefig(results_dir + "sequencess_test_seq2_trials.svg")

#%% FIGURE 5A differential reactivation


def calculate_diff_proba(clf, subj, best_tp, n_shuffles, sigma=1, response="correct"):
    # set persistent random seed per participants
    np.random.seed(utils.get_id(subj)+42)

    # sequence ABC[...] without end wrapping
    full_seq = utils.char2num(settings.seq_12)[:-2]

    # retrieve participant's performance and number of learning block
    perf_test = utils.get_performance(subj, which="test")
    perf_learn = utils.get_performance(subj, which="learning")[-1]
    nblocks = len(utils.get_performance(subj, which="learning"))

    test_x, test_y = testing_img2[subj]


    # get the sequences and the response of this subject
    seq, resp = seqs[subj]

    # get decoding accuracy of this participant for this classifier
    acc, _ = utils.get_decoding_accuracy(subj, clf=clf)

    if acc < 0.3:
        return pd.DataFrame()
    if "120" in subj:
        # we lost the data file for the learning part of DSMR120
        return pd.DataFrame()
    # Train classifier with timepoint from 1)
    tp = best_tp
    clf.fit(X=localizer[subj][0][:, :, tp], y=localizer[subj][1], neg_x=neg_x[subj])

    # match seq and resp to the loaded trials
    # reason: some trials might have been discarded due to artefacts
    # we need to filter these out by using a bit of matching
    i_checked = 0
    seq_filtered = []
    resp_filtered = []
    for y in test_y:
        while y != seq[i_checked][1]:
            i_checked += 1
        seq_filtered.append(seq[i_checked])
        resp_filtered.append(resp[i_checked])
    np.testing.assert_array_equal(np.array(seq_filtered)[:, 1], test_y)
    seq = seq_filtered
    resp = resp_filtered

    probas = np.array([clf.predict_proba(test_x[:, :, t]) for t in range(len(times))])
    # smooth probabilities, as they are quite noisy
    probas = ndimage.gaussian_filter(probas, (sigma, 0, 0))
    probas = np.swapaxes(probas, 0, 1)

    # look at i+2 items
    i_next = 2

    tmp = []
    for shuf in list(range(n_shuffles)):
        proba_diff = []
        for r, proba, curr, (seq1, seq2, *_) in zip(
            resp, probas, test_y, seq, strict=True
        ):
            if response != "both" and r != response:
                continue

            # ignore the items that appear twice in sequence
            if seq2 in [4, 1]:
                continue
            assert curr == seq2, "curr should be seq2, something went wrong"

            # index of current item in sequence
            idx_in_seq = full_seq.index(curr) + len(full_seq)

            # next two items, exclude currently on screen item
            next_items = (full_seq + full_seq + full_seq)[
                idx_in_seq + 1 : idx_in_seq + i_next + 1
            ]
            if seq1 in next_items:
                next_items.remove(seq1)
            # ignore all items in the surrounding, e.g. +-2 of seq2
            ignore = (full_seq + full_seq + full_seq)[
                idx_in_seq - 2 : idx_in_seq + i_next + 1
            ]
            dist_items = [x for x in np.arange(max(full_seq) + 1) if not x in ignore]

            # shuffle the labels, except for the first run
            if shuf > 0:
                all_labels_shuf = np.random.permutation(next_items + dist_items)
                next_items = all_labels_shuf[: len(next_items)]
                dist_items = all_labels_shuf[len(next_items) :]

            # sanity check for permutation
            for item in next_items:
                assert item not in dist_items

            # make sure the on-screen items are not contained in the calculation
            for item in [seq1, seq2]:
                assert item not in next_items, f"{item=} in {next_items=}"
                assert item not in dist_items, f"{item=} in {dist_items=}"

            proba_near_trial = np.mean(proba[:, next_items], -1)
            proba_dist_trial = np.mean(proba[:, dist_items], -1)

            diff = proba_near_trial - proba_dist_trial
            proba_diff.append(diff)

        # mean differential probability across all trials
        proba_diff = np.mean(proba_diff, 0)

        df = pd.DataFrame(
            {
                "diff": proba_diff,
                "timepoint": times,
                "shuffle": shuf,
            }
        )
        tmp.append(df)

    df = pd.concat(tmp, ignore_index=True)

    # store results in dataframe
    df["max_acc"] = acc
    df["subject"] = subj
    df["nblocks"] = nblocks
    df["performance_test"] = perf_test
    df["performance_learning"] = perf_learn
    df["train_tp"] = tp
    df["block"] = 7
    return df


n_shuffles = 10000
sigma = 1

### Calculation
res = Parallel(10)(
    delayed(calculate_diff_proba)(
        clf, subj, best_tp, n_shuffles=n_shuffles, sigma=sigma
    )
    for subj in tqdm(subjects, desc="training classifiers")
)
df_diffs = pd.concat(res, ignore_index=True)

####### Plotting
# first filter out too low decoding accuracy
include = df_diffs["max_acc"] >= min_acc

df_diffs_inc = df_diffs[include]
df_diffs_exc = df_diffs[include == False]

subj_include = df_diffs_inc["subject"].unique()

# lets calculate some p values from the shuffles
# get the mean diff value for each shuffle for each timepoint
df_groupby = df_diffs_inc.groupby(["shuffle", "timepoint"]).mean(True)["diff"].unstack()
df_base_mean = df_groupby.iloc[0, :]
df_shuffles_mean = df_groupby.iloc[1:, :]

# calculate p value based on how often the value is larger
# in the ground truth than we would expect by chance
df_pval = (df_base_mean < df_shuffles_mean).mean(0)
df_base = df_diffs_inc[df_diffs_inc["shuffle"] == 0]
max_diff = df_base.groupby("timepoint").mean(True)
val_max_diff = max_diff["diff"].max()
tp_max_diff = max_diff["diff"].argmax()

# plotting of pvalue over time
fig = plt.figure(figsize=[8, 6])
ax = fig.subplots(1, 1)
sns.despine()
sns.lineplot(data=df_base, x="timepoint", y="diff", ax=ax, legend="brief")

# color significant timepoints with a red band
p_significant = utils.get_streaks(df_pval[df_pval < 0.05].index // 10) * 10
tp_train = times[df_diffs_inc["train_tp"].unique()]
if len(tp_train) == 1:
    tp_train = [tp_train[0] - 5, tp_train[0] + 5]
for xmin, xmax in p_significant:
    ax.axvspan(xmin - 5, xmax + 5, color="red", alpha=0.3)

ax.legend(["Diff. reactivation", "95% conf.", "p<0.05"])
ax.hlines(0, times[0], times[-1], color="gray", linestyle="--", linewidth=1)
ax.set_title("Differential reactivation near vs distant items")
ax.set_ylabel("Differential probability")
ax.set_xlabel("time after cue onset (ms)")
fig.tight_layout()
fig.savefig(results_dir + "/differential probability.png")
fig.savefig(results_dir + "/differential probability.svg")

#%% FIGURE 5A+B differential reactivation graded by distance


@memory.cache
def get_distances(node, maxdist=None):
    graph = networkx.Graph()
    seq = utils.char2num(settings.seq_12)
    for i, _ in enumerate(seq[:-1]):
        graph.add_edge(seq[i], seq[i + 1])

    paths = dict(networkx.all_pairs_shortest_path_length(graph))
    return paths[node]


@memory.cache
def get_distances_directed(node, maxdist=None):
    graph = networkx.DiGraph()
    seq = utils.char2num(settings.seq_12)
    for i, _ in enumerate(seq[:-1]):
        graph.add_edge(seq[i], seq[i + 1])

    paths = dict(networkx.all_pairs_shortest_path_length(graph))
    return paths[node]


def fit(clf, subj, *args, **kwargs):
    np.random.seed(utils.get_id(subj))
    return clf.fit(*args, **kwargs)


sigma = 1

clfs = Parallel(len(subjects) - 1)(
    delayed(fit)(
        clf,
        subj,
        X=localizer[subj][0][:, :, best_tp],
        y=localizer[subj][1],
        neg_x=neg_x[subj],
    )
    for subj in tqdm(subjects, desc="train decoders")
)

df_choice = pd.DataFrame()
df_dist1 = pd.DataFrame()
df_dist2 = pd.DataFrame()

for clf, subj in zip(clfs, tqdm(subjects), strict=True):
    np.random.seed(utils.get_id(subj))

    perf_test = utils.get_performance(subj=subj, which="test")
    perf_learn = utils.get_performance(subj=subj, which="learning")[-1]
    nblocks = len(utils.get_performance(subj=subj, which="learning"))

    test_x, test_y = testing_img2[subj]

    # get the sequences and the response of this subject
    seq, resp = seqs[subj]

    # get decoding accuracy of this participant for this classifier
    acc, _ = utils.get_decoding_accuracy(subj, clf=clf)

    if acc < 0.3:
        print(f"{subj} acc too low: {acc:.3f}")
        continue

    # Train classifier with timepoint from 1)
    tp = best_tp
    # match seq and resp to the loaded trials
    # reason: some trials might have been discarded due to artefacts
    i_checked = 0
    seq_filtered = []
    resp_filtered = []
    for y in test_y:
        while y != seq[i_checked][1]:
            i_checked += 1
        seq_filtered.append(seq[i_checked])
        resp_filtered.append(resp[i_checked])
    np.testing.assert_array_equal(np.array(seq_filtered)[:, 1], test_y)
    seq = seq_filtered
    resp = resp_filtered

    probas = np.array([clf.predict_proba(test_x[:, :, t]) for t in range(len(times))])
    probas = ndimage.gaussian_filter(probas, (sigma, 0, 0))
    probas = np.swapaxes(probas, 0, 1)

    tmp_dist1 = pd.DataFrame()
    tmp_dist2 = pd.DataFrame()
    tmp_choice = pd.DataFrame()
    trial = 0
    for r, proba, curr, (seq1, seq2, seq3, *_, choice) in zip(
        resp, probas, test_y, seq, strict=True
    ):
        # ignore the items that appear twice in sequence

        assert curr == seq2, "curr should be seq2, something went wrong"

        if r == "wrong":
            tmp_choice = pd.concat(
                [
                    tmp_choice,
                    pd.DataFrame(
                        {
                            "proba": [
                                *proba[:, choice],
                                *proba[:, seq3],
                                *(proba[:, choice] - proba[:, seq3]),
                            ],
                            "timepoint": [*times, *times, *times],
                            "which": ["choice"] * len(times)
                            + ["target"] * len(times)
                            + ["diff"] * len(times),
                        }
                    ),
                ]
            )
        ignore = [seq1]
        dists1 = get_distances(curr)
        dists2 = get_distances_directed(curr)

        _proba = []
        _dist = []
        _times = []

        for node, dist in dists1.items():
            if node in ignore:
                continue
            p = proba[:, node]
            _proba.extend(p)
            _dist.extend([dist] * len(p))
            _times.extend(times)
            tmp_dist1 = pd.concat(
                [
                    tmp_dist1,
                    pd.DataFrame(
                        {
                            "proba": _proba,
                            "timepoint": _times,
                            "distance": _dist,
                            "choice": r,
                            "trial": trial,
                        }
                    ),
                ]
            )
        for node, dist in dists2.items():
            if node in ignore:
                continue
            p = proba[:, node]
            _proba.extend(p)
            _dist.extend([dist] * len(p))
            _times.extend(times)
            tmp_dist2 = pd.concat(
                [
                    tmp_dist2,
                    pd.DataFrame(
                        {
                            "proba": _proba,
                            "timepoint": _times,
                            "distance": _dist,
                            "choice": r,
                            "trial": trial,
                        }
                    ),
                ]
            )
        trial += 1
    tmp_dist1["subject"] = subj
    tmp_dist2["subject"] = subj
    tmp_choice["subject"] = subj
    tmp_dist1["performance_test"] = perf_test
    tmp_dist2["performance_test"] = perf_test
    tmp_choice["performance_test"] = perf_test

    df_choice = pd.concat([df_choice, tmp_choice], ignore_index=True)
    df_dist1 = pd.concat([df_dist1, tmp_dist1], ignore_index=True)
    df_dist2 = pd.concat([df_dist2, tmp_dist2], ignore_index=True)

# use directed graph, makes more sense
# df_dist = df_dist1  # undirected graph -> shows u-shape! cool
df_dist = df_dist2  # directed graph -> basically undirected u-graph collapsed

##### Plot by distance
fig2 = plt.figure(figsize=[8, 6])
ax1 = fig2.subplots(1, 1)
sns.despine()
fig2.suptitle("Reactivation strength by distance on directional graph")

react_tp = times[df_dist.groupby("timepoint").mean(True)["proba"].argmax()]

df_peak = df_dist[df_dist["timepoint"] == react_tp]
df_peak = df_peak.groupby(["distance", "subject"]).mean(True)
df_peak = df_peak.reset_index()
df_peak = df_peak[df_peak["distance"] < 5]

ax1.clear()
meanprops = {
    "marker": "o",
    "markerfacecolor": "black",
    "markeredgecolor": "white",
    "markersize": "10",
}

sns.boxplot(
    data=df_peak,
    x="distance",
    y="proba",
    ax=ax1,
    boxprops=dict(alpha=0.3),
    saturation=1,
    width=0.5,
    showmeans=True,
    meanprops=meanprops,
    color="tab:blue",
)
sns.regplot(data=df_peak, x="distance", y="proba", ax=ax1, scatter=False)
# sns.regplot(data=df_peak, x='distance', y='proba', ax=ax)
r, p = pearsonr(df_peak["proba"], df_peak["distance"])
ax1.set_title(f"all trials {r=:.2f} {p=:.3f}")
ax1.set_ylabel("reactivation probability")
ax1.set_xlabel("distance on directed graph")
plt.pause(0.1)
fig.tight_layout()
fig2.savefig(results_dir + "/graph_distance_all.png")
fig2.savefig(results_dir + "/graph_distance_all.svg")

##### Plot correct/wrong answers
fig5 = plt.figure(figsize=[16, 6])
axs = fig5.subplots(1, 2)
sns.despine()
fig5.suptitle("Reactivation strength by trial response")

for i, response in enumerate(["correct", "wrong"]):
    react_tp = times[df_dist.groupby("timepoint").mean(True)["proba"].argmax()]
    idx = react_tp == df_dist["timepoint"]

    df_peak = df_dist[idx].groupby(["distance", "subject", "choice"]).mean(True)
    df_peak = df_peak.reset_index()
    df_peak = df_peak[df_peak.distance < 5]
    df_peak = df_peak[df_peak.choice == response]

    ax = axs[i]
    ax.clear()
    color = "xkcd:blue green" if response == "correct" else "xkcd:dull red"
    sns.boxplot(
        data=df_peak,
        x="distance",
        y="proba",
        ax=ax,
        color=color,
        saturation=1,
        width=0.5,
        boxprops=dict(alpha=0.3),
        showmeans=True,
        meanprops=meanprops,
    )
    sns.regplot(
        data=df_peak, x="distance", y="proba", ax=ax, color=color, scatter=False
    )
    r, p = pearsonr(df_peak["proba"], df_peak["distance"])
    response = response.replace("wrong", "incorrect")
    ax.set_title(f"{response=} {r=:.2f} {p=:.4f}")
    ax.set_ylabel("reactivation probability")
    ax.set_xlabel("distance on directed graph")
    ax.set_ylim(*ax1.get_ylim())

plt.pause(0.1)
fig.tight_layout()
fig5.savefig(results_dir + "/graph_distance_response.png")
fig5.savefig(results_dir + "/graph_distance_response.svg")

df_peak = df_dist[df_dist["timepoint"] == react_tp]
df_peak = df_peak.groupby(["distance", "subject", "choice"]).mean(True)
df_peak = df_peak.reset_index()
df_peak = df_peak[df_peak["distance"] < 5]

df_peak_all = df_peak.groupby(["subject", "distance"]).mean(True).reset_index()

# statistical analysis using ANOVA
model = AnovaRM(df_peak_all, "proba", "subject", within=["distance"])
print("both", model.fit())

for choice in ["correct", "wrong"]:
    df_choice = df_peak[df_peak.choice == choice]
    model = AnovaRM(df_choice, "proba", "subject", ["distance"])
    print(choice, "\n", model.fit())

pg.rm_anova(
    data=df_peak_all, dv="proba", within="distance", subject="subject", detailed=True
)

has_wrong_resp = df_peak[df_peak.choice == "wrong"].subject.unique()
model = AnovaRM(
    df_peak[df_peak.subject.isin(has_wrong_resp)],
    "proba",
    "subject",
    within=["distance", "choice"],
)
print(choice, "\n", model.fit())
