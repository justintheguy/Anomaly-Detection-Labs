import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

np.random.seed(42)


import warnings

warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*no accelerator is found.*"
)

'''
#Ex 1
x_train, _ = make_blobs(
    n_samples=500,
    centers=[(0, 0)],
    cluster_std=1,
    random_state=42
)

projection_vectors = np.random.multivariate_normal(
    mean=[0, 0],
    cov=np.eye(2),
    size=5
)

projection_vectors = (
    projection_vectors
    / np.linalg.norm(
        projection_vectors,
        axis=1,
        keepdims=True
    )
)

print("Projection vectors:")
print(projection_vectors)

print("\nVector lengths:")
print(
    np.linalg.norm(
        projection_vectors,
        axis=1
    )
)

projected_train = []

for vector in projection_vectors:
    projection = x_train @ vector
    projected_train.append(projection)


fig, axes = plt.subplots(
    1,
    5,
    figsize=(18, 4)
)

for i in range(5):

    axes[i].hist(
        projected_train[i],
        bins=10,
        range=(-4, 4)
    )

    axes[i].set_title(
        f"Projection {i + 1}"
    )

plt.tight_layout()
plt.show()


x_test = np.random.uniform(
    low=-3,
    high=3,
    size=(500, 2)
)

n_bins_values = [5, 10, 20, 40]
hist_range = (-5, 5)


fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5)
)

for plot_index, n_bins in enumerate(n_bins_values):
    probabilities_vector = []
    bin_edges_list = []

    for projection in projected_train:

        counts, bin_edges = np.histogram(
            projection,
            bins=n_bins,
            range=hist_range
        )

        probabilities = counts / len(projection)

        probabilities_vector.append(probabilities)
        bin_edges_list.append(bin_edges)



    projected_test = []

    for vector in projection_vectors:
        projection = x_test @ vector
        projected_test.append(projection)


    test_probabilities = np.zeros(
        (x_test.shape[0], 5)
    )

    for i in range(5):

        values = projected_test[i]

        bin_edges = bin_edges_list[i]

        current_probabilities = probabilities_vector[i]

        bin_indices = np.digitize(
            values,
            bin_edges
        ) - 1

        bin_indices = np.clip(
        bin_indices,
        0,
        n_bins - 1
        )

        

        test_probabilities[:, i] = current_probabilities[bin_indices]


    test_scores = np.mean(
        test_probabilities,
        axis=1
    )

    



    scatter = axes[plot_index].scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=test_scores,
        cmap="viridis",
        s=25
    )

    

    axes[plot_index].set_xlabel("x1")
    axes[plot_index].set_ylabel("x2")
    axes[plot_index].set_title(f"Simplified LODA score map for n_bins = {n_bins} ")

plt.tight_layout()
plt.show() 
'''

#Ex 2
''' 
x_train, _ = make_blobs(
    n_samples=[500, 500],
    centers=[(10, 0), (0, 10)],
    cluster_std=1,
    random_state=42
)

iforest = IForest(
    contamination=0.02,
    random_state=42
)

dif = DIF(
    contamination=0.02,
    random_state=42
)

loda = LODA(
    contamination=0.02,
    random_state=42
)

iforest.fit(x_train)
dif.fit(x_train)
loda.fit(x_train)

x_test = np.random.uniform(
    low=-10,
    high=20,
    size=(1000, 2)
)

iforest_scores = iforest.decision_function(x_test)
dif_scores = dif.decision_function(x_test)
loda_scores = loda.decision_function(x_test)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

models = [
    ("IForest", iforest_scores),
    ("DIF", dif_scores),
    ("LODA", loda_scores)
]

for i, (name, scores) in enumerate(models):
    scatter = axes[i].scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=scores,
        cmap="viridis",
        s=25
    )

    fig.colorbar(
        scatter,
        ax=axes[i]
    )

    axes[i].set_xlabel("x1")
    axes[i].set_ylabel("x2")
    axes[i].set_title(name)

dif_configs = [
    [50, 20],
    [100, 50],
    [500, 100]
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

for i, hidden_layers in enumerate(dif_configs):

    dif_model = DIF(
        contamination=0.02,
        hidden_neurons=hidden_layers,
        random_state=42
    )

    dif_model.fit(x_train)

    dif_scores = dif_model.decision_function(x_test)

    scatter = axes[i].scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=dif_scores,
        cmap="viridis",
        s=25
    )

    axes[i].set_title(
        f"DIF hidden_neurons={hidden_layers}"
    )

    axes[i].set_xlabel("x1")
    axes[i].set_ylabel("x2")

    fig.colorbar(
        scatter,
        ax=axes[i]
    )


loda_bins = [5, 10, 20, 40]

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5)
)

for i, n_bins in enumerate(loda_bins):

    loda_model = LODA(
        contamination=0.02,
        n_bins=n_bins,
        random_state=42
    )

    loda_model.fit(x_train)

    loda_scores = loda_model.decision_function(x_test)

    scatter = axes[i].scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=loda_scores,
        cmap="viridis",
        s=25
    )

    axes[i].set_title(
        f"LODA n_bins={n_bins}"
    )

    axes[i].set_xlabel("x1")
    axes[i].set_ylabel("x2")

    fig.colorbar(
        scatter,
        ax=axes[i]
    )



x_train_3d, _ = make_blobs(
    n_samples=[500, 500],
    centers=[
        (0, 10, 0),
        (10, 0, 10)
    ],
    cluster_std=1,
    random_state=42
)

x_test_3d = np.random.uniform(
    low=-10,
    high=20,
    size=(1000, 3)
)

iforest_3d = IForest(
    contamination=0.02,
    random_state=42
)

dif_3d = DIF(
    contamination=0.02,
    random_state=42
)

loda_3d = LODA(
    contamination=0.02,
    random_state=42
)

iforest_3d.fit(x_train_3d)
dif_3d.fit(x_train_3d)
loda_3d.fit(x_train_3d)

iforest_scores_3d = iforest_3d.decision_function(x_test_3d)
dif_scores_3d = dif_3d.decision_function(x_test_3d)
loda_scores_3d = loda_3d.decision_function(x_test_3d)

fig = plt.figure(
    figsize=(20, 6)
)

# IForest
ax1 = fig.add_subplot(
    131,
    projection="3d"
)

scatter1 = ax1.scatter(
    x_test_3d[:, 0],
    x_train_3d[:, 1],
    x_test_3d[:, 2],
    c=iforest_scores_3d,
    cmap="viridis",
    s=20
)

ax1.set_title("IForest - 3D")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("x3")

fig.colorbar(
    scatter1,
    ax=ax1,
    shrink=0.6
)


# DIF
ax2 = fig.add_subplot(
    132,
    projection="3d"
)

scatter2 = ax2.scatter(
    x_test_3d[:, 0],
    x_test_3d[:, 1],
    x_test_3d[:, 2],
    c=dif_scores_3d,
    cmap="viridis",
    s=20
)

ax2.set_title("DIF - 3D")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_zlabel("x3")

fig.colorbar(
    scatter2,
    ax=ax2,
    shrink=0.6
)


# LODA
ax3 = fig.add_subplot(
    133,
    projection="3d"
)

scatter3 = ax3.scatter(
    x_test_3d[:, 0],
    x_test_3d[:, 1],
    x_test_3d[:, 2],
    c=loda_scores_3d,
    cmap="viridis",
    s=20
)

ax3.set_title("LODA - 3D")
ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.set_zlabel("x3")

fig.colorbar(
    scatter3,
    ax=ax3,
    shrink=0.6
)


plt.tight_layout()
plt.show()

'''

#Ex3

data = loadmat("shuttle.mat")

#print("SHUTTLE DATA KEYS:")
#print(data.keys())

x = data["X"]
y = data["y"].ravel()

#print("x shape:", x.shape)
#print("y shape:", y.shape)

#print("Normal samples:", np.sum(y == 0))
#print("Anomalies:", np.sum(y == 1))

#print("Contamination:", np.mean(y))

ba_iforest = []
auc_iforest = []

ba_loda = []
auc_loda = []

ba_dif = []
auc_dif = []

for seed in range(10):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.4,
        random_state=seed,
        stratify=y
    )

    
    x_train, x_test = standardi  zer(
        x_train,
        x_test
    )

   
    iforest = IForest(
        contamination=np.mean(y),
        random_state=42
    )

    loda = LODA(
        contamination=np.mean(y),
        random_state=42
    )

    dif = DIF(
        contamination=np.mean(y),
        random_state=42
    )

    iforest.fit(x_train)
    loda.fit(x_train)
    dif.fit(x_train)

    iforest_pred = iforest.predict(x_test)

    iforest_scores = iforest.decision_function(x_test)

    ba_iforest.append(
        balanced_accuracy_score(
            y_test,
            iforest_pred
        )
    )

    auc_iforest.append(
        roc_auc_score(
            y_test,
            iforest_scores
        )
    )


    loda_pred = loda.predict(x_test)

    loda_scores = loda.decision_function(x_test)

    ba_loda.append(
        balanced_accuracy_score(
            y_test,
            loda_pred
        )
    )

    auc_loda.append(
        roc_auc_score(
            y_test,
            loda_scores
        )
    )



    dif_pred = dif.predict(x_test)

    dif_scores = dif.decision_function(x_test)

    ba_dif.append(
        balanced_accuracy_score(
            y_test,
            dif_pred
        )
    )

    auc_dif.append(
        roc_auc_score(
            y_test,
            dif_scores
        )
    )

    print(f"Finished split {seed + 1}/10")

print("\nMEAN RESULTS")

print("\nIForest")
print("Mean BA:", np.mean(ba_iforest))
print("Mean ROC AUC:", np.mean(auc_iforest))

print("\nLODA")
print("Mean BA:", np.mean(ba_loda))
print("Mean ROC AUC:", np.mean(auc_loda))

print("\nDIF")
print("Mean BA:", np.mean(ba_dif))
print("Mean ROC AUC:", np.mean(auc_dif))