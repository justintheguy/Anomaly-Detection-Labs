import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import make_blobs
from pyod.models.lof import LOF
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

#Ex1
'''
a = 2
b = 1

variances = [0.25, 1, 4, 16]
mean_values = [-5, 0, 5, 10]

x_regular = np.random.normal(
    loc=0,
    scale=1,
    size=100
)

mean = 0
sigma = 0.5
'''
'''
epsilon = np.random.normal(
    loc=mean,
    scale=sigma,
    size=100

)


y_regular = a * x_regular + b + epsilon
'''
'''
x_high_x = np.random.normal(
    loc=0,
    scale=6,
    size=10
)

epsilon_high_x = np.random.normal(
    loc=mean,
    scale=sigma,
    size=10
)

y_high_x = a * x_high_x + b + epsilon_high_x

x_high_y = np.random.normal(
    loc=0,
    scale=1,
    size=10
)

epsilon_high_y = np.random.normal(
    loc=mean,
    scale=6,
    size=10
)

y_high_y = a * x_high_y + b + epsilon_high_y

x_high_xy = np.random.normal(
    loc=0,
    scale=6,
    size=10
)

epsilon_high_xy = np.random.normal(
    loc=mean,
    scale=6,
    size=10
)

y_high_xy =  a * x_high_xy + b+ epsilon_high_xy


x = np.concatenate([
    x_regular,
    x_high_x,
    x_high_y,
    x_high_xy
])

y = np.concatenate([
    y_regular,
    y_high_x,
    y_high_y,
    y_high_xy
])

x_design = np.column_stack([ np.ones(len(x)), x])

H = (x_design @np.linalg.inv(x_design.T @ x_design) @ x_design.T)
leverage_scores = np.diag(H)

top_indices = np.argsort( leverage_scores)[-5:]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9)
)

axes = axes.ravel()

for i, variance in enumerate(variances):
    sigma = np.sqrt(variance)


    epsilon_regular = np.random.normal(
        loc=mean,
        scale=sigma,
        size=100
    )

    y_regular = (
        a * x_regular
        + b
        + epsilon_regular
    )



    epsilon_high_x = np.random.normal(
        loc=mean,
        scale=sigma,
        size=10
    )

    y_high_x = (
        a * x_high_x
        + b
        + epsilon_high_x
    )


    
    epsilon_high_y = np.random.normal(
        loc=mean,
        scale=sigma * 6,
        size=10
    )

    y_high_y = (
        a * x_high_y
        + b
        + epsilon_high_y
    )


   
    epsilon_high_xy = np.random.normal(
        loc=mean,
        scale=sigma * 6,
        size=10
    )

    y_high_xy = (
        a * x_high_xy
        + b
        + epsilon_high_xy
    )


    y = np.concatenate([
        y_regular,
        y_high_x,
        y_high_y,
        y_high_xy
    ])


    axes[i].scatter(
        x,
        y,
        label="Data"
    )

    axes[i].scatter(
        x[top_indices],
        y[top_indices],
        marker="x",
        s=120,
        label="Highest leverage"
    )

    axes[i].set_title(
        f"Noise variance = {variance}"
    )

    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")

    axes[i].legend()



for i, mean in enumerate(mean_values):

    sigma = 1

    epsilon_regular = np.random.normal(
        loc=mean,
        scale=sigma,
        size=100
    )

    y_regular =  a * x_regular + b + epsilon_regular
    


    epsilon_high_x = np.random.normal(
        loc=mean,
        scale=sigma,
        size=10
    )

    y_high_x =   a * x_high_x + b + epsilon_high_x
    

    epsilon_high_y = np.random.normal(
        loc=mean,
        scale=sigma * 6,
        size=10
    )

    y_high_y = a * x_high_y + b + epsilon_high_y



    epsilon_high_xy = np.random.normal(
        loc=mean,
        scale=sigma * 6,
        size=10
    )

    y_high_xy = a * x_high_xy + b + epsilon_high_xy
    


    y = np.concatenate([
        y_regular,
        y_high_x,
        y_high_y,
        y_high_xy
    ])


    axes[i].scatter(
        x,
        y,
        label="Data"
    )

    axes[i].scatter(
        x[top_indices],
        y[top_indices],
        marker="x",
        s=120,
        label="Highest leverage"
    )

    axes[i].set_title(
        f"Noise mean = {mean}"
    )

    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")
    axes[i].legend()

plt.tight_layout()
plt.show()


plt.scatter(
    x,
    y,
    label="Data"
)

plt.scatter(
    x[top_indices],
    y[top_indices],
    marker="x",
    s=120,
    label="Highest leverage"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Leverage scores")
plt.legend()

plt.show()

'''
'''
a = 2
b = -1
c = 3

mean = 0
sigma = 1

x1_regular = np.random.normal(
    loc=0,
    scale=1,
    size=100
)

x2_regular = np.random.normal(
    loc=0,
    scale=1,
    size=100
)

epsilon_regular = np.random.normal(
    loc=mean,
    scale=sigma,
    size=100
)

y_regular =  a * x1_regular  + b * x2_regular + c + epsilon_regular

x1_high_x = np.random.normal(
    loc=0,
    scale=6,
    size=10
)

x2_high_x = np.random.normal(
    loc=0,
    scale=6,
    size=10
)

epsilon_high_x = np.random.normal(
    loc=mean,
    scale=sigma,
    size=10
)

y_high_x =  a * x1_high_x + b * x2_high_x  + c + epsilon_high_x




x1_high_y = np.random.normal(
    loc=0,
    scale=1,
    size=10
)

x2_high_y = np.random.normal(
    loc=0,
    scale=1,
    size=10
)

epsilon_high_y = np.random.normal(
    loc=mean,
    scale=6,
    size=10
)

y_high_y = a * x1_high_y + b * x2_high_y + c + epsilon_high_y



x1_high_xy = np.random.normal(
    loc=0,
    scale=6,
    size=10
)

x2_high_xy = np.random.normal(
    loc=0,
    scale=6,
    size=10
)

epsilon_high_xy = np.random.normal(
    loc=mean,
    scale=6,
    size=10
)

y_high_xy =  a * x1_high_xy+ b * x2_high_xy + c + epsilon_high_xy


x1 = np.concatenate([
    x1_regular,
    x1_high_x,
    x1_high_y,
    x1_high_xy
])

x2 = np.concatenate([
    x2_regular,
    x2_high_x,
    x2_high_y,
    x2_high_xy
])

y = np.concatenate([
    y_regular,
    y_high_x,
    y_high_y,
    y_high_xy
])

x_design = np.column_stack([
    np.ones(len(x1)),
    x1,
    x2
])

H =  x_design @ np.linalg.inv(x_design.T @ x_design) @ x_design.T

leverage_scores = np.diag(H)

top_indices = np.argsort(
    leverage_scores
)[-5:]

# print("Sum of leverage scores:",np.sum(leverage_scores))
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.scatter(
    x1,
    x2,
    y,
    label="Data"
)

ax.scatter(
    x1[top_indices],
    x2[top_indices],
    y[top_indices],
    marker="x",
    s=120,
    label="Highest leverage"
)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

ax.set_title(
    "Leverage scores - two predictors"
)

ax.legend()

plt.show()
'''

'''
#Ex 2

x_train, x_test, y_train, y_test = generate_data_clusters(
    n_train=400,
    n_test=200,
    n_clusters=2,
    n_features=2,
    contamination=0.1
)

neighbor_values = [2, 3, 5, 10, 20, 30]

for k in neighbor_values:
    model = KNN(
        n_neighbors=k,
        contamination=0.1
    )

    model.fit(x_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9)
    )

    axes = axes.ravel()

    axes[0].scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train
    )

    axes[0].set_title(
        "Training data - Ground truth"
    )

    axes[1].scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train_pred
    )

    axes[1].set_title(
        "Training data - KNN predictions"
    )

    axes[2].scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=y_test
    )

    axes[2].set_title(
        "Test data - Ground truth"
    )

    axes[3].scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=y_test_pred
    )

    axes[3].set_title(
        "Test data - KNN predictions"
    )

    plt.tight_layout()
    plt.show()
    
    ba_train = balanced_accuracy_score(
        y_train,
        y_train_pred
    )

    ba_test = balanced_accuracy_score(
        y_test,
        y_test_pred
    )

    print(
    f"k={k}: "
    f"Train BA={ba_train:.4f}, "
    f"Test BA={ba_test:.4f}"
    )


#Ex3

x, _ = make_blobs(
    n_samples=[200, 100],
    centers=[(-10, -10), (10, 10)],
    cluster_std=[2, 6],
    random_state=42
)

n_neighbors = [5, 10, 20, 30]

fig, axes = plt.subplots(
    4,
    2,
    figsize=(12, 5)
)

for i, k in enumerate(n_neighbors):

    knn = KNN(
        n_neighbors=k,
        contamination=0.07
    )

    knn.fit(x)
    y_knn = knn.predict(x)

    lof = LOF(
        n_neighbors=k,
        contamination=0.07
    )

    lof.fit(x)
    y_lof = lof.predict(x)



    axes[i,0].scatter(
        x[:, 0],
        x[:, 1],
        c=y_knn
    )

    axes[i,0].set_title(f"KNN for k={k}")

    axes[i,1].scatter(
        x[:, 0],
        x[:, 1],
        c=y_lof
    )

    axes[i,1].set_title(f"LOF for k ={k}")

plt.tight_layout()
plt.show()
'''  

#Ex4
data = loadmat("cardio.mat")
#print(data.keys())

x = data["X"]
y = data["y"].ravel()

#print("X shape:", x.shape)
#print("y shape:", y.shape)

#print("Number of normal samples:", np.sum(y == 0))
#print("Number of anomalies:", np.sum(y == 1))

contamination = np.mean(y)
#print("Contamination:", contamination)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

#print("Train shape:", x_train.shape)
#print("Test shape:", x_test.shape)

#print("Train contamination:", np.mean(y_train))
#print("Test contamination:", np.mean(y_test))

x_train, x_test = standardizer(
    x_train,
    x_test
)

#print("Train feature means:")
#print(np.mean(x_train, axis=0))

#print("\nTrain feature standard deviations:")
#print(np.std(x_train, axis=0))

neighbor_values = np.linspace(30, 120, 10, dtype=int)

train_scores = np.zeros((x_train.shape[0], len(neighbor_values)))
test_scores = np.zeros((x_test.shape[0], len(neighbor_values)))

for i, k in enumerate(neighbor_values):

    model = KNN(
        n_neighbors=k,
        contamination=contamination
    )

    model.fit(x_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_ba = balanced_accuracy_score(
        y_train,
        y_train_pred
    )

    test_ba = balanced_accuracy_score(
        y_test,
        y_test_pred
    )
    
    #print(
    #    f"For k={k}: "
    #    f"Train BA={train_ba:.4f}, "
    #    f"Test BA={test_ba:.4f}"
    #)
    
    train_scores[:, i] = model.decision_function(x_train)

    test_scores[:, i] = model.decision_function(x_test)


#print("Train score matrix shape:", train_scores.shape)
#print("Test score matrix shape:", test_scores.shape)

train_scores_std, test_scores_std = standardizer( train_scores, test_scores )

train_avg_scores = average(train_scores_std)
test_avg_scores = average(test_scores_std)

train_max_scores = maximization(train_scores_std)
test_max_scores = maximization(test_scores_std)

avg_threshold = np.quantile(
    train_avg_scores,
    1 - contamination
)

max_threshold = np.quantile(
    train_max_scores,
    1 - contamination
)

y_train_avg = (train_avg_scores > avg_threshold).astype(int)
y_test_avg = (test_avg_scores > avg_threshold).astype(int)
y_train_max = (train_max_scores > max_threshold).astype(int)
y_test_max = (test_max_scores > max_threshold).astype(int)

avg_train_ba = balanced_accuracy_score( y_train, y_train_avg)
avg_test_ba = balanced_accuracy_score( y_test, y_test_avg)
max_train_ba = balanced_accuracy_score( y_train, y_train_max)
max_test_ba = balanced_accuracy_score( y_test, y_test_max)

print("AVERAGE strategy")
print(f"Train BA = {avg_train_ba:.4f}")
print(f"Test BA = {avg_test_ba:.4f}")

print("\nMAXIMUM strategy")
print(f"Train BA = {max_train_ba:.4f}")
print(f"Test BA = {max_test_ba:.4f}")



