from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np

#Ex1
'''
x_train, x_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1
)


#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

#print("Number of training anomalies:", sum(y_train))
#print("Number of test anomalies:", sum(y_test))

normal = y_train == 0
anomaly = y_train == 1

plt.scatter(
    x_train[normal, 0],
    x_train[normal, 1],
    label="Normal"
)

plt.scatter(
    x_train[anomaly, 0],
    x_train[anomaly, 1],
    label="Anomaly"
) 

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Training data")
plt.legend()
plt.show() 



#Ex2

#model = KNN(contamination=0.1)
model = KNN(contamination=0.2)
model.fit(x_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_test_pred)

#print(cm)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_test_pred
).ravel()

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
balanced_accuracy = (tpr + tnr) / 2

print("TPR:", tpr)
print("TNR:", tnr)
print("Balanced accuracy:", balanced_accuracy)

test_scores = model.decision_function(x_test)

fpr, tpr, thresholds = roc_curve(
    y_test,
    test_scores
)

plt.figure()

plt.plot(fpr, tpr)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Test Data")

#plt.show()



#Ex3

x_train, x_test, y_train, y_test = generate_data(
    n_train=1000,
    n_test=0,
    n_features=1,
    contamination=0.1
)

x = x_train.ravel()

mean = np.mean(x)
std = np.std(x)

z_scores = (x - mean) / std
abs_z_scores = np.abs(z_scores)
threshold = np.quantile(abs_z_scores, 0.9)

y_pred = (abs_z_scores > threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_train,
    y_pred
).ravel()


tpr = tp / (tp + fn)
tnr = tn / (tn + fp)

balanced_accuracy = (tpr + tnr) / 2

print("Mean:", mean)
print("Standard deviation:", std)
print("Z-score threshold:", threshold)

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("TPR:", tpr)
print("TNR:", tnr)
print("Balanced accuracy:", balanced_accuracy)

'''

#Ex 4

normal_mean = np.array([5, 10])
sigma = np.array([
    [4, 2],
    [2, 9]
])

L = np.linalg.cholesky(sigma)

x_normal = np.random.randn(900, 2)
normal_data = x_normal @ L.T + normal_mean

anomaly_mean = np.array([12, 20])
x_anomaly = np.random.randn(100, 2)
anomaly_data = x_anomaly @ L.T + anomaly_mean

x_train  = np.vstack([
    normal_data,
    anomaly_data
])

normal_labels = np.zeros(900)
anomaly_labels = np.ones(100)

y_train = np.concatenate([
    normal_labels,
    anomaly_labels
])

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

z_scores = (x_train - mean) / std

scores = np.linalg.norm(z_scores, axis=1)

threshold = np.quantile(scores, 0.9)

y_pred = (scores > threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_train,
    y_pred
).ravel()


tpr = tp / (tp + fn)
tnr = tn / (tn + fp)

balanced_accuracy = (tpr + tnr) / 2

print("Mean:", mean)
print("Standard deviation:", std)
print("Z-score threshold:", threshold)

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("TPR:", tpr)
print("TNR:", tnr)
print("Balanced accuracy:", balanced_accuracy)
