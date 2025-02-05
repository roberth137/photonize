import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

c3_histograms = pd.read_hdf('random_forest/CY3_histogram_all.hdf5', key='hist')
a5_histograms = pd.read_hdf('random_forest/A56_histogram_all.hdf5', key='hist')

histograms = pd.concat([c3_histograms, a5_histograms], axis=0, ignore_index=True)

X = histograms.drop(columns=["777"])  # Features (all histogram bins)
Y = histograms["777"]  # Labels (target class)

le = LabelEncoder()
y_encoded = le.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
model.fit(X_train, y_train)


y_pred = model.predict(X_test)




# Classification report
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
