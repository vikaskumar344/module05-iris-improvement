import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Load the Iris flower dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

print("MODULE 05: IRIS PROJECT IMPROVEMENT")

print("\nSample data:")
display(X.head())

print("\nMissing values:")
print(X.isna().sum())

# Separate training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train the model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.2%}")

# Display actual and predicted species
results = pd.DataFrame({
    "Actual species": [
        iris.target_names[int(value)] for value in y_test
    ],
    "Predicted species": [
        iris.target_names[int(value)] for value in predictions
    ]
})

print("\nExample predictions:")
display(results.head(10))

# Chart 1: Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=[0, 1, 2],
    display_labels=iris.target_names,
    cmap="Blues"
)
plt.title("Iris Classification: Test Results")
plt.tight_layout()
plt.show()

# Improvement: explain the model with feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values()

print("\nFeature importance:")
display(importance.sort_values(ascending=False))

# Chart 2: Feature importance
importance.plot(
    kind="barh",
    color="teal",
    figsize=(8, 4)
)
plt.title("Feature Importance in the Trained Tree")
plt.xlabel("Relative importance")
plt.ylabel("Flower measurement")
plt.tight_layout()
plt.show()

print("\nMost important feature:", importance.idxmax())
print("\nThe new chart helps explain this trained model.")
print("It does not automatically improve prediction accuracy.")
print("Feature importance does not prove cause and effect.")
