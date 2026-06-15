from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    x = df.drop(columns="Outcome")
    y = df["Outcome"]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled,
        y,
        test_size=0.2,
        random_state=2,
    )

    classifier = svm.SVC(kernel="linear")
    classifier.fit(x_train, y_train)

    train_accuracy = accuracy_score(y_train, classifier.predict(x_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(x_test))

    sample = np.asarray((2, 197, 70, 45, 543, 30.5, 0.158, 53)).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    prediction = classifier.predict(sample_scaled)[0]
    label = "Diabetic" if prediction == 1 else "Non-diabetic"

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Sample prediction: {label}")


if __name__ == "__main__":
    main()
