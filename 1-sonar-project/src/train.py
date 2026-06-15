from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sonar.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH, header=None)

    x = df.drop(columns=60)
    y = df[60]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.1,
        stratify=y,
        random_state=42,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(x_train))
    test_accuracy = accuracy_score(y_test, model.predict(x_test))

    sample = np.asarray(
        (
            0.0352,
            0.0116,
            0.0191,
            0.0469,
            0.0737,
            0.1185,
            0.1683,
            0.1541,
            0.1466,
            0.2912,
            0.2328,
            0.2237,
            0.2470,
            0.1560,
            0.3491,
            0.3308,
            0.2299,
            0.2203,
            0.2493,
            0.4128,
            0.3158,
            0.6191,
            0.5854,
            0.3395,
            0.2561,
            0.5599,
            0.8145,
            0.6941,
            0.6985,
            0.8660,
            0.5930,
            0.3664,
            0.6750,
            0.8697,
            0.7837,
            0.7552,
            0.5789,
            0.4713,
            0.1252,
            0.6087,
            0.7322,
            0.5977,
            0.3431,
            0.1803,
            0.2378,
            0.3424,
            0.2303,
            0.0689,
            0.0216,
            0.0469,
            0.0426,
            0.0346,
            0.0158,
            0.0154,
            0.0109,
            0.0048,
            0.0095,
            0.0015,
            0.0073,
            0.0067,
        )
    ).reshape(1, -1)

    prediction = model.predict(sample)[0]
    label = "Rock" if prediction == "R" else "Mine"

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Sample prediction: {label}")


if __name__ == "__main__":
    main()
