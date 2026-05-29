import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


# -------------------------------------------------
# Question 1: Dataset generation and visualization
# -------------------------------------------------

def generate_clean_data(
    n_samples=500,
    noise=20,
    random_state=42
):
    """
    Generate a clean synthetic regression dataset.

    Return:
        X, y, true_coef
    """

    X, y, coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=random_state
    )

    return X, y, coef


def add_outliers(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Add artificial outliers to the first n_outliers observations.
    """

    rng = np.random.RandomState(random_state)

    # Make copies
    X_out = X.copy()
    y_out = y.copy()

    # Add outliers
    X_out[:n_outliers] = 10 + 0.75 * rng.normal(size=(n_outliers, 1))
    y_out[:n_outliers] = -15 + 20 * rng.normal(size=n_outliers)

    return X_out, y_out


def plot_dataset_with_outliers(
    X,
    y,
    n_outliers=25
):
    """
    Plot the dataset and highlight outliers.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # Normal data
    ax.scatter(
        X[n_outliers:],
        y[n_outliers:],
        label="Normal Data"
    )

    # Artificial outliers
    ax.scatter(
        X[:n_outliers],
        y[:n_outliers],
        marker='x',
        s=100,
        label="Artificial Outliers"
    )

    ax.set_title("Dataset with Artificial Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


# -------------------------------------------------
# Question 2: Fit regression models
# -------------------------------------------------

def fit_linear_regression(X, y):
    """
    Fit ordinary Linear Regression.
    """

    model = LinearRegression()
    model.fit(X, y)

    return float(model.coef_[0])


def fit_huber_regression(X, y):
    """
    Fit Huber Regression.
    """

    model = HuberRegressor()
    model.fit(X, y)

    return float(model.coef_[0])


def fit_ransac_regression(X, y, random_state=42):
    """
    Fit RANSAC Regression.
    """

    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)

    return float(model.estimator_.coef_[0])


def fit_theilsen_regression(X, y, random_state=42):
    """
    Fit Theil-Sen Regression.
    """

    model = TheilSenRegressor(random_state=random_state)
    model.fit(X, y)

    return float(model.coef_[0])


def coefficient_errors(coef_dict, true_coef):
    """
    Compute absolute coefficient errors.
    """

    errors = {}

    for model_name, coef in coef_dict.items():
        errors[model_name] = abs(coef - true_coef)

    return errors


def best_robust_model(errors):
    """
    Return robust model with smallest error.
    """

    robust_models = {
        "huber_regression": errors["huber_regression"],
        "ransac_regression": errors["ransac_regression"],
        "theilsen_regression": errors["theilsen_regression"]
    }

    best_model = min(robust_models, key=robust_models.get)

    return best_model


def ransac_outlier_summary(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Return:
        total_outliers_detected,
        added_outliers_detected
    """

    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    total_outliers_detected = np.sum(outlier_mask)

    # Count how many artificial outliers were detected
    added_outliers_detected = np.sum(outlier_mask[:n_outliers])

    return int(total_outliers_detected), int(added_outliers_detected)


# -------------------------------------------------
# Question 2: Visualization functions
# -------------------------------------------------

def plot_regression_fits(
    X,
    y,
    random_state=42
):
    """
    Plot fitted regression lines.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(X, y, alpha=0.6, label="Data")

    # Sort X for smooth lines
    sort_idx = np.argsort(X[:, 0])
    X_sorted = X[sort_idx]

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    ax.plot(
        X_sorted,
        lr.predict(X_sorted),
        label="Linear Regression"
    )

    # Huber Regression
    huber = HuberRegressor()
    huber.fit(X, y)
    ax.plot(
        X_sorted,
        huber.predict(X_sorted),
        label="Huber Regression"
    )

    # RANSAC Regression
    ransac = RANSACRegressor(random_state=random_state)
    ransac.fit(X, y)
    ax.plot(
        X_sorted,
        ransac.predict(X_sorted),
        label="RANSAC Regression"
    )

    # Theil-Sen Regression
    theilsen = TheilSenRegressor(random_state=random_state)
    theilsen.fit(X, y)
    ax.plot(
        X_sorted,
        theilsen.predict(X_sorted),
        label="Theil-Sen Regression"
    )

    ax.set_title("Regression Model Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


def plot_ransac_inliers_outliers(
    X,
    y,
    random_state=42
):
    """
    Plot RANSAC inliers and outliers.
    """

    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    fig, ax = plt.subplots(figsize=(8, 6))

    # Inliers
    ax.scatter(
        X[inlier_mask],
        y[inlier_mask],
        label="Inliers"
    )

    # Outliers
    ax.scatter(
        X[outlier_mask],
        y[outlier_mask],
        marker='x',
        s=100,
        label="Outliers"
    )

    ax.set_title("RANSAC Inliers vs Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig
