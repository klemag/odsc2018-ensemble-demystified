from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import pandas as pd
import numpy as np


def get_tree_as_image(dt):
    dot_data = tree.export_graphviz(
        dt,
        out_file=None,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=5,
        impurity=False,
        proportion=True)

    graph = pydotplus.graph_from_dot_data(dot_data)
    return graph.create_png()


def plot_rf_mae(rf, X_test, y_test):
    mae_trees = [
        mean_absolute_error(tree.predict(X_test), y_test)
        for tree in rf.estimators_
    ]
    index_trees = np.arange(len(rf.estimators_))
    mae_ens = mean_absolute_error(rf.predict(X_test), y_test)

    # Plotting time
    plt.figure(figsize=(13, 7))
    plt.bar(x=index_trees, height=mae_trees, color="cornflowerblue")
    plt.ylim(0, 10)
    plt.axhline(
        mae_ens,
        color="tomato",
        linewidth=3,
        linestyle="dashed",
        label="Random Forest MAE")
    plt.xticks(index_trees)
    plt.xlabel("Single Decision Tree")
    plt.ylabel("MAE")
    plt.legend()


def plot_gb_mae(gb, X_test, y_test):
    mae_trees = [
        mean_absolute_error(y_pred, y_test)
        for y_pred in gb.staged_predict(X_test)
    ]
    index_trees = np.arange(len(gb.estimators_))
    mae_ens = mean_absolute_error(gb.predict(X_test), y_test)

    # Plotting time
    plt.figure(figsize=(13, 7))
    plt.bar(x=index_trees, height=mae_trees, color="cornflowerblue")
    plt.ylim(0, 12)
    plt.axhline(
        mae_ens,
        color="tomato",
        linewidth=3,
        linestyle="dashed",
        label="Gradient Boosting MAE")
    plt.xticks(index_trees)
    plt.xlabel("Boosting Stage")
    plt.ylabel("MAE")
    plt.legend()


def plot_n_predictions_rf(rf, X_test, N=10):
    sample = X_test.sample(N, random_state=42)
    predictions = pd.DataFrame(
        [tree.predict(sample).tolist() for tree in rf.estimators_],
        columns=["#{}".format(i) for i in sample.index])
    plt.figure(figsize=(13, 7))
    sns.boxplot(data=predictions)
    plt.xlabel("Index of sample")
    plt.ylabel("Prediction")


def plot_residuals_gb(gb, X_train, y_train, step_size):
    residuals = [y_train - y_pred for y_pred in gb.staged_predict(X_train)]
    res_df = pd.DataFrame(
        residuals[::step_size],
        index=[
            "Stage {}".format(i * step_size + 1)
            for i in range(len(residuals[::step_size]))
        ]).T

    plt.figure(figsize=(13, 7))
    sns.boxplot(data=res_df, showfliers=False)
    plt.xlabel("Stage")
    plt.ylabel("Residuals")
