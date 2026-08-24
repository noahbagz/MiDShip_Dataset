#!/usr/bin/env python3
"""Create reproducible t-SNE plots for source and generated ship designs."""

import argparse
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PIPELINE_DIR.parent
SOURCE_DATA = (
    PROJECT_DIR
    / "MiDShip_Dataset"
    / "Random_Structures"
    / "Dataset_Design_Data.csv"
)

NUM_PARAMETERS = 120
NUM_PERFORMANCE_COLUMNS = 3
NUM_CONSTRAINTS = 25
RANDOM_SEED = 41

CLASS_COLUMNS = [
    "cat tanker",
    "cat container",
    "cat bulkcarrier",
]
CLASS_LABELS = [
    "Tanker",
    "Container ship",
    "Bulk carrier",
]

# Dataset observations use light versions of red, green, and blue. Generated
# observations use the corresponding saturated colors.
DATASET_COLORS = [
    "#efb3b2",
    "#b9dfbd",
    "#b8cae8",
]
GENERATED_COLORS = [
    "#d62728",
    "#2ca02c",
    "#1f77b4",
]

BULKHEAD_MARKERS = {
    0: "o",
    1: "^",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot source and generated designs in one shared t-SNE embedding."
    )
    parser.add_argument(
        "--batch-id",
        default="diverse_dataset_repair_100_002",
    )
    parser.add_argument("--perplexity", type=float, default=45.0)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--categorical-weight",
        type=float,
        default=2.5,
        help=(
            "Weight applied after standardization to ship-class and "
            "longitudinal-bulkhead indicators."
        ),
    )
    return parser.parse_args()


def load_plot_data(batch_id):
    experiment_dir = PIPELINE_DIR / "experiments" / batch_id
    generated_path = experiment_dir / "{}_X_Results.csv".format(batch_id)
    generated_results_path = experiment_dir / "evaluation" / "candidate_results.csv"

    source = pd.read_csv(SOURCE_DATA)
    source.columns = source.columns.str.strip()

    generated = pd.read_csv(generated_path)
    generated.columns = generated.columns.str.strip()

    generated_results = pd.read_csv(generated_results_path)
    generated_results = generated_results.set_index("candidate_index").reindex(
        generated.index
    )

    parameter_columns = list(source.columns[:NUM_PARAMETERS])
    source_constraint_columns = list(
        source.columns[
            NUM_PARAMETERS + NUM_PERFORMANCE_COLUMNS:
            NUM_PARAMETERS + NUM_PERFORMANCE_COLUMNS + NUM_CONSTRAINTS
        ]
    )
    generated_constraint_columns = [
        "pass_{}".format(name)
        for name in source_constraint_columns
    ]

    source_parameters = source[parameter_columns].copy()
    generated_parameters = generated[parameter_columns].copy()
    combined_parameters = pd.concat(
        [source_parameters, generated_parameters],
        ignore_index=True,
    )

    source_class = np.argmax(
        source[CLASS_COLUMNS].to_numpy(dtype=float),
        axis=1,
    )
    generated_class = np.argmax(
        generated[CLASS_COLUMNS].to_numpy(dtype=float),
        axis=1,
    )

    source_satisfied = source[source_constraint_columns].to_numpy(
        dtype=bool
    ).sum(axis=1)
    # Evaluation errors remain present through candidate metadata but have
    # missing pass flags.  Treat those unknown constraints as unsatisfied in
    # the plot instead of allowing NaN-to-bool conversion to count them as true.
    generated_satisfied = generated_results[
        generated_constraint_columns
    ].fillna(False).to_numpy(dtype=bool).sum(axis=1)

    plot_data = pd.DataFrame(
        {
            "data_source": (
                ["Original dataset"] * len(source)
                + ["Generated"] * len(generated)
            ),
            "sample_index": np.concatenate(
                [source.index.to_numpy(), generated.index.to_numpy()]
            ),
            "ship_class_index": np.concatenate(
                [source_class, generated_class]
            ),
            "ship_class": [
                CLASS_LABELS[index]
                for index in np.concatenate([source_class, generated_class])
            ],
            "longitudinal_bulkhead": np.concatenate(
                [
                    source["longitudinal bulkhead bit"].to_numpy(dtype=int),
                    generated["longitudinal bulkhead bit"].to_numpy(dtype=int),
                ]
            ),
            "constraints_satisfied": np.concatenate(
                [source_satisfied, generated_satisfied]
            ),
        }
    )

    return experiment_dir, combined_parameters, parameter_columns, plot_data


def calculate_embedding(
    parameters,
    parameter_columns,
    perplexity,
    random_seed,
    categorical_weight,
):
    """Standardize, compress, and embed every design on the same coordinates."""

    scaled = StandardScaler().fit_transform(parameters.to_numpy(dtype=float))

    # Give the requested class and longitudinal-bulkhead organization a visible
    # role in the neighborhood metric without removing the remaining 116 design
    # parameters from the embedding.
    emphasized_columns = CLASS_COLUMNS + ["longitudinal bulkhead bit"]

    for column in emphasized_columns:
        column_index = parameter_columns.index(column)
        scaled[:, column_index] *= categorical_weight

    pca = PCA(n_components=40, random_state=random_seed)
    compressed = pca.fit_transform(scaled)

    tsne_arguments = {
        "n_components": 2,
        "perplexity": perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "metric": "euclidean",
        "method": "barnes_hut",
        "angle": 0.5,
        "random_state": random_seed,
        "n_jobs": -1,
    }

    # Scikit-learn renamed ``n_iter`` to ``max_iter`` in version 1.5.  Select
    # the supported spelling so the same plotting stage works in both the
    # established ship-structure environment and newer environments.
    if "max_iter" in inspect.signature(TSNE).parameters:
        tsne_arguments["max_iter"] = 1500
    else:
        tsne_arguments["n_iter"] = 1500

    tsne = TSNE(
        **tsne_arguments
    )
    embedding = tsne.fit_transform(compressed)

    return embedding, float(pca.explained_variance_ratio_.sum())


def style_axis(axis):
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")


def save_figure(figure, output_stem):
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def plot_class_and_bulkhead(plot_data, output_dir, batch_id):
    figure, axis = plt.subplots(figsize=(11.0, 8.2), constrained_layout=True)

    # Plot the light source dataset first so saturated generated observations
    # remain visible when they occupy the same t-SNE neighborhoods.
    for class_index, unused_class_label in enumerate(CLASS_LABELS):
        for bulkhead_value, marker in BULKHEAD_MARKERS.items():
            mask = (
                (plot_data["data_source"] == "Original dataset")
                & (plot_data["ship_class_index"] == class_index)
                & (plot_data["longitudinal_bulkhead"] == bulkhead_value)
            )
            axis.scatter(
                plot_data.loc[mask, "tsne_1"],
                plot_data.loc[mask, "tsne_2"],
                s=13,
                marker=marker,
                c=DATASET_COLORS[class_index],
                alpha=0.42,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )

    for class_index, unused_class_label in enumerate(CLASS_LABELS):
        for bulkhead_value, marker in BULKHEAD_MARKERS.items():
            mask = (
                (plot_data["data_source"] == "Generated")
                & (plot_data["ship_class_index"] == class_index)
                & (plot_data["longitudinal_bulkhead"] == bulkhead_value)
            )
            axis.scatter(
                plot_data.loc[mask, "tsne_1"],
                plot_data.loc[mask, "tsne_2"],
                s=58,
                marker=marker,
                c=GENERATED_COLORS[class_index],
                alpha=0.98,
                edgecolors="#202020",
                linewidths=0.45,
                zorder=3,
            )

    class_source_handles = []

    for class_index, class_label in enumerate(CLASS_LABELS):
        class_source_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=DATASET_COLORS[class_index],
                    markeredgecolor="none",
                    markersize=7,
                    label="Dataset - {}".format(class_label),
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=GENERATED_COLORS[class_index],
                    markeredgecolor="#202020",
                    markeredgewidth=0.45,
                    markersize=8,
                    label="Generated - {}".format(class_label),
                ),
            ]
        )

    first_legend = axis.legend(
        handles=class_source_handles,
        title="Dataset and ship class",
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        ncol=2,
        fontsize=9,
        title_fontsize=10,
    )
    axis.add_artist(first_legend)

    bulkhead_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="none",
            markersize=7,
            label="No longitudinal bulkhead",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="none",
            markersize=8,
            label="Longitudinal bulkhead present",
        ),
    ]
    axis.legend(
        handles=bulkhead_handles,
        title="Configuration",
        loc="lower right",
        frameon=True,
        framealpha=0.94,
        fontsize=9,
        title_fontsize=10,
    )

    figure.suptitle(
        "Original and generated ship designs in a shared t-SNE embedding",
        fontsize=14,
        y=1.015,
    )
    num_dataset_designs = int(
        (plot_data["data_source"] == "Original dataset").sum()
    )
    num_generated_designs = int(
        (plot_data["data_source"] == "Generated").sum()
    )
    axis.set_title(
        "Light RGB: {:,} original designs | Saturated RGB: {:,} generated designs".format(
            num_dataset_designs,
            num_generated_designs,
        ),
        fontsize=10,
        color="#4d4d4d",
        pad=12,
    )
    style_axis(axis)

    save_figure(
        figure,
        output_dir / "tsne_class_bulkhead_comparison",
    )


def plot_constraint_feasibility(plot_data, output_dir):
    figure, axis = plt.subplots(figsize=(11.0, 8.2), constrained_layout=True)
    color_norm = Normalize(vmin=0, vmax=NUM_CONSTRAINTS)
    color_map = plt.get_cmap("Greys")

    for data_source, size, alpha, edge_color, line_width, zorder in (
        ("Original dataset", 14, 0.55, "none", 0.0, 1),
        ("Generated", 62, 1.0, "#151515", 0.55, 3),
    ):
        for bulkhead_value, marker in BULKHEAD_MARKERS.items():
            mask = (
                (plot_data["data_source"] == data_source)
                & (plot_data["longitudinal_bulkhead"] == bulkhead_value)
            )
            axis.scatter(
                plot_data.loc[mask, "tsne_1"],
                plot_data.loc[mask, "tsne_2"],
                s=size,
                marker=marker,
                c=plot_data.loc[mask, "constraints_satisfied"],
                cmap=color_map,
                norm=color_norm,
                alpha=alpha,
                edgecolors=edge_color,
                linewidths=line_width,
                rasterized=data_source == "Original dataset",
                zorder=zorder,
            )

    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=color_map),
        ax=axis,
        fraction=0.045,
        pad=0.025,
    )
    colorbar.set_label("Number of constraints satisfied (out of 25)")
    colorbar.set_ticks([0, 5, 10, 15, 20, 25])

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#b5b5b5",
            markeredgecolor="none",
            markersize=5,
            label="Original dataset",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="#151515",
            markeredgewidth=0.55,
            markersize=9,
            label="Generated",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#555555",
            markersize=7,
            label="No longitudinal bulkhead",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#555555",
            markersize=8,
            label="Longitudinal bulkhead present",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        fontsize=9,
    )

    figure.suptitle(
        "Constraint satisfaction across the shared t-SNE embedding",
        fontsize=14,
        y=1.015,
    )
    axis.set_title(
        "Lighter: fewer constraints satisfied | Darker: greater feasibility",
        fontsize=10,
        color="#4d4d4d",
        pad=12,
    )
    style_axis(axis)

    save_figure(
        figure,
        output_dir / "tsne_constraint_feasibility",
    )


def main():
    args = parse_arguments()
    (
        experiment_dir,
        parameters,
        parameter_columns,
        plot_data,
    ) = load_plot_data(args.batch_id)

    embedding, pca_variance = calculate_embedding(
        parameters,
        parameter_columns,
        args.perplexity,
        args.random_seed,
        args.categorical_weight,
    )
    plot_data["tsne_1"] = embedding[:, 0]
    plot_data["tsne_2"] = embedding[:, 1]

    figure_dir = experiment_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_data.to_csv(figure_dir / "tsne_embedding.csv", index=False)
    plot_class_and_bulkhead(plot_data, figure_dir, args.batch_id)
    plot_constraint_feasibility(plot_data, figure_dir)

    print("Embedded {:,} designs.".format(len(plot_data)))
    print("PCA variance retained before t-SNE: {:.3f}".format(pca_variance))
    print("Wrote figures to {}".format(figure_dir))


if __name__ == "__main__":
    main()
