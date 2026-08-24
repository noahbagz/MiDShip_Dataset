#!/usr/bin/env python3
"""Create the requested four-panel seaborn plot in one shared t-SNE space."""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MIDSHIP_DATASET_DIR = PROJECT_ROOT / "MiDShip_Dataset"
DATASET_DIR = MIDSHIP_DATASET_DIR / "Random_Structures"
RUN_DIR = SCRIPT_DIR / "full_parameter_ranges_retrained"
BATCHES_DIR = RUN_DIR / "batches"
REPAIRED_DIR = MIDSHIP_DATASET_DIR / "Repaired_Structures"
FIGURE_DIR = RUN_DIR / "figures"

DATASET_CSV = DATASET_DIR / "Dataset_Design_Data.csv"
REPAIRED_PARAMETER_CSV = (
    REPAIRED_DIR / "repaired_random_design_Parameters.csv"
)

NUM_PARAMETERS = 120
NUM_BATCHES = 5
TSNE_PERPLEXITY = 45.0
CATEGORICAL_WEIGHT = 2.5
RANDOM_SEED = 41
REPAIRED_SUBSET_SIZE = 100

CLASS_COLUMNS = ["cat tanker", "cat container", "cat bulkcarrier"]
CLASS_LABELS = ["Tanker", "Container ship", "Bulk carrier"]
LONGITUDINAL_BULKHEAD_COLUMN = "longitudinal bulkhead bit"

# The light/saturated RGB pairing follows the visual language already used by
# the notebook and the existing constraint-pipeline t-SNE figures.
DATASET_COLORS = ["#efb3b2", "#b9dfbd", "#b8cae8"]
GENERATED_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]
BULKHEAD_MARKERS = {0: "o", 1: "^"}


def sgld_batch_id(batch_number):
    return f"sgld_batch_{batch_number:03d}"


def load_sgld_parameters():
    """Load all 500 requested SGLD candidate vectors."""

    parameter_frames = []
    metadata_frames = []

    for batch_number in range(1, NUM_BATCHES + 1):
        current_batch_id = sgld_batch_id(batch_number)
        batch_dir = BATCHES_DIR / current_batch_id
        candidate_path = batch_dir / f"{current_batch_id}_X_Results.csv"
        candidates = pd.read_csv(candidate_path)
        candidates.columns = candidates.columns.str.strip()
        parameter_frames.append(candidates)
        metadata_frames.append(
            pd.DataFrame(
                {
                    "batch_id": current_batch_id,
                    "candidate_index": np.arange(len(candidates)),
                }
            )
        )

    return (
        pd.concat(parameter_frames, ignore_index=True),
        pd.concat(metadata_frames, ignore_index=True),
    )


def load_repaired_parameters():
    """Load the complete 6,254-vector repaired candidate dataset."""

    parameters = pd.read_csv(REPAIRED_PARAMETER_CSV)
    parameters.columns = parameters.columns.str.strip()

    return (
        parameters.copy().reset_index(drop=True),
        pd.DataFrame({"candidate_index": np.arange(len(parameters))}),
    )


def add_design_metadata(frame, data_source, sample_indices):
    """Attach source, ship class, and bulkhead fields used by the plot."""

    class_indices = np.argmax(
        frame[CLASS_COLUMNS].to_numpy(dtype=float),
        axis=1,
    )

    return pd.DataFrame(
        {
            "data_source": data_source,
            "source_sample_index": sample_indices,
            "ship_class_index": class_indices,
            "ship_class": [CLASS_LABELS[index] for index in class_indices],
            "longitudinal_bulkhead": frame[
                LONGITUDINAL_BULKHEAD_COLUMN
            ].to_numpy(dtype=int),
        }
    )


def load_embedding_input():
    """Combine each population once before fitting the shared embedding."""

    dataset = pd.read_csv(DATASET_CSV)
    dataset.columns = dataset.columns.str.strip()
    parameter_columns = dataset.columns[:NUM_PARAMETERS].to_list()
    dataset_parameters = dataset[parameter_columns].copy()

    sgld_parameters, sgld_indices = load_sgld_parameters()
    repaired_parameters, repaired_indices = load_repaired_parameters()
    sgld_parameters = sgld_parameters[parameter_columns]
    repaired_parameters = repaired_parameters[parameter_columns]

    parameters = pd.concat(
        [dataset_parameters, sgld_parameters, repaired_parameters],
        ignore_index=True,
    )
    metadata = pd.concat(
        [
            add_design_metadata(
                dataset_parameters,
                "Dataset",
                dataset.index.to_numpy(dtype=int),
            ),
            add_design_metadata(
                sgld_parameters,
                "SGLD",
                np.arange(len(sgld_parameters)),
            ),
            add_design_metadata(
                repaired_parameters,
                "Repaired",
                repaired_indices["candidate_index"].to_numpy(dtype=int),
            ),
        ],
        ignore_index=True,
    )

    return parameters, parameter_columns, metadata


def calculate_embedding(parameters, parameter_columns):
    """Standardize, emphasize categories, compress, and fit one t-SNE model."""

    scaled = StandardScaler().fit_transform(
        parameters.to_numpy(dtype=float)
    )

    # Apply the requested 2.5 categorical weight only after standardization so
    # the multiplier has a consistent meaning across binary indicators.
    emphasized_columns = CLASS_COLUMNS + [LONGITUDINAL_BULKHEAD_COLUMN]

    for column in emphasized_columns:
        scaled[:, parameter_columns.index(column)] *= CATEGORICAL_WEIGHT

    pca = PCA(n_components=40, random_state=RANDOM_SEED)
    compressed = pca.fit_transform(scaled)
    tsne_arguments = {
        "n_components": 2,
        "perplexity": TSNE_PERPLEXITY,
        "learning_rate": "auto",
        "init": "pca",
        "metric": "euclidean",
        "method": "barnes_hut",
        "angle": 0.5,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    # Support both the established environment and newer scikit-learn names.
    if "max_iter" in inspect.signature(TSNE).parameters:
        tsne_arguments["max_iter"] = 1500
    else:
        tsne_arguments["n_iter"] = 1500

    embedding = TSNE(**tsne_arguments).fit_transform(compressed)

    # As in the notebook, scale the plotted coordinates to [0, 1].
    embedding = (
        (embedding - embedding.min(axis=0))
        / (embedding.max(axis=0) - embedding.min(axis=0))
    )

    return embedding, float(pca.explained_variance_ratio_.sum())


def plot_population(axis, plot_data, source_name, generated=False):
    """Plot one population by ship class and longitudinal-bulkhead marker."""

    palette = GENERATED_COLORS if generated else DATASET_COLORS
    point_size = 22 if generated else 7
    alpha = 0.92 if generated else 0.34
    edge_color = "#202020" if generated else None
    line_width = 0.25 if generated else 0.0

    for class_index in range(len(CLASS_LABELS)):
        for bulkhead_value, marker in BULKHEAD_MARKERS.items():
            mask = (
                (plot_data["data_source"] == source_name)
                & (plot_data["ship_class_index"] == class_index)
                & (plot_data["longitudinal_bulkhead"] == bulkhead_value)
            )

            sns.scatterplot(
                data=plot_data.loc[mask],
                x="tsne_1",
                y="tsne_2",
                color=palette[class_index],
                marker=marker,
                s=point_size,
                alpha=alpha,
                edgecolor=edge_color,
                linewidth=line_width,
                legend=False,
                ax=axis,
                rasterized=True,
                zorder=3 if generated else 1,
            )


def style_axis(axis, title):
    axis.set_title(title, fontsize=11, pad=10)
    axis.set_xlabel("Dim. 1")
    axis.set_ylabel("Dim. 2")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")


def create_four_panel_plot(plot_data):
    """Render the four requested views without retraining or transforming t-SNE."""

    rng = np.random.default_rng(RANDOM_SEED)
    repaired_rows = plot_data.index[
        plot_data["data_source"] == "Repaired"
    ].to_numpy()
    repaired_subset_rows = rng.choice(
        repaired_rows,
        size=REPAIRED_SUBSET_SIZE,
        replace=False,
    )

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = "10"
    sns.set_theme(style="white", context="paper", font="serif")
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(22, 5.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for axis in axes:
        plot_population(axis, plot_data, "Dataset", generated=False)

    plot_population(axes[1], plot_data, "SGLD", generated=True)
    plot_population(
        axes[2],
        plot_data.loc[
            (plot_data["data_source"] != "Repaired")
            | plot_data.index.isin(repaired_subset_rows)
        ],
        "Repaired",
        generated=True,
    )
    plot_population(axes[3], plot_data, "Repaired", generated=True)

    titles = [
        "a) Rnd. Dataset",
        "b) Rnd. Dataset + 500 Attempted SGLD Candidates",
        "c) Rnd. Dataset + 100 Attempted Repair Candidates",
        "d) Rnd. Dataset + 6,254 Attempted Repair Candidates",
    ]

    for axis, title in zip(axes, titles):
        style_axis(axis, title)

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GENERATED_COLORS[class_index],
            markeredgecolor="none",
            markersize=6,
            label=class_label,
        )
        for class_index, class_label in enumerate(CLASS_LABELS)
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#c7c7c7",
            markeredgecolor="none",
            markersize=5,
            label="Rnd. Dataset (Light RGB)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#4d4d4d",
            markeredgecolor="#202020",
            markersize=7,
            label="Generated (Saturated RGB)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#555555",
            markersize=6,
            label="No Longitudinal Bulkhead",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#555555",
            markersize=7,
            label="Longitudinal Bulkhead",
        ),
    ]
    figure.legend(
        handles=class_handles + source_handles,
        loc="outside lower center",
        ncol=7,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "t-SNE Embedding of Ship Structure Designs",
        fontsize=14,
    )

    return figure, repaired_subset_rows


def main():
    parameters, parameter_columns, plot_data = load_embedding_input()
    embedding, pca_variance = calculate_embedding(
        parameters,
        parameter_columns,
    )
    plot_data["tsne_1"] = embedding[:, 0]
    plot_data["tsne_2"] = embedding[:, 1]

    figure, repaired_subset_rows = create_four_panel_plot(plot_data)
    plot_data["shown_in_repaired_100_panel"] = plot_data.index.isin(
        repaired_subset_rows
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_data.to_csv(FIGURE_DIR / "shared_tsne_embedding.csv", index=False)
    figure.savefig(
        FIGURE_DIR / "shared_tsne_four_panel.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        FIGURE_DIR / "shared_tsne_four_panel.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    print(f"Embedded {len(plot_data):,} designs in one t-SNE model.")
    print(f"PCA variance retained before t-SNE: {pca_variance:.3f}")
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
