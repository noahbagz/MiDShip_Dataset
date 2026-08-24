#!/usr/bin/env python3
"""Train the notebook surrogates and reproduce its SGLD design experiment.

This file is a direct script extraction of the active methods in
``Regression_Training_And_Optimization.ipynb``.  The numerical choices used by
SGLD are intentionally fixed below.  In particular, this script preserves the
notebook's seed selection, seed interpolation, objective/constraint gradient,
noise level, step size, parameter bounds, and final parameter cleaning rules.

The script trains the two neural networks once and then generates five
independent batches of 100 parametric designs.  Model checkpoints and every
candidate artifact are written below ``sgld_generation_pipeline`` so the
longer Rhino stage can be resumed without retraining the networks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


# =============================================================================
# PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "MiDShip_Dataset" / "Random_Structures"
RUN_DIR = SCRIPT_DIR / "full_parameter_ranges_retrained"
MODEL_DIR = SCRIPT_DIR / "models"
TRAINING_DIR = SCRIPT_DIR / "training"
FIGURE_DIR = SCRIPT_DIR / "figures"
BATCHES_DIR = RUN_DIR / "batches"

PARAMETER_CSV = DATASET_DIR / "random_test_design_Parameters_All.csv"
PROPERTY_CSV = DATASET_DIR / "random_test_design_Structural_Properties.csv"
THRESHOLD_CSV = DATASET_DIR / "random_test_design_Constraint_Thresholds.csv"
VALUE_CSV = DATASET_DIR / "random_test_design_Constraint_Values.csv"
ERROR_INDEX_CSV = DATASET_DIR / "random_test_design_error_idx_All.csv"
PARAMETER_RANGE_CSV = DATASET_DIR / "StructuralParameterList_V2_Updated_Ranges.csv"

MODEL_CHECKPOINT = MODEL_DIR / "notebook_surrogate_models.pt"
TRAINING_COMPLETE = TRAINING_DIR / "training_complete.json"


# =============================================================================
# NOTEBOOK EXPERIMENT INVARIANTS -- DO NOT TUNE IN THIS REPLICATION
# =============================================================================

NUM_BATCHES = 5
POPULATION_SIZE = 100
TOP_SEED_COUNT = 25

TRAIN_SPLIT = 0.8
OBJECTIVE_BATCH_SIZE = 64
CONSTRAINT_BATCH_SIZE = 128
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001

OBJECTIVE_HIDDEN_LAYERS = [128, 128]
CONSTRAINT_HIDDEN_LAYERS = [256, 256]
MODEL_DROPOUT = 0.25

SGLD_NUM_STEPS = 100
SGLD_SIGMA = 0.001
SGLD_ALPHA = 0.1
SGLD_CONSTRAINT_OFFSET = 0.7
SGLD_OBJECTIVE_WEIGHT = 0.1
SGLD_SEED_INTERPOLATION = True

PERFORMANCE_COLUMN_INDICES = [2, 6, 11]
PERFORMANCE_LABELS = [
    r"Structural Density [tons/$m^3$]",
    r"Cross-Sectional Area [$m^2$]",
    r"Max. Bending Moment [$Nm$]",
]

CONSTRAINT_PLOT_LABELS = [
    "Min. Double Bot. Height",
    "Min. Bot. Floor Thk.",
    "Min. Bot. Girder Thk.",
    "Max. Bot. Floor Spacing",
    "Min. Inner Bot. Thk.",
    "Min. Hopper Plate Thk.",
    "Min. Bot. Long. Stiff. SM",
    "Min. Bot. Trans. Stiff. SM",
    "Min. Hopper Stiff. SM",
    "Min. Frame SM",
    "Min. Webframe SM",
    "Min. Webframe Depth",
    "Min. Webframe Thk.",
    "Min. Side Stringer SM",
    "Min. Side Stringer Depth",
    "Min. Blkhd. Thk.",
    "Min. Blkhd. Vert. Stiff. SM",
    "Min. Blkhd. Horiz. Stiff. SM",
    "Min. Deck Trans. Stiff. SM",
    "Min. Deck Thk.",
    "Min. Deck Beam SM",
    "Min. Deck Beam Depth",
    "Min. Deck Beam Thk.",
    "Min. Side Shell Thk.",
    "Min. Bot. Shell Thk.",
]


# These parameter index groups are copied from the notebook's
# ``clean_Struct_Params`` cell.  They are part of the SGLD method because the
# continuous SGLD result is sanitized only after the final gradient step.
PARAMETER_INDEX_BIT = np.array(
    [27, 31, 35, 39, 43, 47, 51, 56, 74, 79, 86, 91, 96, 100, 104, 105, 111, 115, 116]
)
PARAMETER_INDEX_CATEGORY = np.array([57, 58, 59])
PARAMETER_INDEX_INTEGER = np.array(
    [15, 16, 17, 18, 19, 20, 21, 22, 70, 75, 82, 87, 92, 106]
)
PARAMETER_INDEX_PLATE_THICKNESS = np.array(
    [7, 8, 9, 10, 11, 12, 23, 29, 33, 37, 41, 45, 49, 52, 54, 62, 65, 68, 69, 72, 77, 80, 81, 84, 89, 94, 98, 102, 107, 109, 113]
)
PARAMETER_INDEX_BRACKET = np.array([60, 61, 63, 64])
PARAMETER_INDEX_CONTAINER = np.array(
    [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
)
PARAMETER_INDEX_BULKCARRIER = np.array(
    [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
)


class MLP(nn.Module):
    """Notebook multilayer perceptron, including its 25% dropout layers."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.fc = nn.ModuleList()
        self.fc.append(self.layer(input_size, hidden_size[0]))

        for layer_index in range(1, len(hidden_size)):
            self.fc.append(
                self.layer(hidden_size[layer_index - 1], hidden_size[layer_index])
            )

        self.fc.append(nn.Linear(hidden_size[-1], output_size))

    @staticmethod
    def layer(dim_in, dim_out, dropout=MODEL_DROPOUT):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, values):
        for layer in self.fc:
            values = layer(values)

        return values


def device_for_notebook():
    """Use the same device rule as the notebook: CUDA when available, else CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_error_indices():
    """Read headerless or ``error_idx``-headed Rhino-error indices."""

    error_values = pd.read_csv(ERROR_INDEX_CSV, header=None).iloc[:, 0]
    error_values = pd.to_numeric(error_values, errors="coerce").dropna()
    return error_values.to_numpy(dtype=int)


def logistic_scaling(values, minimum_values):
    """Apply the exact per-constraint logistic response used in the notebook."""

    scaled = np.zeros_like(values)

    for constraint_index in range(values.shape[1]):
        scaled[:, constraint_index] = (
            2.0
            / (
                1.0
                + np.exp(
                    5.0
                    * values[:, constraint_index]
                    / (minimum_values[constraint_index] + 1.0e-6)
                )
            )
            - 1.0
        )

    return scaled


def load_full_parameter_bounds(parameter_columns):
    """Load the complete 120-parameter range envelope from the range CSV.

    The CSV provides direct ``LL`` and ``UL`` values for 119 parameters. The
    bilge radius ``R_b`` is the sole exception: it is defined strategically as
    0.75--1.0 times ``Db/1000``. For SGLD's single rectangular clipping box,
    use the complete attainable envelope implied by the CSV's double-bottom
    range: 0.75(800)/1000 = 0.6 m through 1.0(3500)/1000 = 3.5 m.
    """

    ranges = pd.read_csv(PARAMETER_RANGE_CSV, encoding="utf-8-sig")
    ranges["name"] = ranges["name"].str.strip()
    normalized_parameter_columns = [name.strip() for name in parameter_columns]

    if ranges["idx"].to_list() != list(range(len(parameter_columns))):
        raise RuntimeError(
            "Parameter-range indices do not align with the 120-element design vector."
        )

    if ranges["name"].to_list() != normalized_parameter_columns:
        raise RuntimeError(
            "Parameter-range names do not align with the design-vector columns."
        )

    lower_bounds = ranges["LL"].to_numpy(dtype=float)
    upper_bounds = ranges["UL"].to_numpy(dtype=float)
    missing_bounds = np.flatnonzero(
        np.isnan(lower_bounds) | np.isnan(upper_bounds)
    )
    bilge_radius_index = normalized_parameter_columns.index("R_b")
    double_bottom_index = normalized_parameter_columns.index("Db")

    if missing_bounds.tolist() != [bilge_radius_index]:
        raise RuntimeError(
            "Only R_b may use a strategic range in the parameter-range CSV."
        )

    lower_bounds[bilge_radius_index] = (
        float(ranges.loc[bilge_radius_index, "Strategic_LL"])
        * lower_bounds[double_bottom_index]
        / 1000.0
    )
    upper_bounds[bilge_radius_index] = (
        float(ranges.loc[bilge_radius_index, "Strategic_UL"])
        * upper_bounds[double_bottom_index]
        / 1000.0
    )

    return lower_bounds, upper_bounds


def load_notebook_training_data():
    """Reproduce the notebook's cleaning and target construction steps."""

    error_indices = load_error_indices()

    # Load all 6,050 design rows before excluding the known Rhino failures.
    full_parameters = pd.read_csv(PARAMETER_CSV)
    properties = pd.read_csv(PROPERTY_CSV)
    thresholds = pd.read_csv(THRESHOLD_CSV)
    values = pd.read_csv(VALUE_CSV)

    valid_parameters = full_parameters.drop(index=error_indices)
    valid_properties = properties.drop(index=error_indices)

    # The objective network predicts the standardized logarithm of the three
    # structural properties chosen in the notebook.
    objective_unscaled = np.log(
        valid_properties.iloc[:, PERFORMANCE_COLUMN_INDICES].to_numpy(dtype=float)
    )
    objective_scaler = StandardScaler()
    objective_targets = objective_scaler.fit_transform(objective_unscaled)

    # A positive raw constraint difference means the threshold exceeds the
    # realized value, and therefore that the constraint is violated.
    constraint_difference_full = thresholds - values
    constraint_difference_valid = constraint_difference_full.drop(index=error_indices)
    constraint_minimum = constraint_difference_full.min(axis=0).to_numpy(dtype=float)
    constraint_targets = logistic_scaling(
        constraint_difference_valid.to_numpy(dtype=float),
        constraint_minimum,
    )

    # The notebook fits the same StandardScaler independently in the objective
    # and constraint sections.  Both see the same valid 6,020 parameter rows.
    parameter_scaler = StandardScaler()
    scaled_parameters = parameter_scaler.fit_transform(
        valid_parameters.to_numpy(dtype=float)
    )

    # Preserve the notebook's top-seed calculation on all 6,050 rows.  It asks
    # for 25 plus the number of error rows, then removes error indices.
    constraints_satisfied = np.sum(
        constraint_difference_full <= 0.0,
        axis=1,
    )
    top_seed_indices = np.argsort(constraints_satisfied)[
        -(TOP_SEED_COUNT + len(error_indices)):
    ]
    error_index_set = set(error_indices.tolist())
    top_seed_indices = np.array(
        [index for index in top_seed_indices if index not in error_index_set],
        dtype=int,
    )

    # This is the one requested departure from the notebook experiment. Use
    # the full parameter ranges maintained for the CAD generator instead of
    # deriving a narrow box from a subset of dataset rows.
    lower_bounds, upper_bounds = load_full_parameter_bounds(
        valid_parameters.columns.to_list()
    )

    return {
        "error_indices": error_indices,
        "full_parameters": full_parameters,
        "valid_parameters": valid_parameters,
        "parameter_scaler": parameter_scaler,
        "scaled_parameters": scaled_parameters,
        "objective_scaler": objective_scaler,
        "objective_targets": objective_targets,
        "constraint_targets": constraint_targets,
        "constraint_labels": constraint_difference_valid.columns.to_list(),
        "top_seed_indices": top_seed_indices,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "parameter_range_csv": PARAMETER_RANGE_CSV,
    }


def create_data_loader(parameters, labels, batch_size, device):
    """Create the shuffled TensorDataset/DataLoader used in the notebook."""

    dataset = TensorDataset(
        torch.tensor(parameters).float().to(device),
        torch.tensor(labels).float(),
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_one_model(
    model,
    train_parameters,
    train_targets,
    batch_size,
    device,
    description,
):
    """Run the notebook's 1,000-epoch Adam/MSE training loop."""

    data_loader = create_data_loader(
        train_parameters,
        train_targets,
        batch_size,
        device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    final_loss = np.nan

    model.train()

    for unused_epoch in tqdm(
        range(NUM_EPOCHS),
        desc=description,
        unit="epoch",
        dynamic_ncols=True,
    ):
        for parameter_batch, target_batch in data_loader:
            parameter_batch = parameter_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            predictions = model(parameter_batch)
            loss = criterion(predictions, target_batch)
            loss.backward()
            optimizer.step()

            final_loss = float(loss.item())

    return final_loss


def calculate_regression_metrics(truth, prediction, remove_constraint_outliers=False):
    """Calculate the notebook's R2, MSE, and normalized RMSE metrics."""

    metric_rows = []
    filtered_truth = []
    filtered_prediction = []

    for output_index in range(truth.shape[1]):
        output_truth = truth[:, output_index]
        output_prediction = prediction[:, output_index]
        num_removed = 0

        if remove_constraint_outliers:
            # Preserve the one-sided outlier test used in notebook cell 30.
            difference = output_truth - output_prediction
            outlier_indices = np.where(
                difference > 3.0 * np.std(difference)
            )[0]
            output_truth = np.delete(output_truth, outlier_indices)
            output_prediction = np.delete(output_prediction, outlier_indices)
            num_removed = len(outlier_indices)

        mse = np.mean((output_truth - output_prediction) ** 2)
        nrmse = np.sqrt(mse) / (output_truth.max() - output_truth.min())

        metric_rows.append(
            {
                "output_index": output_index,
                "r2": r2_score(output_truth, output_prediction),
                "mse": mse,
                "nrmse": nrmse,
                "num_outliers_removed": num_removed,
            }
        )
        filtered_truth.append(output_truth)
        filtered_prediction.append(output_prediction)

    return pd.DataFrame(metric_rows), filtered_truth, filtered_prediction


def save_figure(figure, name):
    """Save publication-resolution PNG and vector PDF versions of one graph."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{name}.png", dpi=450, bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_objective_regression(truth, prediction, metrics):
    """Copy the three-panel structural-property regression graph."""

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = "10"
    colors = [[1.0, 0.5, 0.5], [0.4, 0.8, 0.4], [0.5, 0.5, 1.0]]

    figure, axes = plt.subplots(1, 3, figsize=(10, 3))

    for output_index, (axis, color) in enumerate(zip(axes, colors)):
        minimum = truth[:, output_index].min()
        maximum = truth[:, output_index].max()
        axis.plot([minimum, maximum], [minimum, maximum], "k--", lw=1)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.scatter(
            truth[:, output_index],
            prediction[:, output_index],
            color=color,
            marker="o",
            alpha=0.85,
            s=5,
        )
        axis.set_xlabel("Ground Truth")
        axis.set_ylabel("Prediction")
        axis.set_title(
            f"{PERFORMANCE_LABELS[output_index]}\n"
            + rf"$R^2$: {metrics.loc[output_index, 'r2']:.3f}"
            + "\n"
            + f"NRMSE: {100.0 * metrics.loc[output_index, 'nrmse']:.1f}%",
            pad=25,
        )

    figure.tight_layout()
    save_figure(figure, "performance_metrics_regression")


def plot_constraint_regression(filtered_truth, filtered_prediction, metrics):
    """Copy the notebook's 5-by-5 constraint-surrogate regression graph."""

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = "12"

    figure, axes = plt.subplots(5, 5, figsize=(20, 15))
    axes = axes.flatten()

    for constraint_index in range(len(filtered_truth)):
        truth = filtered_truth[constraint_index]
        prediction = filtered_prediction[constraint_index]
        axis = axes[constraint_index]
        minimum = min(truth)
        maximum = max(truth)

        axis.plot([minimum, maximum], [minimum, maximum], "k--", lw=1)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.scatter(
            truth,
            prediction,
            color=[0.5, 0.5, 1.0],
            marker="o",
            alpha=0.85,
            s=5,
        )
        axis.axvline(x=0, color="r", linestyle="--")
        axis.axhline(y=0, color="r", linestyle="--")
        axis.set_xlabel("Ground Truth")
        axis.set_ylabel("Prediction")
        axis.set_title(
            f"{CONSTRAINT_PLOT_LABELS[constraint_index]}\n"
            + rf"$R^2$: {metrics.loc[constraint_index, 'r2']:.3f}"
            + "\n"
            + f"NRMSE = {100.0 * metrics.loc[constraint_index, 'nrmse']:.1f}%",
            pad=25,
        )

    figure.tight_layout()
    save_figure(figure, "abs_constraint_regression")


def train_models(force=False):
    """Train both notebook models, save graphs, and persist a resumable checkpoint."""

    if MODEL_CHECKPOINT.is_file() and TRAINING_COMPLETE.is_file() and not force:
        print(f"Training checkpoint already exists: {MODEL_CHECKPOINT}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    data = load_notebook_training_data()
    device = device_for_notebook()
    print(f"Using device: {device}")

    # The notebook creates a separate random permutation for each network.
    objective_permutation = np.random.permutation(len(data["objective_targets"]))
    objective_split = int(TRAIN_SPLIT * len(objective_permutation))
    objective_train_indices = objective_permutation[:objective_split]
    objective_test_indices = objective_permutation[objective_split:]

    objective_model = MLP(
        data["scaled_parameters"].shape[1],
        OBJECTIVE_HIDDEN_LAYERS,
        data["objective_targets"].shape[1],
    ).to(device)

    start_time = time.time()
    objective_loss = train_one_model(
        objective_model,
        data["scaled_parameters"][objective_train_indices],
        data["objective_targets"][objective_train_indices],
        OBJECTIVE_BATCH_SIZE,
        device,
        "Training objective surrogate",
    )
    objective_training_seconds = time.time() - start_time

    objective_model.eval()
    with torch.no_grad():
        objective_prediction_scaled = objective_model(
            torch.tensor(
                data["scaled_parameters"][objective_test_indices]
            ).float().to(device)
        ).cpu().numpy()

    objective_prediction = np.exp(
        data["objective_scaler"].inverse_transform(objective_prediction_scaled)
    )
    objective_truth = np.exp(
        data["objective_scaler"].inverse_transform(
            data["objective_targets"][objective_test_indices]
        )
    )
    objective_metrics, unused_truth, unused_prediction = (
        calculate_regression_metrics(objective_truth, objective_prediction)
    )
    objective_metrics["label"] = PERFORMANCE_LABELS
    objective_metrics.to_csv(
        TRAINING_DIR / "objective_surrogate_metrics.csv",
        index=False,
    )
    plot_objective_regression(
        objective_truth,
        objective_prediction,
        objective_metrics,
    )

    constraint_permutation = np.random.permutation(len(data["constraint_targets"]))
    constraint_split = int(TRAIN_SPLIT * len(constraint_permutation))
    constraint_train_indices = constraint_permutation[:constraint_split]
    constraint_test_indices = constraint_permutation[constraint_split:]

    constraint_model = MLP(
        data["scaled_parameters"].shape[1],
        CONSTRAINT_HIDDEN_LAYERS,
        data["constraint_targets"].shape[1],
    ).to(device)

    start_time = time.time()
    constraint_loss = train_one_model(
        constraint_model,
        data["scaled_parameters"][constraint_train_indices],
        data["constraint_targets"][constraint_train_indices],
        CONSTRAINT_BATCH_SIZE,
        device,
        "Training constraint surrogate",
    )
    constraint_training_seconds = time.time() - start_time

    constraint_model.eval()
    with torch.no_grad():
        constraint_prediction = constraint_model(
            torch.tensor(
                data["scaled_parameters"][constraint_test_indices]
            ).float().to(device)
        ).cpu().numpy()

    constraint_truth = data["constraint_targets"][constraint_test_indices]
    (
        constraint_metrics,
        filtered_constraint_truth,
        filtered_constraint_prediction,
    ) = calculate_regression_metrics(
        constraint_truth,
        constraint_prediction,
        remove_constraint_outliers=True,
    )
    constraint_metrics["label"] = data["constraint_labels"]
    constraint_metrics.to_csv(
        TRAINING_DIR / "constraint_surrogate_metrics.csv",
        index=False,
    )
    plot_constraint_regression(
        filtered_constraint_truth,
        filtered_constraint_prediction,
        constraint_metrics,
    )

    # Save exactly the learned weights and random splits.  The scalers are
    # refit from the immutable source CSVs when this checkpoint is loaded.
    torch.save(
        {
            "objective_model_state": objective_model.cpu().state_dict(),
            "constraint_model_state": constraint_model.cpu().state_dict(),
            "objective_permutation": objective_permutation,
            "constraint_permutation": constraint_permutation,
            "objective_final_loss": objective_loss,
            "constraint_final_loss": constraint_loss,
        },
        MODEL_CHECKPOINT,
    )

    training_summary = {
        "status": "complete",
        "device": str(device),
        "num_valid_designs": len(data["valid_parameters"]),
        "num_error_rows_removed": len(data["error_indices"]),
        "objective_training_seconds": objective_training_seconds,
        "constraint_training_seconds": constraint_training_seconds,
        "objective_final_batch_loss": objective_loss,
        "constraint_final_batch_loss": constraint_loss,
        "notebook_random_seed": None,
        "note": "No random seed is set because the notebook does not set one.",
    }
    TRAINING_COMPLETE.write_text(json.dumps(training_summary, indent=2))


def load_models_and_data():
    """Load the trained weights and reconstruct the notebook data transforms."""

    if not MODEL_CHECKPOINT.is_file():
        raise RuntimeError("Train the notebook surrogate models first.")

    data = load_notebook_training_data()
    device = device_for_notebook()
    checkpoint = torch.load(
        MODEL_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    objective_model = MLP(
        data["scaled_parameters"].shape[1],
        OBJECTIVE_HIDDEN_LAYERS,
        data["objective_targets"].shape[1],
    ).to(device)
    constraint_model = MLP(
        data["scaled_parameters"].shape[1],
        CONSTRAINT_HIDDEN_LAYERS,
        data["constraint_targets"].shape[1],
    ).to(device)

    objective_model.load_state_dict(checkpoint["objective_model_state"])
    constraint_model.load_state_dict(checkpoint["constraint_model_state"])
    objective_model.eval()
    constraint_model.eval()

    return data, objective_model, constraint_model, device


def clean_structural_parameters(parameter_set, lower_bounds, upper_bounds):
    """Apply the notebook's post-SGLD clipping and categorical cleanup."""

    parameters = parameter_set.copy().astype(float)

    for row_index in range(len(parameters)):
        parameters[row_index] = np.clip(
            parameters[row_index],
            lower_bounds,
            upper_bounds,
        )
        parameters[row_index, PARAMETER_INDEX_BIT] = (
            parameters[row_index, PARAMETER_INDEX_BIT] >= 0.5
        )
        parameters[row_index, PARAMETER_INDEX_INTEGER] = np.int32(
            parameters[row_index, PARAMETER_INDEX_INTEGER] + 0.5
        )
        parameters[row_index, PARAMETER_INDEX_PLATE_THICKNESS] = np.int32(
            parameters[row_index, PARAMETER_INDEX_PLATE_THICKNESS] + 0.5
        )

        if parameters[row_index, 6] < 1000.0 * parameters[row_index, 4]:
            parameters[row_index, 4] = parameters[row_index, 6] / 1000.0

        ship_class = np.argmax(
            parameters[row_index, PARAMETER_INDEX_CATEGORY]
        )
        parameters[row_index, 57:60] = 0.0
        parameters[row_index, 57 + ship_class] = 1.0

        if parameters[row_index, 57] == 1:
            parameters[row_index, PARAMETER_INDEX_BULKCARRIER] = 0.0
            parameters[row_index, PARAMETER_INDEX_CONTAINER] = 0.0
        elif parameters[row_index, 58] == 1:
            parameters[row_index, PARAMETER_INDEX_BULKCARRIER] = 0.0
            parameters[row_index, PARAMETER_INDEX_BRACKET] = 0.0
        elif parameters[row_index, 59] == 1:
            parameters[row_index, PARAMETER_INDEX_CONTAINER] = 0.0

        if parameters[row_index, 12] < 10.0:
            parameters[row_index, 12] = 0.0
        else:
            parameters[row_index, 50] = 0.0

        if parameters[row_index, 105] <= 0.5:
            parameters[row_index, 105] = 0.0
            parameters[row_index, 106:116] = 0.0
        elif parameters[row_index, 106] <= 0.5:
            parameters[row_index, 105:116] = 0.0
        else:
            parameters[row_index, 105] = 1.0

        if (
            parameters[row_index, 116] <= 0.5
            or parameters[row_index, 117] < 1.0e-2
            or parameters[row_index, 118] < 1.0e-2
        ):
            parameters[row_index, 116] = 0.0
            parameters[row_index, 117:120] = 0.0
        else:
            parameters[row_index, 116] = 1.0

    return parameters


def optimize_step(
    scaled_parameters,
    objective_model,
    constraint_model,
):
    """Perform one unchanged notebook SGLD update in standardized space."""

    with torch.enable_grad():
        scaled_parameters.requires_grad_(True)
        objective_prediction = objective_model(scaled_parameters)
        constraint_prediction = (
            constraint_model(scaled_parameters) + SGLD_CONSTRAINT_OFFSET
        )

        objective_step = (
            torch.sum(objective_prediction[:, 0])
            - sum(objective_prediction[:, 2])
        )
        constraint_step = torch.sum(nn.ReLU()(constraint_prediction))
        step = (
            -constraint_step
            - SGLD_OBJECTIVE_WEIGHT * objective_step
        )
        step.sum().backward()

        output = (
            scaled_parameters
            + SGLD_ALPHA * scaled_parameters.grad
            + SGLD_SIGMA * torch.randn_like(scaled_parameters)
        )

    return output.detach()


def stochastic_gradient_langevin_dynamics(
    initial_parameters,
    data,
    objective_model,
    constraint_model,
    device,
):
    """Run the notebook's interpolated 100-step SGLD design generation."""

    scaled = data["parameter_scaler"].transform(initial_parameters.copy())

    if SGLD_SEED_INTERPOLATION:
        interpolation_pairs = np.random.choice(
            len(scaled),
            size=(len(scaled), 2),
            replace=True,
        )
        interpolation_weights = np.random.rand(len(scaled), 1)
        scaled = (
            scaled[interpolation_pairs[:, 0]] * interpolation_weights
            + scaled[interpolation_pairs[:, 1]]
            * (1.0 - interpolation_weights)
        )

    scaled_tensor = torch.from_numpy(scaled).to(device).float()
    lower_tensor = torch.from_numpy(
        data["parameter_scaler"].transform(
            data["lower_bounds"].reshape(1, -1)
        )
    ).float().to(device)
    upper_tensor = torch.from_numpy(
        data["parameter_scaler"].transform(
            data["upper_bounds"].reshape(1, -1)
        )
    ).float().to(device)

    for unused_step in tqdm(
        range(SGLD_NUM_STEPS),
        desc="SGLD design generation",
        unit="step",
        leave=False,
        dynamic_ncols=True,
    ):
        scaled_tensor = optimize_step(
            scaled_tensor,
            objective_model,
            constraint_model,
        )
        scaled_tensor = torch.clip(
            scaled_tensor,
            lower_tensor,
            upper_tensor,
        )

    generated = data["parameter_scaler"].inverse_transform(
        scaled_tensor.detach().cpu().numpy()
    )

    return clean_structural_parameters(
        generated,
        data["lower_bounds"],
        data["upper_bounds"],
    )


def batch_id(batch_number):
    return f"sgld_batch_{batch_number:03d}"


def generate_one_batch(
    batch_number,
    data,
    objective_model,
    constraint_model,
    device,
):
    """Generate and save one notebook-sized population of 100 designs."""

    current_batch_id = batch_id(batch_number)
    output_dir = BATCHES_DIR / current_batch_id
    parameter_output = output_dir / f"{current_batch_id}_X_Results.csv"

    if parameter_output.is_file() and len(pd.read_csv(parameter_output)) == POPULATION_SIZE:
        print(f"Candidate batch already exists: {current_batch_id}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # The notebook samples with replacement from its 25 best available seed
    # indices and then performs a second random interpolation within the batch.
    selected_seed_indices = np.random.choice(
        data["top_seed_indices"],
        size=POPULATION_SIZE,
        replace=True,
    )
    initial_parameters = data["full_parameters"].values[
        selected_seed_indices,
        :,
    ].astype(float)

    generated = stochastic_gradient_langevin_dynamics(
        initial_parameters,
        data,
        objective_model,
        constraint_model,
        device,
    )

    generated_scaled = data["parameter_scaler"].transform(generated)
    generated_tensor = torch.from_numpy(generated_scaled).to(device).float()

    with torch.no_grad():
        objective_prediction = objective_model(generated_tensor).cpu().numpy()
        constraint_prediction = constraint_model(generated_tensor).cpu().numpy()

    # Preserve cell 39's output transform exactly.  It inverse-standardizes the
    # log-property predictions without exponentiating them before saving.
    objective_prediction = data["objective_scaler"].inverse_transform(
        objective_prediction
    )

    pd.DataFrame(
        generated,
        columns=data["full_parameters"].columns,
    ).to_csv(parameter_output, index=False)
    pd.DataFrame(
        objective_prediction[:, [0, 2]],
        columns=[
            "Structural Weight [kg]",
            "Max Bending Moment [Nm]",
        ],
    ).to_csv(
        output_dir / f"{current_batch_id}_F_Results.csv",
        index=False,
    )
    pd.DataFrame(
        constraint_prediction,
        columns=data["constraint_labels"],
    ).to_csv(
        output_dir / f"{current_batch_id}_G_Results.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "candidate_index": np.arange(POPULATION_SIZE),
            "source_seed_index": selected_seed_indices,
        }
    ).to_csv(output_dir / "seed_indices.csv", index=False)

    (output_dir / "candidate_generation_complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "batch_id": current_batch_id,
                "num_candidates": POPULATION_SIZE,
            },
            indent=2,
        )
    )


def generate_batches():
    """Generate the five requested independent SGLD batches."""

    data, objective_model, constraint_model, device = load_models_and_data()
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    for batch_number in tqdm(
        range(1, NUM_BATCHES + 1),
        desc="Generating SGLD batches",
        unit="batch",
        dynamic_ncols=True,
    ):
        generate_one_batch(
            batch_number,
            data,
            objective_model,
            constraint_model,
            device,
        )


def write_experiment_manifest():
    """Record the fixed notebook settings next to the generated results."""

    manifest = {
        "source_notebook": str(PROJECT_ROOT / "Regression_Training_And_Optimization.ipynb"),
        "num_batches": NUM_BATCHES,
        "population_size": POPULATION_SIZE,
        "top_seed_count": TOP_SEED_COUNT,
        "train_split": TRAIN_SPLIT,
        "objective_batch_size": OBJECTIVE_BATCH_SIZE,
        "constraint_batch_size": CONSTRAINT_BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "objective_hidden_layers": OBJECTIVE_HIDDEN_LAYERS,
        "constraint_hidden_layers": CONSTRAINT_HIDDEN_LAYERS,
        "dropout": MODEL_DROPOUT,
        "sgld_num_steps": SGLD_NUM_STEPS,
        "sgld_sigma": SGLD_SIGMA,
        "sgld_alpha": SGLD_ALPHA,
        "sgld_constraint_offset": SGLD_CONSTRAINT_OFFSET,
        "sgld_objective_weight": SGLD_OBJECTIVE_WEIGHT,
        "sgld_seed_interpolation": SGLD_SEED_INTERPOLATION,
        "random_seed": None,
        "parameter_bound_source": str(PARAMETER_RANGE_CSV),
        "parameter_bound_columns": "LL and UL",
        "bilge_radius_bound": (
            "Full static envelope implied by Strategic_LL/Strategic_UL "
            "and the Db LL/UL: 0.6 to 3.5 m"
        ),
        "departure_from_notebook": (
            "Only the SGLD clipping bounds were replaced at user request."
        ),
        "model_training_policy": (
            "Fresh training for a new full-parameter-range run; reuse only "
            "when resuming that same run."
        ),
        "candidate_sampling_policy": (
            "Five newly sampled independent batches of 100 designs."
        ),
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Reproduce notebook surrogate training and SGLD generation."
    )
    parser.add_argument(
        "--stage",
        choices=("train", "generate", "all"),
        default="all",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if a complete model checkpoint already exists.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    write_experiment_manifest()

    if args.stage in ("train", "all"):
        train_models(force=args.force_retrain)

    if args.stage in ("generate", "all"):
        generate_batches()


if __name__ == "__main__":
    main()
