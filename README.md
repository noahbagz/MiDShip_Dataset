# MiDShip

MiDShip is a multimodal dataset and reproducible codebase for generating,
modifying, and evaluating parametric ship cargo-hold structures. The dataset
connects 120-parameter design vectors to full and mesh-ready CAD geometry,
engineering drawings, structural-element tables, calculated structural
properties, and 25 local scantling constraints derived from the American
Bureau of Shipping (ABS) Marine Vessel Rules.

The current release contains 12,753 successfully generated designs spanning
tankers, containerships, and bulk carriers. The SGLD-generated and repaired
subsets are derived from selected random-design seeds and should therefore be
treated as related subsets rather than statistically independent samples.

<p align="center">
  <img src="github_figures/MidShip_Overview_1.png"
       alt="MiDShip workflow from parametric design vectors to CAD geometry, drawings, and structural evaluations"
       width="100%">
</p>

## Dataset access

The MiDShip dataset is hosted separately because the complete CAD, drawing,
and tabular collection is too large to distribute through this GitHub
repository.

> **Dataset download:** [MiDShip on Hugging Face](https://huggingface.co/datasets/DeCoDELab/MiDShip)

After downloading the dataset, extract it into the repository root using the
following directory name and subset structure:

```text
Ship_Structures/
└── MiDShip_Dataset/
    ├── Random_Structures/
    ├── SGLD_Gen_Structures/
    └── Repaired_Structures/
```

The generation, repair, drawing, and evaluation scripts use these relative
paths. No source-code changes are required when the external dataset is placed
in this location.

## Dataset at a glance

| Subset | Attempted candidates | Retained designs | Fully feasible | Mean constraint violations | Mean nearest-neighbor distance |
|---|---:|---:|---:|---:|---:|
| Random | 6,050 | 6,020 | 0.0% | 13.192 | 3.729 ± 0.052 |
| SGLD-generated | 500 | 496 | 64.9% | 0.409 | 1.144 ± 0.104 |
| Equation-informed repaired | 6,254 | 6,237 | 79.4% | 0.296 | 3.495 ± 0.133 |

These values are the results reported in the accompanying manuscript.
Feasibility requires satisfying all 25 implemented constraints. Mean
nearest-neighbor distance is calculated for 100-design sets after min-max
scaling all 120 parameters using the retained random subset; the reported
range is ± two sample standard deviations across sets.

<p align="center">
  <img src="github_figures/Dataset_Examples.png"
       alt="Examples of containership, tanker, and bulk-carrier structures in MiDShip"
       width="100%">
</p>

## What each design contains

Each retained design is identified by an integer index shared across its
files. Its representations include:

- a 120-parameter design vector describing principal characteristics, ship
  class, plates, stiffeners, girders, bulkheads, and optional hatch geometry;
- a Rhino `.3dm` model and a general `.igs` model;
- a mesh-ready `.igs` model in which smaller stiffeners are represented by
  curves for later line-element meshing;
- a `_Structural_Elements.csv` bill of materials containing element geometry
  and section properties;
- 12 calculated structural properties, including steel weight, centroid,
  longitudinal cross-sectional properties, and maximum bending moment;
- one calculated value and one governing threshold for each of 25 implemented
  structural constraints;
- three engineering-drawing PDFs, three corresponding annotated PDFs with
  component bounding boxes, and two CSV annotation tables.

The 120 parameters are organized into principal characteristics and ship
class, plate dimensions, and stiffener/girder spacing and sizing.

<p align="center">
  <img src="github_figures/Example_Params.png"
       alt="Organization of the 120 MiDShip parametric design variables"
       width="75%">
</p>

## Repository structure

```text
Ship_Structures/
├── Rhino_Macros/                       # Parameter, CAD, and drawing generation
├── equation_repair_pipeline/          # Repair workflow and shared Rhino worker
├── sgld_generation_pipeline/          # Surrogate training and SGLD experiment
├── tools/                              # Constraint evaluation and repair functions
├── Regression_Training_And_Optimization.ipynb
├── StructuralParameterList_V2_Updated_Ranges.csv
└── Autogluon_env.yml
```

The principal maintained components are:

- `MiDShip_Dataset`: the separately downloaded random, SGLD-generated, and
  equation-repaired design subsets used by the pipelines;
- [`Parametric_Structure_V2.py`](Rhino_Macros/Parametric_Structure_V2.py): the
  parametric cargo-hold geometry definition used inside Rhino;
- [`rhino_StructGen.py`](Rhino_Macros/rhino_StructGen.py): structure export and
  structural-element data generation;
- [`rhino_2D_Drawing.py`](Rhino_Macros/rhino_2D_Drawing.py): procedural
  engineering-drawing generation;
- [`batched_structure_generation.py`](equation_repair_pipeline/batched_structure_generation.py):
  the shared restartable Rhino structure-generation supervisor;
- [`Batched_Drawing_Generation.py`](Rhino_Macros/Batched_Drawing_Generation.py):
  the restartable drawing-generation supervisor;
- [`Parametric_Structure_Eval.py`](tools/Parametric_Structure_Eval.py):
  structural-property and constraint evaluation;
- [`repair_parametric_designs.py`](tools/repair_parametric_designs.py): the
  equation-informed parameter repair rules;
- [`sgld_experiment.py`](sgld_generation_pipeline/sgld_experiment.py): the
  notebook-faithful surrogate-training and SGLD candidate-generation method.

The entry points documented below are the maintained dataset-generation
workflows. The two pipeline directories contain only their maintained source
files; models, checkpoints, intermediate batches, figures, and statistics are
generated locally when the corresponding pipeline runs.

## Requirements

The numerical, machine-learning, plotting, and notebook dependencies are
defined in [`Autogluon_env.yml`](Autogluon_env.yml). Create and activate the
environment from the repository root:

```bash
conda env create --file Autogluon_env.yml
conda activate Autogluon_env
```

CAD and drawing generation additionally require Rhino 8 for macOS. The
supervisors currently expect Rhino at `/Applications/Rhino 8.app` and use its
`rhinocode` command. Rhino's script server must be running; if necessary, run
`StartScriptServer` in Rhino before launching a batch. Rhino and its embedded
Python modules are not installed by the Conda environment.

All commands below should be run from the repository root.

## Dataset-generation workflow

Every subset follows the same four-stage data flow:

1. generate or repair parameter vectors;
2. generate full and mesh-ready structures in Rhino;
3. generate engineering drawings in Rhino;
4. evaluate structural properties and the 25 constraints.

### 1. Random subset

Generate the paper-sized random parameter set:

```bash
python Rhino_Macros/Generate_Random_Parameters.py \
    --num-samples 6050 \
    --random-seed 41 \
    --overwrite
```

Generate the structures with automatic Rhino restarts and per-design resume:

```bash
python Rhino_Macros/run_random_structure_generation.py
```

Evaluate the successfully generated structures:

```bash
python tools/evaluate_midship_dataset.py --dataset random
```

### 2. Equation-informed repaired subset

The repair method selects every valid random design with 13 or fewer initial
constraint violations and creates two unique repaired variants. It changes
only parameters associated with observed violations, while preserving the
source hull dimensions, ship class, and hatch configuration. Target
exceedances are sampled independently by constraint family.

Generate repaired parameter vectors without starting Rhino:

```bash
python equation_repair_pipeline/generate_repaired_parameters.py
```

Run the complete resumable repair workflow—candidate generation, Rhino,
exact evaluation, publication into `MiDShip_Dataset`, and t-SNE plotting:

```bash
bash equation_repair_pipeline/run_repaired_dataset_pipeline.sh
```

See [`equation_repair_pipeline/README.md`](equation_repair_pipeline/README.md)
for the repair rules and fixed experiment settings.

### 3. SGLD-generated subset

The SGLD workflow reproduces the active method in
[`Regression_Training_And_Optimization.ipynb`](Regression_Training_And_Optimization.ipynb).
It retrains the structural-property and constraint neural networks, samples
five batches of 100 candidates using the fixed notebook hyperparameters,
generates and evaluates the structures, publishes the aligned outputs, and
creates the comparison plots and summary tables.

Run the complete resumable workflow with:

```bash
bash sgld_generation_pipeline/run_sgld_pipeline.sh
```

Individual stages can be run with `--stage candidates`, `rhino`, `evaluate`,
`publish`, or `analysis`. See
[`sgld_generation_pipeline/README.md`](sgld_generation_pipeline/README.md)
for the fixed model and SGLD settings.

### 4. Drawing generation

Open [`Batched_Drawing_Generation.py`](Rhino_Macros/Batched_Drawing_Generation.py)
and leave exactly one random, repaired, or SGLD configuration block
uncommented near the top of the file. Then run:

```bash
python Rhino_Macros/Batched_Drawing_Generation.py
```

Drawings and annotation tables are written to the selected subset's
`Dataset_Drawings` directory.

### 5. Re-evaluate released data

Recalculate the aligned structural-property, constraint-threshold,
constraint-value, and combined design-data tables for every subset:

```bash
python tools/evaluate_midship_dataset.py --dataset all
```

Use `--dataset random`, `repaired`, or `sgld` to process one subset.

## Long-running Rhino jobs

The structure and drawing supervisors are designed to be resumed instead of
babysat:

- existing complete outputs are discovered and skipped;
- successful Rhino-modified parameter rows are checkpointed after each
  structure;
- Rhino is relaunched after a process failure;
- progress resets the consecutive-failure counter;
- an index that prevents progress three consecutive times is written to a
  persistent failed/skip file so later designs can continue.

Interrupting a run does not discard completed designs. Launch the same command
again to continue from the first incomplete, non-skipped index.

## Constraint evaluation

For each design and constraint, the evaluator saves a calculated constraint
value and the corresponding governing threshold. A minimum-type constraint is
satisfied when

```text
constraint value >= constraint threshold
```

The sole maximum-spacing constraint is sign-transformed before storage so the
same comparison remains valid. The implemented constraints cover selected
plate thicknesses, stiffener and girder section moduli, structural depths, and
member spacing. They are not a complete ship-classification approval process
and do not replace full-vessel loading, finite-element analysis, or review by
a classification society.

## Design-space visualization

The repository includes shared t-SNE plots for the random, SGLD-generated,
and repaired parameter sets. These plots use one joint embedding with
perplexity 45 and a categorical weight of 2.5. They provide a qualitative view
of the parameter distributions; t-SNE is not used as a quantitative measure
of global design-space coverage.

<p align="center">
  <img src="github_figures/shared_tsne_four_panel.png"
       alt="Shared t-SNE comparison of random, SGLD-generated, and repaired MiDShip designs"
       width="100%">
</p>

## License

The MiDShip code repository is released under the GNU General Public License
version 3 (GNU GPL-3.0). Proprietary use requires a separate commercial
license.

For commercial licensing inquiries, contact
[noahbagz@mit.edu](mailto:noahbagz@mit.edu).

## Manuscript and citation

The dataset and generation methods are described in the accompanying
manuscript, **“MiDShip: Multimodal Dataset of Ship Cargo Hold Structures for
Engineering Design.”** A formal citation and archival publication link will
be added when they are available.

## Disclaimer

Disclaimer: This research was funded by the American Bureau of Shipping (ABS). The opinions, findings, conclusions, technical approach, analysis, calculations, and recommendations expressed herein, including any use, interpretation, or derivation of ABS Rule requirements or formulas, are solely those of the author(s) and have not been reviewed, validated, or endorsed by ABS. Nothing in this paper may be relied upon as a statement or interpretation of the ABS Rules or as a substitute for the ABS Rules as published by ABS, which govern in all cases. ABS makes no representation or warranty as to the accuracy or fitness for any purpose of the material herein and assumes no liability arising from its use.
