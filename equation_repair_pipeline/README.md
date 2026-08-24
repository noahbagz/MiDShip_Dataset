# Equation-informed constraint repair pipeline

This directory contains the maintained source files for generating the
repaired subset of MiDShip. The workflow reads the aligned random-design
parameters, constraint thresholds, and measured constraint values from
`MiDShip_Dataset/Random_Structures`.

For every valid random design with 13 or fewer initial violations, the method
creates two unique repaired variants. It modifies only parameters associated
with observed constraint failures while preserving principal hull dimensions,
ship class, and hatch configuration. Exact feasibility is determined only
after Rhino regenerates the structure and the rule-based evaluator recalculates
all 25 constraints.

## Maintained files

| File | Purpose |
|---|---|
| `generate_repaired_parameters.py` | Generate repaired parameter vectors without starting Rhino. |
| `run_repaired_dataset_pipeline.sh` | Launch the complete repaired-dataset workflow. |
| `batched_structure_generation.py` | Generate structures in restartable Rhino batches. This worker is also shared by the random and SGLD workflows. |
| `evaluate_structures.py` | Evaluate staged repaired structures before publication. |
| `plot_tsne_comparison.py` | Generate the repaired-design t-SNE comparison plots. |

The repair equations and main stage orchestration are maintained in
`tools/repair_parametric_designs.py` and
`tools/run_repaired_random_design_pipeline.py`.

## Generate parameters only

From the repository root, run:

```bash
python equation_repair_pipeline/generate_repaired_parameters.py
```

This creates the candidate parameter table and repair-provenance tables in a
local working directory under `equation_repair_pipeline/experiments`.
That directory is generated at runtime and is not required in a clean clone.

## Run the complete pipeline

```bash
bash equation_repair_pipeline/run_repaired_dataset_pipeline.sh
```

The complete workflow performs these stages:

1. generate two repaired variants for every eligible random-design source;
2. generate full and mesh-ready structures in Rhino;
3. checkpoint Rhino-modified parameters after each successful design;
4. restart Rhino after a process failure and resume incomplete work;
5. evaluate the generated structures using the exact constraint equations;
6. publish aligned files into `MiDShip_Dataset/Repaired_Structures`;
7. generate the repaired-design t-SNE plots.

The underlying Python entry point also accepts individual stages:

```bash
python tools/run_repaired_random_design_pipeline.py --stage candidates
python tools/run_repaired_random_design_pipeline.py --stage rhino
python tools/run_repaired_random_design_pipeline.py --stage evaluate
python tools/run_repaired_random_design_pipeline.py --stage publish
python tools/run_repaired_random_design_pipeline.py --stage tsne
```

## Resume behavior

The Rhino stage saves each successful updated parameter row and writes a
completion marker before advancing. Running the same command again skips
complete designs. If one design prevents progress during three consecutive
fresh Rhino processes, its index is recorded and later designs continue.

The generated working directories, logs, completion markers, plots, and
intermediate result tables are outputs of the pipeline rather than required
source files.
