# SGLD design-generation pipeline

This directory contains the maintained source files for reproducing the
surrogate training and stochastic gradient Langevin dynamics (SGLD)-inspired
design experiment extracted from
`Regression_Training_And_Optimization.ipynb`.

The method retrains the structural-property and constraint neural networks,
generates five batches of 100 candidate parameter vectors, generates their
structures in Rhino, evaluates the exact constraints, publishes the completed
designs into `MiDShip_Dataset/SGLD_Gen_Structures`, and calculates the reported
visualizations and statistics.

## Maintained files

| File | Purpose |
|---|---|
| `run_sgld_pipeline.sh` | Launch the complete pipeline with the configured Conda environment. |
| `run_sgld_pipeline.py` | Coordinate stages, checkpoints, and Rhino restart behavior. |
| `sgld_experiment.py` | Train the two neural networks and sample SGLD candidates. |
| `evaluate_sgld_batches.py` | Evaluate generated batch structures exactly. |
| `merge_full_parameter_range_structures.py` | Publish and globally renumber structure files. |
| `merge_retrained_structure_csvs.py` | Merge parameter and error-index tables across batches. |
| `plot_shared_tsne.py` | Generate the shared random/SGLD/repaired t-SNE visualization. |
| `summarize_design_methods.py` | Calculate feasibility, violation, and diversity summaries. |

## Fixed experiment settings

The script preserves the notebook experiment settings:

- five batches of 100 candidates;
- 1,000 neural-network training epochs;
- 25 seed designs;
- 100 SGLD steps;
- `alpha = 0.1`;
- `sigma = 0.001`;
- full parameter ranges from
  `StructuralParameterList_V2_Updated_Ranges.csv`.

Candidate generation intentionally does not set a NumPy or PyTorch random seed
because the source notebook does not set one.

## Run the complete pipeline

From the repository root, run:

```bash
bash sgld_generation_pipeline/run_sgld_pipeline.sh
```

On a clean clone, the pipeline creates its local model, training, batch,
figure, and statistics directories automatically. These generated artifacts
are not required source inputs.

The complete workflow performs these stages:

1. train the structural-property and constraint surrogate networks;
2. save model metrics and regression plots;
3. sample five 100-candidate SGLD batches;
4. generate structures using resumable Rhino batches;
5. evaluate all available structures using the exact constraints;
6. globally renumber and publish the completed dataset files;
7. create the shared t-SNE plot and comparative summary tables.

Individual stages can also be run directly:

```bash
python sgld_generation_pipeline/run_sgld_pipeline.py --stage candidates
python sgld_generation_pipeline/run_sgld_pipeline.py --stage rhino
python sgld_generation_pipeline/run_sgld_pipeline.py --stage evaluate
python sgld_generation_pipeline/run_sgld_pipeline.py --stage publish
python sgld_generation_pipeline/run_sgld_pipeline.py --stage analysis
```

## Resume behavior

Candidate, model, and per-design Rhino checkpoints are preserved during an
interrupted local run. Relaunching the same pipeline skips completed work. If
one candidate remains the first incomplete design after three consecutive
fresh Rhino failures, its index is recorded and generation continues with the
remaining candidates.
