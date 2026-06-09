# Reproducibility Guide

## Environment

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
pip install -r vetderm_agent/requirements_agent.txt
```

CUDA is recommended for full training and MC-dropout evaluation. CPU execution is suitable for reading code, inspecting notebooks, and running small smoke tests.

## Data Status

Raw images, per-image prediction files, probability arrays, dataset manifests, checksums, and figure source tables are intentionally not included in this GitHub-ready folder yet. Full retraining or full metric regeneration requires the private data/source-data package.

Expected local data layout for full reproduction after data access is granted:

```text
data/training_data_deduped_splits/seed42/
  train/
  val/
  test/

data/development_external_2026_05_02/
  Cat/
  Cattles/
  Dog/
```

The internal code uses `Cattles` as the folder label for cattle; manuscript-facing text should use `Cattle`.

## Final Model Training Notebooks

The notebooks under `notebooks/` are publication-oriented records of the final training and comparison workflow:

```text
notebooks/train_unified_sgca.ipynb
notebooks/train_sgca_cross_attention.ipynb
notebooks/compare_final_models.ipynb
```

The final Unified SGCA checkpoint used in the manuscript is a dog-normal robustness fine-tuned checkpoint. The Cross-Attention checkpoint is the 384 px internet-robust run used as the second final model.

## Final Figures

Final manuscript figure files are included in:

```text
figures/model_comparison/
```

Per-image source data and probability arrays used to regenerate these figures are withheld for now.

## Agent Evaluation

Summary outputs for the MC30 decision-support agent and OOD stress test are included under:

```text
results/agent/triage_mc30/
results/agent/ood_domain_gate_mc30/
```

Per-image agent prediction files are withheld at this stage.
