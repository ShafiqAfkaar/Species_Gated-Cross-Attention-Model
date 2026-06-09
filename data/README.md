# Data

Raw image data are not included in this GitHub-ready folder. Dataset manifests, checksums, per-image predictions, probability arrays, and figure source tables are also withheld at this stage.

Expected local layout for full reproduction after data access is granted:

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

The internal folder name `Cattles` is retained for compatibility with trained checkpoints; manuscript-facing text should use `Cattle`.
