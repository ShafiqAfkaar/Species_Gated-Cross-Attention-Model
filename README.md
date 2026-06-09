# SGCA: Species-Gated Cross-Attention for Veterinary Dermatology

This repository contains the GitHub-ready code and release artifacts for the Species-Gated Cross-Attention (SGCA) veterinary dermatology system described in the accompanying manuscript. The system classifies 21 dermatological categories across cats, cattle, and dogs from RGB images and includes a deployment-oriented decision-support agent.

The prototype agent is intended for image-based screening, ranked educational support, uncertainty-aware triage, and unsupported-image rejection. It is not a veterinary diagnosis or treatment-prescription system.

## Model Architecture

### SGCA Architecture

![SGCA architecture](assets/architecture/sgca_architecture_part_a.png)

### Disease Heads and Inference Outputs

![SGCA disease heads and inference](assets/architecture/sgca_disease_heads_and_inference_part_b.png)

### Decision-Support Agent Workflow

![SGCA decision-support agent workflow](assets/architecture/sgca_agent_workflow_part_c.png)

## Repository Layout

```text
sgca/                         Core model definitions and utilities
vetderm_agent/                Prototype decision-support agent
notebooks/                    Publication training and comparison notebooks
scripts/                      Evaluation and reproducibility scripts
results/models/               Final checkpoints and training histories
results/agent/                Summary agent and OOD evaluation outputs
assets/architecture/          Architecture and workflow panels shown above
data/                         Data access notes; raw images are not included
MODEL_CARD.md                 Model card and safety boundary
REPRODUCE.md                  Reproducibility notes
```

## Final Manuscript Metrics

Under species-conditioned decoding, the final Unified SGCA checkpoint achieved 96.5% validation accuracy, 97.5% internal-test accuracy, and 79.5% development-external accuracy. Macro-F1 was 96.6%, 98.0%, and 81.7% on the same splits, with development-external top-3 accuracy of 93.0%. SGCA Cross-Attention achieved 72.8% development-external accuracy, 72.8% macro-F1, and 88.8% top-3 accuracy.

The final Unified SGCA checkpoint is the dog-normal robustness fine-tuned checkpoint selected before final reporting:

```text
results/models/unified_sgca/unified_sgca_best.pt
```

The Cross-Attention checkpoint is:

```text
results/models/sgca_cross_attention/sgca_cross_attention_best.pt
```

## Data and Source-Data Availability

Raw images, per-image prediction files, probability arrays, dataset manifests, checksums, full figure source tables, and manuscript result figures are **not included in this GitHub-ready folder at this stage**. They are archived separately and can be released later through an approved data package or repository release.

## Checkpoints and Large Files

Model checkpoints are approximately 80 MB each. They should be tracked with Git LFS or attached to a GitHub release rather than committed as ordinary Git blobs.

## Scientific Boundary

The development-external cohort is a heterogeneous internet-derived robustness cohort, not a locked prospective clinical validation set. The unsupported-image/domain gate is part of the deployment agent and OOD stress test; it is not used to filter the base model benchmark.
