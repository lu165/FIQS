# Reproducibility Package

This directory contains metadata and records required to reproduce the reported experiments.

## Contents

### model_manifests/

Contains model identification information, including model revision and checksum records.

### hashes/

Contains SHA256 integrity verification files for released experiment resources.

### commands/

Contains representative commands used to run the experiments.

### environment/

Contains software environment information, including:

- Python package versions
- library versions
- model configuration
- model revision information

### run_manifests/

Contains experiment execution records, including seed information and runtime metadata.


## Data and Model Availability

The released repository provides:

- experiment scripts
- configuration files
- evaluation code
- reproducibility metadata

Due to data management and licensing considerations, some benchmark datasets and model weights are not redistributed.

Researchers can reconstruct the experimental pipeline using the released code and publicly available resources, or contact the corresponding author for restricted datasets.

## Experiment Correspondence

The files in this directory correspond to the reported experiments as follows:

- `model_manifests/`:
  Model identification and checksum information.

- `commands/`:
  Representative evaluation commands.

- `run_manifests/`:
  Training configuration and runtime records for the reported LoRA experiments.

Different records may correspond to different random seeds and experimental settings reported in the manuscript.

## Prompt Templates

Prompt templates are implemented directly in the released experiment scripts.

## Dataset Protocol

The experiments use task-specific datasets with predefined evaluation protocols rather than merging all datasets into a single corpus for random train/test splitting.

