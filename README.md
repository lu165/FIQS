# FIQS: Forest Intelligence Question-answering System

This repository provides the revision-stage source code of FIQS,
a unified framework for forestry-oriented large language models.

## Framework

FIQS integrates:

- Task-aware routing
- Allometric Equation Retrieval-Augmented Generation (AllomEq-RAG)
- Specialized biomass estimation
- Forestry knowledge question answering

## Repository Structure

- code/       Core implementation and experiment scripts
- config/     Configuration files
- data/       Data availability information
- requirements.txt   Python dependencies

## Installation

pip install -r requirements.txt

## Configuration

API keys are provided through environment variables. Example:

export DEEPSEEK_API_KEY="your_api_key"

No API keys are included in this repository.

## Running Experiments

Main FIQS experiment:
python code/experiments/run_fiqs.py

Baseline:
python code/experiments/run_baseline.py

Pure retrieval baseline:
python code/experiments/run_pure_rag.py

Ablation experiments:
python code/experiments/run_without_router.py
python code/experiments/run_without_allomeq_rag.py
python code/experiments/run_without_specialized.py

## Data Availability

This repository releases the implementation code, configuration files,
and experiment scripts. Due to licensing restrictions and third-party
data policies, some benchmark datasets and third-party resources are
not redistributed in this repository. The dataset organization and
evaluation protocols are documented in `data/DATASET_PROTOCOL.md`.
Publicly available benchmark resources can be reconstructed following
their original distribution protocols. Experiments requiring restricted
datasets should contact the corresponding author for lawful access.

## License

MIT License

## Reproducibility

To facilitate reproduction and transparent evaluation, this repository provides additional reproducibility materials.

The `reproducibility/` directory contains:

- model identification and checksum information;
- software environment records;
- experiment commands;
- run manifests.

The `data/DATASET_PROTOCOL.md` file documents dataset organization, evaluation protocols, and restricted dataset handling.

Prompt templates are implemented directly in the released experiment scripts.

