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
data policies, the complete datasets used in this study are not
redistributed during the revision stage. Additional materials may be
released after acceptance and publication, subject to applicable
licensing conditions.

## License

MIT License
