# NanoScreen-AI Tumor Delivery Prioritization

## Project Title

Artificial intelligence-assisted prioritization of nanoparticle formulations for efficient tumor delivery


## Overview

This repository provides the public analysis workflow, supplementary materials, and supporting resources for a machine-learning framework designed to prioritize nanoparticle formulations with high predicted tumor delivery efficiency.

The workflow includes dataset curation, high-delivery label construction, model development and optimization, out-of-fold and independent test-set evaluation, SHAP-based model interpretation, virtual formulation screening, cancer-type-specific candidate prioritization, and local working-range estimation.

The model-prioritized candidates are computational predictions intended to support future experimental validation. They should not be interpreted as experimentally validated optimal nanoparticle formulations.

The virtual screening workflow generated 50,000 virtual nanoparticle candidates using chemistry-aware categorical feasibility filtering, real-database Type-MAT-Shape combination constraints, numeric out-of-distribution control, CatBoost-based prediction of high-delivery probability, and candidate ranking.

The finalized screening library contains 50,000 virtual candidates and exports the top 200 prioritized candidates for downstream analysis. The final virtual screening outputs provided in Supplementary Material D.xlsx correspond to the finalized screening run used for manuscript preparation.

The top 20 candidates presented in the main manuscript were selected from the prioritized candidate pool according to predicted high-delivery probability ranking.


## Repository Link

https://github.com/Men-wutao/NanoScreen-AI-Tumor-Delivery-Prioritization


## Repository Structure

```text
NanoScreen-AI-Tumor-Delivery-Prioritization/

├── README.md
├── requirements.txt
│
├── scripts/
│   └── NanoScreen_AI_analysis_workflow_clean.ipynb
│
├── Supplementary materials/
│   ├── Supplementary Material A.docx
│   ├── Supplementary Material B.xlsx
│   ├── Supplementary Material C.xlsx
│   └── Supplementary Material D.xlsx
│
├── dashboard_link.md
│
└── LICENSE
```

## Public Analysis Workflow

The `scripts/` folder contains the cleaned public notebook released for reproducible analysis.

`NanoScreen_AI_analysis_workflow_clean.ipynb` represents the finalized computational workflow used for the manuscript analysis.

The notebook preserves the original analysis logic, including:

- Data loading
- Data preprocessing
- Feature engineering
- Dataset splitting
- High-delivery label construction
- Machine-learning model training
- CatBoost, XGBoost, and LightGBM comparison
- Model evaluation
- Model selection
- SHAP-based interpretation
- Virtual formulation screening
- Candidate generation and ranking
- Local working-range estimation
- Output export


The notebook was cleaned for public release by removing temporary debugging code, redundant exploratory analyses, deprecated experiments, and local environment-dependent paths while preserving the reported computational workflow and results.

The notebook uses repository-relative paths and can be adapted to local execution environments by modifying input/output directories when necessary.

Running the notebook requires the datasets and supplementary files provided in this repository.


## Supplementary Materials

The `Supplementary materials/` folder contains the final supplementary files prepared for the manuscript.


### Supplementary Material A.docx

Word document containing Tables S1–S9 and supplementary information related to data and code availability.


### Supplementary Material B.xlsx

Excel file containing the analytical dataset, data partitioning, high-delivery label definition, endpoint description, threshold sensitivity analysis, variable dictionary, variable ranges, and categorical feature levels.


### Supplementary Material C.xlsx

Excel file containing model evaluation outputs, cross-validation results, out-of-fold predictions, independent test-set predictions, model reports, optimized hyperparameters, and ranking-related metrics.


### Supplementary Material D.xlsx

Excel file containing virtual formulation screening outputs, including:

- 50,000 virtual candidates with model-based screening scores
- Top-ranked candidate lists
- Cancer-type-specific candidate screening results
- Local working-range estimation outputs


The supplementary materials correspond to the finalized analysis workflow used for manuscript preparation.


## Data Source

The analytical dataset and data dictionary used in this study are provided in Supplementary Material B.xlsx.

The file contains:

- Cleaned analytical dataset
- Training and test split information
- High-delivery labels
- Endpoint definition
- Threshold sensitivity analysis
- Variable dictionary
- Variable ranges
- Categorical feature levels


## Dashboard Links

The deployed NanoScreen-AI dashboard is available at:

https://nanoscreen-ai-dashboard.streamlit.app/


The dashboard implementation repository is:

https://github.com/Men-wutao/NanoScreen-AI-Dashboard


The dashboard provides an interactive visualization interface for model-based prioritization results.

The dashboard presents computational predictions and does not represent experimentally validated nanoparticle formulations.


## Important Note

The virtual screening results and local working ranges are model-prioritized outputs designed to guide future experimental validation.

They should not be interpreted as experimentally validated optimal nanoparticle formulations.

The predicted high-delivery probabilities represent model-based prioritization scores generated by the trained machine-learning framework.
