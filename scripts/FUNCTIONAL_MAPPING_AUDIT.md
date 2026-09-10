# Functional Mapping Audit

Audit date: 2026-09-10  
Audit mode: static code review only. No notebooks or scripts were executed. No model training, result recomputation, or figure generation was run.

## Scope

Reviewed public upload scripts:

- `scripts/main_analysis_final.py`
- `scripts/figure_generation_cleaned.py`

`main_analysis_final.py` is the manually organized final public analysis code exported from the finalized Jupyter workflow.

`figure_generation_cleaned.py` is the cleaned public figure-generation code record exported from the figure-generation workflow.

Original notebooks are not included in the upload package.

The cleaned scripts should be interpreted as public code records exported from the finalized analysis workflow. They preserve the major analysis logic but may require manual review of input files, dependencies, runtime requirements, and environment configuration before full re-execution.

---

# Main Analysis Functional Mapping

| Function / workflow module | Original workflow evidence | Cleaned script evidence | Status | Notes |
|---|---|---|---|---|
| Data loading and preprocessing | Main analysis workflow includes dataset loading, cleaning, and preprocessing procedures. | `main_analysis_final.py` contains repository-relative paths, dataset loading, categorical normalization, and preprocessing functions. | Matched | Personal absolute paths were replaced with repository-relative configuration variables. |
| Tumor delivery endpoint extraction | Workflow defines `DE_tumor` as the primary prediction endpoint. | `main_analysis_final.py` retains `TARGET_DE = "DE_tumor"` and endpoint handling. | Matched | Non-target delivery endpoints are removed according to the analysis workflow. |
| Group-aware train/test split | Workflow applies grouped splitting to avoid information leakage. | Group-aware splitting functions are retained in `main_analysis_final.py`. | Matched | Group structure is preserved. |
| q0.75 high-delivery label construction | Primary classification endpoint is constructed using the training-set q0.75 threshold. | Threshold calculation and label generation are retained. | Matched | Threshold is determined using training data only. |
| q0.70/q0.80 threshold sensitivity analysis | Alternative percentile thresholds are evaluated. | Sensitivity analysis logic is retained. | Matched | The public script preserves threshold sensitivity evaluation. |
| Candidate models | LR, KNN, DT, SVM-RBF, XGBoost, LightGBM, DNN, CatBoost, Voting, and Stacking models are evaluated. | Corresponding model training and evaluation blocks are retained. | Matched | Model families are preserved. |
| Optuna hyperparameter optimization | Model optimization is performed using PR-AUC-oriented optimization. | Optimization logic is retained in model workflow blocks. | Matched | Hyperparameter search workflow is preserved. |
| Imbalance handling | None, class-weight, and random oversampling strategies are considered. | Corresponding imbalance branches are retained. | Matched | Original comparison logic is preserved. |
| CV metrics and OOF prediction | Cross-validation metrics and OOF predictions are generated. | OOF prediction and metric export logic are retained. | Matched | OOF-based evaluation is preserved. |
| Primary model selection | Primary model is selected according to OOF performance, emphasizing PR-AUC. | Selection logic is retained. | Matched | Independent test data are not used for model selection. |
| Independent test evaluation | Held-out test evaluation is performed after model selection. | Test evaluation and prediction export are retained. | Matched | Test performance reporting is preserved. |
| Ranking metrics | PR-AUC, ROC-AUC, F1, Precision@K, Recall@K, EF@K and related metrics are calculated. | Ranking metric calculations are retained. | Matched | Screening-oriented evaluation is preserved. |
| CatBoost SHAP interpretation | SHAP interpretation is performed for model explanation. | SHAP calculation and export logic are retained mainly in `main_analysis_final.py`. | Matched | SHAP plotting is not primarily handled by `figure_generation_cleaned.py`. |
| Feature-level SHAP aggregation | Transformed features are aggregated back to original predictors. | Feature aggregation logic is retained. | Matched | Original predictor interpretation is preserved. |

---

# Virtual Screening Functional Mapping

| Function / workflow module | Cleaned script status | Notes |
|---|---|---|
| Virtual candidate generation | Matched | The final public workflow includes the updated candidate-generation logic from the finalized application script. |
| 50,000 candidate generation | Matched | The candidate pool size is configured as 50,000 virtual formulations. |
| Continuous-variable sampling | Matched | Continuous variables are sampled according to the final high-delivery formulation distribution strategy. |
| Material-category generation | Matched | `build_material_library()` constructs observed material-combination libraries from the reference dataset. |
| Type-MAT-Shape joint sampling | Matched | `sample_material_categories()` generates Type, MAT, and Shape jointly from observed real-database Type-MAT-Shape combinations rather than independent categorical sampling. |
| Chemistry-aware feasibility constraint | Matched | Candidate filtering enforces observed Type-MAT-Shape combinations through the material library constraint. |
| Candidate deduplication | Matched | Duplicate candidate formulations are removed before downstream screening. |
| Numerical feasibility filtering | Matched | Absolute-range and variable-validity filtering are retained. |
| Out-of-distribution filtering | Matched | Quantile-based numerical OOD filtering is retained. |
| CatBoost candidate scoring | Matched | Candidates are ranked using predicted high-delivery probability from the selected model. |
| Top candidate export | Matched | Top-ranked candidate tables and screening outputs are retained. |
| Cancer-type-specific screening | Matched | Condition-specific candidate filtering remains available. |
| Local working-range estimation | Matched | Size, Zeta Potential, and Admin working-range estimation logic is retained. |
| Generation metadata | Matched | Metadata records include candidate numbers, filtering information, and Type-MAT-Shape sampling strategy. |

---

# Figure Generation Functional Mapping

| Function / workflow module | Status | Notes |
|---|---|---|
| Dataset characteristic figures | Matched | Dataset distribution and feature landscape panels are retained in `figure_generation_cleaned.py`. |
| Model performance figures | Matched | Model comparison and ranking-related visualization logic are retained. |
| SHAP visualization | Partially matched | SHAP calculation/export is retained in `main_analysis_final.py`; figure-generation script mainly retains data-driven figure panels. |
| Candidate prioritization figures | Matched | Candidate ranking and working-range visualization panels are retained. |
| Figure export logic | Matched | Final figure export procedures are retained. |

---

# Overall Assessment

Most major analytical functions from the original workflow are retained in the cleaned public scripts.

The final public analysis script now reflects the updated virtual-screening workflow, including chemistry-aware Type-MAT-Shape constrained candidate generation and feasibility filtering.

The main limitation is organizational rather than functional: the scripts are cleaned public code records derived from notebook workflows. They may require manual review of data availability, package versions, computational resources, and execution order before complete reproduction in a new environment.

Figure 1 is not generated by Python because it was prepared separately as a schematic workflow figure.

---

# Conclusion

**Ready after documentation consistency review.**

Recommended final checks:

- confirm README descriptions match the current repository structure;
- confirm no removed supplementary-builder scripts are referenced;
- confirm supplementary materials correspond to the final manuscript version;
- verify that public scripts preserve the final analysis workflow without modifying reported results.
