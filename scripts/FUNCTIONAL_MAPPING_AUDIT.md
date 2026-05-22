# Functional Mapping Audit

Audit date: 2026-05-22  
Audit mode: static code review only. No notebooks or scripts were executed. No model training, result recomputation, or figure generation was run.

## Scope

Reviewed public upload scripts:

- `scripts/main_analysis_final.py`
- `scripts/figure_generation_cleaned.py`

`main_analysis_final.py` is the manually organized final public analysis code exported from the finalized Jupyter workflow.

Original notebook evidence is cited by notebook code-cell number from the original local notebooks. The original notebooks are not included in the upload package.

The cleaned scripts should be interpreted as cleaned public code records exported from the notebooks. They preserve the notebook-derived workflow, but they are not guaranteed to be one-click executable in a different environment without manual review of inputs, dependencies, and runtime assumptions.

## Main Analysis Functional Mapping

| Function / workflow module | Original notebook evidence | Cleaned script evidence | Matched / Partially matched / Missing | Notes |
|---|---|---|---|---|
| 1. Data loading and preprocessing | Main notebook cells 0 and 2 contain `PATH_2024`, `read_csv`, `clean_2024`, and preprocessing setup. | `main_analysis_final.py` lines 22, 50, 588, 607. | Matched | Personal paths were replaced by repository-relative path variables. |
| 2. DEtumor target extraction | Main notebook cells 0 and 2 define `TARGET_DE` and target handling. | Lines 56, 210-224, 640-651. | Matched | Target column is `DE_tumor`; labels are built from the training-set threshold. |
| 3. Removal of non-tumor delivery variables | Main notebook cells 0 and 2 include DE-column filtering around target setup. | Lines 214-215 drop other `DE_` columns when only the tumor endpoint is retained. | Matched | This preserves removal of non-target delivery variables. |
| 4. Group-aware train/test split | Main notebook cells 0 and 2 contain `group_train_test_split`; cells also define group columns. | Lines 237-268 define `group_train_test_split`; line 625 applies it. | Matched | Group-aware split is retained. |
| 5. q0.75 high-delivery label construction | Main notebook cells 0 and 2 define `QUANTILE_P = 0.75` and train-only thresholding. | Lines 74, 640-651, 777-786. | Matched | Threshold is computed on training data only. |
| 6. q0.70/q0.80 threshold sensitivity | Main notebook cells 0 and 2 contain `THRESHOLD_CANDIDATES` and `threshold_sensitivity_report`. | Lines 482 and 917-918. | Matched | Sensitivity report logic is retained as code record. |
| 7. Candidate models: LR, KNN, SVM-RBF, DT, XGBoost, LightGBM, DNN, CatBoost, Voting, Stacking | Main notebook cells 11, 13, 15, 17, 19, 21, 23, 25 define single-model families; cells 27 and 29 contain stacking/voting payload logic. | Lines 1353, 2005, 2659, 3355, 3993, 4658, 5364, 6050, 6850-6851, 7536-7537. | Matched | All requested model families are present. |
| 8. Optuna hyperparameter optimization | Main notebook cells 11, 13, 15, 17, 19, 21, 23, 25 import/use `optuna` and optimize PR-AUC. | Lines 1266, 1625-1634 and repeated model blocks. | Matched | Optuna optimization is retained across model blocks. |
| 9. Imbalance settings: none, class_weight, random oversampling | Main notebook cells 0 and 2 define imbalance setup; model cells use `class_weight` and `random_oversample_any`. | Lines 96-97, 346-382, 1015-1042, 1536-1543. | Matched | Baseline/none, class-weight, and random-oversampling paths are retained where applicable. |
| 10. CV metrics and OOF predictions | Main notebook model cells include `oof_prob_train`, fold metrics, and CV summaries. | Lines 1641-1662, 1742-1750, 1807-1827 and repeated model blocks. | Matched | OOF predictions and metrics are exported per model block. |
| 11. Primary-model selection by OOF PR-AUC | Main notebook cells 30 and 31 contain `sort_summary_df_for_selection`, `model_metrics_oof_df`, and `PRIMARY_MODEL_NAME`. | Lines 8278-8311 and 8433-8468. | Matched | Selection sorts OOF metrics with PR-AUC first, followed by ranking tie-breakers. |
| 12. Independent test-set evaluation | Main notebook model cells compute test metrics and export test predictions. | Lines 1773-1787, 1794, 1807-1827 and repeated model blocks; summary lines 8559-8601. | Matched | Test metrics are reported but not used for primary-model selection. |
| 13. Ranking metrics: PR-AUC, ROC-AUC, F1, Precision@K, Recall@K, EF@K, EF5%, EF10% | Main notebook model cells define ranking and threshold metrics; cells 30-31 aggregate them. | Lines 1414-1483, 1773-1787, 8204-8206, 8278-8311. | Matched | Ranking-oriented metrics are retained. |
| 14. CatBoost SHAP interpretation | Main notebook cell 42 contains `import shap`, `TreeExplainer`, CatBoost handling, and SHAP exports. | Lines 8973, 9163, 9294-9304, 9447-9539. | Matched | SHAP uses tree explainer paths, including CatBoost when selected or used as fallback. |
| 15. One-hot SHAP aggregation to original predictors | Main notebook cell 42 contains `sv_group` and grouped feature aggregation. | Lines 9391-9433, 9516, 9589. | Matched | One-hot/transformed features are aggregated back to grouped predictors. |
| 16. Virtual candidate generation | Main notebook cells 47, 49, 52, 54, 56, 57, and 59 contain candidate generation/application code. | Lines 10073-10118 and 10552-10558. | Matched | Candidate generation is retained in the final application block. |
| 17. 50,000 candidate pool | Main notebook candidate cells define `N_CANDIDATES = int(..., 50000)`. | Line 10073 and use at lines 10552-10558. | Matched | Default candidate pool size is 50,000. |
| 18. Deduplication | Main notebook candidate cells contain `drop_duplicates`. | Lines 10560-10563 and 11178. | Matched | Candidate-level deduplication is retained. |
| 19. Feasibility filtering | Main notebook candidate cells include feasibility filters and reports. | Lines 10610-10661 and 10676-10678. | Matched | Absolute range, nonnegative, integer, categorical tuple, and pairwise seen-combination filters are retained. |
| 20. OOD control using 0.01 and 0.99 quantiles | Main notebook candidate cells define `OOD_Q_LOW`, `OOD_Q_HIGH`, and quantile bounds. | Lines 10087-10091, 10603-10605, 10663-10669, 10734-10737. | Matched | Numeric OOD filtering uses 0.01/0.99 bounds by default. |
| 21. Final candidate scoring and ranking | Main notebook candidate cells contain model application and ranking. | Lines 10685-10709. | Matched | Candidates are scored by predicted high-delivery probability and ranked. |
| 22. Top candidate export / top-200 export | Main notebook candidate cells include `TOP_N`, `PAPER_TOPK_LIST`, and paper candidate table exports. | Lines 10074, 10101, 11140-11155. | Matched | `TOP_N` defaults to 200 and exports `generated_top{TOP_N}` plus paper-style tables. |
| 23. Cancer-type-specific / CT / Breast screening | Main notebook candidate cells include `ENABLE_CT_SUBTABLES`, `CT_SUBTABLE_VALUES`, `RANGE_FILTERS`, and `Breast`. | Lines 10105-10111, 10231, 11221-11229, 11252-11256, 11316-11328. | Matched | CT-specific tables and Breast filtering are retained. |
| 24. Local working-range estimation for Size, Zeta Potential, and Admin | Main notebook candidate cells include candidate-level intervals and condition-specific recommendation windows. | Lines 10119-10154, 10747-10848, 10864-10981, 11009-11067. | Matched | Size is handled separately in nm space; Zeta Potential and Admin use local perturbation/window logic. |
| 25. generation_meta / range recommendation / candidate table outputs | Main notebook candidate cells include `generation_meta`, `range_recommendation`, and candidate table exports. | Lines 11009-11067, 11083-11155, 11264-11290, 11347-11418. | Matched | Metadata, range recommendation, top tables, and candidate outputs are retained as code records. |

## Figure Generation Functional Mapping

| Function / workflow module | Original notebook evidence | Cleaned script evidence | Matched / Partially matched / Missing | Notes |
|---|---|---|---|---|
| Fig. 2 dataset characteristics and label distribution plotting logic | Figure notebook cells 2 and 4 contain Figure 2 dataset landscape, endpoint/label distribution, missingness, categorical and continuous panels. | `figure_generation_cleaned.py` lines 896-1373 and 1403-1846. | Matched | Combined and individual panel/source-data export logic is retained. |
| Fig. 3 model comparison and ranking-oriented performance plotting logic | Figure notebook cells 5-14 contain Figure 3 model comparison, PR-AUC, fold PR-AUC, OOF/test metrics, score separation, and ranking panels. | Lines 1868-3896 and 3932-4995. | Matched | Model comparison and screening/ranking panels are retained. |
| Fig. 4 detailed performance and SHAP interpretation plotting logic | Figure notebook cells 15-16 contain Figure 4 performance panels; cells 67-69 contain SHAP plotting/analysis material. | Figure performance panels retained at lines 5019-5950; SHAP plotting is not retained in `figure_generation_cleaned.py` but SHAP interpretation/export is retained in `main_analysis_final.py` lines 9294-9539. | Partially matched | Detailed performance panels are retained in the figure script. SHAP plotting is preserved as part of the main analysis script, not in the figure-generation script. |
| Fig. 5 dashboard-related figure or screenshot arrangement logic, if applicable | Figure notebook search found Figure 5 candidate-prioritization cells 21-22; no dashboard/screenshot-specific evidence was found in the reviewed notebook cells. | Lines 5998-6934 contain Figure 5 candidate prioritization, retention, top-candidate profile, and working-range panels. | Partially matched | Figure 5 candidate-prioritization logic is retained. Dashboard screenshot arrangement was not identified as a retained Python figure-generation module. |
| Final panel export or figure output logic | Figure notebook cells 2, 4, 5-16, 21-22 contain `savefig`, panel export, and source-data export logic. | Lines 380-386, 1348-1364, 1417-1430, 2120-2149, 2814-2848, 3085-3119, 4632-4995, 5824-5950, 6116-6128, 6861-6934. | Matched | Multi-format exports and source-data exports are retained for data-driven panels. |

## Overall Assessment

Most core analytical and figure-generation functionality from the original notebooks is retained in the cleaned public scripts. The main caveat is organizational rather than functional: the cleaned scripts are notebook-derived code records with top-level execution blocks and shared global state, so they document the workflow but do not guarantee one-click reproducibility in a fresh environment.

Figure 1 is correctly excluded from Python-generation requirements because it was prepared separately as a schematic workflow figure.

## Conclusion

**Ready after minor README clarification.**

Recommended clarification: state explicitly that SHAP interpretation and SHAP plot/export code are retained in `main_analysis_final.py`, while `figure_generation_cleaned.py` mainly retains the data-driven Figure 2, Figure 3, Figure 4 performance, and Figure 5 candidate-prioritization panel-generation code. Also state that the cleaned scripts are public code records and should be manually reviewed before full re-execution.
