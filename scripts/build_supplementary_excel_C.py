"""Build Supplementary Excel File C from updated virtual-screening outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATED_OUTPUTS_DIR = PROJECT_ROOT / "updated_outputs"
SUPPLEMENTARY_DIR = PROJECT_ROOT / "supplementary"
OUTPUT_FILE = SUPPLEMENTARY_DIR / "Supplementary_Excel_File_C_virtual_screening_outputs.xlsx"

ALL_CANDIDATES_FILE = UPDATED_OUTPUTS_DIR / "candidates_scored.csv"
TOP_CANDIDATES_FILE = UPDATED_OUTPUTS_DIR / "top_candidates.csv"
REQUIRED_SHEETS = [
    "screening_summary",
    "generation_meta",
    "all_scored_candidates",
    "top_200_candidates",
    "top_20_candidates",
    "top_10_candidates",
    "ct_specific_candidates",
    "ct_specific_ranges",
    "range_detail",
    "range_pretty",
]


def value_to_text(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def item_value_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return pd.DataFrame([{"item": key, "value": value_to_text(value)} for key, value in payload.items()])


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required updated virtual-screening output is missing: {path}")
    return pd.read_csv(path)


def rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    if "rank_model" in ranked.columns:
        ranked = ranked.sort_values("rank_model", ascending=True)
    elif "pred_score" in ranked.columns:
        ranked = ranked.sort_values("pred_score", ascending=False)
        ranked["rank_model"] = range(1, len(ranked) + 1)
    return ranked.reset_index(drop=True)


def screening_summary(all_candidates: pd.DataFrame, top_candidates: pd.DataFrame) -> pd.DataFrame:
    n_requested = 50000
    n_final = len(all_candidates)
    type_mat_shape = (
        all_candidates[["Type", "MAT", "Shape"]].drop_duplicates().shape[0]
        if {"Type", "MAT", "Shape"}.issubset(all_candidates.columns)
        else "not available"
    )
    rows = [
        ("candidate_pool_requested", n_requested),
        ("unique_candidates_after_deduplication", "not separately recorded in updated_outputs"),
        ("candidates_after_feasibility_filtering", "not separately recorded in updated_outputs"),
        ("candidates_after_out_of_distribution_control", n_final),
        ("final_scored_candidates", n_final),
        ("top_candidates_exported", len(top_candidates)),
        ("primary_model", "CatBoost"),
        ("screening_score", "predicted high-delivery probability"),
        ("candidate_generation_strategy", "50,000 virtual nanoparticle candidates were generated and then deduplicated, feasibility-filtered, OOD-filtered, scored, and ranked."),
        ("categorical_feasibility_filtering", "Chemistry-aware categorical feasibility filtering was applied before model scoring."),
        ("type_mat_shape_constraint", f"Generated Type-MAT-Shape combinations were constrained to combinations observed in the real database; retained combinations: {type_mat_shape}."),
        ("ood_filtering", "Numeric out-of-distribution filtering was applied before final CatBoost scoring."),
        ("ranking_strategy", "Candidates were prioritized by descending CatBoost predicted high-delivery probability."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def generation_meta(all_candidates: pd.DataFrame, top_candidates: pd.DataFrame) -> pd.DataFrame:
    score_col = "pred_score" if "pred_score" in all_candidates.columns else "pred_proba_high"
    payload: dict[str, Any] = {
        "input_all_scored_candidates": str(ALL_CANDIDATES_FILE.relative_to(PROJECT_ROOT)),
        "input_top_candidates": str(TOP_CANDIDATES_FILE.relative_to(PROJECT_ROOT)),
        "candidate_pool_requested": 50000,
        "final_scored_candidates": int(len(all_candidates)),
        "top_candidates_exported": int(len(top_candidates)),
        "primary_model": "CatBoost",
        "screening_score_column": score_col,
        "generation_update": "latest nanoparticle virtual screening workflow rerun in Jupyter Notebook",
        "categorical_feasibility_filtering": "chemistry-aware categorical feasibility filtering",
        "combination_constraint": "Type-MAT-Shape combinations constrained to combinations observed in the real database",
        "ood_filtering": "enabled before final scoring",
        "ranking_strategy": "descending CatBoost high-delivery probability",
        "top_candidate_prioritization": "top-ranked candidates exported for manuscript supplementary file",
    }
    if score_col in all_candidates.columns:
        payload.update(
            {
                "minimum_predicted_probability": float(all_candidates[score_col].min()),
                "median_predicted_probability": float(all_candidates[score_col].median()),
                "maximum_predicted_probability": float(all_candidates[score_col].max()),
            }
        )
    if {"Type", "MAT", "Shape"}.issubset(all_candidates.columns):
        payload["retained_type_mat_shape_combinations"] = int(
            all_candidates[["Type", "MAT", "Shape"]].drop_duplicates().shape[0]
        )
    return pd.DataFrame([{"item": key, "value": value_to_text(value)} for key, value in payload.items()])


def paper_candidate_table(frame: pd.DataFrame, topk: int | None = None) -> pd.DataFrame:
    ranked = rank_candidates(frame)
    if topk is not None:
        ranked = ranked.head(topk).copy()
    out = pd.DataFrame()
    out["Rank"] = range(1, len(ranked) + 1)
    column_map = [
        ("Type", "Type"),
        ("MAT", "MAT"),
        ("TS", "TS"),
        ("CT", "CT"),
        ("TM", "TM"),
        ("Shape", "Shape"),
        ("Size", "Size (log10)"),
        ("Size (nm)", "Size (nm)"),
        ("Zeta Potential", "Zeta Potential"),
        ("Admin", "Admin"),
        ("pred_proba_high", "Predicted high-delivery probability"),
        ("rank_model", "Model rank"),
    ]
    for source, target in column_map:
        if source in ranked.columns:
            out[target] = ranked[source].values
    return out


def range_detail(candidates: pd.DataFrame, top_candidates: pd.DataFrame, ct_value: str = "Breast") -> pd.DataFrame:
    if "CT" in top_candidates.columns:
        range_source = rank_candidates(top_candidates[top_candidates["CT"].astype(str) == ct_value]).head(20)
        source_label = f"Top {ct_value} candidates"
    else:
        range_source = pd.DataFrame()
        source_label = f"Top {ct_value} candidates"
    if range_source.empty:
        range_source = rank_candidates(top_candidates).head(20)
        source_label = "Top candidates"

    features = [
        ("Size (nm)", "Size", "nm"),
        ("Zeta Potential", "Zeta Potential", "mV"),
        ("Admin", "Admin", "mg/kg"),
    ]
    rows = []
    for column, parameter, unit in features:
        if column not in range_source.columns:
            continue
        values = pd.to_numeric(range_source[column], errors="coerce").dropna()
        if values.empty:
            continue
        lower = float(values.quantile(0.25))
        median = float(values.median())
        upper = float(values.quantile(0.75))
        rows.append(
            {
                "source": source_label,
                "feature": column,
                "Parameter": parameter,
                "n": int(values.size),
                "q_low": 0.25,
                "q_high": 0.75,
                "lower": lower,
                "median": median,
                "upper": upper,
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
                "Unit": unit,
                "recommended_window": f"{lower:.4g} - {upper:.4g}",
            }
        )
    return pd.DataFrame(rows)


def range_pretty(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=["Parameter", "Recommended window", "Median", "Unit", "Candidates used"])
    pretty = detail[["Parameter", "recommended_window", "median", "Unit", "n"]].copy()
    pretty = pretty.rename(
        columns={
            "recommended_window": "Recommended window",
            "median": "Median",
            "n": "Candidates used",
        }
    )
    return pretty


def style_workbook(writer: pd.ExcelWriter) -> None:
    fill = PatternFill(fill_type="solid", fgColor="D9D9D9")
    side = Side(style="thin", color="000000")
    border = Border(left=side, right=side, top=side, bottom=side)
    body = Font(name="Times New Roman", size=10)
    header = Font(name="Times New Roman", size=10, bold=True)
    alignment = Alignment(vertical="center", wrap_text=True)
    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = "A2"
        for row in worksheet.iter_rows():
            for cell in row:
                cell.font = header if cell.row == 1 else body
                cell.fill = fill if cell.row == 1 else PatternFill(fill_type=None)
                cell.border = border
                cell.alignment = alignment
        for column in worksheet.columns:
            width = max(len(str(cell.value or "")) for cell in column)
            worksheet.column_dimensions[column[0].column_letter].width = min(max(width + 2, 8), 45)


def main() -> None:
    SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)
    all_candidates = rank_candidates(read_csv_required(ALL_CANDIDATES_FILE))
    top_candidates = rank_candidates(read_csv_required(TOP_CANDIDATES_FILE))
    top_200 = paper_candidate_table(top_candidates, 200)
    top_20 = paper_candidate_table(top_candidates, 20)
    top_10 = paper_candidate_table(top_candidates, 10)
    ct_candidates_raw = (
        top_candidates[top_candidates["CT"].astype(str) == "Breast"].copy()
        if "CT" in top_candidates.columns
        else pd.DataFrame()
    )
    if ct_candidates_raw.empty and "CT" in all_candidates.columns:
        ct_candidates_raw = all_candidates[all_candidates["CT"].astype(str) == "Breast"].head(20).copy()
    ct_candidates = paper_candidate_table(ct_candidates_raw, 20)
    detail = range_detail(all_candidates, top_candidates, ct_value="Breast")
    pretty = range_pretty(detail)
    sheets = {
        "screening_summary": screening_summary(all_candidates, top_candidates),
        "generation_meta": generation_meta(all_candidates, top_candidates),
        "all_scored_candidates": all_candidates,
        "top_200_candidates": top_200,
        "top_20_candidates": top_20,
        "top_10_candidates": top_10,
        "ct_specific_candidates": ct_candidates,
        "ct_specific_ranges": pretty.assign(source_file="updated_outputs/top_candidates.csv")[
            ["source_file", "Parameter", "Recommended window", "Median", "Unit", "Candidates used"]
        ],
        "range_detail": detail,
        "range_pretty": pretty,
    }
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for sheet_name in REQUIRED_SHEETS:
            sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
        style_workbook(writer)
    print(f"Generated: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
