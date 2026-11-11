"""Metadata helpers for accession-level splits and labels."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class AccessionLabel:
    """PI-RADS label plus auxiliary metadata for a single accession."""

    accession_id: str
    pirads: int
    cs_pca_label: int  # 0 or 1
    gleason_grade_group: int | None = None
    gleason_label: int | None = None  # 0 or 1 when pathology label is available
    patient_id: str | None = None
    study_datetime: pd.Timestamp | None = None
    h5_path: Path | None = None


PIRADS_PATTERN = re.compile(r"(?:PI-?RADS|Pirads)[:\s]*(\d)", re.IGNORECASE)


def extract_max_pirads(report: object) -> float:
    """Extract the maximum PI-RADS score mentioned inside a report body."""

    if isinstance(report, str):
        matches = PIRADS_PATTERN.findall(report)
        if matches:
            return float(max(int(score) for score in matches))
    return float(np.nan)


def _parse_grade_group(value: object) -> int | None:
    gg = pd.to_numeric(value, errors="coerce")
    if pd.isna(gg):
        return None
    return int(gg)


def _load_pathology_frame(pathology_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(pathology_csv))

    df = df.copy()
    df["AccessionNumber"] = df["mri_accession"].astype(str)
    df["gleason_grade_group"] = df["highest_grade_group"].apply(_parse_grade_group)
    impute_mask = df["biopsy_accession"].notna() & df["gleason_grade_group"].isna()
    df.loc[impute_mask, "gleason_grade_group"] = 1

    grouped = (
        df.groupby("AccessionNumber", as_index=False)["gleason_grade_group"]
        .max()
    )
    labeled = grouped["gleason_grade_group"].notna().sum()
    imputed_rows = int(impute_mask.sum())
    imputed_accessions = int(df.loc[impute_mask, "AccessionNumber"].nunique())
    logger.info(
        "[pathology] accessions=%d with_valid_grade_group=%d imputed_gg1_rows=%d imputed_gg1_accessions=%d",
        len(grouped),
        int(labeled),
        imputed_rows,
        imputed_accessions,
    )
    return grouped


def _load_label_frame(
    csv_path: Path,
    pathology_csv: Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(Path(csv_path))
    required = {"PatientID", "AccessionNumber", "StudyDate", "ReportBody"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["PatientID"] = df["PatientID"].astype(str)
    df["AccessionNumber"] = df["AccessionNumber"].astype(str)
    df["StudyDate"] = pd.to_datetime(df["StudyDate"], errors="coerce")
    df["maxPIRADS"] = df["ReportBody"].apply(extract_max_pirads)
    df = df.dropna(subset=["maxPIRADS", "StudyDate"])
    df["maxPIRADS"] = df["maxPIRADS"].astype(int)
    if pathology_csv is not None:
        pathology_df = _load_pathology_frame(pathology_csv)
        df = df.merge(pathology_df, on="AccessionNumber", how="left")
    else:
        df["gleason_grade_group"] = pd.NA
    return df


def parse_h5_filename(path: Path) -> Dict[str, object]:
    """Extract MRN and scanner metadata encoded inside ESC .h5 filenames."""

    info: Dict[str, object] = {"path": Path(path)}
    parts = Path(path).stem.split("#")
    for part in parts:
        if part.startswith("Site"):
            info["scanner_site"] = part[4:]
        elif part.startswith("F"):
            info["F_field"] = part[1:]
        elif part.startswith("M"):
            info["M_field"] = part[1:]
        elif re.fullmatch(r"D\d{6}", part):
            info["date"] = datetime.strptime(part[1:], "%d%m%y").date().isoformat()
        elif re.fullmatch(r"T\d{6}", part):
            info["time"] = datetime.strptime(part[1:], "%H%M%S").time().isoformat()
        elif "__" in part and part.split("__")[-1].isdigit():
            info["mrn"] = part.split("__")[-1]
        else:
            info.setdefault("modality_parts", []).append(part)

    info["fname_without_site"] = "#".join([part for part in parts if not part.startswith("Site")])

    if "date" in info and "time" in info:
        try:
            info["datetime"] = pd.Timestamp(f"{info['date']}T{info['time']}")
        except Exception:  # pragma: no cover - defensive
            info["datetime"] = pd.NaT
    else:
        info["datetime"] = pd.NaT
    return info


def load_accession_labels(
    csv_path: Path,
    h5_root: Path,
    date_buffer_days: int = 7,
    pathology_csv: Path | None = None,
    gleason_positive_threshold: int = 2,
) -> List[AccessionLabel]:
    """Load accession-level labels and align them with on-disk H5 volumes."""

    df = _load_label_frame(csv_path, pathology_csv=pathology_csv)

    h5_records = _scan_h5_directory(Path(h5_root))
    mrn_lookup = _group_records_by_mrn(h5_records)
    logger.info("[load_labels] Total CSV rows after filtering: %d", len(df))

    labels: List[AccessionLabel] = []
    unmatched_accessions: List[str] = []
    for _, row in df.iterrows():
        accession = row["AccessionNumber"]
        pirads = int(row["maxPIRADS"])
        h5_path = _match_h5_to_accession(row, mrn_lookup, date_buffer_days)
        if h5_path is None:
            unmatched_accessions.append(accession)
            continue
        cs_label = 1 if pirads >= 3 else 0
        grade_group_raw = row.get("gleason_grade_group")
        grade_group = None if pd.isna(grade_group_raw) else int(grade_group_raw)
        gleason_label = (
            1 if grade_group >= gleason_positive_threshold else 0
        ) if grade_group is not None else None
        labels.append(
            AccessionLabel(
                accession_id=accession,
                pirads=pirads,
                cs_pca_label=cs_label,
                gleason_grade_group=grade_group,
                gleason_label=gleason_label,
                patient_id=row["PatientID"],
                study_datetime=row["StudyDate"],
                h5_path=h5_path,
            )
        )

    logger.info(
        "[load_labels] matched_accessions=%d unmatched_accessions=%d",
        len(labels),
        len(unmatched_accessions),
    )
    if unmatched_accessions:
        logger.warning(
            "Unable to align %d accessions to H5 files",
            len(unmatched_accessions),
        )

    return labels


def load_accession_labels_from_csv(
    csv_path: Path,
    pathology_csv: Path | None = None,
    gleason_positive_threshold: int = 2,
) -> List[AccessionLabel]:
    """Load labels directly from CSV without recon H5 matching."""
    df = _load_label_frame(csv_path, pathology_csv=pathology_csv)
    df = df.drop_duplicates(subset=["AccessionNumber"])
    logger.info("[load_labels_csv] Total CSV rows after filtering: %d", len(df))

    labels: List[AccessionLabel] = []
    for _, row in df.iterrows():
        accession = row["AccessionNumber"]
        pirads = int(row["maxPIRADS"])
        cs_label = 1 if pirads >= 3 else 0
        grade_group_raw = row.get("gleason_grade_group")
        grade_group = None if pd.isna(grade_group_raw) else int(grade_group_raw)
        gleason_label = (
            1 if grade_group >= gleason_positive_threshold else 0
        ) if grade_group is not None else None
        labels.append(
            AccessionLabel(
                accession_id=accession,
                pirads=pirads,
                cs_pca_label=cs_label,
                gleason_grade_group=grade_group,
                gleason_label=gleason_label,
                patient_id=row["PatientID"],
                study_datetime=row["StudyDate"],
                h5_path=None,
            )
        )
    return labels


def summarize_label_distribution(labels: Sequence[AccessionLabel]) -> Dict[str, object]:
    """Summarize PI-RADS and Gleason label coverage for a split."""
    total = len(labels)
    pirads_counts = Counter(lbl.pirads for lbl in labels)
    cs_positive = sum(lbl.cs_pca_label for lbl in labels)
    cs_negative = total - cs_positive

    gleason_labeled = [lbl for lbl in labels if lbl.gleason_label is not None]
    gleason_count = len(gleason_labeled)
    gleason_positive = sum(lbl.gleason_label for lbl in gleason_labeled)
    gleason_negative = gleason_count - gleason_positive
    gg_counts = Counter(
        lbl.gleason_grade_group
        for lbl in gleason_labeled
        if lbl.gleason_grade_group is not None
    )

    return {
        "count": total,
        "pirads_positive": int(cs_positive),
        "pirads_negative": int(cs_negative),
        "pirads_distribution": {f"pirads_{k}": int(v) for k, v in sorted(pirads_counts.items())},
        "gleason_labeled_count": int(gleason_count),
        "gleason_unlabeled_count": int(total - gleason_count),
        "gleason_positive": int(gleason_positive),
        "gleason_negative": int(gleason_negative),
        "grade_group_distribution": {f"gg_{k}": int(v) for k, v in sorted(gg_counts.items())},
    }


def list_dicom_accessions(dicom_roots: Sequence[Path]) -> set[str]:
    accessions: set[str] = set()
    for root in dicom_roots:
        root = Path(root)
        for path in root.glob("*.h5"):
            accessions.add(path.stem)
    logger.info("[dicom_scan] Found %d DICOM accessions across %d roots", len(accessions), len(dicom_roots))
    return accessions


def _scan_h5_directory(h5_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    h5_files = sorted(Path(h5_root).glob("*.h5"))
    logger.info("[scan_h5] Found %d .h5 files", len(h5_files))
    parse_failures = 0
    for path in h5_files:
        try:
            parsed = parse_h5_filename(path)
            records.append(parsed)
        except Exception as exc:  # pragma: no cover - defensive
            parse_failures += 1
            logger.debug("Failed to parse one H5 filename: %s", exc)
    records_with_mrn = sum(1 for r in records if r.get("mrn"))
    logger.info(
        "[scan_h5] parsed_records=%d with_mrn=%d parse_failures=%d",
        len(records),
        records_with_mrn,
        parse_failures,
    )
    return records


def _group_records_by_mrn(records: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        mrn = record.get("mrn")
        if not mrn:
            continue
        grouped.setdefault(mrn, []).append(record)
    for recs in grouped.values():
        recs.sort(key=lambda rec: rec.get("datetime") or pd.Timestamp.max)
    logger.info("[group_mrn] Grouped into %d unique MRNs", len(grouped))
    return grouped


def _match_h5_to_accession(row: pd.Series, grouped: Dict[str, List[Dict[str, object]]], buffer_days: int) -> Path | None:
    patient_id = row["PatientID"]
    study_date = row["StudyDate"]
    if pd.isna(study_date):
        return None

    candidates = grouped.get(patient_id)
    if not candidates:
        return None

    study_norm = study_date.normalize()
    best_idx: int | None = None
    best_delta = pd.Timedelta.max
    for idx, record in enumerate(candidates):
        rec_dt = record.get("datetime", pd.NaT)
        if pd.isna(rec_dt):
            continue
        delta = abs(rec_dt.normalize() - study_norm)
        if delta <= pd.Timedelta(days=buffer_days) and delta < best_delta:
            best_delta = delta
            best_idx = idx

    if best_idx is None:
        return None

    record = candidates.pop(best_idx)
    path = record.get("path")
    return Path(path) if path is not None else None


def _label_patient_key(label: AccessionLabel) -> str:
    return label.patient_id if label.patient_id is not None else label.accession_id


def temporal_split(
    labels: Sequence[AccessionLabel],
    train_frac: float = 0.75,
    val_frac: float = 0.15,
) -> Tuple[List[str], List[str], List[str]]:
    """Split accession IDs by patient-level chronology of ``study_datetime``.

    Patients are grouped by ``patient_id`` and assigned to a single split.
    Patient order is determined by each patient's earliest available study date.
    """

    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")

    grouped: Dict[str, List[AccessionLabel]] = defaultdict(list)
    for lbl in labels:
        grouped[_label_patient_key(lbl)].append(lbl)

    patient_order: List[Tuple[str, pd.Timestamp]] = []
    for patient_id, patient_labels in grouped.items():
        valid_dates = [
            dt for dt in (lbl.study_datetime for lbl in patient_labels)
            if dt is not None and not pd.isna(dt)
        ]
        patient_dt = min(valid_dates) if valid_dates else pd.Timestamp.max
        patient_order.append((patient_id, patient_dt))

    patient_order.sort(key=lambda item: (item[1], item[0]))
    total_patients = len(patient_order)
    n_train = int(total_patients * train_frac)
    n_val = int(total_patients * val_frac)

    train_patients = {pid for pid, _ in patient_order[:n_train]}
    val_patients = {pid for pid, _ in patient_order[n_train:n_train + n_val]}
    test_patients = {pid for pid, _ in patient_order[n_train + n_val:]}

    train_ids = [lbl.accession_id for lbl in labels if _label_patient_key(lbl) in train_patients]
    val_ids = [lbl.accession_id for lbl in labels if _label_patient_key(lbl) in val_patients]
    test_ids = [lbl.accession_id for lbl in labels if _label_patient_key(lbl) in test_patients]
    return train_ids, val_ids, test_ids


def consecutive_patient_holdout_split(
    labels: Sequence[AccessionLabel],
    *,
    train_frac: float = 0.85,
    test_frac: float = 0.10,
    test_start_date: str | pd.Timestamp | None = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Temporal split with contiguous test holdout and patient-level leakage control.

    Strategy:
    1. Determine test-start date from explicit ``test_start_date`` or from the
       most recent ``test_frac`` of dated accessions.
    2. Assign test as all accessions with ``study_datetime >= test_start_date``.
    3. Build train/val pool from accessions before test-start date, excluding
       every patient that appears in test.
    4. Sort pool patients by first exam date; earliest patients -> train,
       most recent patients -> val.
    """

    if not labels:
        return [], [], []
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")
    if test_start_date is None and not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1 when test_start_date is not set")

    dated_labels = [
        lbl for lbl in labels
        if lbl.study_datetime is not None and not pd.isna(lbl.study_datetime)
    ]
    if not dated_labels:
        raise ValueError("Cannot split labels: no valid study_datetime values available")

    split_start = pd.Timestamp(test_start_date) if test_start_date is not None else None
    if split_start is not None and pd.isna(split_start):
        raise ValueError(f"Invalid test_start_date '{test_start_date}'")
    if split_start is None:
        dated_labels.sort(key=lambda lbl: (lbl.study_datetime, lbl.accession_id))
        n_test_tail = max(1, int(math.ceil(len(dated_labels) * test_frac)))
        n_test_tail = min(len(dated_labels), n_test_tail)
        split_start = pd.Timestamp(dated_labels[-n_test_tail].study_datetime)

    test_patient_ids: set[str] = set()
    test_ids: List[str] = []
    for label in labels:
        study_dt = label.study_datetime
        patient_key = _label_patient_key(label)
        if study_dt is None or pd.isna(study_dt):
            # Conservative choice: unknown chronology goes to holdout.
            test_patient_ids.add(patient_key)
            test_ids.append(label.accession_id)
            continue
        if pd.Timestamp(study_dt) >= split_start:
            test_patient_ids.add(patient_key)
            test_ids.append(label.accession_id)

    pool_by_patient: Dict[str, List[AccessionLabel]] = defaultdict(list)
    for label in labels:
        patient_key = _label_patient_key(label)
        study_dt = label.study_datetime
        if patient_key in test_patient_ids:
            continue
        if study_dt is None or pd.isna(study_dt):
            continue
        if pd.Timestamp(study_dt) >= split_start:
            continue
        pool_by_patient[patient_key].append(label)

    if not pool_by_patient:
        raise ValueError("No pre-test patients remain for train/val after holdout filtering")

    patient_order: List[Tuple[str, pd.Timestamp]] = []
    for patient_id, patient_labels in pool_by_patient.items():
        first_dt = min(pd.Timestamp(lbl.study_datetime) for lbl in patient_labels)
        patient_order.append((patient_id, first_dt))
    patient_order.sort(key=lambda item: (item[1], item[0]))

    total_pool_patients = len(patient_order)
    if total_pool_patients == 1:
        n_train = 1
    else:
        n_train = int(total_pool_patients * train_frac)
        n_train = max(1, min(total_pool_patients - 1, n_train))

    train_patients = {pid for pid, _ in patient_order[:n_train]}
    val_patients = {pid for pid, _ in patient_order[n_train:]}

    train_ids: List[str] = []
    val_ids: List[str] = []
    for label in labels:
        patient_key = _label_patient_key(label)
        study_dt = label.study_datetime
        if study_dt is None or pd.isna(study_dt):
            continue
        if pd.Timestamp(study_dt) >= split_start:
            continue
        if patient_key in train_patients:
            train_ids.append(label.accession_id)
        elif patient_key in val_patients:
            val_ids.append(label.accession_id)

    excluded_pretest_from_test_patients = 0
    for label in labels:
        study_dt = label.study_datetime
        if study_dt is None or pd.isna(study_dt):
            continue
        if _label_patient_key(label) in test_patient_ids and pd.Timestamp(study_dt) < split_start:
            excluded_pretest_from_test_patients += 1

    logger.info(
        "[split:consecutive_patient_holdout] train=%d val=%d test=%d "
        "test_patients=%d pool_patients=%d excluded_pretest_accessions_from_test_patients=%d",
        len(train_ids),
        len(val_ids),
        len(test_ids),
        len(test_patient_ids),
        total_pool_patients,
        excluded_pretest_from_test_patients,
    )
    return train_ids, val_ids, test_ids
