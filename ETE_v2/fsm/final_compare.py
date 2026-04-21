from __future__ import annotations

from typing import Optional

from fsm.models.patient import PatientInput


FINAL_COMPARE_SOURCE_ACHIEVED = "Achieved"
FINAL_COMPARE_SOURCE_PGP = "PGP"
FINAL_COMPARE_SOURCE_NO_GLASSES = "No Glasses"
FINAL_COMPARE_SOURCE_CURRENT_BASELINE = "Current Baseline"
FINAL_COMPARE_ZERO_SPH = 0.0
FINAL_COMPARE_ZERO_CYL = 0.0
FINAL_COMPARE_ZERO_AXIS = 180.0
FINAL_COMPARE_ZERO_ADD = 0.0

_NO_GLASSES_WEAR_TYPES = {
    "none",
    "no glasses",
    "no current glasses",
    "no current eyewear",
    "does not wear glasses",
}


def _normalized_text(value: object) -> str:
    return str(value or "").strip().lower()


def has_full_lensometry_pair(patient_input: Optional[PatientInput]) -> bool:
    if not patient_input:
        return False
    return bool(
        patient_input.lenso_re
        and patient_input.lenso_le
        and patient_input.lenso_re.has_full_rx()
        and patient_input.lenso_le.has_full_rx()
    )


def is_no_glasses_case(patient_input: Optional[PatientInput]) -> bool:
    if not patient_input:
        return False

    wear_type = _normalized_text(getattr(patient_input, "wear_type", ""))
    satisfaction = _normalized_text(getattr(patient_input, "satisfaction_with_current_rx", ""))

    if satisfaction == "no pgp":
        return True
    return wear_type in _NO_GLASSES_WEAR_TYPES and satisfaction == ""


def resolve_final_compare_current_source(patient_input: Optional[PatientInput]) -> str:
    if has_full_lensometry_pair(patient_input):
        return FINAL_COMPARE_SOURCE_PGP
    if is_no_glasses_case(patient_input):
        return FINAL_COMPARE_SOURCE_NO_GLASSES
    return ""


def final_compare_current_baseline_values(patient_input: Optional[PatientInput]) -> tuple[str, dict]:
    source = resolve_final_compare_current_source(patient_input)
    if source == FINAL_COMPARE_SOURCE_PGP and patient_input:
        lenso_re = patient_input.lenso_re
        lenso_le = patient_input.lenso_le
        return source, {
            "re_sph": lenso_re.sphere if lenso_re else None,
            "re_cyl": lenso_re.cylinder if lenso_re else None,
            "re_axis": lenso_re.axis if lenso_re else None,
            "le_sph": lenso_le.sphere if lenso_le else None,
            "le_cyl": lenso_le.cylinder if lenso_le else None,
            "le_axis": lenso_le.axis if lenso_le else None,
            "add_r": patient_input.lenso_add_r,
            "add_l": patient_input.lenso_add_l,
        }
    if source == FINAL_COMPARE_SOURCE_NO_GLASSES:
        return source, {
            "re_sph": FINAL_COMPARE_ZERO_SPH,
            "re_cyl": FINAL_COMPARE_ZERO_CYL,
            "re_axis": FINAL_COMPARE_ZERO_AXIS,
            "le_sph": FINAL_COMPARE_ZERO_SPH,
            "le_cyl": FINAL_COMPARE_ZERO_CYL,
            "le_axis": FINAL_COMPARE_ZERO_AXIS,
            "add_r": FINAL_COMPARE_ZERO_ADD,
            "add_l": FINAL_COMPARE_ZERO_ADD,
        }
    return "", {
        "re_sph": None,
        "re_cyl": None,
        "re_axis": None,
        "le_sph": None,
        "le_cyl": None,
        "le_axis": None,
        "add_r": None,
        "add_l": None,
    }


def final_compare_second_option_phase_name(source: str) -> str:
    label = source or FINAL_COMPARE_SOURCE_CURRENT_BASELINE
    return f"Final Compare Second Option {label}"


def final_compare_source_short_label(source: str) -> str:
    return source or FINAL_COMPARE_SOURCE_CURRENT_BASELINE
