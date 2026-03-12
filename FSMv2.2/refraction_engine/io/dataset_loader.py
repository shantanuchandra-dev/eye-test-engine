from pathlib import Path
from typing import Optional, List, Union
import pandas as pd

from refraction_engine.models.patient import PatientInput
from refraction_engine.models.prescription import EyePrescription


def _to_bool(v):
    if pd.isna(v):
        return False
    return str(v).strip().lower() in ["true", "1", "yes"]


def _to_float(v) -> Optional[float]:
    if pd.isna(v) or v == "":
        return None
    return float(v)


def _to_int(v) -> Optional[int]:
    if pd.isna(v) or v == "":
        return None
    return int(v)


def _to_str(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _eye_rx(sph, cyl, axis) -> Optional[EyePrescription]:
    sph = _to_float(sph)
    cyl = _to_float(cyl)
    axis = _to_float(axis)

    if sph is None and cyl is None and axis is None:
        return None

    return EyePrescription(
        sphere=sph,
        cylinder=cyl,
        axis=axis
    )


def load_patients(
    patient_csv: Union[str, Path],
    ar_csv: Union[str, Path],
    lenso_csv: Union[str, Path],
) -> List[PatientInput]:

    patient_df = pd.read_csv(patient_csv)
    ar_df = pd.read_csv(ar_csv)
    lenso_df = pd.read_csv(lenso_csv)

    patient_df["visit_id"] = patient_df["visit_id"].astype(str).str.strip()
    ar_df["visit_id"] = ar_df["visit_id"].astype(str).str.strip()
    lenso_df["visit_id"] = lenso_df["visit_id"].astype(str).str.strip()

    ar_map = {str(r["visit_id"]): r for _, r in ar_df.iterrows()}
    lenso_map = {str(r["visit_id"]): r for _, r in lenso_df.iterrows()}

    patients: List[PatientInput] = []

    for _, row in patient_df.iterrows():

        visit_id = str(row["visit_id"]).strip()
        ar = ar_map.get(visit_id)
        lenso = lenso_map.get(visit_id)

        patient = PatientInput(
            visit_id=visit_id,

            age=_to_int(row.get("age_years")),
            occupation=_to_str(row.get("occupation_type")),
            screen_time_hours=_to_float(row.get("screen_time_hours")),
            driving_hours=_to_float(row.get("driving_time_hours")),

            primary_reason=_to_str(row.get("primary_reason")),
            symptoms_text=_to_str(row.get("symptoms_multi")),

            satisfaction_with_current_rx=_to_str(row.get("satisfaction_with_current_rx")),
            wear_type=_to_str(row.get("wear_type")),

            distance_target_preference=_to_str(row.get("distance_target")),
            priority=_to_str(row.get("priority")),
            near_priority_declared=_to_str(row.get("near_priority")),

            last_eye_test_months_ago=_to_float(row.get("last_eye_test_months_ago")),
            rx_change_was_large=_to_bool(row.get("rx_change_was_large")),
            fluctuating_vision_reported=_to_bool(row.get("fluctuating_vision_reported")),

            diabetes=_to_bool(row.get("diabetes")),
            prior_eye_surgery=_to_str(row.get("prior_eye_surgery")),
            keratoconus=_to_bool(row.get("known_keratoconus")),
            amblyopia=_to_bool(row.get("known_amblyopia")),
            infection=_to_bool(row.get("current_eye_infection_or_inflammation")),

            optom_review_flag=False,

            autorefractor_re=_eye_rx(
                ar.get("S_r_AR") if ar is not None else None,
                ar.get("C_r_AR") if ar is not None else None,
                ar.get("A_r_AR") if ar is not None else None
            ),

            autorefractor_le=_eye_rx(
                ar.get("S_l_AR") if ar is not None else None,
                ar.get("C_l_AR") if ar is not None else None,
                ar.get("A_l_AR") if ar is not None else None
            ),

            lenso_re=_eye_rx(
                lenso.get("S_r_Lenso") if lenso is not None else None,
                lenso.get("C_r_Lenso") if lenso is not None else None,
                lenso.get("A_r_Lenso") if lenso is not None else None
            ),

            lenso_le=_eye_rx(
                lenso.get("S_l_Lenso") if lenso is not None else None,
                lenso.get("C_l_Lenso") if lenso is not None else None,
                lenso.get("A_l_Lenso") if lenso is not None else None
            ),

            lenso_add_r=_to_float(
                lenso.get("Add_r_Lenso") if lenso is not None else None
            ),

            lenso_add_l=_to_float(
                lenso.get("Add_l_Lenso") if lenso is not None else None
            )
        )

        patients.append(patient)

    return patients