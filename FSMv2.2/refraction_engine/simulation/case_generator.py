import random
from dataclasses import dataclass

from refraction_engine.engines.derived_variables_engine import DerivedVariablesEngine
from refraction_engine.models.patient import PatientInput
from refraction_engine.models.prescription import EyePrescription


@dataclass
class SyntheticCase:
    case_id: str
    profile_id: str
    patient_input: PatientInput
    truth_rx: dict
    ar_re: EyePrescription
    ar_le: EyePrescription
    lenso_re: EyePrescription
    lenso_le: EyePrescription
    lenso_add_r: float
    lenso_add_l: float
    dv: object


def round_quarter(x):
    return round(x / 0.25) * 0.25


def wrap_axis(axis):
    axis = int(round(axis)) % 180
    if axis < 0:
        axis += 180
    return axis


def generate_truth_rx(rng, profile):
    profile_id = profile["profile_id"]

    if "high_myope" in profile_id:
        sph_re = round_quarter(rng.uniform(-8.0, -4.0))
        sph_le = round_quarter(rng.uniform(-8.0, -4.0))
    elif "hyperope" in profile_id:
        sph_re = round_quarter(rng.uniform(0.0, 3.0))
        sph_le = round_quarter(rng.uniform(0.0, 3.0))
    elif "presbyope" in profile_id:
        sph_re = round_quarter(rng.uniform(-2.5, 1.5))
        sph_le = round_quarter(rng.uniform(-2.5, 1.5))
    else:
        sph_re = round_quarter(rng.uniform(-6.0, 2.0))
        sph_le = round_quarter(rng.uniform(-6.0, 2.0))

    if "young_astigmat" in profile_id:
        cyl_re = round_quarter(rng.uniform(-2.5, -1.0))
        cyl_le = round_quarter(rng.uniform(-2.5, -1.0))
    else:
        cyl_re = round_quarter(rng.uniform(-3.0, 0.0))
        cyl_le = round_quarter(rng.uniform(-3.0, 0.0))

    axis_re = wrap_axis(rng.randint(0, 179))
    axis_le = wrap_axis(rng.randint(0, 179))

    if "mild_anisometrope" in profile_id:
        sph_le = round_quarter(sph_re + rng.uniform(0.5, 1.0))
    elif "moderate_anisometrope" in profile_id:
        sph_le = round_quarter(sph_re + rng.uniform(1.0, 2.0))

    add_r = 0.0
    add_l = 0.0
    if profile["age_years"] >= 45:
        add_r = round_quarter(rng.uniform(0.75, 2.25))
        add_l = add_r

    return {
        "re_sph": sph_re,
        "re_cyl": cyl_re,
        "re_axis": axis_re,
        "le_sph": sph_le,
        "le_cyl": cyl_le,
        "le_axis": axis_le,
        "add_r": add_r,
        "add_l": add_l,
    }


def generate_measurement_from_truth(rng, truth_sph, truth_cyl, truth_axis, quality="normal"):
    if quality == "large_mismatch":
        sph_noise = rng.uniform(-1.5, 1.5)
        cyl_noise = rng.uniform(-1.0, 1.0)
        axis_noise = rng.randint(-20, 20)
    else:
        sph_noise = rng.uniform(-0.5, 0.5)
        cyl_noise = rng.uniform(-0.5, 0.5)
        axis_noise = rng.randint(-10, 10)

    return EyePrescription(
        sphere=round_quarter(truth_sph + sph_noise),
        cylinder=round_quarter(truth_cyl + cyl_noise),
        axis=wrap_axis(truth_axis + axis_noise),
    )


def build_patient_input(case_id, profile, truth, ar_re, ar_le, lenso_re, lenso_le):
    return PatientInput(
        visit_id=case_id,
        age=profile["age_years"],
        occupation=profile["occupation_type"],
        screen_time_hours=profile["screen_time_hours"],
        driving_hours=profile["driving_time_hours"],
        primary_reason=profile["primary_reason"],
        symptoms_text=profile["symptoms_multi"],
        satisfaction_with_current_rx=profile["satisfaction_with_current_rx"],
        wear_type=profile["wear_type"],
        distance_target_preference=profile["distance_target"],
        priority=profile["priority"],
        near_priority_declared=profile["near_priority"],
        last_eye_test_months_ago=profile["last_eye_test_months_ago"],
        rx_change_was_large=profile["rx_change_was_large"],
        fluctuating_vision_reported=profile["fluctuating_vision_reported"],
        diabetes=profile["diabetes"],
        prior_eye_surgery=profile["prior_eye_surgery"],
        keratoconus=profile["known_keratoconus"],
        amblyopia=profile["known_amblyopia"],
        infection=profile["current_eye_infection_or_inflammation"],
        optom_review_flag=False,
        autorefractor_re=ar_re,
        autorefractor_le=ar_le,
        lenso_re=lenso_re,
        lenso_le=lenso_le,
        lenso_add_r=truth["add_r"],
        lenso_add_l=truth["add_l"],
    )


def generate_case(case_id, profile, calibration, rng_seed=1):
    rng = random.Random(rng_seed)

    truth = generate_truth_rx(rng, profile)
    quality = "large_mismatch" if profile["profile_id"] == "large_mismatch_candidate" else "normal"

    ar_re = generate_measurement_from_truth(
        rng, truth["re_sph"], truth["re_cyl"], truth["re_axis"], quality=quality
    )
    ar_le = generate_measurement_from_truth(
        rng, truth["le_sph"], truth["le_cyl"], truth["le_axis"], quality=quality
    )
    lenso_re = generate_measurement_from_truth(
        rng, truth["re_sph"], truth["re_cyl"], truth["re_axis"], quality=quality
    )
    lenso_le = generate_measurement_from_truth(
        rng, truth["le_sph"], truth["le_cyl"], truth["le_axis"], quality=quality
    )

    patient = build_patient_input(case_id, profile, truth, ar_re, ar_le, lenso_re, lenso_le)

    dv_engine = DerivedVariablesEngine(calibration)
    dv = dv_engine.derive(patient)

    return SyntheticCase(
        case_id=case_id,
        profile_id=profile["profile_id"],
        patient_input=patient,
        truth_rx=truth,
        ar_re=ar_re,
        ar_le=ar_le,
        lenso_re=lenso_re,
        lenso_le=lenso_le,
        lenso_add_r=truth["add_r"],
        lenso_add_l=truth["add_l"],
        dv=dv,
    )