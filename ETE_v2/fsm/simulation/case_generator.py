import random
from dataclasses import dataclass

from fsm.engines.derived_variables_engine import DerivedVariablesEngine
from fsm.models.patient import PatientInput
from fsm.models.prescription import EyePrescription


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


def get_measurement_qualities(profile):
    """
    Return separate measurement-quality modes for AR and lenso so the synthetic
    population does not accidentally create too many hidden edge cases.
    """
    profile_id = profile["profile_id"]
    satisfied = profile.get("satisfaction_with_current_rx", "") == "Satisfied"
    edge_case = bool(profile.get("edge_case", False))

    ar_quality = "normal"
    lenso_quality = "normal"

    if profile_id == "large_mismatch_candidate":
        ar_quality = "large_mismatch"
        lenso_quality = "tight"
        return ar_quality, lenso_quality

    if edge_case:
        ar_quality = "moderate_mismatch"

    if satisfied or profile_id in (
        "comfortable_current_glasses",
        "presbyope_distance_priority",
        "presbyope_near_priority",
    ):
        lenso_quality = "tight"

    return ar_quality, lenso_quality


def generate_add_from_age(rng, age_years, profile):
    """
    Generate near add only if BOTH:
    1. Age supports presbyopia
    2. Profile indicates real near demand
    """

    # --- near demand signals ---
    near_priority = profile.get("near_priority", "")
    symptoms = profile.get("symptoms_multi", "").lower()
    screen_time = profile.get("screen_time_hours", 0.0)

    has_near_demand = (
        near_priority == "High"
        or "near" in symptoms
        or screen_time >= 8.0
    )

    if not has_near_demand:
        return 0.0

    # --- age-based add only if demand exists ---
    if age_years >= 55:
        return round_quarter(rng.uniform(1.50, 2.50))
    if age_years >= 50:
        return round_quarter(rng.uniform(1.25, 2.00))
    if age_years >= 45:
        return round_quarter(rng.uniform(0.75, 1.50))

    return 0.0


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

    add_r = generate_add_from_age(rng, profile["age_years"], profile)
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
    if quality == "tight":
        sph_noise = rng.uniform(-0.25, 0.25)
        cyl_noise = rng.uniform(-0.25, 0.25)
        axis_noise = rng.randint(-5, 5)
    elif quality == "moderate_mismatch":
        sph_noise = rng.uniform(-0.75, 0.75)
        cyl_noise = rng.uniform(-0.75, 0.75)
        axis_noise = rng.randint(-15, 15)
    elif quality == "large_mismatch":
        sph_noise = rng.uniform(-1.5, 1.5)
        cyl_noise = rng.uniform(-1.0, 1.0)
        axis_noise = rng.randint(-20, 20)
    else:
        sph_noise = rng.uniform(-0.375, 0.375)
        cyl_noise = rng.uniform(-0.375, 0.375)
        axis_noise = rng.randint(-8, 8)

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
    ar_quality, lenso_quality = get_measurement_qualities(profile)

    ar_re = generate_measurement_from_truth(
        rng, truth["re_sph"], truth["re_cyl"], truth["re_axis"], quality=ar_quality
    )
    ar_le = generate_measurement_from_truth(
        rng, truth["le_sph"], truth["le_cyl"], truth["le_axis"], quality=ar_quality
    )
    lenso_re = generate_measurement_from_truth(
        rng, truth["re_sph"], truth["re_cyl"], truth["re_axis"], quality=lenso_quality
    )
    lenso_le = generate_measurement_from_truth(
        rng, truth["le_sph"], truth["le_cyl"], truth["le_axis"], quality=lenso_quality
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
