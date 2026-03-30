from copy import deepcopy


BASE_PROFILE = {
    "age_years": 28,
    "occupation_type": "Office",
    "screen_time_hours": 4.0,
    "driving_time_hours": 0.5,
    "primary_reason": "Blurred distance",
    "symptoms_multi": "Blurred distance",
    "satisfaction_with_current_rx": "Not satisfied",
    "wear_type": "Single Vision",
    "distance_target": "",
    "priority": "Balanced",
    "near_priority": "",
    "last_eye_test_months_ago": 12.0,
    "rx_change_was_large": False,
    "fluctuating_vision_reported": False,
    "diabetes": False,
    "prior_eye_surgery": "None",
    "known_keratoconus": False,
    "known_amblyopia": False,
    "current_eye_infection_or_inflammation": False,
}


def _make(name: str, population_weight: float, edge_case: bool = False, **updates):
    p = deepcopy(BASE_PROFILE)
    p.update(updates)
    p["profile_id"] = name
    p["population_weight"] = population_weight
    p["edge_case"] = edge_case
    return p


PROFILES = [
    # --- MAINSTREAM PROFILES (~90%) ---
    _make("young_stable_myope", 0.18, age_years=24, priority="Balanced", symptoms_multi="Blurred distance", primary_reason="Blurred distance"),
    _make("young_astigmat", 0.08, age_years=26, priority="Balanced", symptoms_multi="Blurred distance", primary_reason="Blurred distance"),
    _make("high_myope_stable", 0.08, age_years=31, priority="Distance-first", symptoms_multi="Blurred distance", primary_reason="Blurred distance"),
    _make("adult_hyperope", 0.10, age_years=34, priority="Balanced", symptoms_multi="Eye strain", primary_reason="Routine check"),
    _make("presbyope_distance_priority", 0.06, age_years=47, wear_type="Progressive", priority="Distance-first", near_priority="Medium", symptoms_multi="Blurred distance", primary_reason="Blurred distance"),
    _make("presbyope_near_priority", 0.04, age_years=52, wear_type="Progressive", priority="Comfort-first", near_priority="High", symptoms_multi="Blurred near", primary_reason="Blurred near"),
    _make("driver_night_symptoms", 0.05, age_years=37, occupation_type="Driver", driving_time_hours=5.0, priority="Distance-first", symptoms_multi="Night driving difficulty, Glare/halos", primary_reason="Blurred distance"),
    _make("screen_heavy_young_adult", 0.08, age_years=27, screen_time_hours=8.0, near_priority="Medium", symptoms_multi="Eye strain", primary_reason="Routine check"),
    _make("comfortable_current_glasses", 0.12, age_years=32, satisfaction_with_current_rx="Satisfied", symptoms_multi="", primary_reason="Routine check"),
    _make("dissatisfied_current_glasses", 0.08, age_years=32, satisfaction_with_current_rx="Not satisfied", symptoms_multi="Blurred distance", primary_reason="Blurred distance"),
    _make("near_demand_professional", 0.05, age_years=39, occupation_type="Designer", screen_time_hours=8.0, near_priority="High", priority="Balanced", symptoms_multi="Blurred near, Eye strain", primary_reason="Blurred near"),
    _make("conservative_comfort_first", 0.04, age_years=44, priority="Comfort-first", near_priority="Medium", symptoms_multi="Glare/halos", primary_reason="Routine check"),

    # --- EDGE CASES (~10%) ---
    _make("large_mismatch_candidate", 0.02, True, age_years=35, satisfaction_with_current_rx="Satisfied", symptoms_multi="Blurred distance", primary_reason="Blurred distance"),
    _make("moderate_anisometrope", 0.02, True, age_years=28, symptoms_multi="Blurred distance", primary_reason="Blurred distance"),
    _make("mild_anisometrope", 0.02, True, age_years=28, symptoms_multi="Blurred distance", primary_reason="Blurred distance"),
    _make("medical_risk_flagged", 0.01, True, age_years=42, diabetes=True, symptoms_multi="Blurred distance", primary_reason="Blurred distance"),
    _make("fluctuating_vision_adult", 0.01, True, age_years=33, fluctuating_vision_reported=True, symptoms_multi="Fluctuating vision", primary_reason="Blurred distance"),
    _make("high_symptom_stable", 0.01, True, age_years=30, symptoms_multi="Sudden vision change", primary_reason="Blurred distance"),
    _make("low_confidence_like_case", 0.01, True, age_years=29, priority="Comfort-first", symptoms_multi="Blurred distance", primary_reason="Blurred distance"),
]


def get_profiles():
    return [deepcopy(p) for p in PROFILES]