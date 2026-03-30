from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from fsm.config.calibration_loader import CalibrationLoader
from fsm.models.derived_variables import DerivedVariables
from fsm.models.patient import PatientInput
from fsm.models.prescription import EyePrescription


class DerivedVariablesEngine:
    def __init__(self, calibration: CalibrationLoader):
        self.cal = calibration

    def _round_axis_to_phoropter_step(self, axis: Optional[float]) -> Optional[float]:
        if axis is None:
            return None
        step = float(self.cal.get("axis_fixed_step", 5))
        if step <= 0:
            return float(axis)
        rounded = round(float(axis) / step) * step
        wrapped = rounded % 180
        return 180.0 if abs(wrapped) < 1e-9 else float(wrapped)

    def derive(self, patient: PatientInput) -> DerivedVariables:
        age_bucket = self._derive_age_bucket(patient)
        distance_priority = self._derive_distance_priority(patient)
        near_priority = self._derive_near_priority(patient)
        symptom_risk = self._derive_symptom_risk(patient)
        medical_risk = self._derive_medical_risk(patient)
        stability = self._derive_stability(patient)

        mismatch_re = self._derive_mismatch_level(
            patient.autorefractor_re,
            patient.lenso_re,
            eye="re",
        )
        mismatch_le = self._derive_mismatch_level(
            patient.autorefractor_le,
            patient.lenso_le,
            eye="le",
        )

        start_policy = self._derive_start_source_policy(
            patient=patient,
            stability=stability,
            mismatch_re=mismatch_re,
            mismatch_le=mismatch_le,
        )

        start_re = self._derive_start_rx(
            ar=patient.autorefractor_re,
            lenso=patient.lenso_re,
            policy=start_policy,
        )
        start_le = self._derive_start_rx(
            ar=patient.autorefractor_le,
            lenso=patient.lenso_le,
            policy=start_policy,
        )

        add_expected = self._derive_add_expected(
            patient=patient,
            near_priority=near_priority,
        )
        target_distance_va = self._derive_target_distance_va(
            patient=patient,
            symptom_risk=symptom_risk,
            medical_risk=medical_risk,
            stability=stability,
        )
        endpoint_bias = self._derive_endpoint_bias_policy(
            patient=patient,
            age_bucket=age_bucket,
            near_priority=near_priority,
            symptom_risk=symptom_risk,
        )
        step_size_policy = self._derive_step_size_policy(
            stability=stability,
            medical_risk=medical_risk,
            symptom_risk=symptom_risk,
        )

        max_delta_start = self._derive_max_delta_from_start_sph(step_size_policy)
        max_delta_ar = self._derive_max_delta_from_ar_sph(medical_risk)
        branching_guardrails = self._derive_branching_guardrails(
            symptom_risk=symptom_risk,
            medical_risk=medical_risk,
            stability=stability,
        )
        axis_tolerance = self._derive_axis_tolerance(branching_guardrails)
        cyl_tolerance = self._derive_cyl_tolerance(step_size_policy)
        requires_review = self._derive_requires_optom_review(
            patient=patient,
            symptom_risk=symptom_risk,
            medical_risk=medical_risk,
        )
        anomaly_watch = self._derive_anomaly_watch(
            stability=stability,
            start_policy=start_policy,
            mismatch_re=mismatch_re,
            mismatch_le=mismatch_le,
        )
        expected_convergence = self._derive_expected_convergence_time(
            age_bucket=age_bucket,
            stability=stability,
            symptom_risk=symptom_risk,
            medical_risk=medical_risk,
            start_policy=start_policy,
            start_re=start_re,
            start_le=start_le,
        )
        confidence_requirement = self._derive_confidence_requirement(
            stability=stability,
            symptom_risk=symptom_risk,
            medical_risk=medical_risk,
        )
        fog_policy = self._derive_fogging_policy(
            age_bucket=age_bucket,
            stability=stability,
            medical_risk=medical_risk,
            symptom_risk=symptom_risk,
        )
        fog_amount = self._derive_fogging_amount(fog_policy)
        fog_clearance_mode = self._derive_fogging_clearance_mode(
            fog_policy=fog_policy,
            target_distance_va=target_distance_va,
        )
        fog_confirm = self._derive_fogging_required_confirmation(
            confidence_requirement=confidence_requirement
        )
        axis_step_policy = self._derive_axis_step_policy(
            symptom_risk=symptom_risk,
            medical_risk=medical_risk,
            stability=stability,
            start_policy=start_policy,
            start_re=start_re,
            start_le=start_le,
        )
        duochrome_max_flips = self._derive_duochrome_max_flips(branching_guardrails)
        near_test_required = self._derive_near_test_required(
            near_priority=near_priority,
            add_expected=add_expected,
            age_bucket=age_bucket,
            primary_reason=patient.primary_reason or "",
        )

        # FSM v2.3 additions
        jcc_axis_same_required = int(
            self.cal.get("jcc_axis_same_required", 2)
        )

        jcc_axis_max_flips = int(
            self.cal.get("jcc_axis_max_flips", 3)
        )

        axis_lane_re = self._derive_axis_lane_metadata(
            ar=patient.autorefractor_re,
            lenso=patient.lenso_re,
            policy=start_policy,
            start_rx=start_re,
        )
        axis_lane_le = self._derive_axis_lane_metadata(
            ar=patient.autorefractor_le,
            lenso=patient.lenso_le,
            policy=start_policy,
            start_rx=start_le,
        )

        near_binoc_step = float(
            self.cal.get("near_binoc_step_D", 0.25)
        )

        near_binoc_max_plus = int(
            self.cal.get("near_binoc_max_plus_steps", 4)
        )

        near_binoc_max_minus = int(
            self.cal.get("near_binoc_max_minus_steps", 4)
        )

        return DerivedVariables(
            visit_id=patient.visit_id,
            dv_age_bucket=age_bucket,
            dv_distance_priority=distance_priority,
            dv_near_priority=near_priority,
            dv_symptom_risk_level=symptom_risk,
            dv_medical_risk_level=medical_risk,
            dv_stability_level=stability,
            dv_ar_lenso_mismatch_level_RE=mismatch_re,
            dv_ar_lenso_mismatch_level_LE=mismatch_le,
            dv_start_source_policy=start_policy,
            dv_start_rx_RE_sph=start_re.sphere,
            dv_start_rx_RE_cyl=start_re.cylinder,
            dv_start_rx_RE_axis=start_re.axis,
            dv_start_rx_LE_sph=start_le.sphere,
            dv_start_rx_LE_cyl=start_le.cylinder,
            dv_start_rx_LE_axis=start_le.axis,
            dv_add_expected=add_expected,
            dv_target_distance_va=target_distance_va,
            dv_endpoint_bias_policy=endpoint_bias,
            dv_step_size_policy=step_size_policy,
            dv_max_delta_from_start_sph=max_delta_start,
            dv_max_delta_from_ar_sph=max_delta_ar,
            dv_axis_tolerance_deg=axis_tolerance,
            dv_cyl_tolerance_D=cyl_tolerance,
            dv_requires_optom_review=requires_review,
            dv_anomaly_watch=anomaly_watch,
            dv_expected_convergence_time=expected_convergence,
            dv_branching_guardrails=branching_guardrails,
            dv_confidence_requirement=confidence_requirement,
            dv_fogging_policy=fog_policy,
            dv_fogging_amount_D=fog_amount,
            dv_fogging_clearance_mode=fog_clearance_mode,
            dv_fogging_required_confirmation=fog_confirm,
            dv_axis_step_policy=axis_step_policy,
            dv_duochrome_max_flips=duochrome_max_flips,
            dv_near_test_required=near_test_required,

            # FSM v3.1 additions
            dv_accommodation_level=age_bucket,
            # A positive low-fog amount should still be applied at coarse entry
            # even when the policy label is "Low_Fog".
            dv_fogging_required=(fog_amount > 0.0),
            dv_fogging_stop_at_target_va=True,

            dv_jcc_axis_same_required=jcc_axis_same_required,
            dv_jcc_axis_max_flips=jcc_axis_max_flips,
            # Retained for compatibility with older exports/debug consumers.
            # Lane 4 corresponds to the old "quick search" behavior family.
            dv_quick_axis_search_RE=(axis_lane_re["lane_id"] == "LANE_4"),
            dv_quick_axis_search_LE=(axis_lane_le["lane_id"] == "LANE_4"),
            dv_axis_source_used_RE=axis_lane_re["source_used"],
            dv_axis_source_used_LE=axis_lane_le["source_used"],
            dv_axis_cyl_magnitude_for_lane_RE=axis_lane_re["cyl_magnitude"],
            dv_axis_cyl_magnitude_for_lane_LE=axis_lane_le["cyl_magnitude"],
            dv_axis_is_near_cardinal_RE=axis_lane_re["is_near_cardinal"],
            dv_axis_is_near_cardinal_LE=axis_lane_le["is_near_cardinal"],
            dv_axis_lane_id_RE=axis_lane_re["lane_id"],
            dv_axis_lane_id_LE=axis_lane_le["lane_id"],
            dv_axis_lane_name_RE=axis_lane_re["lane_name"],
            dv_axis_lane_name_LE=axis_lane_le["lane_name"],
            dv_axis_step_sequence_RE=axis_lane_re["sequence_text"],
            dv_axis_step_sequence_LE=axis_lane_le["sequence_text"],
            dv_axis_confidence_label_RE=axis_lane_re["confidence_label"],
            dv_axis_confidence_label_LE=axis_lane_le["confidence_label"],
            dv_axis_selection_reason_RE=axis_lane_re["selection_reason"],
            dv_axis_selection_reason_LE=axis_lane_le["selection_reason"],

            dv_near_binoc_step_D=near_binoc_step,
            dv_near_binoc_max_plus_steps=near_binoc_max_plus,
            dv_near_binoc_max_minus_steps=near_binoc_max_minus,
        )

    def derive_as_dict(self, patient: PatientInput) -> dict:
        return asdict(self.derive(patient))

    def _derive_age_bucket(self, patient: PatientInput) -> str:
        age = patient.age or 0
        child_max = self.cal.get("age_child_max", 17)
        pres_min = self.cal.get("age_presbyope_min", 40)
        if age < child_max:
            return "Child"
        if age >= pres_min:
            return "Presbyope"
        return "Adult"

    def _derive_distance_priority(self, patient: PatientInput) -> str:
        drive = patient.driving_hours or 0
        occ = (patient.occupation or "").strip()
        priority = (patient.priority or "").strip()
        drive_high = self.cal.get("driving_hours_high", 2)

        base = "High" if (drive >= drive_high or occ == "Driver") else "Medium"
        if priority == "Comfort-first":
            return "Medium" if base == "High" else "Low"
        return base

    def _derive_near_priority(self, patient: PatientInput) -> str:
        declared = (patient.near_priority_declared or "").strip()
        screen = patient.screen_time_hours or 0
        reason = (patient.primary_reason or "").strip()
        screen_high = self.cal.get("screen_time_high", 6)

        if declared:
            return declared
        if screen >= screen_high or reason == "Blurred near":
            return "High"
        return "Low"

    def _derive_symptom_risk(self, patient: PatientInput) -> str:
        s = (patient.symptoms_text or "").strip()
        high_terms = ["Sudden vision change", "Double vision"]
        moderate_terms = ["Glare/halos", "Night driving difficulty", "Fluctuating vision"]

        if any(term in s for term in high_terms):
            return "High"
        if any(term in s for term in moderate_terms):
            return "Moderate"
        return "None"

    def _derive_medical_risk(self, patient: PatientInput) -> str:
        surgery = (patient.prior_eye_surgery or "").strip()
        if patient.keratoconus or patient.infection:
            return "High"
        if patient.diabetes or surgery not in ("", "None") or patient.amblyopia:
            return "Moderate"
        return "None"

    def _derive_stability(self, patient: PatientInput) -> str:
        last = patient.last_eye_test_months_ago or 0
        uncertain_months = self.cal.get("last_eye_test_uncertain_months", 24)
        if patient.fluctuating_vision_reported or patient.rx_change_was_large:
            return "Unstable"
        if patient.diabetes or last > uncertain_months:
            return "Uncertain"
        return "Stable"

    def _derive_mismatch_level(
        self,
        ar: Optional[EyePrescription],
        lenso: Optional[EyePrescription],
        eye: str,
    ) -> str:
        if ar is None or lenso is None or not ar.has_full_rx() or not lenso.has_full_rx():
            return "Small"

        d_s = abs(float(ar.sphere) - float(lenso.sphere))
        d_c = abs(float(ar.cylinder) - float(lenso.cylinder))
        d_a_raw = abs(float(ar.axis) - float(lenso.axis))
        d_a = min(d_a_raw, 180 - d_a_raw)

        s_med = self.cal.get(f"mismatch_{eye}_sph_medium", 1)
        s_large = self.cal.get(f"mismatch_{eye}_sph_large", 2)
        c_med = self.cal.get(f"mismatch_{eye}_cyl_medium", 0.5)
        c_large = self.cal.get(f"mismatch_{eye}_cyl_large", 1)
        a_med = self.cal.get(f"mismatch_{eye}_axis_medium", 15)
        a_large = self.cal.get(f"mismatch_{eye}_axis_large", 30)

        if d_s > s_large or d_c > c_large or d_a > a_large:
            return "Large"
        if d_s > s_med or d_c > c_med or d_a > a_med:
            return "Medium"
        return "Small"

    def _derive_start_source_policy(
        self,
        patient: PatientInput,
        stability: str,
        mismatch_re: str,
        mismatch_le: str,
    ) -> str:
        sat = (patient.satisfaction_with_current_rx or "").strip()
        reason = (patient.primary_reason or "").strip()

        start_ar = self.cal.get("start_policy_if_only_ar", "Start_AR")
        start_lenso = self.cal.get("start_policy_if_only_lenso", "Start_Lenso")
        start_sat = self.cal.get("start_policy_if_satisfied_small_mismatch", "Start_Lenso")
        start_unsat = self.cal.get("start_policy_if_unsatisfied_or_blur", "Start_AR")
        start_hybrid = self.cal.get("start_policy_if_large_mismatch_or_unstable", "Hybrid")

        has_ar = (
            patient.autorefractor_re is not None and patient.autorefractor_re.sphere is not None
        ) or (
            patient.autorefractor_le is not None and patient.autorefractor_le.sphere is not None
        )
        has_lenso = (
            patient.lenso_re is not None and patient.lenso_re.sphere is not None
        ) or (
            patient.lenso_le is not None and patient.lenso_le.sphere is not None
        )

        if has_ar and not has_lenso:
            return start_ar
        if has_lenso and not has_ar:
            return start_lenso
        if sat == "Satisfied" and mismatch_re == "Small" and mismatch_le == "Small":
            return start_sat
        if sat == "Not satisfied" or reason in ("Blurred distance", "Blurred near"):
            return start_unsat
        if mismatch_re == "Large" or mismatch_le == "Large" or stability == "Unstable":
            return start_hybrid
        return start_ar

    def _derive_start_rx(
        self,
        ar: Optional[EyePrescription],
        lenso: Optional[EyePrescription],
        policy: str,
    ) -> EyePrescription:
        step = float(self.cal.get("hybrid_rounding_step", 0.25))

        def round_to_step(value: float, round_step: float) -> float:
            return round(value / round_step) * round_step

        ar_s = ar.sphere if ar else None
        ar_c = ar.cylinder if ar else None
        ar_a = ar.axis if ar else None
        ln_s = lenso.sphere if lenso else None
        ln_c = lenso.cylinder if lenso else None
        ln_a = lenso.axis if lenso else None

        if policy == "Start_AR":
            return EyePrescription(ar_s, ar_c, self._round_axis_to_phoropter_step(ar_a))
        if policy == "Start_Lenso":
            return EyePrescription(
                ln_s if ln_s is not None else ar_s,
                ln_c if ln_c is not None else ar_c,
                ln_a if ln_a is not None else self._round_axis_to_phoropter_step(ar_a),
            )

        # Hybrid
        sph = (
            round_to_step((ar_s + ln_s) / 2, step)
            if ar_s is not None and ln_s is not None
            else ar_s if ar_s is not None else ln_s
        )
        cyl = (
            round_to_step((ar_c + ln_c) / 2, step)
            if ar_c is not None and ln_c is not None
            else ar_c if ar_c is not None else ln_c
        )
        axis = (
            self._round_axis_to_phoropter_step(ar_a)
            if ar_a is not None
            else ln_a
        )
        return EyePrescription(sph, cyl, axis)

    def _derive_add_expected(self, patient: PatientInput, near_priority: str) -> str:
        age = patient.age or 0
        wear_type = (patient.wear_type or "").strip()
        add_r = patient.lenso_add_r
        add_l = patient.lenso_add_l
        likely_age = self.cal.get("add_likely_min_age", 42)
        possible_age = self.cal.get("add_possible_min_age", 38)

        # Strong evidence of near add
        if (
            wear_type in ("Progressive", "Bifocal")
            or (add_r is not None and add_r > 0)
            or (add_l is not None and add_l > 0)
        ):
            return "Likely"

        # Age-based likelihood ONLY if near demand exists
        if age >= likely_age and near_priority == "High":
            return "Likely"

        # Possible only for borderline age with high near demand
        if possible_age <= age < likely_age and near_priority == "High":
            return "Possible"

        return "None"

    def _derive_target_distance_va(
        self,
        patient: PatientInput,
        symptom_risk: str,
        medical_risk: str,
        stability: str,
    ) -> str:
        target_66 = self.cal.get("default_distance_target", "6/6_target")
        target_69 = self.cal.get("high_risk_distance_target", "6/9_acceptable")

        if symptom_risk == "High" or medical_risk == "High" or stability == "Unstable":
            return target_69
        return target_66

    def _derive_endpoint_bias_policy(
        self,
        patient: PatientInput,
        age_bucket: str,
        near_priority: str,
        symptom_risk: str,
    ) -> str:
        pri = (patient.priority or "").strip()
        occ = (patient.occupation or "").strip()
        symptoms = (patient.symptoms_text or "").strip()

        comfort_bias = self.cal.get("comfort_presbyope_high_near_bias", "Undercorrect")
        driver_bias = self.cal.get("driver_night_issue_bias", "Overcorrect")
        symptom_bias = self.cal.get("high_symptom_bias", "Neutral")

        if symptom_risk == "High":
            return symptom_bias
        if pri == "Comfort-first" and age_bucket == "Presbyope" and near_priority == "High":
            return comfort_bias
        if occ == "Driver" and (
            "Night driving difficulty" in symptoms or "Glare/halos" in symptoms
        ):
            return driver_bias
        return "Neutral"

    def _derive_step_size_policy(
        self,
        stability: str,
        medical_risk: str,
        symptom_risk: str,
    ) -> str:
        if stability == "Unstable" or medical_risk == "High" or symptom_risk == "High":
            return "Conservative"
        if stability == "Stable" and medical_risk == "None" and symptom_risk == "None":
            return "Aggressive"
        return "Standard"

    def _derive_max_delta_from_start_sph(self, step_size_policy: str) -> float:
        if step_size_policy == "Conservative":
            return float(self.cal.get("max_delta_start_conservative", 1.25))
        if step_size_policy == "Standard":
            return float(self.cal.get("max_delta_start_standard", 2))
        return float(self.cal.get("max_delta_start_aggressive", 2.5))

    def _derive_max_delta_from_ar_sph(self, medical_risk: str) -> float:
        if medical_risk == "High":
            return float(self.cal.get("max_delta_ar_high_medical", 2))
        return float(self.cal.get("max_delta_ar_standard", 3))

    def _derive_axis_tolerance(self, branching_guardrails: str) -> float:
        if branching_guardrails == "Strict":
            return float(self.cal.get("axis_tol_strict", 1))
        if branching_guardrails == "Normal":
            return float(self.cal.get("axis_tol_normal", 3))
        return float(self.cal.get("axis_tol_relaxed", 5))

    def _derive_cyl_tolerance(self, step_size_policy: str) -> float:
        if step_size_policy == "Aggressive":
            return float(self.cal.get("cyl_tol_aggressive", 0.5))
        if step_size_policy == "Conservative":
            return float(self.cal.get("cyl_tol_conservative", 0.25))
        return float(self.cal.get("cyl_tol_standard", 0.25))

    def _derive_requires_optom_review(
        self,
        patient: PatientInput,
        symptom_risk: str,
        medical_risk: str,
    ) -> bool:
        patient_flag = bool(patient.optom_review_flag)
        sym_policy = bool(self.cal.get("immediate_review_if_high_symptom", False))
        med_policy = bool(self.cal.get("immediate_review_if_high_medical", False))

        return (
            (symptom_risk == "High" and sym_policy)
            or (medical_risk == "High" and med_policy)
            or patient_flag
        )

    def _derive_anomaly_watch(
        self,
        stability: str,
        start_policy: str,
        mismatch_re: str,
        mismatch_le: str,
    ) -> bool:
        watch_hybrid = bool(self.cal.get("anomaly_watch_if_hybrid", True))
        watch_unstable = bool(self.cal.get("anomaly_watch_if_unstable", True))
        watch_mismatch = bool(self.cal.get("anomaly_watch_if_large_mismatch", True))

        return (
            (start_policy == "Hybrid" and watch_hybrid)
            or (stability != "Stable" and watch_unstable)
            or ((mismatch_re == "Large" or mismatch_le == "Large") and watch_mismatch)
        )

    def _derive_expected_convergence_time(
        self,
        age_bucket: str,
        stability: str,
        symptom_risk: str,
        medical_risk: str,
        start_policy: str,
        start_re: EyePrescription,
        start_le: EyePrescription,
    ) -> str:
        if self._use_slow_axis_policy_for_ar_start(start_policy, start_re, start_le):
            return "Slow"
        if age_bucket == "Adult" and stability == "Stable" and symptom_risk == "None" and medical_risk == "None":
            return "Fast"
        if age_bucket == "Presbyope" or stability != "Stable" or symptom_risk == "High" or medical_risk == "High":
            return "Slow"
        return "Normal"

    def _derive_branching_guardrails(
        self,
        symptom_risk: str,
        medical_risk: str,
        stability: str,
    ) -> str:
        if symptom_risk == "High" or medical_risk == "High" or stability == "Unstable":
            return "Strict"
        if symptom_risk == "Moderate" or medical_risk == "Moderate" or stability == "Uncertain":
            return "Normal"
        return "Relaxed"

    def _derive_confidence_requirement(
        self,
        stability: str,
        symptom_risk: str,
        medical_risk: str,
    ) -> str:
        if stability != "Stable" or symptom_risk == "High" or medical_risk == "High":
            return "High"
        if symptom_risk == "Moderate" or medical_risk == "Moderate":
            return "Medium"
        return "Low"

    def _derive_fogging_policy(
        self,
        age_bucket: str,
        stability: str,
        medical_risk: str,
        symptom_risk: str,
    ) -> str:
        # FSM v3.1 corrected fogging logic
        # Fogging is driven by accommodation risk (young patients) and instability,
        # NOT by presbyopia.

        # High accommodation risk → strong fog
        if age_bucket == "Child":
            return "Strong_Fog"

        # Moderate accommodation / instability → standard fog
        if (
            age_bucket == "Adult"
            and (
                stability != "Stable"
                or medical_risk == "Moderate"
                or symptom_risk == "Moderate"
            )
        ):
            return "Standard_Fog"

        # High clinical risk → standard fog (controlled, not aggressive)
        if medical_risk == "High" or symptom_risk == "High":
            return "Standard_Fog"

        # Presbyopes → low fog (low accommodation, but still a mild plus buffer)
        if age_bucket == "Presbyope":
            return "Low_Fog"

        return "Low_Fog"

    def _derive_fogging_amount(self, fog_policy: str) -> float:
        if fog_policy == "Strong_Fog":
            return float(self.cal.get("strong_fog_amount", 1))
        if fog_policy == "Standard_Fog":
            return float(self.cal.get("standard_fog_amount", 0.75))
        return float(self.cal.get("low_fog_amount", self.cal.get("no_fog_amount", 0)))

    def _derive_fogging_clearance_mode(
        self,
        fog_policy: str,
        target_distance_va: str,
    ) -> str:
        if fog_policy == "Strong_Fog" and target_distance_va == "6/9_acceptable":
            return str(self.cal.get("stepdown_mode_strong_6_9", "Early_Stop_if_Risk"))
        if fog_policy == "Strong_Fog":
            return str(self.cal.get("stepdown_mode_strong_6_6", "StepDown_0.25"))
        return str(self.cal.get("stepdown_mode_standard", "StepDown_0.50_then_0.25"))

    def _derive_fogging_required_confirmation(self, confidence_requirement: str) -> str:
        if confidence_requirement == "Low":
            return "Low"
        if confidence_requirement == "Medium":
            return "Medium"
        return "High"

    def _derive_axis_step_policy(
        self,
        symptom_risk: str,
        medical_risk: str,
        stability: str,
        start_policy: str,
        start_re: EyePrescription,
        start_le: EyePrescription,
    ) -> str:
        if self._use_slow_axis_policy_for_ar_start(start_policy, start_re, start_le):
            return "Slow"
        if symptom_risk == "High" or medical_risk == "High" or stability == "Unstable":
            return "Slow"
        if symptom_risk == "None" and medical_risk == "None" and stability == "Stable":
            return "Fast"
        return "Normal"

    def _derive_axis_lane_metadata(
        self,
        ar: Optional[EyePrescription],
        lenso: Optional[EyePrescription],
        policy: str,
        start_rx: EyePrescription,
    ) -> dict:
        near_cardinal_threshold = float(self.cal.get("axis_near_cardinal_threshold_deg", 10))
        low_cyl_threshold = float(self.cal.get("axis_low_cyl_threshold_d", 0.50))
        meaningful_cyl_threshold = float(self.cal.get("axis_meaningful_cyl_threshold_d", 0.75))

        source_used = self._derive_axis_source_used(ar=ar, lenso=lenso, policy=policy)
        axis_for_start = self._axis_value_for_source(
            ar=ar,
            lenso=lenso,
            start_rx=start_rx,
            source_used=source_used,
        )
        lenso_axis_available = bool(lenso and lenso.axis is not None)
        ar_axis_available = bool(ar and ar.axis is not None)
        lenso_cyl_mag = abs(float(lenso.cylinder)) if lenso and lenso.cylinder is not None else 0.0
        ar_cyl_mag = abs(float(ar.cylinder)) if ar and ar.cylinder is not None else 0.0
        decision_cyl_mag = lenso_cyl_mag if lenso_axis_available else ar_cyl_mag
        if not lenso_axis_available and not ar_axis_available:
            decision_cyl_mag = abs(float(start_rx.cylinder or 0.0))

        is_near_cardinal = self._is_near_cardinal_axis(
            axis_for_start,
            threshold_deg=near_cardinal_threshold,
        )

        reasons = [f"axis_source={source_used or 'Unknown'}"]
        if lenso_axis_available:
            reasons.append("lensometer_axis_available")
        else:
            reasons.append("ar_only_axis")
        if decision_cyl_mag >= meaningful_cyl_threshold:
            reasons.append("meaningful_cylinder")
        elif decision_cyl_mag < low_cyl_threshold:
            reasons.append("low_cylinder")
        else:
            reasons.append("intermediate_cylinder")
        reasons.append("near_cardinal_axis" if is_near_cardinal else "non_cardinal_axis")

        if lenso_axis_available and lenso_cyl_mag >= meaningful_cyl_threshold:
            lane_id = "LANE_1"
            lane_name = "High-confidence axis"
            confidence_label = "High"
            reasons.insert(0, "lensometer_supported_axis")
        elif (not lenso_axis_available) and ar_axis_available and ar_cyl_mag >= meaningful_cyl_threshold and not is_near_cardinal:
            lane_id = "LANE_2"
            lane_name = "Medium-confidence axis"
            confidence_label = "Medium"
            reasons.insert(0, "ar_only_meaningful_non_cardinal")
        elif (not lenso_axis_available) and ar_axis_available and is_near_cardinal and ar_cyl_mag < low_cyl_threshold:
            lane_id = "LANE_4"
            lane_name = "Very low-confidence axis"
            confidence_label = "Very Low"
            reasons.insert(0, "ar_only_near_cardinal_low_cylinder")
        else:
            lane_id = "LANE_3"
            lane_name = "Low-confidence axis"
            confidence_label = "Low"
            reasons.insert(0, "conservative_fallback_lane")

        sequence = self._read_axis_lane_sequence(
            lane_id=lane_id,
            default=[5.0],
        )

        return {
            "source_used": source_used,
            "cyl_magnitude": round(float(decision_cyl_mag), 4),
            "is_near_cardinal": is_near_cardinal,
            "lane_id": lane_id,
            "lane_name": lane_name,
            "confidence_label": confidence_label,
            "sequence_text": self._format_axis_step_sequence(sequence),
            "selection_reason": "; ".join(reasons),
        }

    def _derive_axis_source_used(
        self,
        ar: Optional[EyePrescription],
        lenso: Optional[EyePrescription],
        policy: str,
    ) -> str:
        if policy == "Start_Lenso":
            if lenso and lenso.axis is not None:
                return "Lenso"
            if ar and ar.axis is not None:
                return "AR"

        if policy in ("Start_AR", "Hybrid"):
            if ar and ar.axis is not None:
                return "AR"
            if lenso and lenso.axis is not None:
                return "Lenso"

        if lenso and lenso.axis is not None:
            return "Lenso"
        if ar and ar.axis is not None:
            return "AR"
        return "Unknown"

    def _axis_value_for_source(
        self,
        ar: Optional[EyePrescription],
        lenso: Optional[EyePrescription],
        start_rx: EyePrescription,
        source_used: str,
    ) -> Optional[float]:
        if source_used == "Lenso" and lenso:
            return lenso.axis
        if source_used == "AR" and ar:
            return ar.axis
        return start_rx.axis

    def _read_axis_lane_sequence(self, lane_id: str, default: list[float]) -> list[float]:
        raw = self.cal.get(f"{str(lane_id).lower()}_sequence_deg", "")
        if not raw:
            fallback_map = {
                "LANE_1": "10,5",
                "LANE_2": "20,10,5",
                "LANE_3": "30,20,10,5",
                "LANE_4": "45,30,20,10,5",
            }
            raw = fallback_map.get(lane_id, "")

        sequence = []
        for token in str(raw).replace("|", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = float(token)
            except ValueError:
                continue
            if value > 0:
                sequence.append(value)

        return sequence or list(default)

    def _format_axis_step_sequence(self, sequence: list[float]) -> str:
        parts = []
        for step in sequence:
            step_value = float(step)
            if abs(step_value - round(step_value)) < 1e-9:
                parts.append(str(int(round(step_value))))
            else:
                parts.append(f"{step_value:g}")
        return ",".join(parts)

    def _use_slow_axis_policy_for_ar_start(
        self,
        start_policy: str,
        start_re: EyePrescription,
        start_le: EyePrescription,
    ) -> bool:
        if start_policy != "Start_AR":
            return False
        return self._is_cardinal_axis(start_re.axis) or self._is_cardinal_axis(start_le.axis)

    def _is_cardinal_axis(self, axis: Optional[float]) -> bool:
        if axis is None:
            return False
        return abs(float(axis) % 180.0) < 1e-9

    def _is_near_cardinal_axis(self, axis: Optional[float], threshold_deg: float) -> bool:
        if axis is None:
            return False
        axis_value = float(axis) % 180.0
        distances = (
            abs(axis_value - 0.0),
            abs(axis_value - 90.0),
            abs(axis_value - 180.0),
        )
        return min(distances) <= float(threshold_deg)

    def _derive_duochrome_max_flips(self, branching_guardrails: str) -> int:
        if branching_guardrails == "Strict":
            return int(self.cal.get("duochrome_max_flips_strict", 3))
        if branching_guardrails == "Normal":
            return int(self.cal.get("duochrome_max_flips_normal", 4))
        return int(self.cal.get("duochrome_max_flips_relaxed", 5))

    def _derive_near_test_required(
        self,
        near_priority: str,
        add_expected: str,
        age_bucket: str,
        primary_reason: str,
    ) -> bool:
        allow_possible = bool(self.cal.get("trigger_near_if_dv_add_expected_possible", True))
        allow_likely = bool(self.cal.get("trigger_near_if_dv_add_expected_likely", True))
        allow_force = bool(self.cal.get("allow_high_near_priority_to_force_near", False))

        return (
            (add_expected == "Likely" and allow_likely)
            or (add_expected == "Possible" and near_priority == "High" and allow_possible)
            or (age_bucket == "Presbyope" and near_priority == "High")
            or (near_priority == "High" and allow_force)
            or ("near" in primary_reason.lower())
        )
