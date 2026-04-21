import sys
import unittest
import csv
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fsm.config.calibration_loader import CalibrationLoader
from fsm.audio.response_matching import (
    ENGLISH_VARIANT_LIBRARY,
    HINDI_VARIANT_LIBRARY,
    localized_option_label,
    localized_voice_prompt,
    match_response,
)
from fsm.engines.derived_variables_engine import DerivedVariablesEngine
from fsm.engines.refraction_fsm_engine import RefractionFSMEngine
from fsm.models.fsm_runtime import FSMRuntimeRow
from fsm.models.patient import PatientInput
from fsm.models.prescription import EyePrescription
from ete_io.outputs import build_session_metadata, write_voice_utterances_csv
from session_orchestrator import SessionOrchestrator
from api_server import app, sessions


CALIBRATION_PATH = ROOT / "config" / "calibration.csv"


class AxisLanePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = CalibrationLoader(CALIBRATION_PATH)

    def _rx(self, sphere: float, cylinder: float, axis: float) -> EyePrescription:
        return EyePrescription(sphere=sphere, cylinder=cylinder, axis=axis)

    def _patient(
        self,
        *,
        visit_id: str,
        ar_re: EyePrescription,
        lenso_re: Optional[EyePrescription],
        ar_le: Optional[EyePrescription] = None,
        lenso_le: Optional[EyePrescription] = None,
        satisfaction: str = "Not satisfied",
        primary_reason: str = "Blurred distance",
        age: int = 30,
        near_priority: str = "",
        wear_type: str = "",
        lenso_add_r: Optional[float] = None,
        lenso_add_l: Optional[float] = None,
    ) -> PatientInput:
        return PatientInput(
            visit_id=visit_id,
            age=age,
            primary_reason=primary_reason,
            satisfaction_with_current_rx=satisfaction,
            wear_type=wear_type,
            near_priority_declared=near_priority,
            driving_hours=1,
            screen_time_hours=2,
            last_eye_test_months_ago=12,
            autorefractor_re=ar_re,
            autorefractor_le=ar_le or self._rx(-1.25, -0.75, 80),
            lenso_re=lenso_re,
            lenso_le=lenso_le,
            lenso_add_r=lenso_add_r,
            lenso_add_l=lenso_add_l,
        )

    def _derive(self, patient: PatientInput):
        return DerivedVariablesEngine(self.calibration).derive(patient)

    def _derive_with_calibration_overrides(self, patient: PatientInput, **overrides):
        with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv", delete=False) as tmp:
            writer = csv.writer(tmp)
            with CALIBRATION_PATH.open(newline="") as source:
                reader = csv.DictReader(source)
                writer.writerow(reader.fieldnames)
                for row in reader:
                    if row["Parameter_Key"] in overrides:
                        row["Value"] = str(overrides[row["Parameter_Key"]])
                    writer.writerow([row[field] for field in reader.fieldnames])
            temp_path = Path(tmp.name)

        try:
            calibration = CalibrationLoader(temp_path)
            return DerivedVariablesEngine(calibration).derive(patient), calibration
        finally:
            temp_path.unlink(missing_ok=True)

    def _advance_to_axis_re(self, dv) -> FSMRuntimeRow:
        engine = RefractionFSMEngine(self.calibration)
        current = engine.initialize_row("axis-seq-visit", dv)
        guard = 0
        while current.state != "E":
            self.assertLess(guard, 12)
            finalized = engine.apply_response(current, "CLEAR", dv)
            next_row = engine._build_next_row(finalized, dv)
            self.assertIsNotNone(next_row)
            current = next_row
            guard += 1
        return current

    def _final_compare_seed_row(self, dv) -> FSMRuntimeRow:
        engine = RefractionFSMEngine(self.calibration)
        return engine._row_for_state(
            step=30,
            visit_id="final-compare",
            state="K",
            dv=dv,
            re_sph=-1.00,
            re_cyl=-0.50,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.75,
            add_l=0.75,
            chart_param="20_20_20",
            phase_step_count=1,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
            final_compare_enabled=True,
            final_compare_round=0,
            final_compare_option_source="Achieved",
            final_compare_current_source="PGP",
            final_compare_current_re_sph=-2.00,
            final_compare_current_re_cyl=-0.50,
            final_compare_current_re_axis=50,
            final_compare_current_le_sph=-1.50,
            final_compare_current_le_cyl=-0.25,
            final_compare_current_le_axis=180,
            final_compare_current_add_r=0.5,
            final_compare_current_add_l=0.5,
        )

    def test_lane_1_selected_when_lenso_axis_available_with_meaningful_cylinder(self):
        dv = self._derive(
            self._patient(
                visit_id="lane-1",
                ar_re=self._rx(-2.00, -1.00, 45),
                lenso_re=self._rx(-2.25, -1.00, 40),
            )
        )
        self.assertEqual(dv.dv_axis_lane_id_RE, "LANE_1")
        self.assertEqual(dv.dv_axis_step_sequence_RE, "10,5")
        self.assertEqual(dv.dv_axis_confidence_label_RE, "High")
        self.assertGreaterEqual(dv.dv_axis_cyl_magnitude_for_lane_RE, 0.75)
        self.assertEqual(dv.dv_axis_tolerance_deg_RE, 5.0)

    def test_lane_2_selected_for_ar_only_meaningful_non_cardinal_axis(self):
        dv = self._derive(
            self._patient(
                visit_id="lane-2",
                ar_re=self._rx(-1.50, -1.25, 47),
                lenso_re=None,
            )
        )
        self.assertEqual(dv.dv_axis_lane_id_RE, "LANE_2")
        self.assertEqual(dv.dv_axis_step_sequence_RE, "20,10,5")
        self.assertFalse(dv.dv_axis_is_near_cardinal_RE)
        self.assertEqual(dv.dv_start_rx_RE_axis, 45.0)
        self.assertEqual(dv.dv_axis_tolerance_deg_RE, 5.0)

    def test_lane_3_selected_for_ar_only_intermediate_confidence_case(self):
        dv = self._derive(
            self._patient(
                visit_id="lane-3",
                ar_re=self._rx(-1.50, -0.40, 55),
                lenso_re=None,
            )
        )
        self.assertEqual(dv.dv_axis_lane_id_RE, "LANE_3")
        self.assertEqual(dv.dv_axis_step_sequence_RE, "30,20,10")
        self.assertIn("conservative_fallback_lane", dv.dv_axis_selection_reason_RE)
        self.assertEqual(dv.dv_axis_tolerance_deg_RE, 10.0)

    def test_lane_4_selected_for_ar_only_near_cardinal_low_cylinder_case(self):
        dv = self._derive(
            self._patient(
                visit_id="lane-4",
                ar_re=self._rx(-1.50, -0.12, 4),
                lenso_re=None,
            )
        )
        self.assertEqual(dv.dv_axis_lane_id_RE, "LANE_4")
        self.assertEqual(dv.dv_axis_step_sequence_RE, "45,30,20,10")
        self.assertTrue(dv.dv_axis_is_near_cardinal_RE)
        self.assertLess(dv.dv_axis_cyl_magnitude_for_lane_RE, 0.50)

    def test_axis_calibration_is_consistent_with_live_lane_policy(self):
        self.assertEqual(float(self.calibration.get("axis_fixed_step", 0)), 5.0)
        self.assertEqual(float(self.calibration.get("axis_rounding_step", 0)), 5.0)
        keys = {row["parameter_key"] for row in self.calibration.get_snapshot()}
        self.assertNotIn("axis_tol_strict", keys)
        self.assertNotIn("axis_tol_normal", keys)
        self.assertNotIn("axis_tol_relaxed", keys)
        self.assertEqual(float(self.calibration.get("axis_meaningful_cyl_threshold_d", 0)), 0.5)
        self.assertEqual(int(self.calibration.get("jcc_axis_same_required", 0)), 1)

    def test_meaningful_cylinder_axis_tolerance_adds_live_five_degree_terminal_step(self):
        patient = self._patient(
            visit_id="axis-meaningful-five-degree-terminal-step",
            ar_re=self._rx(-1.50, -1.25, 47),
            lenso_re=None,
        )
        dv, calibration = self._derive_with_calibration_overrides(patient)
        self.assertEqual(dv.dv_axis_tolerance_deg_RE, 5.0)
        self.assertEqual(dv.dv_axis_step_sequence_RE, "20,10,5")

        engine = RefractionFSMEngine(calibration)
        current = engine.initialize_row("axis-meaningful-five-degree-terminal-step", dv)
        guard = 0
        while current.state != "E":
            self.assertLess(guard, 12)
            finalized = engine.apply_response(current, "CLEAR", dv)
            current = engine._build_next_row(finalized, dv)
            self.assertIsNotNone(current)
            guard += 1

        first = engine.apply_response(current, "ONE", dv)
        self.assertEqual(first.next_state, "E")
        self.assertEqual(abs(first.da_re), 20.0)
        second_row = engine._build_next_row(first, dv)
        self.assertIsNotNone(second_row)

        second = engine.apply_response(second_row, "TWO", dv)
        self.assertEqual(second.next_state, "E")
        self.assertEqual(abs(second.da_re), 10.0)
        third_row = engine._build_next_row(second, dv)
        self.assertIsNotNone(third_row)

        third = engine.apply_response(third_row, "ONE", dv)
        self.assertEqual(third.next_state, "E")
        self.assertEqual(abs(third.da_re), 5.0)
        fourth_row = engine._build_next_row(third, dv)
        self.assertIsNotNone(fourth_row)

        final = engine.apply_response(fourth_row, "TWO", dv)
        self.assertEqual(final.next_state, "F")
        self.assertEqual(final.da_re, 0.0)

    def test_reversal_progression_uses_each_lane_sequence_without_skipping(self):
        cases = [
            ("LANE_1", self._rx(-2.00, -1.00, 45), self._rx(-2.25, -1.00, 40), [10.0, 5.0, 0.0]),
            ("LANE_2", self._rx(-1.50, -1.25, 47), None, [20.0, 10.0, 5.0, 0.0]),
            ("LANE_3", self._rx(-1.50, -0.40, 55), None, [30.0, 20.0, 10.0, 0.0]),
            ("LANE_4", self._rx(-1.50, -0.12, 4), None, [45.0, 30.0, 20.0, 10.0, 0.0]),
        ]

        for lane_id, ar_re, lenso_re, expected_steps in cases:
            with self.subTest(lane_id=lane_id):
                dv = self._derive(
                    self._patient(
                        visit_id=f"progress-{lane_id}",
                        ar_re=ar_re,
                        lenso_re=lenso_re,
                    )
                )
                axis_row = self._advance_to_axis_re(dv)
                engine = RefractionFSMEngine(self.calibration)
                axis_row.re_axis = float(dv.dv_start_rx_RE_axis or 0.0) + 15.0

                current = axis_row
                responses = ["ONE"] + ["TWO" if i % 2 else "ONE" for i in range(1, len(expected_steps))]
                seen_steps = []
                for response in responses:
                    finalized = engine.apply_response(current, response, dv)
                    seen_steps.append(abs(finalized.da_re))
                    if finalized.next_state != "E":
                        break
                    next_row = engine._build_next_row(finalized, dv)
                    self.assertIsNotNone(next_row)
                    current = next_row

                self.assertEqual(seen_steps, expected_steps)

    def test_axis_phase_converges_on_reversal_at_ten_degrees(self):
        dv = self._derive(
            self._patient(
                visit_id="axis-ten-reversal-exit",
                ar_re=self._rx(-1.50, -0.40, 47),
                lenso_re=None,
            )
        )
        self.assertEqual(dv.dv_axis_tolerance_deg_RE, 10.0)

        engine = RefractionFSMEngine(self.calibration)
        axis_row = self._advance_to_axis_re(dv)
        first = engine.apply_response(axis_row, "ONE", dv)
        second_row = engine._build_next_row(first, dv)
        self.assertIsNotNone(second_row)
        second = engine.apply_response(second_row, "TWO", dv)
        third_row = engine._build_next_row(second, dv)
        self.assertIsNotNone(third_row)
        third = engine.apply_response(third_row, "ONE", dv)
        fourth_row = engine._build_next_row(third, dv)
        self.assertIsNotNone(fourth_row)
        finalized = engine.apply_response(fourth_row, "TWO", dv)
        self.assertEqual(finalized.next_state, "F")
        self.assertEqual(finalized.da_re, 0.0)

    def test_axis_tolerance_uses_start_cylinder_not_lensometer_confidence_cylinder(self):
        dv = self._derive(
            self._patient(
                visit_id="axis-start-cyl-tolerance",
                ar_re=self._rx(-1.50, -1.00, 47),
                lenso_re=self._rx(-1.25, -0.25, 50),
                satisfaction="Not satisfied",
            )
        )

        self.assertEqual(dv.dv_start_source_policy, "Start_AR")
        self.assertEqual(dv.dv_start_rx_RE_cyl, -1.00)
        self.assertEqual(dv.dv_axis_cyl_magnitude_for_lane_RE, 0.25)
        self.assertEqual(dv.dv_axis_cyl_magnitude_for_tolerance_RE, 1.00)
        self.assertEqual(dv.dv_axis_tolerance_deg_RE, 5.0)
        self.assertTrue(dv.dv_axis_step_sequence_RE.endswith(",5"))

    def test_axis_phase_converges_on_single_same_response(self):
        dv = self._derive(
            self._patient(
                visit_id="axis-same-exit",
                ar_re=self._rx(-1.50, -1.25, 47),
                lenso_re=None,
            )
        )

        engine = RefractionFSMEngine(self.calibration)
        axis_row = self._advance_to_axis_re(dv)
        finalized = engine.apply_response(axis_row, "SAME", dv)

        self.assertEqual(finalized.next_state, "F")
        self.assertEqual(finalized.da_re, 0.0)

    def test_non_axis_power_phase_behavior_is_unchanged(self):
        dv = self._derive(
            self._patient(
                visit_id="power-regression",
                ar_re=self._rx(-2.00, -1.25, 47),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = self._advance_to_axis_re(dv)
        first = engine.apply_response(current, "ONE", dv)
        first_row = engine._build_next_row(first, dv)
        self.assertIsNotNone(first_row)
        second = engine.apply_response(first_row, "TWO", dv)
        second_row = engine._build_next_row(second, dv)
        self.assertIsNotNone(second_row)
        third = engine.apply_response(second_row, "ONE", dv)
        third_row = engine._build_next_row(third, dv)
        self.assertIsNotNone(third_row)
        axis_done = engine.apply_response(third_row, "TWO", dv)
        self.assertEqual(axis_done.next_state, "F")
        power_row = engine._build_next_row(axis_done, dv)
        self.assertIsNotNone(power_row)
        self.assertEqual(power_row.state, "F")

        power_response = engine.apply_response(power_row, "ONE", dv)
        self.assertAlmostEqual(power_response.dc_re, 0.25)

    def test_jcc_power_terminal_negative_reversal_keeps_previous_power(self):
        dv = self._derive(
            self._patient(
                visit_id="power-terminal-negative-reversal",
                ar_re=self._rx(-2.00, -1.25, 47),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = engine._row_for_state(
            step=12,
            visit_id="power-terminal-negative-reversal",
            state="F",
            dv=dv,
            re_sph=-1.50,
            re_cyl=-1.25,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="20_20_20",
            phase_step_count=2,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=1,
            axis_step=10.0,
            jcc_power_start_re_cyl=-1.0,
        )
        current.response_value = "ONE"

        finalized = engine.apply_response(current, "TWO", dv)

        self.assertEqual(finalized.next_state, "G")
        self.assertEqual(finalized.duo_flip, 2)
        self.assertAlmostEqual(finalized.dc_re, 0.0)
        self.assertAlmostEqual(finalized.ds_re, 0.0)

        next_row = engine._build_next_row(finalized, dv)
        self.assertIsNotNone(next_row)
        self.assertEqual(next_row.state, "G")
        self.assertAlmostEqual(next_row.re_cyl, -1.25)
        self.assertAlmostEqual(next_row.re_sph, -1.50)

    def test_duochrome_terminal_red_reversal_keeps_previous_power(self):
        dv = self._derive(
            self._patient(
                visit_id="duochrome-terminal-red-reversal",
                ar_re=self._rx(-2.00, -1.25, 47),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        terminal_flip_count = int(dv.dv_duochrome_max_flips) - 1
        current = engine._row_for_state(
            step=15,
            visit_id="duochrome-terminal-red-reversal",
            state="G",
            dv=dv,
            re_sph=-1.25,
            re_cyl=-1.25,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="20_20_20",
            phase_step_count=2,
            same_streak=0,
            prev_axis_response="",
            duo_iter=1,
            duo_flip=terminal_flip_count,
            axis_step=10.0,
        )
        current.response_value = "GREEN"

        finalized = engine.apply_response(current, "RED", dv)

        self.assertEqual(finalized.next_state, "C")
        self.assertEqual(finalized.duo_flip, terminal_flip_count + 1)
        self.assertAlmostEqual(finalized.ds_re, 0.0)

        next_row = engine._build_next_row(finalized, dv)
        self.assertIsNotNone(next_row)
        self.assertEqual(next_row.state, "C")
        self.assertAlmostEqual(next_row.re_sph, -1.25)

    def test_duochrome_same_after_green_adds_plus_until_red_then_converges(self):
        dv = self._derive(
            self._patient(
                visit_id="duochrome-green-same-red",
                ar_re=self._rx(-2.00, -1.25, 47),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = engine._row_for_state(
            step=15,
            visit_id="duochrome-green-same-red",
            state="G",
            dv=dv,
            re_sph=-1.25,
            re_cyl=-1.25,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="20_20_20",
            phase_step_count=1,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
        )
        current.response_value = "GREEN"

        same_after_green = engine.apply_response(current, "SAME", dv)
        self.assertEqual(same_after_green.next_state, "G")
        self.assertEqual(same_after_green.duo_same_anchor_response, "GREEN")
        self.assertAlmostEqual(same_after_green.ds_re, 0.25)

        next_row = engine._build_next_row(same_after_green, dv)
        self.assertIsNotNone(next_row)
        self.assertAlmostEqual(next_row.re_sph, -1.00)

        red_recovery = engine.apply_response(next_row, "RED", dv)
        self.assertEqual(red_recovery.next_state, "C")
        self.assertAlmostEqual(red_recovery.ds_re, -0.25)

        final_row = engine._build_next_row(red_recovery, dv)
        self.assertIsNotNone(final_row)
        self.assertEqual(final_row.state, "C")
        self.assertAlmostEqual(final_row.re_sph, -1.25)

    def test_near_line_letters_map_to_clear_without_stimulus_payload(self):
        result = match_response(
            transcript="A P E O R F D Z",
            state="P",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            question="Please read the last line. Is it clear, blurry, or should I repeat?",
            language="en",
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "CLEAR")
        self.assertEqual(result.method, "letter_reading")

    def test_near_starts_from_lenso_add_and_presbyope_first_blurry_jumps_to_point_seven_five(self):
        patient = self._patient(
            visit_id="near-lenso-start",
            ar_re=self._rx(-2.00, -1.25, 47),
            lenso_re=self._rx(-2.00, -1.25, 47),
            ar_le=self._rx(-1.25, -0.75, 80),
            lenso_le=self._rx(-1.25, -0.75, 80),
            age=52,
            primary_reason="Blurred near",
            near_priority="High",
            wear_type="Progressive",
            lenso_add_r=1.25,
            lenso_add_l=1.00,
        )
        dv = self._derive(patient)
        self.assertTrue(dv.dv_near_test_required)
        self.assertAlmostEqual(dv.dv_near_start_add_r, 1.25)
        self.assertAlmostEqual(dv.dv_near_start_add_l, 1.00)

        engine = RefractionFSMEngine(self.calibration)
        k_row = engine._row_for_state(
            step=30,
            visit_id="near-lenso-start",
            state="K",
            dv=dv,
            re_sph=-1.25,
            re_cyl=-1.25,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="20_20_20",
            phase_step_count=1,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
        )
        finalized_k = engine.apply_response(k_row, "SAME", dv)
        self.assertEqual(finalized_k.next_state, "P")
        p_row = engine._build_next_row(finalized_k, dv)
        self.assertIsNotNone(p_row)
        self.assertAlmostEqual(p_row.add_r, 1.25)
        self.assertAlmostEqual(p_row.add_l, 1.00)

        no_lenso_add_patient = self._patient(
            visit_id="near-presbyope-jump",
            ar_re=self._rx(-2.00, -1.25, 47),
            lenso_re=None,
            age=52,
            primary_reason="Blurred near",
            near_priority="High",
        )
        jump_dv = self._derive(no_lenso_add_patient)
        p_start = RefractionFSMEngine(self.calibration)._row_for_state(
            step=30,
            visit_id="near-presbyope-jump",
            state="P",
            dv=jump_dv,
            re_sph=-1.25,
            re_cyl=-1.25,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="near",
            phase_step_count=1,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
        )
        first_blurry = engine.apply_response(p_start, "BLURRY", jump_dv)
        self.assertAlmostEqual(first_blurry.dadd_r, 0.75)

    def test_logging_contains_axis_lane_metadata(self):
        orchestrator = SessionOrchestrator(calibration_path=str(CALIBRATION_PATH))
        orchestrator.phoropter_auto_dispatch = False

        payload = {
            "age": 30,
            "primary_reason": "Blurred distance",
            "satisfaction": "Not satisfied",
            "ar_re_sph": -1.50,
            "ar_re_cyl": -0.25,
            "ar_re_axis": 4,
            "ar_le_sph": -1.25,
            "ar_le_cyl": -0.75,
            "ar_le_axis": 80,
            "lenso_re_sph": None,
            "lenso_re_cyl": None,
            "lenso_re_axis": None,
            "lenso_le_sph": None,
            "lenso_le_cyl": None,
            "lenso_le_axis": None,
        }
        orchestrator.initialize(payload, session_id="orch-axis", phoropter_id="")
        while orchestrator.current_row and orchestrator.current_row.state != "E":
            orchestrator.process_response("CLEAR")
        orchestrator.process_response("ONE")

        self.assertTrue(orchestrator.session_history)
        last_row = orchestrator.session_history[-1]
        self.assertEqual(last_row["axis_lane_id"], "LANE_3")
        self.assertEqual(last_row["axis_step_sequence"], "30,20,10")
        self.assertEqual(last_row["axis_source_used"], "AR")
        self.assertIn(
            "Axis lane selected",
            " ".join(entry["message"] for entry in orchestrator.conversation_log),
        )

    def test_debug_derived_variables_hide_redundant_and_compatibility_fields(self):
        orchestrator = SessionOrchestrator(calibration_path=str(CALIBRATION_PATH))
        orchestrator.phoropter_auto_dispatch = False

        payload = {
            "age": 30,
            "primary_reason": "Blurred distance",
            "satisfaction": "Not satisfied",
            "ar_re_sph": -1.50,
            "ar_re_cyl": -0.25,
            "ar_re_axis": 4,
            "ar_le_sph": -1.25,
            "ar_le_cyl": -0.75,
            "ar_le_axis": 80,
            "lenso_re_sph": None,
            "lenso_re_cyl": None,
            "lenso_re_axis": None,
            "lenso_le_sph": None,
            "lenso_le_cyl": None,
            "lenso_le_axis": None,
        }
        orchestrator.initialize(payload, session_id="orch-dv-filter", phoropter_id="")
        dv = orchestrator.get_derived_variables()

        self.assertNotIn("dv_axis_tolerance_deg", dv)
        self.assertNotIn("dv_cyl_tolerance_D", dv)
        self.assertNotIn("dv_fogging_required_confirmation", dv)
        self.assertNotIn("dv_axis_step_policy", dv)
        self.assertNotIn("dv_accommodation_level", dv)
        self.assertNotIn("dv_fogging_stop_at_target_va", dv)
        self.assertNotIn("dv_quick_axis_search_RE", dv)
        self.assertNotIn("dv_quick_axis_search_LE", dv)

        self.assertEqual(dv["dv_axis_lane_id_RE"], "LANE_3")
        self.assertEqual(dv["dv_axis_step_sequence_RE"], "30,20,10")

    def test_existing_start_source_and_start_rx_logic_is_preserved(self):
        dv = self._derive(
            self._patient(
                visit_id="hybrid-preserved",
                ar_re=self._rx(-5.00, -1.25, 30),
                lenso_re=self._rx(-1.00, -0.25, 80),
                satisfaction="Satisfied",
                primary_reason="Routine check",
            )
        )
        self.assertEqual(dv.dv_start_source_policy, "Hybrid")
        self.assertEqual(dv.dv_start_rx_RE_axis, 30)

    def test_low_fog_policy_label_is_used_for_presbyopes(self):
        patient = self._patient(
            visit_id="low-fog",
            ar_re=self._rx(-1.00, -0.50, 45),
            lenso_re=None,
        )
        patient.age = 52
        dv = self._derive(patient)
        self.assertEqual(dv.dv_fogging_policy, "Low_Fog")

    def test_distance_target_override_from_patient_input_is_ignored(self):
        patient = self._patient(
            visit_id="risk-derived-target",
            ar_re=self._rx(-1.00, -0.50, 45),
            lenso_re=None,
            satisfaction="Satisfied",
            primary_reason="Routine check",
        )
        patient.distance_target_preference = "6/9_acceptable"
        dv = self._derive(patient)
        self.assertEqual(dv.dv_target_distance_va, "6/6_target")

    def test_coarse_sphere_starts_from_third_smallest_chart(self):
        dv = self._derive(
            self._patient(
                visit_id="coarse-start",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = engine.initialize_row("coarse-start", dv)
        self.assertEqual(current.chart_param, "70_60_50")
        self.assertEqual(current.question, "Please read the line. If the letters are not clear, say blurry, or repeat.")

    def test_partial_letter_reading_maps_to_blurry_with_accuracy_confidence(self):
        result = match_response(
            transcript="E G N",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="E G N D H",
            question="Please read the line. If the letters are not clear, say blurry, or repeat.",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "BLURRY")
        self.assertEqual(result.method, "letter_reading_partial_blurry")
        self.assertAlmostEqual(result.confidence, 0.6)

    def test_very_poor_letter_reading_maps_to_repeat_with_accuracy_confidence(self):
        result = match_response(
            transcript="E O",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="E G N D H",
            question="Please read the line. If the letters are not clear, say blurry, or repeat.",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "REPEAT")
        self.assertEqual(result.method, "letter_reading_partial_repeat")
        self.assertAlmostEqual(result.confidence, 0.2)

    def test_english_clear_response_maps_to_clear_in_line_reading_phase(self):
        result = match_response(
            transcript="clear",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="E G N D H",
            question="Please read the line. If the letters are not clear, say blurry, or repeat.",
            language="en",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "CLEAR")
        self.assertEqual(result.method, "clarity_intent")

    def test_hindi_localized_question_uses_clarity_only_not_letter_reading(self):
        question = 'क्या अक्षर साफ हैं, धुंधले हैं, या फिर से?'

        line_result = match_response(
            transcript="ए पी ई ओ एफ",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="A P E O F",
            question=question,
            language="hi",
        )
        self.assertFalse(line_result.accepted)

        blurry_result = match_response(
            transcript="धुंधला है",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="A P E O F",
            question=question,
            language="hi",
        )
        self.assertTrue(blurry_result.accepted)
        self.assertEqual(blurry_result.response_value, "BLURRY")

        repeat_result = match_response(
            transcript="फिर से",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="A P E O F",
            question=question,
            language="hi",
        )
        self.assertTrue(repeat_result.accepted)
        self.assertEqual(repeat_result.response_value, "REPEAT")

    def test_hindi_mixed_clear_and_blurry_transcript_biases_to_blurry(self):
        result = match_response(
            transcript="धुँधला सा धन लाएँ धुँधला है साफ़ है",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="A P E O F",
            question='क्या अक्षर साफ हैं, धुंधले हैं, या फिर से?',
            language="hi",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "BLURRY")
        self.assertEqual(result.method, "line_phase_clarity_override")

    def test_noisy_hindi_blurry_variant_maps_correctly(self):
        result = match_response(
            transcript="Thula Hai",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="H N F Z C",
            question="Read the line, say blurry, or repeat.",
            language="hi",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "BLURRY")

    def test_hinglish_repeat_variant_maps_correctly(self):
        result = match_response(
            transcript="Firse",
            state="G",
            available_options=["GREEN", "RED", "SAME", "REPEAT"],
            question="Green side, red side, both same, or repeat?",
            language="hi",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "REPEAT")

    def test_hindi_transliterated_blurry_phrase_maps_correctly(self):
        result = match_response(
            transcript="dhundla hai",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            question="क्या अक्षर साफ हैं, धुंधले हैं, या फिर से?",
            language="hi",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "BLURRY")

    def test_hindi_variant_library_has_broad_coverage_per_response(self):
        for label in ["CLEAR", "BLURRY", "REPEAT", "ONE", "TWO", "SAME", "RED", "GREEN", "TOP", "BOTTOM"]:
            self.assertGreaterEqual(
                len(HINDI_VARIANT_LIBRARY[label]),
                50,
                f"{label} should have at least 50 Hindi transliterated variants",
            )

    def test_english_variant_library_has_broad_coverage_per_response(self):
        for label in ["CLEAR", "BLURRY", "REPEAT", "ONE", "TWO", "SAME", "RED", "GREEN", "TOP", "BOTTOM"]:
            self.assertGreaterEqual(
                len(ENGLISH_VARIANT_LIBRARY[label]),
                15,
                f"{label} should have broad English variants",
            )

    def test_hindi_logged_english_script_variants_map_correctly(self):
        blurry = match_response(
            transcript="Jule Hain",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            question="Read the line, say blurry, or repeat.",
            language="hi",
        )
        clear = match_response(
            transcript="Saath Hai",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            question="Read the line, say blurry, or repeat.",
            language="hi",
        )
        self.assertTrue(blurry.accepted)
        self.assertEqual(blurry.response_value, "BLURRY")
        self.assertTrue(clear.accepted)
        self.assertEqual(clear.response_value, "CLEAR")

    def test_hindi_negative_clear_phrase_maps_to_blurry(self):
        result = match_response(
            transcript="Saif Nahi Hai",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            question="Read it now, or say still blurry.",
            language="hi",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "BLURRY")

    def test_hindi_better_phrase_maps_to_clear(self):
        result = match_response(
            transcript="हां बेहतर है",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            question="Did it get better than before? Say yes or no.",
            language="hi",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "CLEAR")

    def test_hindi_clear_loanword_maps_to_clear(self):
        result = match_response(
            transcript="क्लियर",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            question="Read it now, or say still blurry.",
            language="hi",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "CLEAR")

    def test_hindi_transliterated_option_phrases_map_correctly(self):
        first = match_response(
            transcript="pehla vikalp",
            state="E",
            available_options=["ONE", "TWO", "SAME", "REPEAT"],
            question="पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से?",
            language="hi",
        )
        same = match_response(
            transcript="dono saman",
            state="E",
            available_options=["ONE", "TWO", "SAME", "REPEAT"],
            question="पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से?",
            language="hi",
        )
        self.assertTrue(first.accepted)
        self.assertEqual(first.response_value, "ONE")
        self.assertTrue(same.accepted)
        self.assertEqual(same.response_value, "SAME")

        compressed_same = match_response(
            transcript="donosaman",
            state="E",
            available_options=["ONE", "TWO", "SAME", "REPEAT"],
            question="पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से?",
            language="hi",
        )
        split_same = match_response(
            transcript="do no saman",
            state="E",
            available_options=["ONE", "TWO", "SAME", "REPEAT"],
            question="पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से?",
            language="hi",
        )
        devanagari_same = match_response(
            transcript="दोनोंसमान",
            state="E",
            available_options=["ONE", "TWO", "SAME", "REPEAT"],
            question="पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से?",
            language="hi",
        )
        devanagari_same_with_hai = match_response(
            transcript="दोनोंसमान है",
            state="E",
            available_options=["ONE", "TWO", "SAME", "REPEAT"],
            question="पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से?",
            language="hi",
        )
        self.assertTrue(compressed_same.accepted)
        self.assertEqual(compressed_same.response_value, "SAME")
        self.assertTrue(split_same.accepted)
        self.assertEqual(split_same.response_value, "SAME")
        self.assertTrue(devanagari_same.accepted)
        self.assertEqual(devanagari_same.response_value, "SAME")
        self.assertTrue(devanagari_same_with_hai.accepted)
        self.assertEqual(devanagari_same_with_hai.response_value, "SAME")

    def test_hindi_transliterated_directional_variants_map_correctly(self):
        red = match_response(
            transcript="laal side better",
            state="G",
            available_options=["GREEN", "RED", "SAME", "REPEAT"],
            question="हरा साइड, लाल साइड, दोनों समान, या फिर से?",
            language="hi",
        )
        bottom = match_response(
            transcript="neeche ki line better",
            state="K",
            available_options=["BOTTOM", "TOP", "SAME", "REPEAT"],
            question="नीचे की लाइन, ऊपर की लाइन, दोनों समान, या फिर से?",
            language="hi",
        )
        self.assertTrue(red.accepted)
        self.assertEqual(red.response_value, "RED")
        self.assertTrue(bottom.accepted)
        self.assertEqual(bottom.response_value, "BOTTOM")

    def test_duochrome_spatial_side_variants_map_correctly(self):
        right_side = match_response(
            transcript="right side",
            state="G",
            available_options=["GREEN", "RED", "SAME", "REPEAT"],
            question="Green side, red side, both same, or repeat?",
            language="en",
        )
        left_side = match_response(
            transcript="left side better",
            state="G",
            available_options=["GREEN", "RED", "SAME", "REPEAT"],
            question="Green side, red side, both same, or repeat?",
            language="en",
        )

        self.assertTrue(right_side.accepted)
        self.assertEqual(right_side.response_value, "RED")
        self.assertTrue(left_side.accepted)
        self.assertEqual(left_side.response_value, "GREEN")

    def test_logged_compressed_english_variants_map_correctly(self):
        green = match_response(
            transcript="Greenside",
            state="G",
            available_options=["GREEN", "RED", "SAME", "REPEAT"],
            question="Green side, red side, both same, or repeat?",
            language="en",
        )
        same = match_response(
            transcript="BothSame",
            state="E",
            available_options=["ONE", "TWO", "SAME", "REPEAT"],
            question="First option, second option, both same, or repeat?",
            language="en",
        )
        first = match_response(
            transcript="FirstOption",
            state="E",
            available_options=["ONE", "TWO", "SAME", "REPEAT"],
            question="First option, second option, both same, or repeat?",
            language="en",
        )
        self.assertTrue(green.accepted)
        self.assertEqual(green.response_value, "GREEN")
        self.assertTrue(same.accepted)
        self.assertEqual(same.response_value, "SAME")
        self.assertTrue(first.accepted)
        self.assertEqual(first.response_value, "ONE")

    def test_coarse_compare_question_does_not_use_line_reading_match(self):
        result = match_response(
            transcript="ए पी ई",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="A P E O F",
            question="Did it get better than before? Say yes or no.",
            language="hi",
        )
        self.assertFalse(result.accepted)

    def test_coarse_compare_clearer_but_still_blurry_maps_to_clear(self):
        result = match_response(
            transcript="they became clear but still bloody",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="A P E O F",
            question="Did it get better than before? Say yes or no.",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "CLEAR")
        self.assertEqual(result.method, "clarity_intent")

    def test_coarse_compare_yes_maps_to_clear(self):
        result = match_response(
            transcript="yes",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="A P E O F",
            question="Did it get better than before? Say yes or no.",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "CLEAR")

    def test_coarse_compare_no_maps_to_blurry(self):
        result = match_response(
            transcript="no",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="A P E O F",
            question="Did it get better than before? Say yes or no.",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "BLURRY")

    def test_coarse_compare_got_worse_maps_to_blurry(self):
        result = match_response(
            transcript="they got worse",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="A P E O F",
            question="Did it get better than before? Say yes or no.",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "BLURRY")
        self.assertEqual(result.method, "clarity_intent")

    def test_coarse_chart_dispatch_uses_last_line_subitem(self):
        dv = self._derive(
            self._patient(
                visit_id="coarse-chart-dispatch",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        orchestrator = SessionOrchestrator(calibration_path=str(CALIBRATION_PATH))
        orchestrator.phoropter_auto_dispatch = False
        orchestrator.derived_variables = dv
        orchestrator.current_row = RefractionFSMEngine(self.calibration).initialize_row(
            "coarse-chart-dispatch", dv
        )

        selection = orchestrator._chart_selection_for_state("B")
        self.assertEqual(selection["chart_items"], ["chart_12", "50"])

    def test_distance_va_confirmation_chart_dispatch_uses_last_line_subitem(self):
        dv = self._derive(
            self._patient(
                visit_id="va-chart-dispatch",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        orchestrator = SessionOrchestrator(calibration_path=str(CALIBRATION_PATH))
        orchestrator.phoropter_auto_dispatch = False
        orchestrator.derived_variables = dv
        orchestrator.current_row = RefractionFSMEngine(self.calibration)._row_for_state(
            step=10,
            visit_id="va-chart-dispatch",
            state="C",
            dv=dv,
            re_sph=-1.50,
            re_cyl=-0.75,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="20_20_20",
            phase_step_count=1,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
        )

        selection = orchestrator._chart_selection_for_state("C")
        self.assertEqual(selection["chart_items"], ["chart_15", "20_3"])

    def test_distance_va_confirmation_uses_line_prompt(self):
        dv = self._derive(
            self._patient(
                visit_id="va-line-prompt",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        row = engine._row_for_state(
            step=10,
            visit_id="va-line-prompt",
            state="C",
            dv=dv,
            re_sph=-1.50,
            re_cyl=-0.75,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="20_20_20",
            phase_step_count=1,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
        )
        self.assertEqual(row.question, "Please read the line. If the letters are not clear, say blurry, or repeat.")

    def test_line_prompt_shortens_when_other_eye_has_already_heard_it(self):
        dv = self._derive(
            self._patient(
                visit_id="cross-eye-line-prompt",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        re_row = engine.initialize_row("cross-eye-line-prompt", dv)
        self.assertEqual(re_row.question, "Please read the line. If the letters are not clear, say blurry, or repeat.")

        le_row = engine._row_for_state(
            step=20,
            visit_id="cross-eye-line-prompt",
            state="D",
            dv=dv,
            re_sph=-1.50,
            re_cyl=-0.75,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="70_60_50",
            phase_step_count=1,
            same_streak=0,
            prev_axis_response="",
            prompt_memory=re_row.prompt_memory,
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
        )
        self.assertEqual(le_row.question, "Read the line, say blurry, or repeat.")

    def test_new_chart_line_first_exposure_keeps_blurry_instruction(self):
        dv = self._derive(
            self._patient(
                visit_id="new-chart-line-prompt",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = engine.initialize_row("new-chart-line-prompt", dv)

        confirmed = engine.apply_response(current, "CLEAR", dv)
        next_row = engine._build_next_row(confirmed, dv)
        self.assertIsNotNone(next_row)
        self.assertEqual(next_row.chart_param, "40_30_25")
        self.assertEqual(next_row.question, "Read the line, say blurry, or repeat.")

    def test_coarse_step_back_transition_pushes_power_to_phoropter(self):
        orchestrator = SessionOrchestrator(calibration_path=str(CALIBRATION_PATH))
        orchestrator.phoropter_auto_dispatch = False
        payload = {
            "age": 30,
            "primary_reason": "Blurred distance",
            "satisfaction": "Not satisfied",
            "ar_re_sph": -1.50,
            "ar_re_cyl": -0.75,
            "ar_re_axis": 45,
            "ar_le_sph": -1.25,
            "ar_le_cyl": -0.75,
            "ar_le_axis": 80,
            "lenso_re_sph": None,
            "lenso_re_cyl": None,
            "lenso_re_axis": None,
            "lenso_le_sph": None,
            "lenso_le_cyl": None,
            "lenso_le_axis": None,
        }
        orchestrator.initialize(payload, session_id="orch-stepback", phoropter_id="")
        orchestrator.phoropter_auto_dispatch = True
        orchestrator.phoropter_id = "stub"

        calls = []

        def fake_send_chart_for_state(state):
            calls.append(("chart_state", state))
            return {"ok": True}

        def fake_send_chart(tab, chart_items):
            calls.append(("chart", tab, tuple(chart_items)))
            return {"ok": True}

        def fake_send_power_with_prev(include_add=False):
            calls.append(("power", include_add, orchestrator.current_row.state, orchestrator.current_row.re_sph))
            return {"ok": True}

        def fake_send_jcc(action):
            calls.append(("jcc", action))
            return {"ok": True}

        orchestrator._send_chart_for_state = fake_send_chart_for_state
        orchestrator._send_chart = fake_send_chart
        orchestrator._send_power_with_prev = fake_send_power_with_prev
        orchestrator._send_jcc = fake_send_jcc
        orchestrator._capture_screenshot = lambda: None

        orchestrator.process_response("CLEAR")
        orchestrator.process_response("BLURRY")
        calls.clear()
        response = orchestrator.process_response("BLURRY")

        self.assertEqual(response["state"], "E")
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0], ("power", False, "E", -1.25))
        self.assertEqual(calls[1], ("chart", "Chart1", ("chart_19",)))

    def test_coarse_blurry_then_more_blurry_steps_back_and_exits_phase(self):
        dv = self._derive(
            self._patient(
                visit_id="coarse-stepback",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = engine.initialize_row("coarse-stepback", dv)
        blurred = engine.apply_response(current, "BLURRY", dv)
        self.assertAlmostEqual(blurred.ds_re, -0.25)
        self.assertFalse(blurred.coarse_compare_mode)
        self.assertEqual(blurred.next_state, "B")

        next_row = engine._build_next_row(blurred, dv)
        self.assertIsNotNone(next_row)
        self.assertEqual(next_row.state, "B")
        self.assertEqual(next_row.chart_param, "70_60_50")
        self.assertEqual(next_row.question, "Read the line, say blurry, or repeat.")

    def test_coarse_compare_starts_only_from_20_25_chart(self):
        dv = self._derive(
            self._patient(
                visit_id="coarse-compare-gated",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = engine._row_for_state(
            step=5,
            visit_id="coarse-compare-gated",
            state="B",
            dv=dv,
            re_sph=-1.50,
            re_cyl=-0.75,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="40_30_25",
            phase_step_count=2,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
        )

        blurred = engine.apply_response(current, "BLURRY", dv)
        self.assertAlmostEqual(blurred.ds_re, -0.25)
        self.assertTrue(blurred.coarse_compare_mode)
        compare_row = engine._build_next_row(blurred, dv)
        self.assertIsNotNone(compare_row)
        self.assertEqual(compare_row.question, "Did it get better than before? Say yes or no.")

        worse = engine.apply_response(compare_row, "BLURRY", dv)
        self.assertAlmostEqual(worse.ds_re, 0.25)
        self.assertEqual(worse.coarse_last_confirmed_chart_re, "40_30_25")
        self.assertEqual(worse.next_state, "E")

    def test_coarse_clearer_branch_returns_to_reading_and_advances_chart(self):
        dv = self._derive(
            self._patient(
                visit_id="coarse-clearer",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = engine._row_for_state(
            step=5,
            visit_id="coarse-clearer",
            state="B",
            dv=dv,
            re_sph=-1.50,
            re_cyl=-0.75,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="40_30_25",
            phase_step_count=2,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
        )

        blurred = engine.apply_response(current, "BLURRY", dv)
        compare_row = engine._build_next_row(blurred, dv)
        self.assertIsNotNone(compare_row)

        clearer = engine.apply_response(compare_row, "CLEAR", dv)
        self.assertEqual(clearer.next_state, "B")
        reading_row = engine._build_next_row(clearer, dv)
        self.assertIsNotNone(reading_row)
        self.assertFalse(reading_row.coarse_compare_mode)
        self.assertTrue(reading_row.coarse_recheck_mode)
        self.assertEqual(reading_row.question, "Can you read the line now, or is it still blurry?")
        self.assertEqual(reading_row.chart_param, "40_30_25")

        confirmed = engine.apply_response(reading_row, "CLEAR", dv)
        self.assertEqual(confirmed.coarse_last_confirmed_chart_re, "40_30_25")
        advanced_row = engine._build_next_row(confirmed, dv)
        self.assertIsNotNone(advanced_row)
        self.assertFalse(advanced_row.coarse_recheck_mode)
        self.assertEqual(advanced_row.chart_param, "20_20_20")

    def test_coarse_recheck_prompt_shortens_after_first_followup(self):
        dv = self._derive(
            self._patient(
                visit_id="coarse-recheck-short",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = engine._row_for_state(
            step=5,
            visit_id="coarse-recheck-short",
            state="B",
            dv=dv,
            re_sph=-1.50,
            re_cyl=-0.75,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="40_30_25",
            phase_step_count=2,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
        )

        blurred = engine.apply_response(current, "BLURRY", dv)
        compare_row = engine._build_next_row(blurred, dv)
        clearer = engine.apply_response(compare_row, "CLEAR", dv)
        recheck_row = engine._build_next_row(clearer, dv)
        self.assertIsNotNone(recheck_row)
        self.assertEqual(recheck_row.question, "Can you read the line now, or is it still blurry?")

        repeated = engine.apply_response(recheck_row, "REPEAT", dv)
        repeated_row = engine._build_next_row(repeated, dv)
        self.assertIsNotNone(repeated_row)
        self.assertEqual(repeated_row.question, "Read it now, or say still blurry.")

    def test_axis_prompt_shortens_after_first_question(self):
        dv = self._derive(
            self._patient(
                visit_id="prompt-shortening",
                ar_re=self._rx(-1.50, -1.25, 47),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = self._advance_to_axis_re(dv)
        self.assertIn("Please compare the two dot patterns.", current.question)

        first_repeat = engine.apply_response(current, "REPEAT", dv)
        second_row = engine._build_next_row(first_repeat, dv)
        self.assertIsNotNone(second_row)
        self.assertEqual(second_row.question, "Which is better, or are both same?")

    def test_jcc_power_prompt_is_short_after_axis_on_same_eye(self):
        dv = self._derive(
            self._patient(
                visit_id="jcc-same-eye-shortening",
                ar_re=self._rx(-1.50, -1.25, 47),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        axis_row = self._advance_to_axis_re(dv)
        self.assertIn("Please compare the two dot patterns.", axis_row.question)

        first = engine.apply_response(axis_row, "ONE", dv)
        first_row = engine._build_next_row(first, dv)
        self.assertIsNotNone(first_row)
        second = engine.apply_response(first_row, "TWO", dv)
        second_row = engine._build_next_row(second, dv)
        self.assertIsNotNone(second_row)
        third = engine.apply_response(second_row, "ONE", dv)
        third_row = engine._build_next_row(third, dv)
        self.assertIsNotNone(third_row)
        axis_done = engine.apply_response(third_row, "TWO", dv)
        self.assertEqual(axis_done.next_state, "F")
        power_row = engine._build_next_row(axis_done, dv)
        self.assertIsNotNone(power_row)
        self.assertEqual(power_row.question, "Which is better, or are both same?")

    def test_bare_option_does_not_map_in_comparison_phase(self):
        result = match_response(
            transcript="option",
            state="E",
            available_options=["ONE", "TWO", "SAME", "REPEAT"],
            question="Which is better, or are both same?",
            language="en",
        )
        self.assertFalse(result.accepted)

    def test_both_alone_maps_to_same_in_comparison_phases(self):
        for state, options in [
            ("E", ["ONE", "TWO", "SAME", "REPEAT"]),
            ("G", ["GREEN", "RED", "SAME", "REPEAT"]),
            ("K", ["BOTTOM", "TOP", "SAME", "REPEAT"]),
        ]:
            with self.subTest(state=state):
                result = match_response(
                    transcript="both",
                    state=state,
                    available_options=options,
                )
                self.assertTrue(result.accepted)
                self.assertEqual(result.response_value, "SAME")

    def test_duochrome_short_prompt_mentions_both_same(self):
        dv = self._derive(
            self._patient(
                visit_id="duochrome-short",
                ar_re=self._rx(-1.50, -1.25, 47),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        row = engine._row_for_state(
            step=20,
            visit_id="duochrome-short",
            state="G",
            dv=dv,
            re_sph=-1.50,
            re_cyl=-1.25,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="20_20_20",
            phase_step_count=3,
            same_streak=0,
            prev_axis_response="",
            prompt_memory={"session:duochrome_compare": 1},
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
        )
        self.assertEqual(row.question, "Green side, red side, both same, or repeat?")

    def test_post_duochrome_distance_va_confirmation_records_last_read_line(self):
        dv = self._derive(
            self._patient(
                visit_id="va-confirm",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = engine._row_for_state(
            step=12,
            visit_id="va-confirm",
            state="C",
            dv=dv,
            re_sph=-1.50,
            re_cyl=-0.75,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.0,
            add_l=0.0,
            chart_param="20_20_20",
            phase_step_count=1,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            coarse_compare_mode=False,
            coarse_last_confirmed_chart_re="40_30_25",
            coarse_last_confirmed_chart_le="",
            distance_va_re_chart="",
            distance_va_le_chart="",
            distance_va_re_line="",
            distance_va_le_line="",
            va_confirm_ceiling_chart="40_30_25",
            axis_step=10.0,
        )

        smaller_unreadable = engine.apply_response(current, "BLURRY", dv)
        self.assertEqual(smaller_unreadable.next_state, "C")
        next_row = engine._build_next_row(smaller_unreadable, dv)
        self.assertIsNotNone(next_row)
        self.assertEqual(next_row.chart_param, "40_30_25")

        confirmed = engine.apply_response(next_row, "CLEAR", dv)
        self.assertEqual(confirmed.next_state, "D")
        self.assertEqual(confirmed.distance_va_re_chart, "40_30_25")
        self.assertEqual(confirmed.distance_va_re_line, "20/25")

    def test_final_compare_starts_after_binocular_completion_when_lenso_is_available(self):
        dv = self._derive(
            self._patient(
                visit_id="final-compare-start",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=self._rx(-2.00, -0.50, 50),
                ar_le=self._rx(-1.25, -0.75, 80),
                lenso_le=self._rx(-1.50, -0.25, 180),
                satisfaction="Satisfied",
                primary_reason="Routine check",
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = self._final_compare_seed_row(dv)

        finalized = engine.apply_response(current, "SAME", dv)
        self.assertEqual(finalized.next_state, "S")
        next_row = engine._build_next_row(finalized, dv)
        self.assertIsNotNone(next_row)
        self.assertEqual(next_row.state, "S")
        self.assertEqual(next_row.final_compare_round, 1)
        self.assertEqual(next_row.re_sph, -1.00)
        self.assertEqual(next_row.re_axis, 45)
        self.assertEqual(next_row.final_compare_achieved_re_sph, -1.00)
        self.assertEqual(next_row.final_compare_achieved_le_axis, 80)
        self.assertEqual(next_row.chart_param, "20_20_20")
        self.assertEqual(next_row.final_compare_current_source, "PGP")

    def test_final_compare_two_confirmations_accept_achieved_rx(self):
        dv = self._derive(
            self._patient(
                visit_id="final-compare-accept",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=self._rx(-2.00, -0.50, 50),
                ar_le=self._rx(-1.25, -0.75, 80),
                lenso_le=self._rx(-1.50, -0.25, 180),
                satisfaction="Satisfied",
                primary_reason="Routine check",
            )
        )
        engine = RefractionFSMEngine(self.calibration)

        current = self._final_compare_seed_row(dv)
        finalized = engine.apply_response(current, "SAME", dv)
        current = engine._build_next_row(finalized, dv)
        self.assertEqual(current.state, "S")

        finalized = engine.apply_response(current, "AUTO_ADVANCE", dv)
        current = engine._build_next_row(finalized, dv)
        self.assertEqual(current.state, "T")
        self.assertEqual(current.re_sph, -2.00)
        self.assertEqual(current.final_compare_option_source, "PGP")

        finalized = engine.apply_response(current, "AUTO_ADVANCE", dv)
        current = engine._build_next_row(finalized, dv)
        self.assertEqual(current.state, "U")

        finalized = engine.apply_response(current, "ONE", dv)
        self.assertEqual(finalized.next_state, "S")
        self.assertEqual(finalized.final_compare_choice_round_1, "ONE")
        current = engine._build_next_row(finalized, dv)
        self.assertEqual(current.final_compare_round, 2)
        self.assertEqual(current.state, "S")
        self.assertEqual(current.re_sph, -1.00)

        finalized = engine.apply_response(current, "AUTO_ADVANCE", dv)
        current = engine._build_next_row(finalized, dv)
        finalized = engine.apply_response(current, "AUTO_ADVANCE", dv)
        current = engine._build_next_row(finalized, dv)

        finalized = engine.apply_response(current, "ONE", dv)
        self.assertEqual(finalized.next_state, "END")
        self.assertEqual(finalized.final_compare_choice_round_2, "ONE")
        self.assertEqual(finalized.patient_accepted_achieved_over_current_rx, "Yes")

    def test_final_compare_repeat_restarts_same_round_from_option_one(self):
        dv = self._derive(
            self._patient(
                visit_id="final-compare-repeat",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=self._rx(-2.00, -0.50, 50),
                ar_le=self._rx(-1.25, -0.75, 80),
                lenso_le=self._rx(-1.50, -0.25, 180),
            )
        )
        engine = RefractionFSMEngine(self.calibration)

        current = self._final_compare_seed_row(dv)
        finalized = engine.apply_response(current, "SAME", dv)
        current = engine._build_next_row(finalized, dv)
        finalized = engine.apply_response(current, "AUTO_ADVANCE", dv)
        current = engine._build_next_row(finalized, dv)
        finalized = engine.apply_response(current, "AUTO_ADVANCE", dv)
        current = engine._build_next_row(finalized, dv)

        self.assertEqual(current.state, "U")
        self.assertEqual(current.final_compare_round, 1)

        finalized = engine.apply_response(current, "REPEAT", dv)
        self.assertEqual(finalized.next_state, "S")
        next_row = engine._build_next_row(finalized, dv)
        self.assertIsNotNone(next_row)
        self.assertEqual(next_row.state, "S")
        self.assertEqual(next_row.final_compare_round, 1)
        self.assertEqual(next_row.re_sph, -1.00)
        self.assertEqual(next_row.final_compare_choice_round_1, "")

    def test_orchestrator_final_compare_repeat_restarts_step_instead_of_reasking_u(self):
        dv = self._derive(
            self._patient(
                visit_id="final-compare-orch-repeat",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=self._rx(-2.00, -0.50, 50),
                ar_le=self._rx(-1.25, -0.75, 80),
                lenso_le=self._rx(-1.50, -0.25, 180),
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        orchestrator = SessionOrchestrator(calibration_path=str(CALIBRATION_PATH))
        orchestrator.phoropter_auto_dispatch = False
        orchestrator.derived_variables = dv

        current = self._final_compare_seed_row(dv)
        current = engine._build_next_row(engine.apply_response(current, "SAME", dv), dv)
        current = engine._build_next_row(engine.apply_response(current, "AUTO_ADVANCE", dv), dv)
        current = engine._build_next_row(engine.apply_response(current, "AUTO_ADVANCE", dv), dv)
        orchestrator.current_row = current

        response = orchestrator.process_response("REPEAT", input_method="Button")
        self.assertEqual(response["state"], "S")
        self.assertEqual(response["final_compare"]["round"], 1)
        self.assertEqual(response["prescription"]["right"]["sph"], -1.00)

    def test_final_compare_chart_dispatch_uses_20_20_last_line(self):
        dv = self._derive(
            self._patient(
                visit_id="final-compare-chart",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=self._rx(-2.00, -0.50, 50),
                ar_le=self._rx(-1.25, -0.75, 80),
                lenso_le=self._rx(-1.50, -0.25, 180),
            )
        )
        orchestrator = SessionOrchestrator(calibration_path=str(CALIBRATION_PATH))
        orchestrator.phoropter_auto_dispatch = False
        orchestrator.derived_variables = dv
        orchestrator.current_row = RefractionFSMEngine(self.calibration)._row_for_state(
            step=30,
            visit_id="final-compare-chart",
            state="S",
            dv=dv,
            re_sph=-2.00,
            re_cyl=-0.50,
            re_axis=50,
            le_sph=-1.50,
            le_cyl=-0.25,
            le_axis=180,
            add_r=0.5,
            add_l=0.5,
            chart_param="20_20_20",
            phase_step_count=1,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
            final_compare_enabled=True,
            final_compare_round=1,
        )

        selection = orchestrator._chart_selection_for_state("S")
        self.assertEqual(selection["chart_items"], ["chart_15", "20_3"])

    def test_session_initialize_adds_pre_exam_forehead_bar_preface(self):
        orchestrator = SessionOrchestrator(calibration_path=str(CALIBRATION_PATH))
        orchestrator.phoropter_auto_dispatch = False

        payload = {
            "age": 30,
            "primary_reason": "Routine check",
            "satisfaction": "No PGP",
            "wear_type": "None",
            "priority": "Standard",
            "near_priority": "Medium",
            "ar_re_sph": -1.50,
            "ar_re_cyl": -0.50,
            "ar_re_axis": 45,
            "ar_le_sph": -1.25,
            "ar_le_cyl": -0.75,
            "ar_le_axis": 80,
        }

        response = orchestrator.initialize(payload, session_id="preface-init", phoropter_id="")
        self.assertEqual(
            response["preface_prompt"],
            "Your eye test is about to begin. Please rest your forehead gently against the forehead bar and look straight ahead.",
        )

    def test_session_initialize_supports_lenso_only_start(self):
        orchestrator = SessionOrchestrator(calibration_path=str(CALIBRATION_PATH))
        orchestrator.phoropter_auto_dispatch = False

        payload = {
            "age": 34,
            "primary_reason": "Routine check",
            "satisfaction": "Satisfied",
            "wear_type": "Single vision",
            "priority": "Standard",
            "near_priority": "Medium",
            "lenso_re_sph": -2.00,
            "lenso_re_cyl": -0.75,
            "lenso_re_axis": 45,
            "lenso_le_sph": -1.50,
            "lenso_le_cyl": -0.50,
            "lenso_le_axis": 80,
            "lenso_add_r": 0.0,
            "lenso_add_l": 0.0,
        }

        orchestrator.initialize(payload, session_id="lenso-only-init", phoropter_id="")
        self.assertIsNone(orchestrator.ar_re)
        self.assertIsNone(orchestrator.ar_le)
        self.assertEqual(orchestrator.derived_variables.dv_start_source_policy, "Start_Lenso")

    def test_final_compare_metadata_includes_acceptance_flag(self):
        end_time = build_session_metadata(
            session_id="final-compare-meta",
            phoropter_id="stub",
            session_start_time=datetime(2026, 3, 27, 12, 0, 0),
            session_end_time=datetime(2026, 3, 27, 12, 10, 0),
            completion_status="completed",
            rows=[
                {
                    "re_sph": -1.00,
                    "re_cyl": -0.50,
                    "re_axis": 45,
                    "add_r": 0.75,
                    "le_sph": -1.25,
                    "le_cyl": -0.75,
                    "le_axis": 80,
                    "add_l": 0.75,
                    "distance_va_re_chart": "20_20_20",
                    "distance_va_re_line": "20/20",
                    "distance_va_le_chart": "20_20_20",
                    "distance_va_le_line": "20/20",
                    "final_compare_enabled": True,
                    "final_compare_choice_round_1": "ONE",
                    "final_compare_choice_round_2": "ONE",
                    "patient_accepted_achieved_over_current_rx": "Yes",
                    "final_compare_achieved_re_sph": -1.00,
                    "final_compare_achieved_re_cyl": -0.50,
                    "final_compare_achieved_re_axis": 45,
                    "final_compare_achieved_add_r": 0.75,
                    "final_compare_achieved_le_sph": -1.25,
                    "final_compare_achieved_le_cyl": -0.75,
                    "final_compare_achieved_le_axis": 80,
                    "final_compare_achieved_add_l": 0.75,
                    "final_compare_current_re_sph": -2.00,
                    "final_compare_current_re_cyl": -0.50,
                    "final_compare_current_re_axis": 50,
                    "final_compare_current_add_r": 0.50,
                    "final_compare_current_le_sph": -1.50,
                    "final_compare_current_le_cyl": -0.25,
                    "final_compare_current_le_axis": 180,
                    "final_compare_current_add_l": 0.50,
                }
            ],
            lensometry={
                "right": {"sph": -2.00, "cyl": -0.50, "axis": 50},
                "left": {"sph": -1.50, "cyl": -0.25, "axis": 180},
            },
        )

        comparison = end_time["final_rx_comparison"]
        self.assertTrue(comparison["ran"])
        self.assertEqual(comparison["current_source"], "PGP")
        self.assertEqual(comparison["round_1_choice"], "ONE")
        self.assertEqual(comparison["accepted_achieved_over_current_rx"], "Yes")
        self.assertEqual(comparison["selected_prescribed_rx_source"], "Achieved")
        self.assertEqual(end_time["final_prescription"]["right"]["sph"], -1.00)
        self.assertEqual(end_time["pgp_rx"]["right"]["sph"], -2.00)
        self.assertEqual(end_time["current_rx"]["right"]["sph"], -2.00)

    def test_final_compare_metadata_selects_current_rx_when_achieved_not_accepted(self):
        metadata = build_session_metadata(
            session_id="final-compare-meta-current",
            phoropter_id="stub",
            session_start_time=datetime(2026, 3, 27, 12, 0, 0),
            session_end_time=datetime(2026, 3, 27, 12, 10, 0),
            completion_status="completed",
            rows=[
                {
                    "re_sph": -2.00,
                    "re_cyl": -0.50,
                    "re_axis": 50,
                    "add_r": 0.50,
                    "le_sph": -1.50,
                    "le_cyl": -0.25,
                    "le_axis": 180,
                    "add_l": 0.50,
                    "distance_va_re_chart": "20_20_20",
                    "distance_va_re_line": "20/20",
                    "distance_va_le_chart": "20_20_20",
                    "distance_va_le_line": "20/20",
                    "final_compare_enabled": True,
                    "final_compare_choice_round_1": "TWO",
                    "final_compare_choice_round_2": "TWO",
                    "patient_accepted_achieved_over_current_rx": "No",
                    "final_compare_achieved_re_sph": -1.00,
                    "final_compare_achieved_re_cyl": -0.50,
                    "final_compare_achieved_re_axis": 45,
                    "final_compare_achieved_add_r": 0.75,
                    "final_compare_achieved_le_sph": -1.25,
                    "final_compare_achieved_le_cyl": -0.75,
                    "final_compare_achieved_le_axis": 80,
                    "final_compare_achieved_add_l": 0.75,
                    "final_compare_current_re_sph": -2.00,
                    "final_compare_current_re_cyl": -0.50,
                    "final_compare_current_re_axis": 50,
                    "final_compare_current_add_r": 0.50,
                    "final_compare_current_le_sph": -1.50,
                    "final_compare_current_le_cyl": -0.25,
                    "final_compare_current_le_axis": 180,
                    "final_compare_current_add_l": 0.50,
                }
            ],
            lensometry={
                "right": {"sph": -2.00, "cyl": -0.50, "axis": 50},
                "left": {"sph": -1.50, "cyl": -0.25, "axis": 180},
            },
        )

        self.assertEqual(metadata["final_rx_comparison"]["current_source"], "PGP")
        self.assertEqual(metadata["final_rx_comparison"]["selected_prescribed_rx_source"], "PGP")
        self.assertEqual(metadata["final_prescription"]["right"]["sph"], -2.00)
        self.assertEqual(metadata["achieved_prescription"]["right"]["sph"], -1.00)

    def test_no_glasses_final_compare_seeds_zero_baseline(self):
        orchestrator = SessionOrchestrator(calibration_path=str(CALIBRATION_PATH))
        orchestrator.phoropter_auto_dispatch = False

        payload = {
            "patient_name": "No Glasses Patient",
            "age": 28,
            "primary_reason": "Routine check",
            "satisfaction": "No PGP",
            "wear_type": "None",
            "priority": "Standard",
            "near_priority": "Medium",
            "ar_re_sph": -1.50,
            "ar_re_cyl": -0.50,
            "ar_re_axis": 45,
            "ar_le_sph": -1.25,
            "ar_le_cyl": -0.75,
            "ar_le_axis": 80,
        }

        orchestrator.initialize(payload, session_id="no-glasses-final-compare", phoropter_id="")
        self.assertTrue(orchestrator.current_row.final_compare_enabled)
        self.assertEqual(orchestrator.current_row.final_compare_current_source, "No Glasses")
        self.assertEqual(orchestrator.current_row.final_compare_current_re_sph, 0.0)
        self.assertEqual(orchestrator.current_row.final_compare_current_re_cyl, 0.0)
        self.assertEqual(orchestrator.current_row.final_compare_current_re_axis, 180.0)
        self.assertEqual(orchestrator.current_row.final_compare_current_le_sph, 0.0)
        self.assertEqual(orchestrator.current_row.final_compare_current_le_cyl, 0.0)
        self.assertEqual(orchestrator.current_row.final_compare_current_le_axis, 180.0)

    def test_no_glasses_final_compare_uses_zero_option_and_one_zero_choice_keeps_zero(self):
        dv = self._derive(
            self._patient(
                visit_id="no-glasses-final-compare-outcome",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
                ar_le=self._rx(-1.25, -0.75, 80),
                lenso_le=None,
                satisfaction="No PGP",
                wear_type="None",
                primary_reason="Routine check",
            )
        )
        engine = RefractionFSMEngine(self.calibration)

        current = engine._row_for_state(
            step=30,
            visit_id="no-glasses-final-compare",
            state="K",
            dv=dv,
            re_sph=-1.00,
            re_cyl=-0.50,
            re_axis=45,
            le_sph=-1.25,
            le_cyl=-0.75,
            le_axis=80,
            add_r=0.75,
            add_l=0.75,
            chart_param="20_20_20",
            phase_step_count=1,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=10.0,
            final_compare_enabled=True,
            final_compare_round=0,
            final_compare_option_source="Achieved",
            final_compare_current_source="No Glasses",
            final_compare_current_re_sph=0.0,
            final_compare_current_re_cyl=0.0,
            final_compare_current_re_axis=180.0,
            final_compare_current_le_sph=0.0,
            final_compare_current_le_cyl=0.0,
            final_compare_current_le_axis=180.0,
            final_compare_current_add_r=0.0,
            final_compare_current_add_l=0.0,
        )

        finalized = engine.apply_response(current, "SAME", dv)
        current = engine._build_next_row(finalized, dv)
        self.assertEqual(current.state, "S")

        finalized = engine.apply_response(current, "AUTO_ADVANCE", dv)
        current = engine._build_next_row(finalized, dv)
        self.assertEqual(current.state, "T")
        self.assertEqual(current.phase_name, "Final Compare Second Option No Glasses")
        self.assertEqual(current.re_sph, 0.0)
        self.assertEqual(current.re_cyl, 0.0)
        self.assertEqual(current.re_axis, 180.0)

        finalized = engine.apply_response(current, "AUTO_ADVANCE", dv)
        current = engine._build_next_row(finalized, dv)
        finalized = engine.apply_response(current, "TWO", dv)
        self.assertEqual(finalized.final_compare_choice_round_1, "TWO")
        self.assertEqual(finalized.patient_accepted_achieved_over_current_rx, "No")

        current = engine._build_next_row(finalized, dv)
        finalized = engine.apply_response(current, "AUTO_ADVANCE", dv)
        current = engine._build_next_row(finalized, dv)
        finalized = engine.apply_response(current, "AUTO_ADVANCE", dv)
        current = engine._build_next_row(finalized, dv)
        finalized = engine.apply_response(current, "ONE", dv)

        self.assertEqual(finalized.next_state, "END")
        self.assertEqual(finalized.final_compare_choice_round_2, "ONE")
        self.assertEqual(finalized.patient_accepted_achieved_over_current_rx, "No")

    def test_final_compare_metadata_uses_no_glasses_source_when_zero_baseline_selected(self):
        metadata = build_session_metadata(
            session_id="final-compare-meta-no-glasses",
            phoropter_id="stub",
            session_start_time=datetime(2026, 3, 27, 12, 0, 0),
            session_end_time=datetime(2026, 3, 27, 12, 10, 0),
            completion_status="completed",
            rows=[
                {
                    "re_sph": 0.0,
                    "re_cyl": 0.0,
                    "re_axis": 180.0,
                    "add_r": 0.0,
                    "le_sph": 0.0,
                    "le_cyl": 0.0,
                    "le_axis": 180.0,
                    "add_l": 0.0,
                    "distance_va_re_chart": "20_20_20",
                    "distance_va_re_line": "20/20",
                    "distance_va_le_chart": "20_20_20",
                    "distance_va_le_line": "20/20",
                    "final_compare_enabled": True,
                    "final_compare_current_source": "No Glasses",
                    "final_compare_choice_round_1": "TWO",
                    "final_compare_choice_round_2": "ONE",
                    "patient_accepted_achieved_over_current_rx": "No",
                    "final_compare_achieved_re_sph": -1.00,
                    "final_compare_achieved_re_cyl": -0.50,
                    "final_compare_achieved_re_axis": 45,
                    "final_compare_achieved_add_r": 0.75,
                    "final_compare_achieved_le_sph": -1.25,
                    "final_compare_achieved_le_cyl": -0.75,
                    "final_compare_achieved_le_axis": 80,
                    "final_compare_achieved_add_l": 0.75,
                    "final_compare_current_re_sph": 0.0,
                    "final_compare_current_re_cyl": 0.0,
                    "final_compare_current_re_axis": 180.0,
                    "final_compare_current_add_r": 0.0,
                    "final_compare_current_le_sph": 0.0,
                    "final_compare_current_le_cyl": 0.0,
                    "final_compare_current_le_axis": 180.0,
                    "final_compare_current_add_l": 0.0,
                }
            ],
            patient_input=PatientInput(
                visit_id="meta-no-glasses",
                satisfaction_with_current_rx="No PGP",
                wear_type="None",
            ),
        )

        self.assertEqual(metadata["final_rx_comparison"]["current_source"], "No Glasses")
        self.assertEqual(metadata["final_rx_comparison"]["selected_prescribed_rx_source"], "No Glasses")
        self.assertEqual(metadata["final_prescription"]["right"]["sph"], 0.0)
        self.assertEqual(metadata["current_rx"]["right"]["axis"], 180.0)

    def test_transition_preface_is_added_after_coarse_completion(self):
        orchestrator = SessionOrchestrator(calibration_path=str(CALIBRATION_PATH))
        orchestrator.phoropter_auto_dispatch = False

        payload = {
            "patient_name": "Asha",
            "age": 30,
            "primary_reason": "Routine check",
            "satisfaction": "Satisfied",
            "ar_re_sph": -1.00,
            "ar_re_cyl": -0.50,
            "ar_re_axis": 45,
            "ar_le_sph": -1.25,
            "ar_le_cyl": -0.75,
            "ar_le_axis": 80,
            "lenso_re_sph": None,
            "lenso_re_cyl": None,
            "lenso_re_axis": None,
            "lenso_le_sph": None,
            "lenso_le_cyl": None,
            "lenso_le_axis": None,
        }
        orchestrator.initialize(payload, session_id="preface", phoropter_id="")
        response = None
        while response is None or response["state"] != "E":
            response = orchestrator.process_response("CLEAR")

        self.assertEqual(response["state"], "E")
        self.assertEqual(
            response["preface_prompt"],
            "You are doing great. Please blink a few times. About 9 minutes left.",
        )

    def test_failed_voice_utterance_export_preserves_match_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "voice.csv"
            write_voice_utterances_csv(
                [],
                [
                    {
                        "timestamp": "2026-03-26T00:00:00Z",
                        "step": 4,
                        "state": "B",
                        "phase_name": "Coarse Sphere RE",
                        "transcript": "EGNOA",
                        "alternatives": ["EGNOA"],
                        "language": "en",
                        "canonical_label": "REPEAT",
                        "match_confidence": 0.6,
                        "match_method": "letter_reading_partial_repeat",
                        "stimulus_letters": "E G N D H",
                    },
                ],
                "diag-session",
                output_path,
            )

            with output_path.open() as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Canonical_Label"], "REPEAT")
            self.assertEqual(rows[0]["Confidence"], "0.6")
            self.assertEqual(rows[0]["Match_Method"], "letter_reading_partial_repeat")
            self.assertEqual(rows[0]["Stimulus_Letters"], "E G N D H")

    def test_metadata_includes_customer_phone_and_serialized_patient_phone(self):
        patient = PatientInput(
            visit_id="phone-meta",
            patient_name="Asha",
            phone_number="9876543210",
        )
        metadata = build_session_metadata(
            session_id="phone-meta",
            phoropter_id="stub",
            session_start_time=datetime(2026, 3, 27, 12, 0, 0),
            session_end_time=datetime(2026, 3, 27, 12, 10, 0),
            completion_status="completed",
            rows=[],
            customer_name="Asha",
            customer_phone="9876543210",
            patient_input=patient,
        )

        self.assertEqual(metadata["customer_name"], "Asha")
        self.assertEqual(metadata["customer_phone"], "9876543210")
        self.assertEqual(metadata["patient_input"]["phone_number"], "9876543210")


class ApiPathResolutionTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def tearDown(self):
        sessions.clear()

    def test_session_intake_uses_app_relative_calibration_path(self):
        response = self.client.post(
            "/api/session/intake",
            json={
                "phoropter_id": "",
                "patient": {
                    "patient_name": "Path Test",
                    "age": 29,
                    "primary_reason": "Routine check",
                    "priority": "Standard",
                    "near_priority": "Medium",
                    "ar_re_sph": -1.00,
                    "ar_re_cyl": -0.50,
                    "ar_re_axis": 45,
                    "ar_le_sph": -1.25,
                    "ar_le_cyl": -0.75,
                    "ar_le_axis": 90,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("session_id", payload)
        self.assertIn("question", payload)

    def test_session_intake_rejects_missing_objective_data(self):
        response = self.client.post(
            "/api/session/intake",
            json={
                "phoropter_id": "",
                "patient": {
                    "patient_name": "No Objective",
                    "age": 29,
                    "primary_reason": "Routine check",
                    "priority": "Standard",
                    "near_priority": "Medium",
                },
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()["error"],
            "Please enter AR and/or lensometry values before starting the test.",
        )

    def test_session_intake_persists_language_and_phone_number(self):
        response = self.client.post(
            "/api/session/intake",
            json={
                "phoropter_id": "",
                "language": "hi",
                "patient": {
                    "patient_name": "Asha",
                    "phone_number": "9876543210",
                    "language": "hi",
                    "age": 29,
                    "primary_reason": "Routine check",
                    "priority": "Standard",
                    "near_priority": "Medium",
                    "ar_re_sph": -1.00,
                    "ar_re_cyl": -0.50,
                    "ar_re_axis": 45,
                    "ar_le_sph": -1.25,
                    "ar_le_cyl": -0.75,
                    "ar_le_axis": 90,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["language"], "hi")
        self.assertEqual(sessions[payload["session_id"]].session_language, "hi")
        self.assertEqual(sessions[payload["session_id"]].patient_input.phone_number, "9876543210")


class HindiLocalizationTests(unittest.TestCase):
    def test_hindi_localizes_coarse_read_prompt(self):
        prompt = localized_voice_prompt(
            state="B",
            language="hi",
            retry=False,
            fallback_question="Please read the line. If the letters are not clear, say blurry, or repeat.",
        )
        self.assertEqual(prompt, "क्या अक्षर साफ हैं, धुंधले हैं, या फिर से?")

    def test_hindi_localizes_short_coarse_read_prompt(self):
        prompt = localized_voice_prompt(
            state="B",
            language="hi",
            retry=False,
            fallback_question="Read the line, say blurry, or repeat.",
        )
        self.assertEqual(prompt, "साफ, धुंधला, या फिर से?")

    def test_hindi_localizes_coarse_compare_prompt(self):
        prompt = localized_voice_prompt(
            state="B",
            language="hi",
            retry=False,
            fallback_question="Did it get better than before? Say yes or no.",
        )
        self.assertEqual(prompt, "क्या यह पहले से बेहतर हुआ? हाँ या नहीं कहिए।")

    def test_hindi_localizes_coarse_recheck_prompt(self):
        prompt = localized_voice_prompt(
            state="B",
            language="hi",
            retry=False,
            fallback_question="Read it now, or say still blurry.",
        )
        self.assertEqual(prompt, "साफ, अभी भी धुंधला, या फिर से?")

    def test_hindi_localizes_jcc_short_prompt(self):
        prompt = localized_voice_prompt(
            state="F",
            language="hi",
            retry=False,
            fallback_question="Which is better, or are both same?",
        )
        self.assertEqual(prompt, "कौन बेहतर है, या दोनों समान हैं?")

    def test_hindi_localizes_duochrome_and_bino_short_prompts(self):
        duo_prompt = localized_voice_prompt(
            state="G",
            language="hi",
            retry=False,
            fallback_question="Green side, red side, both same, or repeat?",
        )
        bino_prompt = localized_voice_prompt(
            state="K",
            language="hi",
            retry=False,
            fallback_question="Bottom line, top line, both same, or repeat?",
        )
        self.assertEqual(duo_prompt, "हरा साइड, लाल साइड, दोनों समान, या फिर से?")
        self.assertEqual(bino_prompt, "नीचे की लाइन, ऊपर की लाइन, दोनों समान, या फिर से?")

    def test_hindi_option_labels_are_contextual(self):
        self.assertEqual(
            localized_option_label(
                "CLEAR",
                "hi",
                state="B",
                question="Did it get better than before? Say yes or no.",
            ),
            "हाँ",
        )
        self.assertEqual(
            localized_option_label(
                "BLURRY",
                "hi",
                state="B",
                question="Can you read the line now, or is it still blurry?",
            ),
            "अभी भी धुंधला",
        )
        self.assertEqual(
            localized_option_label(
                "ONE",
                "hi",
                state="E",
                question="First, second, or both same?",
            ),
            "पहला विकल्प",
        )
        self.assertEqual(
            localized_option_label(
                "SAME",
                "hi",
                state="G",
                question="Green side, red side, both same, or repeat?",
            ),
            "दोनों समान",
        )

    def test_voice_labels_api_returns_hindi_question_and_labels(self):
        client = app.test_client()
        response = client.post(
            "/api/voice/labels",
            json={
                "state": "E",
                "language": "hi",
                "question": "Please compare the two dot patterns. Which one is clearer or sharper? say first option, second option, both same, or repeat.",
                "options": ["ONE", "TWO", "SAME", "REPEAT"],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(
            payload["question"],
            "कृपया दोनों डॉट पैटर्न की तुलना कीजिए। कौन सा ज़्यादा साफ या शार्प दिख रहा है? पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से कहिए।",
        )
        self.assertEqual(
            [item["localized"] for item in payload["labels"]],
            ["पहला विकल्प", "दूसरा विकल्प", "दोनों समान", "फिर से"],
        )


if __name__ == "__main__":
    unittest.main()
