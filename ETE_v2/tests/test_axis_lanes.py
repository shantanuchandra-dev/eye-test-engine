import sys
import unittest
import csv
import tempfile
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fsm.config.calibration_loader import CalibrationLoader
from fsm.audio.response_matching import (
    localized_option_label,
    localized_voice_prompt,
    match_response,
)
from fsm.engines.derived_variables_engine import DerivedVariablesEngine
from fsm.engines.refraction_fsm_engine import RefractionFSMEngine
from fsm.models.fsm_runtime import FSMRuntimeRow
from fsm.models.patient import PatientInput
from fsm.models.prescription import EyePrescription
from ete_io.outputs import write_voice_utterances_csv
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
    ) -> PatientInput:
        return PatientInput(
            visit_id=visit_id,
            age=30,
            primary_reason=primary_reason,
            satisfaction_with_current_rx=satisfaction,
            driving_hours=1,
            screen_time_hours=2,
            last_eye_test_months_ago=12,
            autorefractor_re=ar_re,
            autorefractor_le=ar_le or self._rx(-1.25, -0.75, 80),
            lenso_re=lenso_re,
            lenso_le=lenso_le,
        )

    def _derive(self, patient: PatientInput):
        return DerivedVariablesEngine(self.calibration).derive(patient)

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

    def test_lane_3_selected_for_ar_only_intermediate_confidence_case(self):
        dv = self._derive(
            self._patient(
                visit_id="lane-3",
                ar_re=self._rx(-1.50, -0.50, 55),
                lenso_re=None,
            )
        )
        self.assertEqual(dv.dv_axis_lane_id_RE, "LANE_3")
        self.assertEqual(dv.dv_axis_step_sequence_RE, "30,20,10,5")
        self.assertIn("conservative_fallback_lane", dv.dv_axis_selection_reason_RE)

    def test_lane_4_selected_for_ar_only_near_cardinal_low_cylinder_case(self):
        dv = self._derive(
            self._patient(
                visit_id="lane-4",
                ar_re=self._rx(-1.50, -0.25, 4),
                lenso_re=None,
            )
        )
        self.assertEqual(dv.dv_axis_lane_id_RE, "LANE_4")
        self.assertEqual(dv.dv_axis_step_sequence_RE, "45,30,20,10,5")
        self.assertTrue(dv.dv_axis_is_near_cardinal_RE)
        self.assertLess(dv.dv_axis_cyl_magnitude_for_lane_RE, 0.50)

    def test_reversal_progression_uses_each_lane_sequence_without_skipping(self):
        cases = [
            ("LANE_1", self._rx(-2.00, -1.00, 45), self._rx(-2.25, -1.00, 40), [10.0, 5.0]),
            ("LANE_2", self._rx(-1.50, -1.25, 47), None, [20.0, 10.0, 5.0]),
            ("LANE_3", self._rx(-1.50, -0.50, 55), None, [30.0, 20.0, 10.0, 5.0]),
            ("LANE_4", self._rx(-1.50, -0.25, 4), None, [45.0, 30.0, 20.0, 10.0, 5.0]),
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
        axis_done = engine.apply_response(current, "SAME", dv)
        self.assertEqual(axis_done.next_state, "F")
        power_row = engine._build_next_row(axis_done, dv)
        self.assertIsNotNone(power_row)
        self.assertEqual(power_row.state, "F")

        power_response = engine.apply_response(power_row, "ONE", dv)
        self.assertAlmostEqual(power_response.dc_re, 0.25)

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
        self.assertEqual(last_row["axis_lane_id"], "LANE_4")
        self.assertEqual(last_row["axis_step_sequence"], "45,30,20,10,5")
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

        self.assertEqual(dv["dv_axis_lane_id_RE"], "LANE_4")
        self.assertEqual(dv["dv_axis_step_sequence_RE"], "45,30,20,10,5")

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
        self.assertEqual(current.question, "Can you read all the letters in this line, or are they blurry?")

    def test_partial_letter_reading_maps_to_blurry_with_accuracy_confidence(self):
        result = match_response(
            transcript="E G N",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="E G N D H",
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
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "REPEAT")
        self.assertEqual(result.method, "letter_reading_partial_repeat")
        self.assertAlmostEqual(result.confidence, 0.2)

    def test_coarse_compare_clearer_but_still_blurry_maps_to_clear(self):
        result = match_response(
            transcript="they became clear but still bloody",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="A P E O F",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.response_value, "CLEAR")
        self.assertEqual(result.method, "clarity_intent")

    def test_coarse_compare_got_worse_maps_to_blurry(self):
        result = match_response(
            transcript="they got worse",
            state="B",
            available_options=["CLEAR", "BLURRY", "REPEAT"],
            stimulus_letters="A P E O F",
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
            axis_step=5.0,
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
            axis_step=5.0,
        )
        self.assertEqual(row.question, "Please read the line.")

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
        self.assertEqual(re_row.question, "Can you read all the letters in this line, or are they blurry?")

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
            axis_step=5.0,
        )
        self.assertEqual(le_row.question, "Read the line.")

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

        orchestrator.process_response("BLURRY")
        calls.clear()
        response = orchestrator.process_response("BLURRY")

        self.assertEqual(response["state"], "E")
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0], ("power", False, "E", -1.0))
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
        base_sphere = current.re_sph

        blurred = engine.apply_response(current, "BLURRY", dv)
        self.assertAlmostEqual(blurred.ds_re, -0.25)
        self.assertTrue(blurred.coarse_compare_mode)
        self.assertEqual(blurred.next_state, "B")

        compare_row = engine._build_next_row(blurred, dv)
        self.assertIsNotNone(compare_row)
        self.assertEqual(compare_row.state, "B")
        self.assertEqual(compare_row.question, "Did the line become clearer or more blurry?")

        worse = engine.apply_response(compare_row, "BLURRY", dv)
        self.assertAlmostEqual(worse.ds_re, 0.25)
        self.assertEqual(worse.next_state, "E")
        self.assertAlmostEqual((compare_row.re_sph or 0.0) + worse.ds_re, base_sphere or 0.0)

    def test_coarse_clearer_branch_returns_to_reading_and_advances_chart(self):
        dv = self._derive(
            self._patient(
                visit_id="coarse-clearer",
                ar_re=self._rx(-1.50, -0.75, 45),
                lenso_re=None,
            )
        )
        engine = RefractionFSMEngine(self.calibration)
        current = engine.initialize_row("coarse-clearer", dv)

        blurred = engine.apply_response(current, "BLURRY", dv)
        compare_row = engine._build_next_row(blurred, dv)
        self.assertIsNotNone(compare_row)

        clearer = engine.apply_response(compare_row, "CLEAR", dv)
        self.assertEqual(clearer.next_state, "B")
        reading_row = engine._build_next_row(clearer, dv)
        self.assertIsNotNone(reading_row)
        self.assertFalse(reading_row.coarse_compare_mode)
        self.assertEqual(reading_row.chart_param, "70_60_50")

        confirmed = engine.apply_response(reading_row, "CLEAR", dv)
        self.assertEqual(confirmed.coarse_last_confirmed_chart_re, "70_60_50")
        advanced_row = engine._build_next_row(confirmed, dv)
        self.assertIsNotNone(advanced_row)
        self.assertEqual(advanced_row.chart_param, "40_30_25")

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
        self.assertIn("Please compare the two choices.", current.question)

        first_repeat = engine.apply_response(current, "REPEAT", dv)
        second_row = engine._build_next_row(first_repeat, dv)
        self.assertIsNotNone(second_row)
        self.assertEqual(second_row.question, "First, second, or both same?")

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
        self.assertIn("Please compare the two choices.", axis_row.question)

        axis_done = engine.apply_response(axis_row, "SAME", dv)
        self.assertEqual(axis_done.next_state, "F")
        power_row = engine._build_next_row(axis_done, dv)
        self.assertIsNotNone(power_row)
        self.assertEqual(power_row.question, "First option, second option, or both same?")

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
            axis_step=5.0,
        )
        self.assertEqual(row.question, "Red side, green side, or both same?")

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
            axis_step=5.0,
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
        self.assertIn("You are doing great Asha.", response["preface_prompt"])
        self.assertIn("minutes left", response["preface_prompt"])

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


class HindiLocalizationTests(unittest.TestCase):
    def test_hindi_localizes_coarse_read_prompt(self):
        prompt = localized_voice_prompt(
            state="B",
            language="hi",
            retry=False,
            fallback_question="Can you read all the letters in this line, or are they blurry?",
        )
        self.assertEqual(prompt, "क्या आप इस लाइन के सभी अक्षर पढ़ पा रहे हैं, या वे धुंधले हैं?")

    def test_hindi_localizes_coarse_compare_prompt(self):
        prompt = localized_voice_prompt(
            state="B",
            language="hi",
            retry=False,
            fallback_question="Did the line become clearer or more blurry?",
        )
        self.assertEqual(prompt, "क्या लाइन ज़्यादा साफ हुई या और धुंधली हुई?")

    def test_hindi_localizes_jcc_short_prompt(self):
        prompt = localized_voice_prompt(
            state="F",
            language="hi",
            retry=False,
            fallback_question="First option, second option, or both same?",
        )
        self.assertEqual(prompt, "पहला विकल्प, दूसरा विकल्प, या दोनों समान?")

    def test_hindi_localizes_duochrome_and_bino_short_prompts(self):
        duo_prompt = localized_voice_prompt(
            state="G",
            language="hi",
            retry=False,
            fallback_question="Red side, green side, or both same?",
        )
        bino_prompt = localized_voice_prompt(
            state="K",
            language="hi",
            retry=False,
            fallback_question="Top line, bottom line, or both same?",
        )
        self.assertEqual(duo_prompt, "लाल साइड, हरा साइड, या दोनों समान?")
        self.assertEqual(bino_prompt, "ऊपर की लाइन, नीचे की लाइन, या दोनों समान?")

    def test_hindi_option_labels_are_contextual(self):
        self.assertEqual(
            localized_option_label(
                "CLEAR",
                "hi",
                state="B",
                question="Did the line become clearer or more blurry?",
            ),
            "ज़्यादा साफ",
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
                question="Red side, green side, or both same?",
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
                "question": "Please compare the two choices. Which one is clearer or sharper? say first option, second option, both same, or repeat.",
                "options": ["ONE", "TWO", "SAME", "REPEAT"],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(
            payload["question"],
            "कृपया दोनों विकल्पों की तुलना कीजिए। कौन सा ज़्यादा साफ या शार्प दिख रहा है? पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से कहिए।",
        )
        self.assertEqual(
            [item["localized"] for item in payload["labels"]],
            ["पहला विकल्प", "दूसरा विकल्प", "दोनों समान", "फिर से"],
        )


if __name__ == "__main__":
    unittest.main()
