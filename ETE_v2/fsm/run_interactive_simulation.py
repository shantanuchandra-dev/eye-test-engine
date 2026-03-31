from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsm.audio.local_stt import create_local_transcriber, record_audio_clip
from fsm.audio.response_matching import match_response
from fsm.config.calibration_loader import CalibrationLoader
from fsm.engines.derived_variables_engine import DerivedVariablesEngine
from fsm.engines.refraction_fsm_engine import RefractionFSMEngine
from fsm.models.patient import PatientInput
from fsm.models.prescription import EyePrescription
from fsm.simulation.common import seed_final_compare_context
from fsm.simulation.result_writer import create_run_folder, save_json, save_trace_csv

APP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = APP_ROOT / "results"
DEFAULT_CALIBRATION_PATH = APP_ROOT / "config" / "calibration.csv"
DEFAULT_HF_MODEL_PATH = APP_ROOT / "models" / "whisper-large-v3-turbo-hf"
DEFAULT_CT2_MODEL_PATH = APP_ROOT / "models" / "whisper-large-v3-turbo-ct2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the interactive FSM refraction simulator in keyboard or fully local voice mode."
    )
    parser.add_argument(
        "--input-mode",
        choices=["keyboard", "voice-local-fw", "voice-local-hf"],
        default="keyboard",
        help="How to capture responses for each interactive FSM prompt.",
    )
    parser.add_argument(
        "--voice-record-seconds",
        type=float,
        default=5.0,
        help="Recording window per voice response attempt.",
    )
    parser.add_argument(
        "--voice-samplerate",
        type=int,
        default=16000,
        help="Microphone recording sample rate.",
    )
    parser.add_argument(
        "--voice-reprompt-limit",
        type=int,
        default=2,
        help="How many low-confidence voice retries to allow before falling back to keyboard.",
    )
    parser.add_argument(
        "--voice-cpu-threads",
        type=int,
        default=4,
        help="CPU thread cap for local voice transcription.",
    )
    parser.add_argument(
        "--hf-model-path",
        default=str(DEFAULT_HF_MODEL_PATH),
        help="Local path for the Hugging Face Whisper Turbo model.",
    )
    parser.add_argument(
        "--ct2-model-path",
        default=str(DEFAULT_CT2_MODEL_PATH),
        help="Local path for the CTranslate2-converted Whisper Turbo model.",
    )
    parser.add_argument(
        "--voice-speak-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Speak the question aloud before each voice response attempt.",
    )
    parser.add_argument(
        "--voice-beep-before-record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Play a cue immediately before each voice recording window.",
    )
    return parser


def default_interactive_patient(run_id: str) -> PatientInput:
    return PatientInput(
        visit_id=run_id,
        patient_name="Interactive Simulation",
        age=44,
        occupation="",
        screen_time_hours=7.0,
        driving_hours=1.0,
        primary_reason="Blurred distance",
        symptoms_text="Blurred distance, Blurred near",
        satisfaction_with_current_rx="Not satisfied",
        wear_type="Progressive",
        distance_target_preference="",
        priority="Comfort-first",
        near_priority_declared="High",
        last_eye_test_months_ago=18.0,
        rx_change_was_large=True,
        fluctuating_vision_reported=False,
        diabetes=False,
        prior_eye_surgery="None",
        keratoconus=False,
        amblyopia=False,
        infection=False,
        optom_review_flag=False,
        autorefractor_re=EyePrescription(-2.25, -1.25, 90.0),
        autorefractor_le=EyePrescription(-2.00, -1.00, 75.0),
        lenso_re=EyePrescription(-2.50, -1.50, 100.0),
        lenso_le=EyePrescription(-2.00, -1.25, 85.0),
        lenso_add_r=1.0,
        lenso_add_l=1.0,
    )


def _available_options(current) -> list[str]:
    options = [
        current.opt_1,
        current.opt_2,
        current.opt_3,
        current.opt_4,
        current.opt_5,
        current.opt_6,
    ]
    return [option for option in options if option not in ("", None)]


def _print_profile(dv, run_id: str, input_mode: str) -> None:
    print("\n===== INTERACTIVE FSM REFRACTION SIMULATOR =====\n")
    print(f"Run ID: {run_id}")

    print("\nPATIENT PROFILE")
    print(f"Input Mode: {input_mode}")
    print(f"Age Bucket: {dv.dv_age_bucket}")
    print(f"Distance Priority: {dv.dv_distance_priority}")
    print(f"Near Priority: {dv.dv_near_priority}")
    print(f"Stability: {dv.dv_stability_level}")
    print(f"Symptom Risk: {dv.dv_symptom_risk_level}")
    print(f"Medical Risk: {dv.dv_medical_risk_level}")
    print(f"Start Source Policy: {dv.dv_start_source_policy}")
    print(f"Target Distance VA: {dv.dv_target_distance_va}")
    print(f"Near Test Required: {dv.dv_near_test_required}")
    print(f"Expected ADD: {dv.dv_add_expected}")
    print(f"Fogging Policy: {dv.dv_fogging_policy}")
    print(f"Fogging Amount: {dv.dv_fogging_amount_D}")
    print(f"Confidence Requirement: {dv.dv_confidence_requirement}")
    print(f"Start RX RE: {dv.dv_start_rx_RE_sph:.2f} / {dv.dv_start_rx_RE_cyl:.2f} x {dv.dv_start_rx_RE_axis}")
    print(f"Start RX LE: {dv.dv_start_rx_LE_sph:.2f} / {dv.dv_start_rx_LE_cyl:.2f} x {dv.dv_start_rx_LE_axis}")


def _keyboard_response(options: list[str]) -> tuple[str, dict]:
    user = input("\nSelect option number (or type value): ").strip()

    if user.isdigit():
        idx = int(user) - 1
        if idx < len(options):
            response = options[idx]
        else:
            raise ValueError("Invalid option")
    else:
        response = user

    return response, {
        "input_mode": "keyboard",
        "raw_input_text": user,
        "normalized_input_text": user.strip().lower(),
        "response_match_method": "manual",
        "response_match_confidence": 1.0,
        "response_match_canonical_label": response,
        "response_match_accepted": True,
        "response_attempt_count": 1,
        "response_audio_path": None,
        "stt_backend": None,
    }


def _voice_response(
    *,
    transcriber,
    current,
    options: list[str],
    results_folder: Path,
    record_seconds: float,
    samplerate: int,
    reprompt_limit: int,
    speak_prompts: bool,
    beep_before_record: bool,
) -> tuple[str, dict]:
    audio_dir = results_folder / "voice_audio"

    for attempt in range(1, reprompt_limit + 2):
        audio_path = audio_dir / f"step_{current.step:03d}_attempt_{attempt}.wav"

        if speak_prompts:
            prompt_to_speak = current.question.strip()
            if attempt > 1:
                prompt_to_speak = (
                    f"{match.reprompt_text if 'match' in locals() else current.question.strip()} "
                    f"You have {record_seconds:.0f} seconds to answer."
                )
            _speak_text(prompt_to_speak)

        if beep_before_record:
            _play_beep()
            time.sleep(0.15)

        print(f"\nVOICE INPUT: listening for {record_seconds:.1f}s ...")
        record_audio_clip(
            output_path=audio_path,
            seconds=record_seconds,
            samplerate=samplerate,
        )

        try:
            transcript = transcriber.transcribe(audio_path)
        except Exception as exc:
            print(f"Voice transcription failed: {exc}")
            transcript = ""

        match = match_response(
            transcript=transcript,
            state=current.state,
            available_options=options,
            question=current.question,
        )

        print(f"Heard: {transcript!r}")
        if match.accepted and match.response_value is not None:
            print(
                "Matched response: "
                f"{match.response_value} ({match.method}, confidence={match.confidence:.2f})"
            )
            return match.response_value, {
                "input_mode": transcriber.backend_name,
                "raw_input_text": transcript,
                "normalized_input_text": match.normalized_text,
                "response_match_method": match.method,
                "response_match_confidence": match.confidence,
                "response_match_canonical_label": match.canonical_label,
                "response_match_accepted": True,
                "response_attempt_count": attempt,
                "response_audio_path": str(audio_path),
                "stt_backend": transcriber.backend_name,
                "prompt_spoken": speak_prompts,
                "beep_before_record": beep_before_record,
                "voice_record_seconds": float(record_seconds),
            }

        print(
            "Could not confidently map the voice response"
            f" ({match.method}, confidence={match.confidence:.2f}, reason={match.reason})."
        )
        print(match.reprompt_text)

    print("\nFalling back to keyboard input for this step.")
    response, meta = _keyboard_response(options)
    meta["input_mode"] = f"{transcriber.backend_name}_fallback_keyboard"
    meta["stt_backend"] = transcriber.backend_name
    meta["response_attempt_count"] = reprompt_limit + 1
    meta["prompt_spoken"] = speak_prompts
    meta["beep_before_record"] = beep_before_record
    meta["voice_record_seconds"] = float(record_seconds)
    return response, meta


def _speak_text(text: str) -> None:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return

    if sys.platform.startswith("win"):
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$text = [Console]::In.ReadToEnd(); "
            "if (-not [string]::IsNullOrWhiteSpace($text)) { "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Speak($text); "
            "$speaker.Dispose() }"
        )
        command = ["powershell", "-NoProfile", "-Command", script]
        run_kwargs = {
            "input": cleaned,
            "text": True,
        }
    else:
        command = ["say", cleaned]
        run_kwargs = {}

    try:
        subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **run_kwargs,
        )
    except Exception:
        pass


def _play_beep() -> None:
    try:
        subprocess.run(
            ["osascript", "-e", "beep"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        sys.stdout.write("\a")
        sys.stdout.flush()


def main() -> None:
    args = build_parser().parse_args()

    calibration = CalibrationLoader(str(DEFAULT_CALIBRATION_PATH))
    engine = RefractionFSMEngine(calibration)
    dv_engine = DerivedVariablesEngine(calibration)

    results_folder, run_id = create_run_folder(RESULTS_ROOT, "interactive")
    patient = default_interactive_patient(run_id)
    dv = dv_engine.derive(patient)

    current = engine.initialize_row(
        visit_id=run_id,
        dv=dv,
        ar_re=patient.autorefractor_re,
        ar_le=patient.autorefractor_le,
    )
    seed_final_compare_context(current, patient)

    transcriber = None
    if args.input_mode != "keyboard":
        print(f"\nLoading local voice backend: {args.input_mode} ...")
        transcriber = create_local_transcriber(
            backend=args.input_mode,
            hf_model_path=args.hf_model_path,
            ct2_model_path=args.ct2_model_path,
            cpu_threads=args.voice_cpu_threads,
        )
        print("Local voice backend ready.")

    rows = []

    _print_profile(dv, run_id, args.input_mode)

    while True:
        print("\n--------------------------------")
        print(f"STEP: {current.step}")
        print(f"STATE: {current.state} | {current.phase_name}")
        print(f"CHART: {current.chart_param}")
        print(f"EYE: {current.eye}")

        print("\nCURRENT POWER")

        re_add = current.add_r if current.add_r is not None else 0.0
        le_add = current.add_l if current.add_l is not None else 0.0

        print(f"RE Distance: {current.re_sph:.2f} / {current.re_cyl:.2f} x {current.re_axis}")
        print(f"LE Distance: {current.le_sph:.2f} / {current.le_cyl:.2f} x {current.le_axis}")

        if current.state in ("P", "Q", "R"):
            print(f"RE Add: +{re_add:.2f}")
            print(f"LE Add: +{le_add:.2f}")

        print("\nQUESTION:")
        print(current.question)

        options = _available_options(current)

        print("\nOPTIONS:")
        for i, option in enumerate(options, start=1):
            print(f"{i}. {option}")

        try:
            if args.input_mode == "keyboard":
                response, meta = _keyboard_response(options)
            else:
                response, meta = _voice_response(
                    transcriber=transcriber,
                    current=current,
                    options=options,
                    results_folder=results_folder,
                    record_seconds=args.voice_record_seconds,
                    samplerate=args.voice_samplerate,
                    reprompt_limit=args.voice_reprompt_limit,
                    speak_prompts=args.voice_speak_prompts,
                    beep_before_record=args.voice_beep_before_record,
                )
        except ValueError as exc:
            print(str(exc))
            continue

        finalized = engine.apply_response(
            current=current,
            response_value=response,
            dv=dv,
            ar_re=patient.autorefractor_re,
            ar_le=patient.autorefractor_le,
        )

        row_dict = finalized.__dict__.copy()
        row_dict["test_id"] = run_id
        row_dict.update(meta)
        rows.append(row_dict)

        print(f"\nResponse recorded: {response}")
        print(f"Next state: {finalized.next_state}")

        if finalized.next_state in ("END", "ESCALATE"):
            print("\nTEST TERMINATED:", finalized.next_state)
            break

        next_row = engine._build_next_row(finalized, dv)
        if next_row is None:
            print("\nFSM finished")
            break

        current = next_row

    trace_path = save_trace_csv(rows, results_folder, "trace.csv")

    summary = {
        "test_id": run_id,
        "simulation_type": "interactive",
        "input_mode": args.input_mode,
        "voice_cpu_threads": int(args.voice_cpu_threads),
        "voice_record_seconds": float(args.voice_record_seconds),
        "total_steps": len(rows),
        "final_state": rows[-1]["state"] if rows else None,
        "termination_state": rows[-1]["next_state"] if rows else None,
        "final_re_sph": rows[-1]["re_sph"] if rows else None,
        "final_re_cyl": rows[-1]["re_cyl"] if rows else None,
        "final_re_axis": rows[-1]["re_axis"] if rows else None,
        "final_le_sph": rows[-1]["le_sph"] if rows else None,
        "final_le_cyl": rows[-1]["le_cyl"] if rows else None,
        "final_le_axis": rows[-1]["le_axis"] if rows else None,
        "trace_file": str(trace_path),
    }
    save_json(summary, results_folder, "summary.json")

    print(f"\nSimulation saved to {trace_path}")


if __name__ == "__main__":
    main()
