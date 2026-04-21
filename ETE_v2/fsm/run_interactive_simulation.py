from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsm.audio.local_stt import (
    create_local_transcriber,
    record_audio_clip,
    record_audio_until_silence,
    resolve_input_device,
)
from fsm.audio.response_matching import (
    infer_response_language,
    localized_language_selection_prompt,
    localized_voice_prompt,
    match_language_choice,
    match_response,
)
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
DEFAULT_PRIMARY_CT2_MODEL_PATH = APP_ROOT / "models" / "whisper-base-ct2"


DISTANCE_CHART_STIMULI = {
    "200_150": [["E", "N", "H"], ["S", "L", "C"]],
    "200150": [["E", "N", "H"], ["S", "L", "C"]],
    "100_80": [["H", "B", "V"], ["P", "H", "T"]],
    "100_90": [["H", "B", "V"], ["P", "H", "T"]],
    "10080": [["H", "B", "V"], ["P", "H", "T"]],
    "70_60_50": [["V", "L", "N", "E", "A"], ["D", "A", "O", "F", "C"], ["E", "G", "N", "D", "H"]],
    "706050": [["V", "L", "N", "E", "A"], ["D", "A", "O", "F", "C"], ["E", "G", "N", "D", "H"]],
    "40_30_25": [["F", "Z", "B", "D", "E"], ["O", "F", "L", "C", "T"], ["A", "P", "E", "O", "F"]],
    "403025": [["F", "Z", "B", "D", "E"], ["O", "F", "L", "C", "T"], ["A", "P", "E", "O", "F"]],
    "20_15_10": [["T", "Z", "V", "E", "C"], ["O", "H", "P", "N", "T"], ["V", "L", "F", "T", "H"]],
    "201510": [["T", "Z", "V", "E", "C"], ["O", "H", "P", "N", "T"], ["V", "L", "F", "T", "H"]],
    "20_20_20": [["E", "V", "O", "T", "L"], ["T", "B", "G", "A", "B"], ["H", "N", "F", "Z", "C"]],
    "202020": [["E", "V", "O", "T", "L"], ["T", "B", "G", "A", "B"], ["H", "N", "F", "Z", "C"]],
    "25_20_15": [["D", "F", "N", "P", "T"], ["P", "H", "U", "N", "T"], ["F", "D", "S", "L", "N"]],
    "252015": [["D", "F", "N", "P", "T"], ["P", "H", "U", "N", "T"], ["F", "D", "S", "L", "N"]],
}

NEAR_LAST_LINE_LETTERS = "A P E O R F D Z"

GAMEPAD_PROFILE_MAPS = {
    "xbox_abxy": {
        1: "1",  # B -> option 1
        0: "2",  # A -> option 2
        2: "3",  # X -> option 3
        3: "4",  # Y -> option 4
    }
}

GAMEPAD_HID_XBOX_BUTTON_MASKS = {
    "xbox_abxy": {
        0x2000: "1",  # B -> option 1
        0x1000: "2",  # A -> option 2
        0x4000: "3",  # X -> option 3
        0x8000: "4",  # Y -> option 4
    }
}

GAMEPAD_XINPUT_XBOX_BUTTON_MASKS = {
    "xbox_abxy": {
        0x2000: "1",  # B -> option 1
        0x1000: "2",  # A -> option 2
        0x4000: "3",  # X -> option 3
        0x8000: "4",  # Y -> option 4
    }
}


def _int_arg_autobase(value: str) -> int:
    return int(str(value), 0)


class _CascadeTranscriber:
    backend_name = "cascade_faster_whisper"

    def __init__(self, *, primary_transcriber, fallback_transcriber, accept_confidence: float) -> None:
        self.primary_transcriber = primary_transcriber
        self.fallback_transcriber = fallback_transcriber
        self.accept_confidence = float(accept_confidence)
        self.requested_language = getattr(primary_transcriber, "requested_language", None)
        self.primary_model_path = getattr(primary_transcriber, "model_path", None)
        self.fallback_model_path = getattr(fallback_transcriber, "model_path", None)

    def transcribe_result(
        self,
        audio_path: str | Path,
        *,
        language_override: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ):
        return self.primary_transcriber.transcribe_result(
            audio_path,
            language_override=language_override,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )


def _resolve_model_path(requested_path: str, model_dir_name: str) -> str:
    path = Path(requested_path).expanduser()
    if path.exists():
        return str(path)

    search_roots = []
    for candidate in (APP_ROOT, APP_ROOT.parent, *APP_ROOT.parents):
        if candidate not in search_roots:
            search_roots.append(candidate)

    for root in search_roots:
        direct = root / "models" / model_dir_name
        if direct.exists():
            return str(direct)

        fsm_root = root / "FSM"
        if fsm_root.exists():
            for model_path in sorted(fsm_root.glob(f"*/models/{model_dir_name}")):
                if model_path.exists():
                    return str(model_path)

    return requested_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the interactive FSM refraction simulator in keyboard or fully local voice mode."
    )
    parser.add_argument(
        "--input-mode",
        choices=["keyboard", "voice-local-fw", "voice-local-hf", "voice-local-cascade"],
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
        "--voice-input-device",
        default="default",
        help="Microphone input device index/name. Use 'auto' to choose a likely real microphone.",
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
        "--voice-primary-ct2-model-path",
        default=str(DEFAULT_PRIMARY_CT2_MODEL_PATH),
        help="Primary fast CT2 model path for voice-local-cascade mode.",
    )
    parser.add_argument(
        "--cascade-accept-confidence",
        type=float,
        default=0.78,
        help="Matcher confidence needed to accept the primary cascade transcript without fallback.",
    )
    parser.add_argument(
        "--voice-fw-optimized-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use low-latency Faster Whisper decode settings for the interactive simulator.",
    )
    parser.add_argument(
        "--voice-fw-internal-vad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run Faster Whisper's internal VAD. Usually off because simulator VAD already trims speech.",
    )
    parser.add_argument(
        "--voice-fw-max-new-tokens",
        type=int,
        default=24,
        help="Maximum generated tokens for each short ophthalmic voice response when optimized decode is enabled.",
    )
    parser.add_argument(
        "--voice-language",
        default="auto",
        help="Whisper language code/name to force, or 'auto' for language detection.",
    )
    parser.add_argument(
        "--voice-ask-language-at-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask English/Hindi before the first FSM step when --voice-language auto is used.",
    )
    parser.add_argument(
        "--voice-capture-mode",
        choices=["vad", "fixed"],
        default="vad",
        help="Use fixed clip recording or stop automatically when the speaker falls silent.",
    )
    parser.add_argument(
        "--voice-start-timeout-seconds",
        type=float,
        default=2.5,
        help="How long VAD waits for speech to begin before ending the attempt.",
    )
    parser.add_argument(
        "--voice-end-silence-seconds",
        type=float,
        default=2.0,
        help="How much trailing silence ends a VAD response after speech is detected.",
    )
    parser.add_argument(
        "--voice-min-speech-seconds",
        type=float,
        default=0.25,
        help="Minimum detected speech before silence can end the recording.",
    )
    parser.add_argument(
        "--voice-max-speech-seconds",
        type=float,
        default=4.0,
        help="Safety cap after speech starts, used only if VAD never sees trailing silence.",
    )
    parser.add_argument(
        "--voice-silence-threshold",
        type=float,
        default=0.015,
        help="Input activity threshold for VAD capture.",
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
    parser.add_argument(
        "--gamepad-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Listen for native gamepad button presses as live manual overrides during voice prompts.",
    )
    parser.add_argument(
        "--gamepad-profile",
        choices=sorted(GAMEPAD_PROFILE_MAPS.keys()),
        default="xbox_abxy",
        help="Button-to-option mapping profile for native gamepad input.",
    )
    parser.add_argument(
        "--gamepad-driver",
        choices=["auto", "xinput", "pygame", "hidapi"],
        default="auto",
        help="Native controller backend to use when gamepad input is enabled.",
    )
    parser.add_argument(
        "--gamepad-device-index",
        type=int,
        default=0,
        help="Zero-based controller index to use when native gamepad input is enabled.",
    )
    parser.add_argument(
        "--gamepad-vendor-id",
        type=_int_arg_autobase,
        default=None,
        help="Optional USB vendor ID filter for HID gamepad discovery, for example 0x045e.",
    )
    parser.add_argument(
        "--gamepad-product-id",
        type=_int_arg_autobase,
        default=None,
        help="Optional USB product ID filter for HID gamepad discovery, for example 0x028e.",
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


def _displayed_distance_chart_lines(state: str, chart_param: str) -> list[list[str]] | None:
    normalized = str(chart_param or "").replace("-", "_").replace("/", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    letters = DISTANCE_CHART_STIMULI.get(normalized) or DISTANCE_CHART_STIMULI.get(normalized.replace("_", ""))
    if not letters:
        return None
    if state in {"B", "C", "D", "L"}:
        return [letters[-1]]
    return letters


def _stimulus_letters_for_row(current) -> str:
    if current.state in {"P", "Q", "R"}:
        return NEAR_LAST_LINE_LETTERS

    lines = _displayed_distance_chart_lines(current.state, current.chart_param)
    if not lines:
        return ""
    return "\n".join(" ".join(line) for line in lines)


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds or 0.0), 0.0)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:05.2f}s"
    hours, remaining_minutes = divmod(int(minutes), 60)
    return f"{hours}h {remaining_minutes:02d}m {remaining_seconds:05.2f}s"


def _is_repeat_response(response_value: Optional[str]) -> bool:
    return str(response_value or "").strip().upper() in {"REPEAT", "__REPEAT__"}


def _language_display_name(language_code: Optional[str]) -> str:
    if not language_code or language_code == "auto":
        return "Auto"
    normalized = str(language_code).strip().lower()
    if normalized == "en":
        return "English"
    if normalized == "hi":
        return "Hindi"
    return str(language_code)


def _localized_listen_suffix(language_code: Optional[str], record_seconds: float) -> str:
    if language_code == "hi":
        return f" आपके पास जवाब देने के लिए {record_seconds:.0f} सेकंड हैं।"
    return f" You have {record_seconds:.0f} seconds to answer."


def _dedupe_words(words: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for word in words:
        cleaned = " ".join(str(word or "").replace("_", " ").split()).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _phase_stt_guidance(current, options: list[str], stimulus_letters: str) -> tuple[str, str]:
    state = str(getattr(current, "state", "") or "")
    base_terms = [
        "clear",
        "blurry",
        "repeat",
        "yes",
        "no",
        "better",
        "same",
        "both same",
        "first option",
        "second option",
        "option one",
        "option two",
        "green side",
        "red side",
        "right side",
        "top line",
        "bottom line",
        "English",
        "Hindi",
        "साफ",
        "धुंधला",
        "दोहराएं",
        "हाँ",
        "नहीं",
        "पहला विकल्प",
        "दूसरा विकल्प",
        "दोनों समान",
        "हरा",
        "लाल",
        "ऊपर",
        "नीचे",
        "अंग्रेजी",
        "हिंदी",
    ]
    phase_terms = list(options)
    if state in {"B", "C", "D", "L", "P", "Q", "R"} and stimulus_letters:
        phase_terms.append(stimulus_letters)
    if state in {"F", "G"}:
        phase_terms.extend(["first option", "second option", "both same"])
    if state == "H":
        phase_terms.extend(["green side", "red side", "right side", "both same"])
    if state == "I":
        phase_terms.extend(["top line", "bottom line", "both same"])

    hotwords = ", ".join(_dedupe_words(phase_terms + base_terms))
    initial_prompt = (
        "Short eye test response. Expected words are ophthalmic choices, not general dictation: "
        f"{hotwords}."
    )
    return initial_prompt, hotwords


def _phase_voice_capture_settings(
    *,
    state: str,
    default_record_seconds: float,
    default_end_silence_seconds: float,
) -> tuple[float, float]:
    if state in {"B", "D"}:
        return max(default_record_seconds, 15.0), default_end_silence_seconds
    return default_record_seconds, default_end_silence_seconds


def _capture_audio_attempt(
    *,
    audio_path: Path,
    capture_mode: str,
    record_seconds: float,
    samplerate: int,
    input_device: Optional[int | str],
    start_timeout_seconds: float,
    end_silence_seconds: float,
    min_speech_seconds: float,
    max_speech_seconds: Optional[float],
    silence_threshold: float,
    stop_requested=None,
) -> dict:
    if capture_mode == "fixed":
        selected_device, selected_device_index, selected_device_name, selected_device_reason = resolve_input_device(
            requested_device=input_device,
            samplerate=samplerate,
            channels=1,
            dtype="float32",
        )
        recording_started = time.perf_counter()
        record_audio_clip(
            output_path=audio_path,
            seconds=record_seconds,
            samplerate=samplerate,
            input_device=selected_device,
            stop_requested=stop_requested,
        )
        recording_elapsed = time.perf_counter() - recording_started
        return {
            "audio_path": str(audio_path),
            "recording_seconds": recording_elapsed,
            "speech_detected": None,
            "stop_reason": "manual_override" if stop_requested is not None and stop_requested() else "fixed_window",
            "peak_amplitude": None,
            "input_device_index": selected_device_index,
            "input_device_name": selected_device_name,
            "input_device_selection": selected_device_reason,
        }

    capture_result = record_audio_until_silence(
        output_path=audio_path,
        max_seconds=record_seconds,
        samplerate=samplerate,
        input_device=input_device,
        silence_threshold=silence_threshold,
        end_silence_seconds=end_silence_seconds,
        min_speech_seconds=min_speech_seconds,
        max_speech_seconds=max_speech_seconds,
        start_timeout_seconds=start_timeout_seconds,
        stop_requested=stop_requested,
    )
    return {
        "audio_path": capture_result.output_path,
        "recording_seconds": capture_result.duration_seconds,
        "speech_detected": capture_result.speech_detected,
        "stop_reason": capture_result.stop_reason,
        "peak_amplitude": capture_result.peak_amplitude,
        "input_device_index": capture_result.input_device_index,
        "input_device_name": capture_result.input_device_name,
        "input_device_selection": capture_result.input_device_selection,
        "noise_floor_rms": capture_result.noise_floor_rms,
        "speech_start_threshold_rms": capture_result.speech_start_threshold_rms,
        "speech_continue_threshold_rms": capture_result.speech_continue_threshold_rms,
    }


def _manual_override_response(
    *,
    selected_key: str,
    current,
    options: list[str],
    stimulus_letters: Optional[str],
    override_source: str,
    override_detail: Optional[str] = None,
) -> tuple[str, dict]:
    idx = int(selected_key) - 1
    if idx < 0 or idx >= len(options):
        raise ValueError("Invalid manual override option")

    transcript = options[idx]
    match = match_response(
        transcript=transcript,
        state=current.state,
        available_options=options,
        question=current.question,
        stimulus_letters=stimulus_letters,
    )
    if not match.accepted or match.response_value is None:
        raise ValueError(match.reprompt_text)

    return match.response_value, {
        "input_mode": override_source,
        "raw_input_text": transcript,
        "normalized_input_text": match.normalized_text,
        "response_match_method": override_source,
        "response_match_confidence": 1.0,
        "response_match_canonical_label": match.canonical_label,
        "response_match_accepted": True,
        "response_audio_path": None,
        "stt_backend": None,
        "stimulus_letters": stimulus_letters,
        "manual_override_used": True,
        "manual_override_key": selected_key,
        "manual_override_source": override_source,
        "manual_override_detail": override_detail,
    }


class _ManualOverrideListener:
    def __init__(
        self,
        valid_keys: list[str],
        *,
        enable_gamepad: bool = False,
        gamepad_profile: str = "xbox_abxy",
        gamepad_driver: str = "auto",
        gamepad_device_index: int = 0,
        gamepad_vendor_id: Optional[int] = None,
        gamepad_product_id: Optional[int] = None,
    ):
        self.valid_keys = {str(key) for key in valid_keys}
        self._stop_event = threading.Event()
        self._selected_key: Optional[str] = None
        self._selected_source: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self.enabled = False
        self.enable_gamepad = enable_gamepad
        self.gamepad_profile = gamepad_profile
        self.gamepad_driver = gamepad_driver
        self.gamepad_device_index = max(int(gamepad_device_index), 0)
        self.gamepad_vendor_id = gamepad_vendor_id
        self.gamepad_product_id = gamepad_product_id
        self.keyboard_enabled = False
        self.gamepad_enabled = False
        self.gamepad_name: Optional[str] = None
        self.gamepad_backend: Optional[str] = None
        self._pygame = None
        self._joystick = None
        self._gamepad_button_map: dict[int, str] = {}
        self._gamepad_button_state: dict[int, bool] = {}
        self._hid_device = None
        self._hid_button_mask_map: dict[int, str] = {}
        self._hid_button_state: dict[int, bool] = {}
        self._xinput = None
        self._xinput_user_index: Optional[int] = None
        self._xinput_button_mask_map: dict[int, str] = {}
        self._xinput_button_state: dict[int, bool] = {}

    def start(self) -> "_ManualOverrideListener":
        if not self.valid_keys:
            return self
        self.keyboard_enabled = bool(hasattr(sys.stdin, "isatty") and sys.stdin.isatty())
        if self.enable_gamepad:
            self._init_gamepad()
        if not self.keyboard_enabled and not self.gamepad_enabled:
            return self
        try:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self.enabled = True
        except Exception:
            self.enabled = False
        return self

    def stop(self) -> Optional[str]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        self._shutdown_gamepad()
        return self._selected_key

    def stop_requested(self) -> bool:
        return self._selected_key is not None or self._stop_event.is_set()

    def selected_source(self) -> Optional[str]:
        return self._selected_source

    def _run(self) -> None:
        if self.keyboard_enabled and sys.platform.startswith("win"):
            self._run_windows()
        elif self.keyboard_enabled:
            self._run_posix()
        else:
            self._run_gamepad_only()

    def _init_gamepad(self) -> None:
        if self.gamepad_driver != "auto":
            backends = [self.gamepad_driver]
        elif sys.platform.startswith("win"):
            backends = ["xinput", "hidapi", "pygame"]
        else:
            backends = ["pygame", "hidapi"]
        for backend in backends:
            if backend == "xinput" and self._init_gamepad_xinput():
                return
            if backend == "pygame" and self._init_gamepad_pygame():
                return
            if backend == "hidapi" and self._init_gamepad_hidapi():
                return
        self._shutdown_gamepad()

    def _init_gamepad_xinput(self) -> bool:
        if not sys.platform.startswith("win"):
            return False
        try:
            import ctypes

            class _XInputGamepad(ctypes.Structure):
                _fields_ = [
                    ("wButtons", ctypes.c_ushort),
                    ("bLeftTrigger", ctypes.c_ubyte),
                    ("bRightTrigger", ctypes.c_ubyte),
                    ("sThumbLX", ctypes.c_short),
                    ("sThumbLY", ctypes.c_short),
                    ("sThumbRX", ctypes.c_short),
                    ("sThumbRY", ctypes.c_short),
                ]

            class _XInputState(ctypes.Structure):
                _fields_ = [
                    ("dwPacketNumber", ctypes.c_ulong),
                    ("Gamepad", _XInputGamepad),
                ]

            dll = None
            for library_name in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"):
                try:
                    dll = ctypes.WinDLL(library_name)
                    break
                except OSError:
                    continue
            if dll is None:
                return False

            dll.XInputGetState.argtypes = [ctypes.c_uint, ctypes.POINTER(_XInputState)]
            dll.XInputGetState.restype = ctypes.c_uint
            user_index = int(self.gamepad_device_index)
            state = _XInputState()
            if int(dll.XInputGetState(user_index, ctypes.byref(state))) != 0:
                return False

            mask_map = {
                int(mask): str(mapped_key)
                for mask, mapped_key in GAMEPAD_XINPUT_XBOX_BUTTON_MASKS.get(self.gamepad_profile, {}).items()
                if str(mapped_key) in self.valid_keys
            }
            if not mask_map:
                return False

            self._xinput = {"ctypes": ctypes, "dll": dll, "state_cls": _XInputState}
            self._xinput_user_index = user_index
            self._xinput_button_mask_map = mask_map
            self._xinput_button_state = {mask: False for mask in mask_map}
            self.gamepad_name = f"XInput Controller {user_index}"
            self.gamepad_backend = "xinput"
            self.gamepad_enabled = True
            return True
        except Exception:
            self._shutdown_gamepad()
            return False

    def _init_gamepad_pygame(self) -> bool:
        try:
            os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            import pygame

            pygame.display.init()
            pygame.joystick.init()
            pygame.event.pump()
            if int(pygame.joystick.get_count()) <= self.gamepad_device_index:
                return False

            joystick = pygame.joystick.Joystick(self.gamepad_device_index)
            joystick.init()
            button_map = {
                int(button_index): str(mapped_key)
                for button_index, mapped_key in GAMEPAD_PROFILE_MAPS.get(self.gamepad_profile, {}).items()
                if str(mapped_key) in self.valid_keys
            }
            if not button_map:
                return False

            self._pygame = pygame
            self._joystick = joystick
            self._gamepad_button_map = button_map
            self._gamepad_button_state = {button_index: False for button_index in button_map}
            self.gamepad_name = str(joystick.get_name() or "Gamepad")
            self.gamepad_backend = "pygame"
            self.gamepad_enabled = True
            return True
        except Exception:
            self._shutdown_gamepad()
            return False

    def _init_gamepad_hidapi(self) -> bool:
        try:
            import hid

            devices = []
            for device in hid.enumerate():
                usage_page = int(device.get("usage_page") or 0)
                usage = int(device.get("usage") or 0)
                if usage_page != 1 or usage not in {4, 5}:
                    continue
                vendor_id = int(device.get("vendor_id") or 0)
                product_id = int(device.get("product_id") or 0)
                if self.gamepad_vendor_id is not None and vendor_id != int(self.gamepad_vendor_id):
                    continue
                if self.gamepad_product_id is not None and product_id != int(self.gamepad_product_id):
                    continue
                devices.append(device)

            if len(devices) <= self.gamepad_device_index:
                return False

            device_info = devices[self.gamepad_device_index]
            hid_device = hid.device()
            hid_device.open_path(device_info["path"])
            hid_device.set_nonblocking(1)
            mask_map = {
                int(mask): str(mapped_key)
                for mask, mapped_key in GAMEPAD_HID_XBOX_BUTTON_MASKS.get(self.gamepad_profile, {}).items()
                if str(mapped_key) in self.valid_keys
            }
            if not mask_map:
                hid_device.close()
                return False

            self._hid_device = hid_device
            self._hid_button_mask_map = mask_map
            self._hid_button_state = {mask: False for mask in mask_map}
            manufacturer = str(device_info.get("manufacturer_string") or "").strip()
            product = str(device_info.get("product_string") or "").strip() or "Gamepad"
            self.gamepad_name = f"{manufacturer} {product}".strip()
            self.gamepad_backend = "hidapi"
            self.gamepad_enabled = True
            return True
        except Exception:
            self._shutdown_gamepad()
            return False

    def _shutdown_gamepad(self) -> None:
        if self._joystick is not None:
            try:
                self._joystick.quit()
            except Exception:
                pass
        if self._pygame is not None:
            try:
                self._pygame.joystick.quit()
            except Exception:
                pass
            try:
                self._pygame.quit()
            except Exception:
                pass
        if self._hid_device is not None:
            try:
                self._hid_device.close()
            except Exception:
                pass
        self._joystick = None
        self._pygame = None
        self._hid_device = None
        self._xinput = None
        self._xinput_user_index = None
        self._gamepad_button_map = {}
        self._gamepad_button_state = {}
        self._hid_button_mask_map = {}
        self._hid_button_state = {}
        self._xinput_button_mask_map = {}
        self._xinput_button_state = {}
        self.gamepad_enabled = False
        self.gamepad_backend = None

    def _poll_gamepad(self) -> bool:
        if not self.gamepad_enabled:
            return False
        try:
            if self.gamepad_backend == "xinput":
                return self._poll_gamepad_xinput()
            if self.gamepad_backend == "pygame":
                return self._poll_gamepad_pygame()
            if self.gamepad_backend == "hidapi":
                return self._poll_gamepad_hidapi()
        except Exception:
            self._shutdown_gamepad()
        return False

    def _poll_gamepad_pygame(self) -> bool:
        if self._pygame is None or self._joystick is None:
            return False
        self._pygame.event.pump()
        for button_index, mapped_key in self._gamepad_button_map.items():
            pressed = bool(self._joystick.get_button(button_index))
            previous = self._gamepad_button_state.get(button_index, False)
            self._gamepad_button_state[button_index] = pressed
            if pressed and not previous and mapped_key in self.valid_keys:
                self._selected_key = mapped_key
                self._selected_source = "gamepad"
                self._stop_event.set()
                return True
        return False

    def _poll_gamepad_hidapi(self) -> bool:
        if self._hid_device is None:
            return False
        data = self._hid_device.read(64)
        if not data:
            return False
        button_word = None
        if len(data) >= 4:
            button_word = int(data[2]) | (int(data[3]) << 8)
        elif len(data) >= 2:
            button_word = int(data[0]) | (int(data[1]) << 8)
        if button_word is None:
            return False
        for mask, mapped_key in self._hid_button_mask_map.items():
            pressed = bool(button_word & int(mask))
            previous = self._hid_button_state.get(mask, False)
            self._hid_button_state[mask] = pressed
            if pressed and not previous and mapped_key in self.valid_keys:
                self._selected_key = mapped_key
                self._selected_source = "gamepad"
                self._stop_event.set()
                return True
        return False

    def _poll_gamepad_xinput(self) -> bool:
        if self._xinput is None or self._xinput_user_index is None:
            return False
        ctypes = self._xinput["ctypes"]
        dll = self._xinput["dll"]
        state = self._xinput["state_cls"]()
        if int(dll.XInputGetState(int(self._xinput_user_index), ctypes.byref(state))) != 0:
            return False
        button_word = int(state.Gamepad.wButtons)
        for mask, mapped_key in self._xinput_button_mask_map.items():
            pressed = bool(button_word & int(mask))
            previous = self._xinput_button_state.get(mask, False)
            self._xinput_button_state[mask] = pressed
            if pressed and not previous and mapped_key in self.valid_keys:
                self._selected_key = mapped_key
                self._selected_source = "gamepad"
                self._stop_event.set()
                return True
        return False

    def _run_windows(self) -> None:
        import msvcrt

        while not self._stop_event.is_set():
            if self._poll_gamepad():
                return
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in self.valid_keys:
                    self._selected_key = char
                    self._selected_source = "keyboard"
                    self._stop_event.set()
                    return
            time.sleep(0.02)

    def _run_posix(self) -> None:
        import select
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        new_settings = termios.tcgetattr(fd)
        new_settings[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(fd, termios.TCSANOW, new_settings)
        try:
            while not self._stop_event.is_set():
                if self._poll_gamepad():
                    return
                readable, _w, _x = select.select([sys.stdin], [], [], 0.05)
                if not readable:
                    continue
                char = sys.stdin.read(1)
                if char in self.valid_keys:
                    self._selected_key = char
                    self._selected_source = "keyboard"
                    self._stop_event.set()
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old_settings)

    def _run_gamepad_only(self) -> None:
        while not self._stop_event.is_set():
            if self._poll_gamepad():
                return
            time.sleep(0.02)


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


def _run_stt_match_once(
    *,
    stt_transcriber,
    audio_path: Path,
    current,
    options: list[str],
    stimulus_letters: str,
    language_hint: Optional[str],
) -> dict:
    initial_prompt, hotwords = _phase_stt_guidance(current, options, stimulus_letters)
    stt_started = time.perf_counter()
    try:
        transcript_result = stt_transcriber.transcribe_result(
            audio_path,
            language_override=language_hint,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )
        transcript = transcript_result.text
    except Exception as exc:
        print(f"Voice transcription failed with {getattr(stt_transcriber, 'backend_name', 'stt')}: {exc}")
        transcript_result = None
        transcript = ""
    stt_seconds = time.perf_counter() - stt_started

    detected_language = getattr(transcript_result, "detected_language", None)
    detected_language_probability = getattr(transcript_result, "language_probability", None)
    inferred_language = infer_response_language(
        transcript=transcript,
        detected_language=detected_language,
        detected_language_probability=detected_language_probability,
    )

    match_started = time.perf_counter()
    match = match_response(
        transcript=transcript,
        state=current.state,
        available_options=options,
        question=current.question,
        stimulus_letters=stimulus_letters,
    )
    match_seconds = time.perf_counter() - match_started

    return {
        "transcript_result": transcript_result,
        "transcript": transcript,
        "detected_language": detected_language,
        "detected_language_probability": detected_language_probability,
        "inferred_language": inferred_language,
        "match": match,
        "stt_seconds": stt_seconds,
        "match_seconds": match_seconds,
        "backend_name": getattr(stt_transcriber, "backend_name", "stt"),
        "stt_initial_prompt": initial_prompt,
        "stt_hotwords": hotwords,
    }


def _cascade_primary_is_good_enough(match, accept_confidence: float) -> bool:
    if not match.accepted or match.response_value is None:
        return False
    if _is_repeat_response(match.response_value):
        return True
    return float(match.confidence or 0.0) >= float(accept_confidence)


def _run_stt_match_with_optional_cascade(
    *,
    transcriber,
    audio_path: Path,
    current,
    options: list[str],
    stimulus_letters: str,
    language_hint: Optional[str],
) -> dict:
    primary_transcriber = getattr(transcriber, "primary_transcriber", transcriber)
    fallback_transcriber = getattr(transcriber, "fallback_transcriber", None)
    accept_confidence = float(getattr(transcriber, "accept_confidence", 1.0))

    primary = _run_stt_match_once(
        stt_transcriber=primary_transcriber,
        audio_path=audio_path,
        current=current,
        options=options,
        stimulus_letters=stimulus_letters,
        language_hint=language_hint,
    )
    primary_match = primary["match"]

    cascade_meta = {
        "cascade_enabled": fallback_transcriber is not None,
        "cascade_used": False,
        "cascade_primary_backend": primary["backend_name"],
        "cascade_fallback_backend": getattr(fallback_transcriber, "backend_name", None),
        "cascade_primary_model_path": getattr(primary_transcriber, "model_path", None),
        "cascade_fallback_model_path": getattr(fallback_transcriber, "model_path", None),
        "cascade_accept_confidence": accept_confidence if fallback_transcriber is not None else None,
        "cascade_primary_transcript": primary["transcript"],
        "cascade_primary_match_method": primary_match.method,
        "cascade_primary_match_confidence": primary_match.confidence,
        "cascade_primary_canonical_label": primary_match.canonical_label,
        "cascade_primary_accepted": primary_match.accepted,
        "cascade_primary_stt_seconds": round(primary["stt_seconds"], 3),
        "cascade_primary_match_seconds": round(primary["match_seconds"], 3),
        "cascade_fallback_transcript": None,
        "cascade_fallback_match_method": None,
        "cascade_fallback_match_confidence": None,
        "cascade_fallback_canonical_label": None,
        "cascade_fallback_accepted": None,
        "cascade_fallback_stt_seconds": 0.0,
        "cascade_fallback_match_seconds": 0.0,
        "cascade_reason": "",
    }

    if fallback_transcriber is None:
        primary["cascade_meta"] = cascade_meta
        primary["stt_seconds_total"] = primary["stt_seconds"]
        primary["match_seconds_total"] = primary["match_seconds"]
        primary["selected_backend_name"] = primary["backend_name"]
        primary["selected_cascade_stage"] = "primary"
        return primary

    if _cascade_primary_is_good_enough(primary_match, accept_confidence):
        cascade_meta["cascade_reason"] = "primary_confident"
        primary["cascade_meta"] = cascade_meta
        primary["stt_seconds_total"] = primary["stt_seconds"]
        primary["match_seconds_total"] = primary["match_seconds"]
        primary["selected_backend_name"] = primary["backend_name"]
        primary["selected_cascade_stage"] = "primary"
        return primary

    cascade_meta["cascade_used"] = True
    cascade_meta["cascade_reason"] = (
        "primary_not_accepted"
        if not primary_match.accepted or primary_match.response_value is None
        else "primary_low_confidence"
    )
    print(
        "Cascade fallback: "
        f"primary heard {primary['transcript']!r}, "
        f"match={primary_match.canonical_label}, confidence={float(primary_match.confidence or 0.0):.2f}"
    )

    fallback = _run_stt_match_once(
        stt_transcriber=fallback_transcriber,
        audio_path=audio_path,
        current=current,
        options=options,
        stimulus_letters=stimulus_letters,
        language_hint=language_hint,
    )
    fallback_match = fallback["match"]
    cascade_meta.update(
        {
            "cascade_fallback_transcript": fallback["transcript"],
            "cascade_fallback_match_method": fallback_match.method,
            "cascade_fallback_match_confidence": fallback_match.confidence,
            "cascade_fallback_canonical_label": fallback_match.canonical_label,
            "cascade_fallback_accepted": fallback_match.accepted,
            "cascade_fallback_stt_seconds": round(fallback["stt_seconds"], 3),
            "cascade_fallback_match_seconds": round(fallback["match_seconds"], 3),
        }
    )
    fallback["cascade_meta"] = cascade_meta
    fallback["stt_seconds_total"] = primary["stt_seconds"] + fallback["stt_seconds"]
    fallback["match_seconds_total"] = primary["match_seconds"] + fallback["match_seconds"]
    fallback["selected_backend_name"] = fallback["backend_name"]
    fallback["selected_cascade_stage"] = "fallback"
    return fallback


def _voice_response(
    *,
    transcriber,
    current,
    options: list[str],
    stimulus_letters: str,
    results_folder: Path,
    prompt_language: str,
    stt_language_hint: Optional[str],
    language_locked: bool,
    capture_mode: str,
    record_seconds: float,
    samplerate: int,
    input_device: Optional[int | str],
    reprompt_limit: int,
    speak_prompts: bool,
    beep_before_record: bool,
    start_timeout_seconds: float,
    end_silence_seconds: float,
    min_speech_seconds: float,
    max_speech_seconds: Optional[float],
    silence_threshold: float,
    gamepad_enabled: bool,
    gamepad_profile: str,
    gamepad_driver: str,
    gamepad_device_index: int,
    gamepad_vendor_id: Optional[int],
    gamepad_product_id: Optional[int],
) -> tuple[str, dict]:
    audio_dir = results_folder / "voice_audio"
    total_recording_seconds = 0.0
    total_stt_seconds = 0.0
    total_match_seconds = 0.0
    total_processing_seconds = 0.0
    current_prompt_language = prompt_language
    current_stt_language_hint = stt_language_hint
    effective_record_seconds, effective_end_silence_seconds = _phase_voice_capture_settings(
        state=current.state,
        default_record_seconds=record_seconds,
        default_end_silence_seconds=end_silence_seconds,
    )
    last_timing_meta: dict = {}
    capture_meta: dict = {}
    detected_language = None
    detected_language_probability = None
    inferred_language = None
    last_cascade_meta: dict = {
        "cascade_enabled": bool(getattr(transcriber, "fallback_transcriber", None) is not None),
        "cascade_used": False,
    }

    failed_attempts = 0
    attempt = 0
    while failed_attempts <= reprompt_limit:
        attempt += 1
        attempt_started = time.perf_counter()
        prompt_seconds = 0.0
        beep_seconds = 0.0
        audio_path = audio_dir / f"step_{current.step:03d}_attempt_{attempt}.wav"

        if speak_prompts:
            prompt_started = time.perf_counter()
            prompt_to_speak = localized_voice_prompt(
                state=current.state,
                language=current_prompt_language,
                retry=attempt > 1,
                fallback_question=current.question.strip(),
            )
            if attempt > 1:
                prompt_to_speak = f"{prompt_to_speak}{_localized_listen_suffix(current_prompt_language, effective_record_seconds)}"
            _speak_text(prompt_to_speak)
            prompt_seconds = time.perf_counter() - prompt_started

        if beep_before_record:
            beep_started = time.perf_counter()
            _play_beep()
            time.sleep(0.15)
            beep_seconds = time.perf_counter() - beep_started

        if capture_mode == "vad":
            print(f"\nVOICE INPUT: listening until you stop speaking (max {effective_record_seconds:.1f}s) ...")
        else:
            print(f"\nVOICE INPUT: listening for {effective_record_seconds:.1f}s ...")

        manual_listener = _ManualOverrideListener(
            [str(index) for index in range(1, len(options) + 1)],
            enable_gamepad=gamepad_enabled,
            gamepad_profile=gamepad_profile,
            gamepad_driver=gamepad_driver,
            gamepad_device_index=gamepad_device_index,
            gamepad_vendor_id=gamepad_vendor_id,
            gamepad_product_id=gamepad_product_id,
        ).start()
        try:
            capture_meta = _capture_audio_attempt(
                audio_path=audio_path,
                capture_mode=capture_mode,
                record_seconds=effective_record_seconds,
                samplerate=samplerate,
                input_device=input_device,
                start_timeout_seconds=start_timeout_seconds,
                end_silence_seconds=effective_end_silence_seconds,
                min_speech_seconds=min_speech_seconds,
                max_speech_seconds=max_speech_seconds,
                silence_threshold=silence_threshold,
                stop_requested=manual_listener.stop_requested if manual_listener.enabled else None,
            )
        finally:
            selected_manual_key = manual_listener.stop()
        capture_seconds = float(capture_meta.get("recording_seconds") or 0.0)
        total_recording_seconds += capture_seconds

        if selected_manual_key is not None:
            match_started = time.perf_counter()
            response_value, meta = _manual_override_response(
                selected_key=selected_manual_key,
                current=current,
                options=options,
                stimulus_letters=stimulus_letters,
                override_source=(
                    "manual_gamepad" if manual_listener.selected_source() == "gamepad" else "manual_button"
                ),
                override_detail=(
                    manual_listener.gamepad_name if manual_listener.selected_source() == "gamepad" else None
                ),
            )
            match_seconds = time.perf_counter() - match_started
            total_match_seconds += match_seconds
            total_processing_seconds += match_seconds
            total_seconds = time.perf_counter() - attempt_started
            print(f"Manual response detected: option {selected_manual_key}")
            print(
                "TIMING: "
                f"STT=0.000s | matching={match_seconds:.3f}s | overall={total_seconds:.3f}s "
                f"(recording={capture_seconds:.3f}s, capture={capture_meta.get('stop_reason')})"
            )
            print(
                "VAD: "
                f"noise={float(capture_meta.get('noise_floor_rms') or 0.0):.4f} | "
                f"start={float(capture_meta.get('speech_start_threshold_rms') or 0.0):.4f} | "
                f"continue={float(capture_meta.get('speech_continue_threshold_rms') or 0.0):.4f}"
            )
            meta.update(
                {
                    "input_mode": (
                        f"{transcriber.backend_name}_manual_gamepad"
                        if meta.get("manual_override_source") == "manual_gamepad"
                        else f"{transcriber.backend_name}_manual_button"
                    ),
                    "response_attempt_count": attempt,
                    "response_audio_path": capture_meta.get("audio_path"),
                    "stt_backend": transcriber.backend_name,
                    "stt_orchestrator": transcriber.backend_name,
                    "requested_language": getattr(transcriber, "requested_language", None),
                    "prompt_language_used": prompt_language,
                    "prompt_language_next": current_prompt_language,
                    "stt_language_hint_used": stt_language_hint,
                    "stt_language_hint_next": current_stt_language_hint,
                    "prompt_spoken": speak_prompts,
                    "beep_before_record": beep_before_record,
                    "voice_capture_mode": capture_mode,
                    "voice_record_seconds": float(effective_record_seconds),
                    "voice_capture_stop_reason": capture_meta.get("stop_reason"),
                    "voice_speech_detected": capture_meta.get("speech_detected"),
                    "voice_peak_amplitude": capture_meta.get("peak_amplitude"),
                    "voice_input_device_index": capture_meta.get("input_device_index"),
                    "voice_input_device_name": capture_meta.get("input_device_name"),
                    "voice_input_device_selection": capture_meta.get("input_device_selection"),
                    "voice_noise_floor_rms": capture_meta.get("noise_floor_rms"),
                    "voice_speech_start_threshold_rms": capture_meta.get("speech_start_threshold_rms"),
                    "voice_speech_continue_threshold_rms": capture_meta.get("speech_continue_threshold_rms"),
                    "voice_prompt_seconds": round(prompt_seconds, 3),
                    "voice_beep_seconds": round(beep_seconds, 3),
                    "voice_capture_seconds": round(capture_seconds, 3),
                    "recording_seconds": round(total_recording_seconds, 3),
                    "recording_display": _format_duration(total_recording_seconds),
                    "stt_seconds": 0.0,
                    "stt_display": _format_duration(0.0),
                    "response_match_seconds": round(total_match_seconds, 3),
                    "matcher_seconds": round(total_match_seconds, 3),
                    "matcher_display": _format_duration(total_match_seconds),
                    "audio_processing_seconds": round(total_processing_seconds, 3),
                    "audio_processing_display": _format_duration(total_processing_seconds),
                    "voice_attempt_total_seconds": round(total_seconds, 3),
                    "voice_step_total_seconds": round(total_recording_seconds + total_processing_seconds, 3),
                    "voice_step_total_display": _format_duration(total_recording_seconds + total_processing_seconds),
                    **last_cascade_meta,
                }
            )
            return response_value, meta

        stt_match = _run_stt_match_with_optional_cascade(
            transcriber=transcriber,
            audio_path=audio_path,
            current=current,
            options=options,
            stimulus_letters=stimulus_letters,
            language_hint=current_stt_language_hint,
        )
        transcript = stt_match["transcript"]
        match = stt_match["match"]
        stt_seconds = float(stt_match["stt_seconds_total"])
        match_seconds = float(stt_match["match_seconds_total"])
        total_stt_seconds += stt_seconds
        selected_backend_name = stt_match.get("selected_backend_name") or getattr(transcriber, "backend_name", None)
        selected_cascade_stage = stt_match.get("selected_cascade_stage")
        cascade_meta = stt_match.get("cascade_meta", {})
        last_cascade_meta = dict(cascade_meta)
        selected_input_mode = (
            f"{transcriber.backend_name}_{selected_cascade_stage}"
            if cascade_meta.get("cascade_enabled") and selected_cascade_stage
            else selected_backend_name
        )

        detected_language = stt_match["detected_language"]
        detected_language_probability = stt_match["detected_language_probability"]
        inferred_language = stt_match["inferred_language"]
        if detected_language is not None or inferred_language is not None:
            language_parts = []
            if detected_language is not None:
                detected_label = _language_display_name(detected_language)
                if detected_language_probability is not None:
                    detected_label = f"{detected_label} ({float(detected_language_probability):.2f})"
                language_parts.append(f"STT={detected_label}")
            if inferred_language is not None:
                language_parts.append(f"Inferred={_language_display_name(inferred_language)}")
            print("Language guess:", " | ".join(language_parts))

        if (
            not language_locked
            and getattr(transcriber, "requested_language", None) == "auto"
            and inferred_language in {"en", "hi"}
        ):
            if inferred_language != current_prompt_language or inferred_language != current_stt_language_hint:
                current_prompt_language = inferred_language
                current_stt_language_hint = inferred_language
                print(f"Switching session language to {_language_display_name(inferred_language)} for prompts and STT.")

        total_match_seconds += match_seconds

        processing_seconds = stt_seconds + match_seconds
        total_processing_seconds += processing_seconds
        total_seconds = time.perf_counter() - attempt_started
        step_total_seconds = total_recording_seconds + total_processing_seconds
        last_timing_meta = {
            "voice_prompt_seconds": round(prompt_seconds, 3),
            "voice_beep_seconds": round(beep_seconds, 3),
            "voice_capture_seconds": round(capture_seconds, 3),
            "recording_seconds": round(total_recording_seconds, 3),
            "recording_display": _format_duration(total_recording_seconds),
            "stt_seconds": round(total_stt_seconds, 3),
            "stt_display": _format_duration(total_stt_seconds),
            "response_match_seconds": round(total_match_seconds, 3),
            "matcher_seconds": round(total_match_seconds, 3),
            "matcher_display": _format_duration(total_match_seconds),
            "audio_processing_seconds": round(total_processing_seconds, 3),
            "audio_processing_display": _format_duration(total_processing_seconds),
            "voice_attempt_total_seconds": round(total_seconds, 3),
            "voice_step_total_seconds": round(step_total_seconds, 3),
            "voice_step_total_display": _format_duration(step_total_seconds),
        }

        print(f"Heard: {transcript!r}")
        print(
            "TIMING: "
            f"STT={stt_seconds:.3f}s | "
            f"matching={match_seconds:.3f}s | "
            f"overall={total_seconds:.3f}s "
            f"(recording={capture_seconds:.3f}s, capture={capture_meta.get('stop_reason')})"
        )
        print(
            "VAD: "
            f"noise={float(capture_meta.get('noise_floor_rms') or 0.0):.4f} | "
            f"start={float(capture_meta.get('speech_start_threshold_rms') or 0.0):.4f} | "
            f"continue={float(capture_meta.get('speech_continue_threshold_rms') or 0.0):.4f}"
        )
        if match.accepted and match.response_value is not None:
            if _is_repeat_response(match.response_value):
                print("Repeating the question.")
                continue
            print(
                "Matched response: "
                f"{match.response_value} ({match.method}, confidence={match.confidence:.2f})"
            )
            return match.response_value, {
                "input_mode": selected_input_mode,
                "raw_input_text": transcript,
                "normalized_input_text": match.normalized_text,
                "response_match_method": match.method,
                "response_match_confidence": match.confidence,
                "response_match_canonical_label": match.canonical_label,
                "response_match_accepted": True,
                "response_attempt_count": attempt,
                "response_audio_path": capture_meta.get("audio_path"),
                "stt_backend": selected_backend_name,
                "stt_orchestrator": transcriber.backend_name,
                "requested_language": getattr(transcriber, "requested_language", None),
                "detected_input_language": detected_language,
                "detected_input_language_probability": detected_language_probability,
                "inferred_input_language": inferred_language,
                "prompt_language_used": prompt_language,
                "prompt_language_next": current_prompt_language,
                "stt_language_hint_used": stt_language_hint,
                "stt_language_hint_next": current_stt_language_hint,
                "stimulus_letters": stimulus_letters,
                "prompt_spoken": speak_prompts,
                "beep_before_record": beep_before_record,
                "voice_capture_mode": capture_mode,
                "voice_record_seconds": float(effective_record_seconds),
                "voice_capture_stop_reason": capture_meta.get("stop_reason"),
                "voice_speech_detected": capture_meta.get("speech_detected"),
                "voice_peak_amplitude": capture_meta.get("peak_amplitude"),
                "voice_input_device_index": capture_meta.get("input_device_index"),
                "voice_input_device_name": capture_meta.get("input_device_name"),
                "voice_input_device_selection": capture_meta.get("input_device_selection"),
                "voice_noise_floor_rms": capture_meta.get("noise_floor_rms"),
                "voice_speech_start_threshold_rms": capture_meta.get("speech_start_threshold_rms"),
                "voice_speech_continue_threshold_rms": capture_meta.get("speech_continue_threshold_rms"),
                "manual_override_used": False,
                "manual_override_key": None,
                "manual_override_source": None,
                "manual_override_detail": None,
                **cascade_meta,
                **last_timing_meta,
            }

        print(
            "Could not confidently map the voice response"
            f" ({match.method}, confidence={match.confidence:.2f}, reason={match.reason})."
        )
        print(match.reprompt_text)
        failed_attempts += 1

    print("\nFalling back to keyboard input for this step.")
    response, meta = _keyboard_response(options)
    meta["input_mode"] = f"{transcriber.backend_name}_fallback_keyboard"
    meta["stt_backend"] = transcriber.backend_name
    meta["stt_orchestrator"] = transcriber.backend_name
    meta["requested_language"] = getattr(transcriber, "requested_language", None)
    meta["detected_input_language"] = detected_language
    meta["detected_input_language_probability"] = detected_language_probability
    meta["inferred_input_language"] = inferred_language
    meta["prompt_language_used"] = prompt_language
    meta["prompt_language_next"] = current_prompt_language
    meta["stt_language_hint_used"] = stt_language_hint
    meta["stt_language_hint_next"] = current_stt_language_hint
    meta["response_attempt_count"] = reprompt_limit + 1
    meta["stimulus_letters"] = stimulus_letters
    meta["prompt_spoken"] = speak_prompts
    meta["beep_before_record"] = beep_before_record
    meta["voice_capture_mode"] = capture_mode
    meta["voice_record_seconds"] = float(effective_record_seconds)
    meta["voice_capture_stop_reason"] = capture_meta.get("stop_reason")
    meta["voice_speech_detected"] = capture_meta.get("speech_detected")
    meta["voice_peak_amplitude"] = capture_meta.get("peak_amplitude")
    meta["voice_input_device_index"] = capture_meta.get("input_device_index")
    meta["voice_input_device_name"] = capture_meta.get("input_device_name")
    meta["voice_input_device_selection"] = capture_meta.get("input_device_selection")
    meta["voice_noise_floor_rms"] = capture_meta.get("noise_floor_rms")
    meta["voice_speech_start_threshold_rms"] = capture_meta.get("speech_start_threshold_rms")
    meta["voice_speech_continue_threshold_rms"] = capture_meta.get("speech_continue_threshold_rms")
    meta["manual_override_used"] = False
    meta["manual_override_key"] = None
    meta["manual_override_source"] = None
    meta["manual_override_detail"] = None
    meta.update(last_cascade_meta)
    meta.update(last_timing_meta)
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


def _voice_language_choice(
    *,
    transcriber,
    results_folder: Path,
    capture_mode: str,
    record_seconds: float,
    samplerate: int,
    input_device: Optional[int | str],
    reprompt_limit: int,
    speak_prompts: bool,
    beep_before_record: bool,
    start_timeout_seconds: float,
    end_silence_seconds: float,
    min_speech_seconds: float,
    max_speech_seconds: Optional[float],
    silence_threshold: float,
) -> tuple[str, dict]:
    audio_dir = results_folder / "voice_audio"

    for attempt in range(1, reprompt_limit + 2):
        audio_path = audio_dir / f"language_choice_attempt_{attempt}.wav"
        prompt_text = localized_language_selection_prompt(retry=attempt > 1)

        print("\nLANGUAGE SELECTION")
        print(prompt_text)

        if speak_prompts:
            _speak_text(prompt_text)
        if beep_before_record:
            _play_beep()
            time.sleep(0.15)

        if capture_mode == "vad":
            print(f"\nVOICE INPUT: listening until you stop speaking (max {record_seconds:.1f}s) ...")
        else:
            print(f"\nVOICE INPUT: listening for {record_seconds:.1f}s ...")

        capture_started = time.perf_counter()
        capture_meta = _capture_audio_attempt(
            audio_path=audio_path,
            capture_mode=capture_mode,
            record_seconds=record_seconds,
            samplerate=samplerate,
            input_device=input_device,
            start_timeout_seconds=start_timeout_seconds,
            end_silence_seconds=end_silence_seconds,
            min_speech_seconds=min_speech_seconds,
            max_speech_seconds=max_speech_seconds,
            silence_threshold=silence_threshold,
        )
        capture_seconds = time.perf_counter() - capture_started

        stt_started = time.perf_counter()
        try:
            transcript_result = transcriber.transcribe_result(audio_path)
            transcript = transcript_result.text
        except Exception as exc:
            print(f"Voice transcription failed during language selection: {exc}")
            transcript_result = None
            transcript = ""
        stt_seconds = time.perf_counter() - stt_started

        detected_language = getattr(transcript_result, "detected_language", None)
        detected_language_probability = getattr(transcript_result, "language_probability", None)

        match_started = time.perf_counter()
        match = match_language_choice(
            transcript=transcript,
            detected_language=detected_language,
            detected_language_probability=detected_language_probability,
        )
        match_seconds = time.perf_counter() - match_started

        print(f"Heard: {transcript!r}")
        print(
            "TIMING: "
            f"STT={stt_seconds:.3f}s | matching={match_seconds:.3f}s | "
            f"overall={(capture_seconds + stt_seconds + match_seconds):.3f}s "
            f"(recording={float(capture_meta.get('recording_seconds') or 0.0):.3f}s, "
            f"capture={capture_meta.get('stop_reason')})"
        )

        if match.accepted and match.language_code in {"en", "hi"}:
            print(f"Language selected: {_language_display_name(match.language_code)}")
            return match.language_code, {
                "language_selection_method": match.method,
                "language_selection_confidence": match.confidence,
                "language_selection_transcript": transcript,
                "language_selection_audio_path": capture_meta.get("audio_path"),
                "language_selection_attempt_count": attempt,
                "language_selection_detected_language": detected_language,
                "language_selection_detected_language_probability": detected_language_probability,
                "language_selection_capture_stop_reason": capture_meta.get("stop_reason"),
                "language_selection_recording_seconds": capture_meta.get("recording_seconds"),
                "language_selection_stt_seconds": round(stt_seconds, 3),
                "language_selection_match_seconds": round(match_seconds, 3),
            }

        print(match.reprompt_text)

    print("Language selection fallback: English")
    return "en", {
        "language_selection_method": "fallback",
        "language_selection_confidence": 0.0,
    }


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
    prompt_language = "en" if args.voice_language == "auto" else args.voice_language
    stt_language_hint = None if args.voice_language == "auto" else args.voice_language
    language_locked = args.voice_language != "auto"
    language_selection_meta = {}
    if args.input_mode != "keyboard":
        hf_model_path = _resolve_model_path(args.hf_model_path, DEFAULT_HF_MODEL_PATH.name)
        ct2_model_path = _resolve_model_path(args.ct2_model_path, DEFAULT_CT2_MODEL_PATH.name)
        primary_ct2_model_path = _resolve_model_path(
            args.voice_primary_ct2_model_path,
            DEFAULT_PRIMARY_CT2_MODEL_PATH.name,
        )
        fw_decode_kwargs = {
            "fw_vad_filter": bool(args.voice_fw_internal_vad),
            "fw_condition_on_previous_text": not bool(args.voice_fw_optimized_decode),
            "fw_temperature": 0.0 if args.voice_fw_optimized_decode else None,
            "fw_best_of": 1 if args.voice_fw_optimized_decode else None,
            "fw_without_timestamps": bool(args.voice_fw_optimized_decode),
            "fw_max_new_tokens": (
                int(args.voice_fw_max_new_tokens)
                if args.voice_fw_optimized_decode and int(args.voice_fw_max_new_tokens) > 0
                else None
            ),
        }
        print(f"\nLoading local voice backend: {args.input_mode} ...")
        if args.input_mode == "voice-local-fw":
            print(f"CT2 Model Path: {ct2_model_path}")
        if args.input_mode == "voice-local-hf":
            print(f"HF Model Path: {hf_model_path}")
        if args.input_mode == "voice-local-cascade":
            print(f"Primary CT2 Model Path: {primary_ct2_model_path}")
            print(f"Fallback CT2 Model Path: {ct2_model_path}")
            primary_path_exists = Path(primary_ct2_model_path).expanduser().exists()
            fallback_path_exists = Path(ct2_model_path).expanduser().exists()
            if not primary_path_exists:
                print(
                    "Cascade primary model not found; using the fallback CT2 model directly. "
                    "Install/pass a smaller CT2 model with --voice-primary-ct2-model-path for speed gains."
                )
                primary_transcriber = create_local_transcriber(
                    backend="voice-local-fw",
                    ct2_model_path=ct2_model_path,
                    cpu_threads=args.voice_cpu_threads,
                    language=args.voice_language,
                    **fw_decode_kwargs,
                )
                fallback_transcriber = None
            else:
                primary_transcriber = create_local_transcriber(
                    backend="voice-local-fw",
                    ct2_model_path=primary_ct2_model_path,
                    cpu_threads=args.voice_cpu_threads,
                    language=args.voice_language,
                    **fw_decode_kwargs,
                )
                fallback_transcriber = None
                if fallback_path_exists and Path(primary_ct2_model_path).resolve() != Path(ct2_model_path).resolve():
                    fallback_transcriber = create_local_transcriber(
                        backend="voice-local-fw",
                        ct2_model_path=ct2_model_path,
                        cpu_threads=args.voice_cpu_threads,
                        language=args.voice_language,
                        **fw_decode_kwargs,
                    )
            transcriber = _CascadeTranscriber(
                primary_transcriber=primary_transcriber,
                fallback_transcriber=fallback_transcriber,
                accept_confidence=args.cascade_accept_confidence,
            )
        else:
            transcriber = create_local_transcriber(
                backend=args.input_mode,
                hf_model_path=hf_model_path,
                ct2_model_path=ct2_model_path,
                cpu_threads=args.voice_cpu_threads,
                language=args.voice_language,
                **fw_decode_kwargs,
            )
        print("Local voice backend ready.")
        if args.voice_language == "auto" and args.voice_ask_language_at_start:
            prompt_language, language_selection_meta = _voice_language_choice(
                transcriber=transcriber,
                results_folder=results_folder,
                capture_mode=args.voice_capture_mode,
                record_seconds=args.voice_record_seconds,
                samplerate=args.voice_samplerate,
                input_device=args.voice_input_device,
                reprompt_limit=args.voice_reprompt_limit,
                speak_prompts=args.voice_speak_prompts,
                beep_before_record=args.voice_beep_before_record,
                start_timeout_seconds=args.voice_start_timeout_seconds,
                end_silence_seconds=args.voice_end_silence_seconds,
                min_speech_seconds=args.voice_min_speech_seconds,
                max_speech_seconds=args.voice_max_speech_seconds,
                silence_threshold=args.voice_silence_threshold,
            )
            stt_language_hint = prompt_language
            language_locked = False

    rows = []

    _print_profile(dv, run_id, args.input_mode)
    if args.input_mode != "keyboard":
        print("\nVOICE SETTINGS")
        print(f"Test Language: {_language_display_name(prompt_language)}")
        print(f"Capture Mode: {args.voice_capture_mode}")
        print(f"Input Device: {args.voice_input_device}")
        print(f"FW Optimized Decode: {'ON' if args.voice_fw_optimized_decode else 'OFF'}")
        print(f"FW Internal VAD: {'ON' if args.voice_fw_internal_vad else 'OFF'}")
        if args.voice_fw_optimized_decode:
            print(f"FW Max New Tokens: {args.voice_fw_max_new_tokens}")
        if args.input_mode == "voice-local-cascade":
            print(f"Cascade Primary Model: {getattr(transcriber, 'primary_model_path', None)}")
            print(f"Cascade Fallback Model: {getattr(transcriber, 'fallback_model_path', None)}")
            print(f"Cascade Accept Confidence: {args.cascade_accept_confidence:.2f}")
        print(f"Native Gamepad Input: {'ON' if args.gamepad_enabled else 'OFF'}")
        if args.gamepad_enabled:
            print(f"Gamepad Profile: {args.gamepad_profile}")
            print(f"Gamepad Driver: {args.gamepad_driver}")

    while True:
        print("\n--------------------------------")
        print(f"STEP: {current.step}")
        print(f"STATE: {current.state} | {current.phase_name}")
        print(f"CHART: {current.chart_param}")
        print(f"EYE: {current.eye}")

        stimulus_letters = _stimulus_letters_for_row(current)
        if stimulus_letters:
            print("\nDISPLAYED LETTERS:")
            print(stimulus_letters)

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
                    stimulus_letters=stimulus_letters,
                    results_folder=results_folder,
                    prompt_language=prompt_language,
                    stt_language_hint=stt_language_hint,
                    language_locked=language_locked,
                    capture_mode=args.voice_capture_mode,
                    record_seconds=args.voice_record_seconds,
                    samplerate=args.voice_samplerate,
                    input_device=args.voice_input_device,
                    reprompt_limit=args.voice_reprompt_limit,
                    speak_prompts=args.voice_speak_prompts,
                    beep_before_record=args.voice_beep_before_record,
                    start_timeout_seconds=args.voice_start_timeout_seconds,
                    end_silence_seconds=args.voice_end_silence_seconds,
                    min_speech_seconds=args.voice_min_speech_seconds,
                    max_speech_seconds=args.voice_max_speech_seconds,
                    silence_threshold=args.voice_silence_threshold,
                    gamepad_enabled=args.gamepad_enabled,
                    gamepad_profile=args.gamepad_profile,
                    gamepad_driver=args.gamepad_driver,
                    gamepad_device_index=args.gamepad_device_index,
                    gamepad_vendor_id=args.gamepad_vendor_id,
                    gamepad_product_id=args.gamepad_product_id,
                )
        except ValueError as exc:
            print(str(exc))
            continue

        meta["stimulus_letters"] = stimulus_letters
        if args.input_mode != "keyboard":
            prompt_language = meta.get("prompt_language_next") or prompt_language
            stt_language_hint = meta.get("stt_language_hint_next") or stt_language_hint
            meta.update(language_selection_meta)

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
        "voice_language": args.voice_language,
        "prompt_language_final": prompt_language,
        "ct2_model_path": _resolve_model_path(args.ct2_model_path, DEFAULT_CT2_MODEL_PATH.name),
        "voice_primary_ct2_model_path": _resolve_model_path(
            args.voice_primary_ct2_model_path,
            DEFAULT_PRIMARY_CT2_MODEL_PATH.name,
        ),
        "cascade_accept_confidence": float(args.cascade_accept_confidence),
        "cascade_primary_model_path": getattr(transcriber, "primary_model_path", None) if transcriber else None,
        "cascade_fallback_model_path": getattr(transcriber, "fallback_model_path", None) if transcriber else None,
        "voice_capture_mode": args.voice_capture_mode,
        "voice_input_device": args.voice_input_device,
        "voice_cpu_threads": int(args.voice_cpu_threads),
        "voice_record_seconds": float(args.voice_record_seconds),
        "voice_start_timeout_seconds": float(args.voice_start_timeout_seconds),
        "voice_end_silence_seconds": float(args.voice_end_silence_seconds),
        "voice_min_speech_seconds": float(args.voice_min_speech_seconds),
        "voice_max_speech_seconds": float(args.voice_max_speech_seconds),
        "voice_silence_threshold": float(args.voice_silence_threshold),
        "gamepad_enabled": bool(args.gamepad_enabled),
        "gamepad_profile": args.gamepad_profile,
        "gamepad_driver": args.gamepad_driver,
        "gamepad_device_index": int(args.gamepad_device_index),
        "gamepad_vendor_id": args.gamepad_vendor_id,
        "gamepad_product_id": args.gamepad_product_id,
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
