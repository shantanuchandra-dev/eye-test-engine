from __future__ import annotations

from collections import deque
import os
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import psutil
import soundfile as sf


def configure_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def configure_cpu_threads(threads: Optional[int]) -> None:
    if threads is None or threads <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(int(threads))
    os.environ["MKL_NUM_THREADS"] = str(int(threads))


@dataclass
class MemorySample:
    rss_bytes: int
    timestamp: float


@dataclass
class TranscriptionResult:
    backend: str
    model_path: str
    audio_path: str
    requested_language: str
    text: str
    load_seconds: float
    transcribe_seconds: float
    total_seconds: float
    peak_rss_bytes: int
    peak_rss_gb: float


@dataclass
class TranscriptResult:
    text: str
    requested_language: str
    detected_language: Optional[str]
    language_probability: Optional[float]


@dataclass
class AudioCaptureResult:
    output_path: str
    duration_seconds: float
    speech_detected: bool
    stop_reason: str
    peak_amplitude: float
    input_device_index: Optional[int] = None
    input_device_name: Optional[str] = None
    input_device_selection: Optional[str] = None
    noise_floor_rms: Optional[float] = None
    speech_start_threshold_rms: Optional[float] = None
    speech_continue_threshold_rms: Optional[float] = None


class BaseLocalTranscriber:
    backend_name: str = "base"

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language_override: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> str:
        return self.transcribe_result(
            audio_path,
            language_override=language_override,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        ).text

    def transcribe_result(
        self,
        audio_path: str | Path,
        *,
        language_override: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> TranscriptResult:
        raise NotImplementedError


@lru_cache(maxsize=1)
def _whisper_language_maps() -> tuple[dict[str, str], dict[str, str]]:
    from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

    code_to_name: dict[str, str] = {}
    for language_name, language_code in TO_LANGUAGE_CODE.items():
        code_to_name.setdefault(language_code, language_name)
    return TO_LANGUAGE_CODE, code_to_name


def _resolve_requested_language(language: Optional[str]) -> tuple[Optional[str], Optional[str], str]:
    if language is None:
        return None, None, "auto"

    normalized = str(language).strip().lower().replace("_", "-")
    if normalized in {"", "auto", "detect", "auto-detect"}:
        return None, None, "auto"

    language_to_code, code_to_name = _whisper_language_maps()
    if normalized in code_to_name:
        return normalized, code_to_name[normalized], normalized
    if normalized in language_to_code:
        language_code = language_to_code[normalized]
        return language_code, code_to_name[language_code], language_code

    raise ValueError(
        f"Unsupported Whisper language: {language}. Use a language code like 'en' or 'hi', "
        "a Whisper language name like 'english' or 'hindi', or 'auto'."
    )


def _current_rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def _load_audio_for_hf(audio_path: str | Path, target_sample_rate: int = 16000) -> np.ndarray:
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)

    if getattr(audio, "size", 0) == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    if int(sample_rate) != int(target_sample_rate):
        import torch
        import torchaudio.functional as F

        waveform = torch.from_numpy(np.asarray(audio)).unsqueeze(0)
        resampled = F.resample(waveform, int(sample_rate), int(target_sample_rate))
        audio = resampled.squeeze(0).numpy()

    return np.asarray(audio, dtype=np.float32)


def _coerce_input_device_reference(device: Optional[int | str]) -> Optional[int | str]:
    if device is None:
        return "__DEFAULT__"

    normalized = str(device).strip()
    if not normalized or normalized.lower() == "default":
        return "__DEFAULT__"
    if normalized.lower() == "auto":
        return "__AUTO__"
    if normalized.lstrip("+-").isdigit():
        return int(normalized)
    return normalized


def _default_input_device_reference(sd) -> Optional[int | str]:
    default_device = getattr(sd.default, "device", None)
    candidate = default_device
    if default_device is not None and not isinstance(default_device, (str, bytes)):
        try:
            candidate = default_device[0]
        except Exception:
            candidate = default_device

    if candidate in (None, "", -1):
        return None
    return int(candidate) if isinstance(candidate, (int, float)) else candidate


def _input_device_is_virtual(name: str) -> bool:
    lowered = str(name or "").strip().lower()
    virtual_markers = (
        "voicemeeter",
        "stereo mix",
        "line in",
        "sound mapper",
        "primary sound capture driver",
        "wave out",
        "what u hear",
        "cable output",
        "loopback",
    )
    return any(marker in lowered for marker in virtual_markers)


def _input_device_hostapi_priority(hostapi_name: str) -> int:
    lowered = str(hostapi_name or "").strip().lower()
    if "wasapi" in lowered:
        return 3
    if "directsound" in lowered:
        return 2
    if lowered == "mme" or lowered.endswith(" mme"):
        return 1
    if "wdm-ks" in lowered:
        return -1
    return 0


def _input_device_priority(
    name: str,
    hostapi_name: str,
    default_samplerate: float,
    requested_samplerate: int,
) -> tuple[int, int, int, int]:
    lowered = str(name or "").strip().lower()
    preferred_markers = ("headset microphone", "microphone", "mic")
    is_preferred = any(marker in lowered for marker in preferred_markers)
    exact_rate_match = int(round(float(default_samplerate or 0.0))) == int(requested_samplerate)
    hostapi_priority = _input_device_hostapi_priority(hostapi_name)
    return (1 if is_preferred else 0, hostapi_priority, 1 if exact_rate_match else 0, len(lowered))


def resolve_input_device(
    *,
    requested_device: Optional[int | str],
    samplerate: int,
    channels: int = 1,
    dtype: str = "float32",
) -> tuple[Optional[int | str], Optional[int], Optional[str], str]:
    import sounddevice as sd

    normalized_request = _coerce_input_device_reference(requested_device)

    def validate(device_ref: int | str) -> tuple[Optional[int | str], Optional[int], Optional[str]]:
        sd.check_input_settings(device=device_ref, samplerate=samplerate, channels=channels, dtype=dtype)
        device_info = sd.query_devices(device_ref, "input")
        device_name = str(device_info.get("name") or "").strip() or str(device_ref)
        device_index: Optional[int] = None
        if isinstance(device_ref, int):
            device_index = int(device_ref)
        else:
            for idx, info in enumerate(sd.query_devices()):
                if str(info.get("name") or "").strip() == device_name and int(info.get("max_input_channels") or 0) > 0:
                    device_index = idx
                    break
        return device_ref, device_index, device_name

    if normalized_request not in {"__DEFAULT__", "__AUTO__"}:
        device_ref, device_index, device_name = validate(normalized_request)
        return device_ref, device_index, device_name, "explicit"

    default_device = _default_input_device_reference(sd)
    if normalized_request == "__DEFAULT__":
        if default_device is None:
            return None, None, "System default input", "default"
        device_ref, device_index, device_name = validate(default_device)
        return device_ref, device_index, device_name, "default"

    if default_device is not None:
        try:
            device_ref, device_index, device_name = validate(default_device)
            default_hostapi_name = ""
            if device_index is not None:
                device_info = sd.query_devices(device_index)
                hostapi_index = int(device_info.get("hostapi") or 0)
                default_hostapi_name = str(sd.query_hostapis(hostapi_index).get("name") or "")
            if (
                not _input_device_is_virtual(device_name)
                and _input_device_hostapi_priority(default_hostapi_name) >= 0
            ):
                return device_ref, device_index, device_name, "default"
        except Exception:
            pass

    candidates: list[tuple[tuple[int, int, int, int], int, str]] = []
    hostapis = sd.query_hostapis()
    for idx, device_info in enumerate(sd.query_devices()):
        if int(device_info.get("max_input_channels") or 0) <= 0:
            continue

        device_name = str(device_info.get("name") or "").strip() or f"Input device {idx}"
        if _input_device_is_virtual(device_name):
            continue

        hostapi_index = int(device_info.get("hostapi") or 0)
        hostapi_name = str(hostapis[hostapi_index].get("name") or "")
        if _input_device_hostapi_priority(hostapi_name) < 0:
            continue

        try:
            validate(idx)
        except Exception:
            continue

        priority = _input_device_priority(
            device_name,
            hostapi_name,
            float(device_info.get("default_samplerate") or 0.0),
            samplerate,
        )
        candidates.append((priority, idx, device_name))

    if candidates:
        candidates.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[0][3], item[1]))
        _priority, device_index, device_name = candidates[0]
        return device_index, device_index, device_name, "auto_microphone"

    if default_device is not None:
        device_ref, device_index, device_name = validate(default_device)
        return device_ref, device_index, device_name, "default_fallback"

    raise RuntimeError(
        f"No usable microphone input device was found for {samplerate} Hz mono float32 capture."
    )


def record_audio_clip(
    *,
    output_path: str | Path,
    seconds: float = 2.0,
    samplerate: int = 16000,
    input_device: Optional[int | str] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    frame_seconds: float = 0.1,
) -> Path:
    import sounddevice as sd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_device, _device_index, _device_name, _selection_reason = resolve_input_device(
        requested_device=input_device,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
    )

    blocksize = max(1, int(samplerate * frame_seconds))
    target_frames = max(1, int(seconds * samplerate))
    captured_chunks: list[np.ndarray] = []
    captured_frames = 0

    with sd.InputStream(
        device=selected_device,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
    ) as stream:
        while captured_frames < target_frames:
            if stop_requested is not None and stop_requested():
                break

            frames_to_read = min(blocksize, target_frames - captured_frames)
            frames, _overflowed = stream.read(frames_to_read)
            chunk = np.asarray(frames, dtype=np.float32).reshape(-1)
            if chunk.size == 0:
                continue

            captured_chunks.append(chunk.copy())
            captured_frames += int(chunk.size)

            if stop_requested is not None and stop_requested():
                break

    if captured_chunks:
        audio = np.concatenate(captured_chunks).astype(np.float32).reshape(-1, 1)
    else:
        audio = np.zeros((max(1, int(samplerate * 0.1)), 1), dtype=np.float32)

    sf.write(str(output_path), audio, samplerate)
    return output_path


def record_audio_until_silence(
    *,
    output_path: str | Path,
    max_seconds: float = 5.0,
    samplerate: int = 16000,
    input_device: Optional[int | str] = None,
    silence_threshold: float = 0.015,
    end_silence_seconds: float = 0.8,
    min_speech_seconds: float = 0.25,
    max_speech_seconds: Optional[float] = None,
    start_timeout_seconds: float = 2.5,
    frame_seconds: float = 0.1,
    pre_speech_seconds: float = 0.25,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> AudioCaptureResult:
    import sounddevice as sd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_device, device_index, device_name, selection_reason = resolve_input_device(
        requested_device=input_device,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
    )

    blocksize = max(1, int(samplerate * frame_seconds))
    preroll_chunks: deque[np.ndarray] = deque(maxlen=max(1, int(pre_speech_seconds / frame_seconds)))
    waiting_chunks: list[np.ndarray] = []
    captured_chunks: list[np.ndarray] = []
    bootstrap_rms_values: list[float] = []
    speech_detected = False
    speech_candidate_duration = 0.0
    speech_duration = 0.0
    trailing_silence = 0.0
    peak_amplitude = 0.0
    stop_reason = "max_duration"
    capture_started = time.perf_counter()
    noise_floor = max(float(silence_threshold) * 0.35, 0.0005)
    bootstrap_seconds = min(0.35, max(0.0, float(start_timeout_seconds) * 0.4))
    last_start_threshold = max(float(silence_threshold), noise_floor * 2.2)
    last_continue_threshold = max(float(silence_threshold), noise_floor * 1.6)

    with sd.InputStream(
        device=selected_device,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
    ) as stream:
        while True:
            if stop_requested is not None and stop_requested():
                stop_reason = "manual_override"
                if not speech_detected and waiting_chunks:
                    captured_chunks.extend(waiting_chunks)
                    preroll_chunks.clear()
                    waiting_chunks.clear()
                break

            frames, _overflowed = stream.read(blocksize)
            chunk = np.asarray(frames, dtype=np.float32).reshape(-1)
            if chunk.size == 0:
                continue

            chunk_duration = float(chunk.size) / float(samplerate)
            elapsed = time.perf_counter() - capture_started

            chunk_peak = float(np.max(np.abs(chunk)))
            chunk_rms = float(np.sqrt(np.mean(np.square(chunk))))
            peak_amplitude = max(peak_amplitude, chunk_peak)
            # Use RMS rather than peak amplitude for VAD. Peak-driven VAD is too
            # sensitive to room clicks/electrical noise and can keep the capture
            # alive until max_seconds even after the patient has stopped speaking.
            if elapsed <= bootstrap_seconds:
                bootstrap_rms_values.append(chunk_rms)
            elif bootstrap_rms_values:
                noise_floor = max(noise_floor, float(np.median(np.asarray(bootstrap_rms_values, dtype=np.float32))))
                bootstrap_rms_values.clear()

            speech_start_threshold = max(
                float(silence_threshold) * 1.35,
                noise_floor + 0.012,
                noise_floor * 2.2,
            )
            speech_continue_threshold = max(
                float(silence_threshold) * 1.50,
                noise_floor + 0.015,
                noise_floor * 2.8,
            )
            last_start_threshold = speech_start_threshold
            last_continue_threshold = speech_continue_threshold
            is_speech = (
                chunk_rms >= (speech_continue_threshold if speech_detected else speech_start_threshold)
            )

            if speech_detected:
                captured_chunks.append(chunk.copy())
                if is_speech:
                    speech_duration += chunk_duration
                    trailing_silence = 0.0
                    if max_speech_seconds is not None and speech_duration >= float(max_speech_seconds):
                        stop_reason = "max_speech_segment"
                        break
                else:
                    noise_floor = (noise_floor * 0.95) + (chunk_rms * 0.05)
                    trailing_silence += chunk_duration
                    if speech_duration >= min_speech_seconds and trailing_silence >= end_silence_seconds:
                        stop_reason = "silence_after_speech"
                        break
            else:
                preroll_chunks.append(chunk.copy())
                waiting_chunks.append(chunk.copy())
                if elapsed <= bootstrap_seconds:
                    # Short ambient bootstrap prevents steady room/device noise
                    # from immediately becoming a "speech" segment.
                    noise_floor = (noise_floor * 0.85) + (chunk_rms * 0.15)
                elif is_speech:
                    speech_candidate_duration += chunk_duration
                    if speech_candidate_duration < min_speech_seconds:
                        continue
                    speech_detected = True
                    captured_chunks.extend(waiting_chunks)
                    preroll_chunks.clear()
                    waiting_chunks.clear()
                    speech_duration += speech_candidate_duration
                    speech_candidate_duration = 0.0
                    trailing_silence = 0.0
                elif elapsed >= min(max_seconds, start_timeout_seconds):
                    stop_reason = "start_timeout"
                    captured_chunks.extend(waiting_chunks)
                    preroll_chunks.clear()
                    waiting_chunks.clear()
                    break
                else:
                    speech_candidate_duration = 0.0
                    noise_floor = (noise_floor * 0.90) + (chunk_rms * 0.10)

            if elapsed >= max_seconds:
                stop_reason = "max_duration"
                if not speech_detected:
                    captured_chunks.extend(waiting_chunks)
                    preroll_chunks.clear()
                    waiting_chunks.clear()
                break

    if captured_chunks:
        captured_audio = np.concatenate(captured_chunks).astype(np.float32)
    else:
        captured_audio = np.zeros(max(1, int(samplerate * 0.1)), dtype=np.float32)

    sf.write(str(output_path), captured_audio.reshape(-1, 1), samplerate)
    duration_seconds = float(captured_audio.size) / float(samplerate)
    return AudioCaptureResult(
        output_path=str(output_path),
        duration_seconds=duration_seconds,
        speech_detected=speech_detected,
        stop_reason=stop_reason,
        peak_amplitude=peak_amplitude,
        input_device_index=device_index,
        input_device_name=device_name,
        input_device_selection=selection_reason,
        noise_floor_rms=noise_floor,
        speech_start_threshold_rms=last_start_threshold,
        speech_continue_threshold_rms=last_continue_threshold,
    )


class HFWhisperTurboTranscriber(BaseLocalTranscriber):
    backend_name = "hf_whisper_turbo"

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "cpu",
        language: Optional[str] = "auto",
    ) -> None:
        configure_offline_env()
        self.model_path = str(model_path)
        self.device = device
        self.language_code, self.language_name, self.requested_language = _resolve_requested_language(language)

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self._torch_dtype = torch.float32
        pipeline_device = -1 if device == "cpu" else device

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            local_files_only=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            dtype=self._torch_dtype,
        )
        processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self._sampling_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self._torch_dtype,
            device=pipeline_device,
        )

    def transcribe_result(
        self,
        audio_path: str | Path,
        *,
        language_override: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> TranscriptResult:
        audio_array = _load_audio_for_hf(audio_path, target_sample_rate=self._sampling_rate)
        language_code, language_name, requested_language = _resolve_requested_language(
            self.requested_language if language_override is None else language_override
        )
        if language_name is not None:
            result = self._pipe(audio_array, generate_kwargs={"language": language_name})
        else:
            result = self._pipe(audio_array)
        return TranscriptResult(
            text=str(result["text"]).strip(),
            requested_language=requested_language,
            detected_language=language_code,
            language_probability=None,
        )


class FasterWhisperTurboTranscriber(BaseLocalTranscriber):
    backend_name = "faster_whisper_turbo"

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: Optional[int] = None,
        language: Optional[str] = "auto",
        vad_filter: bool = True,
        condition_on_previous_text: bool = True,
        temperature: Optional[float] = None,
        best_of: Optional[int] = None,
        without_timestamps: bool = False,
        max_new_tokens: Optional[int] = None,
        hotwords: Optional[str] = None,
    ) -> None:
        configure_offline_env()
        from faster_whisper import WhisperModel

        self.model_path = str(model_path)
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.vad_filter = bool(vad_filter)
        self.condition_on_previous_text = bool(condition_on_previous_text)
        self.temperature = temperature
        self.best_of = best_of
        self.without_timestamps = bool(without_timestamps)
        self.max_new_tokens = max_new_tokens
        self.hotwords = hotwords
        self.language_code, _, self.requested_language = _resolve_requested_language(language)
        self._model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )

    def transcribe_result(
        self,
        audio_path: str | Path,
        *,
        language_override: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> TranscriptResult:
        runtime_language_code, _, runtime_requested_language = _resolve_requested_language(
            self.requested_language if language_override is None else language_override
        )
        transcribe_kwargs = {
            "beam_size": 1,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": self.condition_on_previous_text,
            "without_timestamps": self.without_timestamps,
        }
        if self.temperature is not None:
            transcribe_kwargs["temperature"] = float(self.temperature)
        if self.best_of is not None:
            transcribe_kwargs["best_of"] = int(self.best_of)
        if self.max_new_tokens is not None and int(self.max_new_tokens) > 0:
            transcribe_kwargs["max_new_tokens"] = int(self.max_new_tokens)
        runtime_hotwords = hotwords or self.hotwords
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        if runtime_hotwords:
            transcribe_kwargs["hotwords"] = runtime_hotwords
        if runtime_language_code is not None:
            transcribe_kwargs["language"] = runtime_language_code

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*encountered in matmul",
                category=RuntimeWarning,
            )
            segments, info = self._model.transcribe(
                str(audio_path),
                **transcribe_kwargs,
            )
        return TranscriptResult(
            text=" ".join(segment.text.strip() for segment in segments).strip(),
            requested_language=runtime_requested_language,
            detected_language=getattr(info, "language", runtime_language_code),
            language_probability=getattr(info, "language_probability", None),
        )


def create_local_transcriber(
    *,
    backend: str,
    hf_model_path: str | Path = "models/whisper-large-v3-turbo-hf",
    ct2_model_path: str | Path = "models/whisper-large-v3-turbo-ct2",
    cpu_threads: Optional[int] = None,
    language: Optional[str] = "auto",
    fw_vad_filter: bool = True,
    fw_condition_on_previous_text: bool = True,
    fw_temperature: Optional[float] = None,
    fw_best_of: Optional[int] = None,
    fw_without_timestamps: bool = False,
    fw_max_new_tokens: Optional[int] = None,
    fw_hotwords: Optional[str] = None,
) -> BaseLocalTranscriber:
    configure_offline_env()
    configure_cpu_threads(cpu_threads)

    if backend == "voice-local-hf":
        return HFWhisperTurboTranscriber(
            model_path=hf_model_path,
            device="cpu",
            language=language,
        )
    if backend == "voice-local-fw":
        return FasterWhisperTurboTranscriber(
            model_path=ct2_model_path,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            language=language,
            vad_filter=fw_vad_filter,
            condition_on_previous_text=fw_condition_on_previous_text,
            temperature=fw_temperature,
            best_of=fw_best_of,
            without_timestamps=fw_without_timestamps,
            max_new_tokens=fw_max_new_tokens,
            hotwords=fw_hotwords,
        )

    raise ValueError(f"Unsupported local transcriber backend: {backend}")


def _measure_peak_rss(run_fn: Callable[[], str], poll_interval_s: float = 0.05) -> tuple[str, int]:
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)

    text_holder: list[str] = []
    error_holder: list[BaseException] = []
    done = False

    def wrapped() -> None:
        nonlocal done
        try:
            text_holder.append(run_fn())
        except BaseException as exc:  # pragma: no cover - surfaced to caller
            error_holder.append(exc)
        finally:
            done = True

    import threading

    worker = threading.Thread(target=wrapped, daemon=True)
    worker.start()

    while worker.is_alive():
        try:
            peak_rss = max(peak_rss, int(process.memory_info().rss))
        except psutil.Error:
            pass
        time.sleep(poll_interval_s)

    worker.join()
    if error_holder:
        raise error_holder[0]

    peak_rss = max(peak_rss, _current_rss_bytes())
    return (text_holder[0] if text_holder else ""), peak_rss


def transcribe_with_transformers(
    *,
    model_path: str | Path,
    audio_path: str | Path,
    device: str = "cpu",
    language: Optional[str] = "auto",
) -> TranscriptionResult:
    configure_offline_env()

    model_path = str(model_path)
    audio_path = str(audio_path)

    load_started = time.perf_counter()

    transcriber = HFWhisperTurboTranscriber(
        model_path=model_path,
        device=device,
        language=language,
    )
    load_seconds = time.perf_counter() - load_started

    transcribe_started = time.perf_counter()

    transcript_result: Optional[TranscriptResult] = None

    def run_transcribe() -> str:
        nonlocal transcript_result
        transcript_result = transcriber.transcribe_result(audio_path)
        return transcript_result.text

    text, peak_rss_bytes = _measure_peak_rss(run_transcribe)
    transcribe_seconds = time.perf_counter() - transcribe_started

    return TranscriptionResult(
        backend="hf_whisper_turbo",
        model_path=model_path,
        audio_path=audio_path,
        requested_language=transcriber.requested_language,
        text=text,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        total_seconds=load_seconds + transcribe_seconds,
        peak_rss_bytes=peak_rss_bytes,
        peak_rss_gb=peak_rss_bytes / (1024 ** 3),
    )


def transcribe_with_faster_whisper(
    *,
    model_path: str | Path,
    audio_path: str | Path,
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: Optional[int] = None,
    language: Optional[str] = "auto",
) -> TranscriptionResult:
    configure_offline_env()

    model_path = str(model_path)
    audio_path = str(audio_path)

    load_started = time.perf_counter()

    transcriber = FasterWhisperTurboTranscriber(
        model_path=model_path,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        language=language,
    )
    load_seconds = time.perf_counter() - load_started

    transcribe_started = time.perf_counter()

    transcript_result: Optional[TranscriptResult] = None

    def run_transcribe() -> str:
        nonlocal transcript_result
        transcript_result = transcriber.transcribe_result(audio_path)
        return transcript_result.text

    text, peak_rss_bytes = _measure_peak_rss(run_transcribe)
    transcribe_seconds = time.perf_counter() - transcribe_started

    return TranscriptionResult(
        backend="faster_whisper_turbo",
        model_path=model_path,
        audio_path=audio_path,
        requested_language=transcriber.requested_language,
        text=text,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        total_seconds=load_seconds + transcribe_seconds,
        peak_rss_bytes=peak_rss_bytes,
        peak_rss_gb=peak_rss_bytes / (1024 ** 3),
    )
