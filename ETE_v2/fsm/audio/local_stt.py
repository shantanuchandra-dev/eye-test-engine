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


class BaseLocalTranscriber:
    backend_name: str = "base"

    def transcribe(self, audio_path: str | Path, *, language_override: Optional[str] = None) -> str:
        return self.transcribe_result(audio_path, language_override=language_override).text

    def transcribe_result(
        self,
        audio_path: str | Path,
        *,
        language_override: Optional[str] = None,
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


def record_audio_clip(
    *,
    output_path: str | Path,
    seconds: float = 2.0,
    samplerate: int = 16000,
    stop_requested: Optional[Callable[[], bool]] = None,
    frame_seconds: float = 0.1,
) -> Path:
    import sounddevice as sd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blocksize = max(1, int(samplerate * frame_seconds))
    target_frames = max(1, int(seconds * samplerate))
    captured_chunks: list[np.ndarray] = []
    captured_frames = 0

    with sd.InputStream(
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
    silence_threshold: float = 0.015,
    end_silence_seconds: float = 0.8,
    min_speech_seconds: float = 0.25,
    start_timeout_seconds: float = 2.5,
    frame_seconds: float = 0.1,
    pre_speech_seconds: float = 0.25,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> AudioCaptureResult:
    import sounddevice as sd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blocksize = max(1, int(samplerate * frame_seconds))
    preroll_chunks: deque[np.ndarray] = deque(maxlen=max(1, int(pre_speech_seconds / frame_seconds)))
    waiting_chunks: list[np.ndarray] = []
    captured_chunks: list[np.ndarray] = []
    speech_detected = False
    speech_duration = 0.0
    trailing_silence = 0.0
    peak_amplitude = 0.0
    stop_reason = "max_duration"
    capture_started = time.perf_counter()

    with sd.InputStream(
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
            chunk_level = max(chunk_rms, chunk_peak * 0.5)
            peak_amplitude = max(peak_amplitude, chunk_peak)
            is_speech = chunk_level >= silence_threshold

            if speech_detected:
                captured_chunks.append(chunk.copy())
                if is_speech:
                    speech_duration += chunk_duration
                    trailing_silence = 0.0
                else:
                    trailing_silence += chunk_duration
                    if speech_duration >= min_speech_seconds and trailing_silence >= end_silence_seconds:
                        stop_reason = "silence_after_speech"
                        break
            else:
                preroll_chunks.append(chunk.copy())
                waiting_chunks.append(chunk.copy())
                if is_speech:
                    speech_detected = True
                    captured_chunks.extend(list(preroll_chunks))
                    preroll_chunks.clear()
                    waiting_chunks.clear()
                    speech_duration += chunk_duration
                    trailing_silence = 0.0
                elif elapsed >= min(max_seconds, start_timeout_seconds):
                    stop_reason = "start_timeout"
                    captured_chunks.extend(waiting_chunks)
                    preroll_chunks.clear()
                    waiting_chunks.clear()
                    break

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
    ) -> None:
        configure_offline_env()
        from faster_whisper import WhisperModel

        self.model_path = str(model_path)
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
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
    ) -> TranscriptResult:
        runtime_language_code, _, runtime_requested_language = _resolve_requested_language(
            self.requested_language if language_override is None else language_override
        )
        transcribe_kwargs = {
            "beam_size": 1,
            "vad_filter": True,
        }
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
