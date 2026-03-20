"""Deepgram STT client for Eye Test Engine voice pipeline.

Streams audio to Deepgram's WebSocket API for real-time transcription.
Returns final transcripts via a callback. Uses nova-3 model.

Requires: DEEPGRAM_API_KEY environment variable.
"""

import asyncio
import json
import os
from typing import Callable, Optional

import websockets

# Deepgram language codes for supported languages
DEEPGRAM_LANG_MAP = {
    "en": "en-IN",   # Default to Indian English for better accent handling
    "hi": "hi",
    "te": "te",
    "ta": "ta",
    "kn": "kn",
    "mr": "mr",
    "ml": "ml",
    "gu": "gu",
    "bn": "bn",
    "pa": "pa",
}


class DeepgramSTTClient:
    """Streams audio to Deepgram for real-time STT.

    Usage:
        client = DeepgramSTTClient(api_key, lang="en", on_transcript=callback)
        await client.connect()
        await client.send_audio(audio_bytes)  # call repeatedly
        await client.close()

    The on_transcript callback receives (transcript: str, is_final: bool).
    """

    def __init__(
        self,
        api_key: str,
        lang: str = "en",
        model: str = "nova-3",
        on_transcript: Optional[Callable] = None,
        on_vad: Optional[Callable] = None,
        sample_rate: int = 16000,
    ):
        self._api_key = api_key
        self._lang = DEEPGRAM_LANG_MAP.get(lang, "en-IN")
        self._model = model
        self._on_transcript = on_transcript  # async callable(transcript, is_final)
        self._on_vad = on_vad  # async callable(is_speaking: bool)
        self._sample_rate = sample_rate
        self._ws = None
        self._receive_task = None
        self._connected = False
        self._speaking = False

    async def connect(self):
        """Open WebSocket connection to Deepgram."""
        params = (
            f"model={self._model}"
            f"&language={self._lang}"
            f"&encoding=linear16"
            f"&sample_rate={self._sample_rate}"
            f"&channels=1"
            f"&punctuate=true"
            f"&interim_results=true"
            f"&endpointing=300"
            f"&vad_events=true"
            f"&smart_format=true"
        )
        url = f"wss://api.deepgram.com/v1/listen?{params}"

        headers = {"Authorization": f"Token {self._api_key}"}

        try:
            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            self._connected = True
            self._receive_task = asyncio.create_task(self._receive_loop())
            print(f"[DEEPGRAM] Connected: model={self._model} lang={self._lang}")
        except Exception as e:
            print(f"[DEEPGRAM] Connection failed: {e}")
            self._connected = False
            raise

    async def send_audio(self, audio_bytes: bytes):
        """Send raw PCM audio bytes to Deepgram."""
        if not self._connected or not self._ws:
            return
        try:
            await self._ws.send(audio_bytes)
        except Exception as e:
            print(f"[DEEPGRAM] Send error: {e}")
            self._connected = False

    async def close(self):
        """Close the Deepgram connection."""
        self._connected = False
        if self._ws:
            try:
                # Send close message per Deepgram protocol
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._receive_task = None

    async def _receive_loop(self):
        """Receive transcription results from Deepgram."""
        try:
            async for message in self._ws:
                if not self._connected:
                    break

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "Results":
                    await self._handle_result(data)
                elif msg_type == "SpeechStarted":
                    if self._on_vad and not self._speaking:
                        self._speaking = True
                        await self._on_vad(True)
                elif msg_type == "UtteranceEnd":
                    if self._on_vad and self._speaking:
                        self._speaking = False
                        await self._on_vad(False)
                elif msg_type == "Metadata":
                    pass  # Connection metadata
                elif msg_type == "Error":
                    print(f"[DEEPGRAM] Error: {data.get('description', data)}")

        except websockets.exceptions.ConnectionClosed:
            print("[DEEPGRAM] Connection closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[DEEPGRAM] Receive error: {e}")
        finally:
            self._connected = False

    async def _handle_result(self, data: dict):
        """Process a transcription result."""
        channel = data.get("channel", {})
        alternatives = channel.get("alternatives", [])
        if not alternatives:
            return

        transcript = alternatives[0].get("transcript", "").strip()
        is_final = data.get("is_final", False)
        speech_final = data.get("speech_final", False)

        if not transcript:
            return

        # We care about final transcripts (after endpointing)
        if self._on_transcript and (is_final or speech_final):
            await self._on_transcript(transcript, True)
        elif self._on_transcript and not is_final:
            # Interim — could send to UI for live preview
            await self._on_transcript(transcript, False)

    @property
    def is_connected(self) -> bool:
        return self._connected
