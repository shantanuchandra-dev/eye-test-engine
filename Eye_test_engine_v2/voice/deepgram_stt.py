"""Optional Deepgram pre-recorded STT (linear16 PCM, 16 kHz mono)."""

from __future__ import annotations

import httpx

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"


def _extract_transcript(payload: dict) -> str:
    """Parse Deepgram /v1/listen JSON into a single transcript string."""
    results = payload.get("results") or {}
    channels = results.get("channels") or []
    if not channels:
        return ""
    alts = (channels[0] or {}).get("alternatives") or []
    if not alts:
        return ""
    text = (alts[0] or {}).get("transcript") or ""
    return str(text).strip()


async def transcribe_pcm16_le(
    pcm_bytes: bytes,
    api_key: str,
    *,
    language: str = "en",
    model: str = "nova-2",
    timeout_sec: float = 60.0,
) -> str:
    """
    Transcribe raw little-endian int16 PCM (16 kHz, mono).

    Args:
        api_key: Deepgram API key (never send from the browser).
        language: Pipeline lang code 'en' or 'hi' (mapped for Deepgram).
        model: e.g. nova-2, nova-2-general, etc.
    """
    if not pcm_bytes:
        return ""
    lang = "hi" if language == "hi" else "en"
    params = {
        "model": model,
        "language": lang,
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": "1",
        "smart_format": "true",
    }
    headers = {
        "Authorization": f"Token {api_key.strip()}",
        "Content-Type": "application/octet-stream",
    }
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        r = await client.post(
            DEEPGRAM_LISTEN_URL,
            params=params,
            content=pcm_bytes,
            headers=headers,
        )
    r.raise_for_status()
    data = r.json()
    return _extract_transcript(data)
