/**
 * ETE v2 — Main Test Screen JavaScript
 *
 * Drives the eye test session: receives questions from the FSM backend,
 * displays options, sends responses, updates prescription table and phase progress.
 */

const API = window.BACKEND_URL || '';
const LOGS_PASSWORD = 'Shantanu';
const TTS_VOICE_STORAGE_KEY = 'ete_tts_voice_name';
// Single-switch rollback point for the duplex audio layer.
// Set back to "legacy" to restore the original half-duplex behavior.
const AUDIO_TURN_MODE = 'duplex';
// Keep early controller/keyboard answers, but disable early voice interruption.
const EARLY_VOICE_BARGE_IN_ENABLED = false;

// ── TTS (Browser SpeechSynthesis) ──
let ttsEnabled = true;
let ttsSelectedVoiceName = null; // null = auto; string = pinned voice name
let _ttsHtmlAudio = null;

/** Sarvam bulbul cloud voice (must match fsm_tts_phrases.SARVAM_AI_OPTUM_VOICES / API speaker ids). */
const SARVAM_CLOUD_VOICES = [
  { id: 'shruti', label: 'AI Optum: Shruti' },
  { id: 'ishita', label: 'AI Optum: Ishita' },
];
const SARVAM_DEFAULT_SPEAKER_ID = 'ishita';
const LS_SARVAM_CLOUD_SPEAKER = 'eteSarvamCloudSpeaker';

function loadSarvamCloudSpeaker() {
  try {
    const v = localStorage.getItem(LS_SARVAM_CLOUD_SPEAKER);
    if (v && SARVAM_CLOUD_VOICES.some((x) => x.id === v)) return v;
  } catch (e) { /* ignore */ }
  return SARVAM_DEFAULT_SPEAKER_ID;
}

let sarvamCloudSpeaker = loadSarvamCloudSpeaker();

function syncSarvamCloudVoiceSelect() {
  const sel = document.getElementById('sarvamCloudVoiceSelect');
  if (!sel) return;
  if (SARVAM_CLOUD_VOICES.some((x) => x.id === sarvamCloudSpeaker)) {
    sel.value = sarvamCloudSpeaker;
  } else {
    sarvamCloudSpeaker = SARVAM_DEFAULT_SPEAKER_ID;
    sel.value = sarvamCloudSpeaker;
  }
}

function setSarvamCloudSpeaker(id) {
  if (!SARVAM_CLOUD_VOICES.some((x) => x.id === id)) return;
  sarvamCloudSpeaker = id;
  try {
    localStorage.setItem(LS_SARVAM_CLOUD_SPEAKER, id);
  } catch (e) { /* ignore */ }
  syncSarvamCloudVoiceSelect();
  console.log(`[TTS] Sarvam cloud voice: ${id}`);
}

function ttsClipUrl(hash) {
  const sp = encodeURIComponent(sarvamCloudSpeaker);
  return `${API}/api/tts-sarvam/${hash}.mp3?speaker=${sp}`;
}

/** Load MP3 from Sarvam: GET by hash (cache or server-resolved phrase), then POST /synthesize for any text. */
async function fetchSarvamSpeechBlob(text, sinceGen) {
  if (_ttsStale(sinceGen)) return null;
  const hash = await sha256Hex(text);
  if (hash) {
    try {
      const r = await fetch(ttsClipUrl(hash));
      if (r.ok) {
        const blob = await r.blob();
        if (!_ttsStale(sinceGen)) return blob;
      }
    } catch (e) { /* try POST */ }
  }
  if (_ttsStale(sinceGen)) return null;
  try {
    const r = await fetch(`${API}/api/tts-sarvam/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: (text || '').normalize('NFC'),
        speaker: sarvamCloudSpeaker,
      }),
    });
    if (!r.ok) return null;
    const blob = await r.blob();
    return _ttsStale(sinceGen) ? null : blob;
  } catch (e) {
    return null;
  }
}

const TTS_RETRY_PREFIXES_EN = ['Let me repeat. ', 'I could not hear you. ', "I didn't catch that. "];
const TTS_RETRY_PREFIXES_HI = ['फिर से सुनिए। ', 'सुनाई नहीं दिया। ', 'समझ नहीं आया। '];

function stopHtmlTtsAudio() {
  if (_ttsHtmlAudio) {
    try {
      _ttsHtmlAudio.onended = null;
      _ttsHtmlAudio.onerror = null;
      _ttsHtmlAudio.pause();
      _ttsHtmlAudio.currentTime = 0;
      _ttsHtmlAudio.removeAttribute('src');
      _ttsHtmlAudio.load();
    } catch (e) { /* ignore */ }
    _ttsHtmlAudio = null;
  }
}

/** True if this TTS request was invalidated by stopExamTtsAndTimers() (terminal / reset). */
function _ttsStale(sinceGen) {
  return sinceGen != null && sinceGen !== _ttsAbortGeneration;
}

/** True while browser or cloud (HTMLAudio) TTS is active (do not start mic / shortcuts). */
function isTtsActive() {
  return (
    ('speechSynthesis' in window && speechSynthesis.speaking)
    || !!_ttsHtmlAudio
  );
}

/** Mic may commit a response only after the post-beep gate; JCC flip1 is observation-only (TTS says "first option", etc.). */
function voiceSubmissionAllowed(state) {
  if (state === 'LANG_SELECT') return !isTtsActive();
  return _inputEnabled && _flipState !== 'flip1' && !isTtsActive();
}

/** Abort browser STT and stop Whisper recording so stale callbacks cannot submit after a new prompt. */
function stopVoiceCaptureHard() {
  voiceRecording = false;
  if (recognition) {
    try { recognition.abort(); } catch (e) { /* ignore */ }
    recognition = null;
  }
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    try { mediaRecorder.stop(); } catch (e) { /* ignore */ }
  }
}

/** Invalidates in-flight speakQuestion callbacks when the exam hits a terminal state. */
let _ttsAbortGeneration = 0;
const _ttsGestureAbortCallbacks = [];

function registerTtsGestureAbort(cleanup) {
  _ttsGestureAbortCallbacks.push(cleanup);
}

/** Stop all exam TTS, pending gesture retries, and JCC auto-flip timers (call when test ends or session resets). */
function stopExamTtsAndTimers() {
  _ttsAbortGeneration += 1;
  const pending = _ttsGestureAbortCallbacks.splice(0);
  for (const cb of pending) {
    try { cb(); } catch (e) { /* ignore */ }
  }
  stopHtmlTtsAudio();
  if ('speechSynthesis' in window) speechSynthesis.cancel();
  stopVoiceCaptureHard();
  if (_autoFlipTimer) {
    clearTimeout(_autoFlipTimer);
    _autoFlipTimer = null;
  }
  _flipState = null;
  updateFlipIndicator(null);
}

async function sha256Hex(str) {
  if (!crypto || !crypto.subtle) return null;
  const normalized = (str || '').normalize('NFC');
  const buf = new TextEncoder().encode(normalized);
  const hashBuffer = await crypto.subtle.digest('SHA-256', buf);
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

function splitRetryPrefix(text) {
  const all = [...TTS_RETRY_PREFIXES_EN, ...TTS_RETRY_PREFIXES_HI];
  for (const p of all) {
    if (text.startsWith(p)) return [p, text.slice(p.length)];
  }
  return null;
}

/** Same strings as handleAutoFlip baseFlip1Text — used with transition preface compound cache. */
const FLIP1_TTS_BODY_EN = 'Here is the first option. Take your time and look carefully.';
const FLIP1_TTS_BODY_HI = 'यह पहला विकल्प है। आराम से ध्यान से देखिए।';
const FLIP2_PREFIX_EN = 'And now the second option.';
const FLIP2_PREFIX_HI = 'अब दूसरा विकल्प है।';

function _escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Split session_orchestrator transition preface (no patient name) + JCC flip 1 body for two-part cloud TTS play. */
function splitTransitionPrefaceFlipBody(text) {
  const t = (text || '').replace(/\s+/g, ' ').trim();
  const enRe = new RegExp(
    `^(You are doing great\\. Please blink a few times\\. About \\d+ (?:minute|minutes) left\\.)\\s+${_escapeRegExp(FLIP1_TTS_BODY_EN)}$`,
  );
  let m = t.match(enRe);
  if (m) return [m[1].trim(), FLIP1_TTS_BODY_EN];
  const hiRe = new RegExp(
    `^(आप बहुत अच्छा कर रहे हैं। कृपया कुछ बार पलक झपकाइए। लगभग \\d+ मिनट बाकी हैं।)\\s+${_escapeRegExp(FLIP1_TTS_BODY_HI)}$`,
    'u',
  );
  m = t.match(hiRe);
  if (m) return [m[1].trim(), FLIP1_TTS_BODY_HI];
  return null;
}

/**
 * Transition preface may include a patient name (uncached). Split into generic preface + body
 * so we play two cached clips (same strings as fsm_tts_phrases preface-only + question).
 */
function splitGenericTransitionPrefaceAndBody(text) {
  const t = (text || '').replace(/\s+/g, ' ').trim();
  // Optional patient name between "great" / "हैं" and the period (must also match generic, no-name preface).
  const enRe =
    /^You are doing great(?:\s+(.+?))?\. Please blink a few times\. About (\d+) (?:minute|minutes) left\.\s+([\s\S]+)$/;
  let m = t.match(enRe);
  if (m) {
    const n = parseInt(m[2], 10);
    const body = m[3].trim();
    const lab = n === 1 ? 'minute' : 'minutes';
    const genericPreface = `You are doing great. Please blink a few times. About ${n} ${lab} left.`;
    return [genericPreface, body];
  }
  const hiRe =
    /^आप बहुत अच्छा कर रहे हैं(?:\s+(.+?))?। कृपया कुछ बार पलक झपकाइए। लगभग (\d+) मिनट बाकी हैं।\s+([\s\S]+)$/u;
  m = t.match(hiRe);
  if (m) {
    const n = parseInt(m[2], 10);
    const body = m[3].trim();
    const genericPreface = `आप बहुत अच्छा कर रहे हैं। कृपया कुछ बार पलक झपकाइए। लगभग ${n} मिनट बाकी हैं।`;
    return [genericPreface, body];
  }
  return null;
}

/** JCC flip 2: prefix + question (DOM often shows short "First option…" while cache uses normalized EN). */
function splitFlip2PrefixAndBody(text) {
  const t = (text || '').replace(/\s+/g, ' ').trim();
  const enRe = new RegExp(`^${_escapeRegExp(FLIP2_PREFIX_EN)}\\s+([\\s\\S]+)$`);
  let m = t.match(enRe);
  if (m) return [FLIP2_PREFIX_EN, m[1].trim()];
  const hiRe = new RegExp(`^${_escapeRegExp(FLIP2_PREFIX_HI)}\\s+([\\s\\S]+)$`, 'u');
  m = t.match(hiRe);
  if (m) return [FLIP2_PREFIX_HI, m[1].trim()];
  return null;
}

function playBlobAudio(blob, onEnd) {
  return new Promise((resolve, reject) => {
    stopHtmlTtsAudio();
    speechSynthesis.cancel();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    _ttsHtmlAudio = audio;
    let urlRevoked = false;
    const revoke = () => {
      if (urlRevoked) return;
      urlRevoked = true;
      URL.revokeObjectURL(url);
    };
    audio.onended = () => {
      revoke();
      if (_ttsHtmlAudio === audio) _ttsHtmlAudio = null;
      if (onEnd) onEnd();
      resolve();
    };
    audio.onerror = (e) => {
      revoke();
      if (_ttsHtmlAudio === audio) _ttsHtmlAudio = null;
      reject(e);
    };

    const tryPlay = () => {
      audio.play().catch((err) => {
        // Without a prior user gesture, HTMLAudio often rejects with NotAllowedError while
        // speechSynthesis still works — which caused browser TTS first, then cached MP3 on a later step.
        if (err && err.name === 'NotAllowedError') {
          let gestureCleaned = false;
          let gestureTo = null;
          function cleanupWait() {
            if (gestureCleaned) return;
            gestureCleaned = true;
            if (gestureTo) clearTimeout(gestureTo);
            document.removeEventListener('pointerdown', resume, true);
            document.removeEventListener('keydown', resume, true);
          }
          function resume() {
            cleanupWait();
            audio.play().catch(reject);
          }
          registerTtsGestureAbort(cleanupWait);
          document.addEventListener('pointerdown', resume, { once: true, capture: true });
          document.addEventListener('keydown', resume, { once: true, capture: true });
          // If the user never interacts, fall back to browser TTS after a delay (tryPlayCached → speakQuestionBrowserFallback).
          gestureTo = setTimeout(() => {
            cleanupWait();
            revoke();
            if (_ttsHtmlAudio === audio) _ttsHtmlAudio = null;
            reject(err);
          }, 8000);
          return;
        }
        reject(err);
      });
    };
    tryPlay();
  });
}

async function tryPlayCachedCloudExactTTS(text, onEnd, sinceGen = null) {
  if (_ttsStale(sinceGen)) return false;
  const blob = await fetchSarvamSpeechBlob(text, sinceGen);
  if (!blob) return false;
  try {
    await playBlobAudio(blob, onEnd);
    console.log('[TTS] Sarvam (exact)');
    return true;
  } catch (e) {
    console.warn('[TTS] Sarvam clip failed:', e);
    return false;
  }
}

async function tryPlayCachedCompoundTTS(text, onEnd, sinceGen = null) {
  if (_ttsStale(sinceGen)) return false;
  const sp = splitRetryPrefix(text);
  if (!sp) return false;
  const [prefix, rawBody] = sp;
  let body = rawBody.trim();
  if (!body) return false;
  // DOM question is often the short FSM line; cloud cache uses buildSpokenQuestionText() normalization.
  if ((sessionLanguage || 'en') !== 'hi') {
    body = buildSpokenQuestionText(currentState || {}, body, '');
  }
  if (_ttsStale(sinceGen)) return false;
  const [b1, b2] = await Promise.all([
    fetchSarvamSpeechBlob(prefix, sinceGen),
    fetchSarvamSpeechBlob(body, sinceGen),
  ]);
  if (!b1 || !b2) return false;
  if (_ttsStale(sinceGen)) return false;
  try {
    await playBlobAudio(b1, null);
    if (_ttsStale(sinceGen)) return false;
    await playBlobAudio(b2, onEnd);
    console.log('[TTS] Sarvam (retry prefix + question)');
    return true;
  } catch (e) {
    console.warn('[TTS] Sarvam compound failed:', e);
    return false;
  }
}

async function tryPlayCachedTransitionFlipTTS(text, onEnd, sinceGen = null) {
  if (_ttsStale(sinceGen)) return false;
  const sp = splitTransitionPrefaceFlipBody(text);
  if (!sp) return false;
  const [preface, body] = sp;
  if (_ttsStale(sinceGen)) return false;
  const [b1, b2] = await Promise.all([
    fetchSarvamSpeechBlob(preface, sinceGen),
    fetchSarvamSpeechBlob(body, sinceGen),
  ]);
  if (!b1 || !b2) return false;
  if (_ttsStale(sinceGen)) return false;
  try {
    await playBlobAudio(b1, null);
    if (_ttsStale(sinceGen)) return false;
    await playBlobAudio(b2, onEnd);
    console.log('[TTS] Sarvam (transition preface + flip 1)');
    return true;
  } catch (e) {
    console.warn('[TTS] Transition+flip compound failed:', e);
    return false;
  }
}

async function tryPlayCachedGenericTransitionCompoundTTS(text, onEnd, sinceGen = null) {
  if (_ttsStale(sinceGen)) return false;
  const sp = splitGenericTransitionPrefaceAndBody(text);
  if (!sp) return false;
  const [preface, rawBody] = sp;
  let body = rawBody.trim();
  if (!body) return false;
  if ((sessionLanguage || 'en') !== 'hi') {
    body = buildSpokenQuestionText(currentState || {}, body, '');
  }
  if (_ttsStale(sinceGen)) return false;
  const [b1, b2] = await Promise.all([
    fetchSarvamSpeechBlob(preface, sinceGen),
    fetchSarvamSpeechBlob(body, sinceGen),
  ]);
  if (!b1 || !b2) return false;
  if (_ttsStale(sinceGen)) return false;
  try {
    await playBlobAudio(b1, null);
    if (_ttsStale(sinceGen)) return false;
    await playBlobAudio(b2, onEnd);
    console.log('[TTS] Sarvam (generic transition preface + question)');
    return true;
  } catch (e) {
    console.warn('[TTS] Generic transition compound failed:', e);
    return false;
  }
}

async function tryPlayCachedFlip2CompoundTTS(text, onEnd, sinceGen = null) {
  if (_ttsStale(sinceGen)) return false;
  const sp = splitFlip2PrefixAndBody(text);
  if (!sp) return false;
  const [prefix, rawBody] = sp;
  let body = rawBody.trim();
  if (!body) return false;
  if ((sessionLanguage || 'en') !== 'hi') {
    body = buildSpokenQuestionText(currentState || {}, body, '');
  }
  if (_ttsStale(sinceGen)) return false;
  const [b1, b2] = await Promise.all([
    fetchSarvamSpeechBlob(prefix, sinceGen),
    fetchSarvamSpeechBlob(body, sinceGen),
  ]);
  if (!b1 || !b2) return false;
  if (_ttsStale(sinceGen)) return false;
  try {
    await playBlobAudio(b1, null);
    if (_ttsStale(sinceGen)) return false;
    await playBlobAudio(b2, onEnd);
    console.log('[TTS] Sarvam (flip 2 prefix + question)');
    return true;
  } catch (e) {
    console.warn('[TTS] Flip2 compound failed:', e);
    return false;
  }
}

async function tryPlayCachedTTS(text, onEnd, sinceGen = null) {
  if (await tryPlayCachedCloudExactTTS(text, onEnd, sinceGen)) return true;
  if (await tryPlayCachedCompoundTTS(text, onEnd, sinceGen)) return true;
  if (await tryPlayCachedTransitionFlipTTS(text, onEnd, sinceGen)) return true;
  if (await tryPlayCachedFlip2CompoundTTS(text, onEnd, sinceGen)) return true;
  if (await tryPlayCachedGenericTransitionCompoundTTS(text, onEnd, sinceGen)) return true;
  return false;

try {
  ttsSelectedVoiceName = localStorage.getItem(TTS_VOICE_STORAGE_KEY) || null;
} catch (e) {
  ttsSelectedVoiceName = null;
}

// Fixed voice list: only these voices are shown in the dropdown
const ALLOWED_VOICES = [
  { match: v => v.name === 'Samantha',          label: 'Samantha (US)',       group: '🇺🇸 US English', disabled: false },
  { match: v => v.name === 'Google US English', label: 'Google US English',   group: '🇺🇸 US English', disabled: false },
  { match: v => v.name === 'Rishi',             label: 'Rishi (IN English)',  group: '🇮🇳 Indian',     disabled: false },
  { match: v => v.name === 'Google हिन्दी',       label: 'Google Hindi',        group: '🇮🇳 Indian',     disabled: false },
];

// Coming-soon Indian voices (shown disabled in dropdown)
const COMING_SOON_VOICES = [
  { label: 'Google Bengali',   group: '🇮🇳 Indian' },
  { label: 'Google Tamil',     group: '🇮🇳 Indian' },
  { label: 'Google Telugu',    group: '🇮🇳 Indian' },
  { label: 'Google Gujarati',  group: '🇮🇳 Indian' },
  { label: 'Google Kannada',   group: '🇮🇳 Indian' },
  { label: 'Google Malayalam', group: '🇮🇳 Indian' },
  { label: 'Google Marathi',   group: '🇮🇳 Indian' },
];

const ENGLISH_VOICE_PREFERENCE_ORDER = [
  'Samantha',
  'Ava',
  'Allison',
  'Karen',
  'Moira',
  'Tessa',
  'Veena',
  'Rishi',
  'Daniel',
  'Alex',
  'Google UK English Female',
  'Google US English',
];

const HINDI_VOICE_PREFERENCE_ORDER = [
  'Google हिन्दी',
];

function isSafariBrowser() {
  const ua = navigator.userAgent || '';
  const vendor = navigator.vendor || '';
  return /Safari/i.test(ua) && /Apple/i.test(vendor) && !/Chrome|CriOS|Edg|OPR|Firefox|FxiOS/i.test(ua);
}

function isDuplexTurnMode() {
  return AUDIO_TURN_MODE === 'duplex';
}

function scoreVoiceForTTS(voice, lang = 'en') {
  if (!voice) return Number.NEGATIVE_INFINITY;

  const name = (voice.name || '').toLowerCase();
  const voiceLang = (voice.lang || '').toLowerCase();
  let score = 0;

  if (lang === 'hi') {
    const idx = HINDI_VOICE_PREFERENCE_ORDER.findIndex(pref => pref.toLowerCase() === name);
    if (idx >= 0) score += 500 - idx * 25;
    if (voiceLang === 'hi-in') score += 120;
    else if (voiceLang.startsWith('hi')) score += 90;
    if (voice.default) score += 15;
    if (voice.localService) score += 10;
    return score;
  }

  const idx = ENGLISH_VOICE_PREFERENCE_ORDER.findIndex(pref => pref.toLowerCase() === name);
  if (idx >= 0) score += 600 - idx * 20;
  if (voiceLang === 'en-in') score += 120;
  else if (voiceLang === 'en-us') score += 110;
  else if (voiceLang === 'en-gb') score += 100;
  else if (voiceLang.startsWith('en')) score += 80;
  if (voice.default) score += 15;
  if (voice.localService) score += 10;
  if (name.includes('google us english')) score -= 30;
  return score;
}

function getBestAvailableVoice(voices, lang = 'en') {
  const pool = (voices || []).filter(v => (v.lang || '').toLowerCase().startsWith(lang === 'hi' ? 'hi' : 'en'));
  if (!pool.length) return null;

  let best = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const voice of pool) {
    const score = scoreVoiceForTTS(voice, lang);
    if (score > bestScore) {
      best = voice;
      bestScore = score;
    }
  }
  return best;
}

function getTTSProfile(profileKey = 'default', lang = 'en') {
  const englishProfiles = {
    default: { rate: 1, pitch: 1.02, volume: 0.95 },
    guide: { rate: 1, pitch: 1.03, volume: 0.96 },
    reading: { rate: 1, pitch: 1.01, volume: 0.95 },
    comparison: { rate: 1, pitch: 1.0, volume: 0.95 },
    retry: { rate: 1, pitch: 1.0, volume: 0.95 },
    flip1: { rate: 1, pitch: 0.99, volume: 0.95 },
    flip2: { rate: 1, pitch: 1.0, volume: 0.95 },
    terminal: { rate: 1, pitch: 1.05, volume: 0.97 },
  };
  const hindiProfiles = {
    default: { rate: 1, pitch: 1.0, volume: 0.97 },
    guide: { rate: 1, pitch: 1.01, volume: 0.97 },
    reading: { rate: 1, pitch: 1.0, volume: 0.97 },
    comparison: { rate: 1, pitch: 1.0, volume: 0.97 },
    retry: { rate: 1, pitch: 1.0, volume: 0.97 },
    flip1: { rate: 1, pitch: 0.99, volume: 0.97 },
    flip2: { rate: 1, pitch: 1.0, volume: 0.97 },
    terminal: { rate: 1, pitch: 1.03, volume: 0.98 },
  };
  const profiles = lang === 'hi' ? hindiProfiles : englishProfiles;
  return profiles[profileKey] || profiles.default;
}

function getQuestionTTSProfile(data) {
  if (!data) return 'default';
  if (['B', 'C', 'D', 'L'].includes(data.state)) return 'reading';
  if (['E', 'F', 'G', 'H', 'I', 'J', 'K'].includes(data.state)) return 'comparison';
  return 'default';
}

function getStaticQuestionSpeech(localizedQuestion) {
  return (localizedQuestion || '').replace(/\s+/g, ' ').trim();
}

function getStaticFlipPrompt(stage, baseQuestion = '') {
  const prompt = (baseQuestion || '').toLowerCase();
  const useShortForm = !(
    prompt.includes('please compare the two dot patterns')
    || prompt.includes('please compare the two choices')
  );
  const useCondensedSecondPrompt = _jccPromptExposureCount > 3;

  if (sessionLanguage === 'hi') {
    if (stage === 'flip1') {
      return useShortForm
        ? 'पहला विकल्प'
        : 'कृपया दोनों डॉट पैटर्न की तुलना कीजिए। कौन सा ज़्यादा साफ या शार्प दिख रहा है? यह पहला विकल्प है।';
    }
    if (useCondensedSecondPrompt) {
      return 'दूसरा विकल्प। कौन बेहतर है, या दोनों समान हैं?';
    }
    return useShortForm
      ? 'दूसरा विकल्प। कौन बेहतर है? पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से?'
      : 'यह दूसरा विकल्प है। कौन बेहतर है, पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से कहिए।';
  }

  if (stage === 'flip1') {
    return useShortForm
      ? 'First option'
      : 'Please compare the two dot patterns. Which one is clearer or sharper? This is the first option.';
  }
  if (useCondensedSecondPrompt) {
    return 'Second option. Which is better, or are both same?';
  }
  return useShortForm
    ? 'Second option. Which is better? First option, second option, both same or repeat?'
    : 'This is the second option. Which is better, first option, second option, both same, or repeat.';
}

function getStaticTerminalSpeech({ isEscalate, compareRan, acceptedAchieved }) {
  if (isEscalate) {
    return sessionLanguage === 'hi'
      ? 'इस टेस्ट को ऑप्टोमेट्रिस्ट की समीक्षा की आवश्यकता है।'
      : 'This test requires optometrist review.';
  }
  if (compareRan && acceptedAchieved) {
    return sessionLanguage === 'hi'
      ? 'बधाई हो। आपका आई टेस्ट पूरा हो गया है। आपने पी जी पी की तुलना में प्राप्त आर एक्स को पसंद किया। धन्यवाद।'
      : 'Congratulations. Your eye test is complete. You preferred the achieved prescription over the PGP. Thank you.';
  }
  if (compareRan) {
    return sessionLanguage === 'hi'
      ? 'बधाई हो। आपका आई टेस्ट पूरा हो गया है। आपने पी जी पी को पसंद किया। धन्यवाद।'
      : 'Congratulations. Your eye test is complete. You preferred the PGP. Thank you.';
  }
  return sessionLanguage === 'hi'
    ? 'बधाई हो! आपका आई टेस्ट पूरा हो गया है। धन्यवाद।'
    : 'Congratulations. Your eye test is complete. Thank you.';
}

function populateTTSVoiceDropdown() {
  const sel = document.getElementById('ttsVoiceSelect');
  if (!sel) return;
  const voices = speechSynthesis.getVoices();

  sel.innerHTML = '';

  const groups = {};

  const getOrCreateGroup = (groupLabel) => {
    if (!groups[groupLabel]) {
      groups[groupLabel] = document.createElement('optgroup');
      groups[groupLabel].label = groupLabel;
      sel.appendChild(groups[groupLabel]);
    }
    return groups[groupLabel];
  };

  // Active voices
  ALLOWED_VOICES.forEach(({ match, label, group }) => {
    const voice = voices.find(match);
    if (!voice) return;
    const opt = document.createElement('option');
    opt.value = voice.name;
    opt.textContent = label;
    getOrCreateGroup(group).appendChild(opt);
  });

  // Coming-soon voices (disabled)
  COMING_SOON_VOICES.forEach(({ label, group }) => {
    const opt = document.createElement('option');
    opt.value = '__coming_soon__';
    opt.textContent = `${label} — Coming Soon`;
    opt.disabled = true;
    opt.style.color = '#94a3b8';
    getOrCreateGroup(group).appendChild(opt);
  });

  // Restore previous selection
  if (ttsSelectedVoiceName) {
    sel.value = ttsSelectedVoiceName;
  }

  // Default: Samantha
  if (!ttsSelectedVoiceName) {
    const preferred = getBestAvailableVoice(voices, 'en');
    ttsSelectedVoiceName = preferred ? preferred.name : null;
    if (ttsSelectedVoiceName) sel.value = ttsSelectedVoiceName;
  }

  console.log(`[TTS] Voice dropdown ready. Selected: ${ttsSelectedVoiceName}`);
}

function setTTSVoice(name) {
  if (name === '__coming_soon__') {
    // Snap back to previous valid selection
    const sel = document.getElementById('ttsVoiceSelect');
    if (sel && ttsSelectedVoiceName) sel.value = ttsSelectedVoiceName;
    showComingSoonToast();
    return;
  }
  ttsSelectedVoiceName = name;
  try { localStorage.setItem(TTS_VOICE_STORAGE_KEY, name || ''); } catch (e) {}
  console.log(`[TTS] Voice pinned to: ${name}`);
}

function showComingSoonToast() {
  const existing = document.getElementById('comingSoonToast');
  if (existing) existing.remove();
  const toast = document.createElement('div');
  toast.id = 'comingSoonToast';
  toast.textContent = '🚧 Coming Soon';
  toast.style.cssText = `
    position:fixed; bottom:24px; left:50%; transform:translateX(-50%);
    background:#1e293b; color:#f8fafc; padding:10px 22px;
    border-radius:8px; font-size:0.85rem; font-weight:600;
    box-shadow:0 4px 16px rgba(0,0,0,0.3); z-index:9999;
    animation: fadeInUp 0.2s ease;
  `;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2200);
}

function speakQuestion(text, langOverride, onEnd, profileKey = 'default') {
  const ttsSessionId = ++_ttsSessionId;

  if (!ttsEnabled || !('speechSynthesis' in window)) {
    console.log('[TTS] Disabled or unavailable');
    if (onEnd) onEnd();
    return;
  }

  // Clear only active/pending utterances so transitions sound less clipped.
  if (speechSynthesis.speaking || speechSynthesis.pending) {
    speechSynthesis.cancel();
  }
  speechSynthesis.resume();

  const doSpeak = () => {
    const lang = langOverride || sessionLanguage || 'en';
    const speechText = (text || '').replace(/\s+/g, ' ').trim();
    const utterance = new SpeechSynthesisUtterance(speechText);
    const profile = getTTSProfile(profileKey, lang);
    utterance.rate = profile.rate;
    utterance.pitch = profile.pitch;
    utterance.volume = profile.volume;
    const voices = speechSynthesis.getVoices();
    console.log(`[TTS] Speaking (${lang}, ${profileKey}): "${speechText.substring(0, 50)}..." [${voices.length} voices available]`);

    if (lang === 'hi') {
      utterance.lang = 'hi-IN';
      const hiVoice = getBestAvailableVoice(voices, 'hi');
      if (hiVoice) { utterance.voice = hiVoice; console.log(`[TTS] Hindi voice: ${hiVoice.name}`); }
    } else {
      // Use pinned voice from dropdown; set utterance.lang to match voice locale
      const pinned = ttsSelectedVoiceName
        ? voices.find(v => v.name === ttsSelectedVoiceName)
        : null;
      if (pinned) {
        utterance.voice = pinned;
        utterance.lang = pinned.lang;
        console.log(`[TTS] Using pinned voice: ${pinned.name} (${pinned.lang})`);
      } else {
        const fallback = getBestAvailableVoice(voices, 'en');
        if (fallback) { utterance.voice = fallback; utterance.lang = fallback.lang; }
        console.log(`[TTS] Fallback voice: ${fallback ? fallback.name : 'browser default'}`);
      }
    }

    const startupGuardMs = lang === 'hi' ? 2500 : 800;
    let speechStarted = false;

    // Guard against double-fire (cancel() on a previous utterance can fire onend)
    let callbackFired = false;
    const fireOnEnd = () => {
      if (ttsSessionId !== _ttsSessionId) return;
      if (callbackFired) return;
      callbackFired = true;
      clearInterval(keepAlive);
      clearTimeout(startupGuard);
      if (onEnd) onEnd();
    };

    utterance.onstart = () => {
      if (ttsSessionId !== _ttsSessionId) return;
      speechStarted = true;
      console.log('[TTS] Started speaking');
      clearTimeout(startupGuard);
    };
    utterance.onend = () => {
      if (ttsSessionId !== _ttsSessionId) return;
      console.log('[TTS] Finished speaking');
      fireOnEnd();
    };
    utterance.onerror = (e) => {
      if (ttsSessionId !== _ttsSessionId) return;
      console.error('[TTS] Error:', e.error);
      fireOnEnd();
    };

    speechSynthesis.speak(utterance);

    // Chrome workaround: Chrome sometimes pauses speech after 15s.
    // Use resume() only (no pause first) to avoid the audible stutter on Indian voices.
    const keepAlive = setInterval(() => {
      if (!speechSynthesis.speaking) { clearInterval(keepAlive); return; }
      speechSynthesis.resume();
    }, 10000);

    // Safety fallback: if the browser silently blocks TTS (autoplay policy),
    // neither onend nor onerror fires.
    const startupGuard = setTimeout(() => {
      if (ttsSessionId !== _ttsSessionId) return;
      if (!speechStarted && !speechSynthesis.speaking && !speechSynthesis.pending) {
        console.warn(`[TTS] Speech did not start within ${startupGuardMs}ms — firing callback as fallback.`);
        fireOnEnd();
      }
    }, startupGuardMs);
  };

  // Voices may not be loaded yet — wait with timeout
  const voices = speechSynthesis.getVoices();
  if (voices.length > 0) {
    // Small delay after cancel() — Chrome needs this
    setTimeout(doSpeak, 50);
  } else {
    console.log('[TTS] Waiting for voices to load...');
    let attempts = 0;
    const waitForVoices = () => {
      attempts++;
      if (speechSynthesis.getVoices().length > 0) {
        setTimeout(doSpeak, 50);
      } else if (attempts < 30) {
        setTimeout(waitForVoices, 100);
      } else {
        console.warn('[TTS] Voices never loaded after 3s, speaking anyway');
        setTimeout(doSpeak, 50);
      }
    };
    setTimeout(waitForVoices, 50);
  }
}

let _spokenFlowGeneration = 0;

function invalidateSpokenFlow() {
  _spokenFlowGeneration += 1;
}

function estimateSpeechFloorMs(text, lang) {
  const cleaned = (text || '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return 0;
  const wordCount = cleaned.split(' ').filter(Boolean).length;
  const punctuationCount = (cleaned.match(/[,.!?]/g) || []).length;
  if (lang === 'hi') {
    return Math.min(12000, Math.max(2200, 900 + wordCount * 340 + punctuationCount * 140));
  }
  return 0;
}

function speakQuestionWithStableFollowup(text, langOverride, onEnd, profileKey = 'default') {
  if (!ttsEnabled || !('speechSynthesis' in window)) {
    if (onEnd) onEnd();
    return;
  }

  _spokenFlowGeneration += 1;
  speakQuestion(text, langOverride, () => {
    if (onEnd) onEnd();
  }, profileKey);
}

// Preload voices and populate dropdown (needed on some browsers)
if ('speechSynthesis' in window) {
  speechSynthesis.getVoices();
  speechSynthesis.onvoiceschanged = () => populateTTSVoiceDropdown();
  // Fallback for browsers that load voices synchronously
  setTimeout(populateTTSVoiceDropdown, 300);
}

// ── Distance chart stimuli (from FSMv3.1_R2) ──
const DISTANCE_CHART_STIMULI = {
  '200_150':     [['E','N','H'], ['S','L','C']],
  '200150':      [['E','N','H'], ['S','L','C']],
  '100_80':      [['H','B','V'], ['P','H','T']],
  '100_90':      [['H','B','V'], ['P','H','T']],
  '10080':       [['H','B','V'], ['P','H','T']],
  '70_60_50':    [['V','L','N','E','A'], ['D','A','O','F','C'], ['E','G','N','D','H']],
  '706050':      [['V','L','N','E','A'], ['D','A','O','F','C'], ['E','G','N','D','H']],
  '40_30_25':    [['F','Z','B','D','E'], ['O','F','L','C','T'], ['A','P','E','O','F']],
  '403025':      [['F','Z','B','D','E'], ['O','F','L','C','T'], ['A','P','E','O','F']],
  '20_15_10':    [['T','Z','V','E','C'], ['O','H','P','N','T'], ['V','L','F','T','H']],
  '201510':      [['T','Z','V','E','C'], ['O','H','P','N','T'], ['V','L','F','T','H']],
  '20_20_20':    [['E','V','O','T','L'], ['T','B','G','A','B'], ['H','N','F','Z','C']],
  '202020':      [['E','V','O','T','L'], ['T','B','G','A','B'], ['H','N','F','Z','C']],
  '25_20_15':    [['D','F','N','P','T'], ['P','H','U','N','T'], ['F','D','S','L','N']],
  '252015':      [['D','F','N','P','T'], ['P','H','U','N','T'], ['F','D','S','L','N']],
};

function getDisplayedDistanceChartLines(state, chartParam) {
  const normalized = (chartParam || '').replace(/[-\/]/g, '_').replace(/_+/g, '_');
  const letters = DISTANCE_CHART_STIMULI[normalized] || DISTANCE_CHART_STIMULI[normalized.replace(/_/g, '')];
  if (!letters) return null;
  if (['B', 'C', 'D', 'L'].includes(state)) {
    return [letters[letters.length - 1]];
  }
  return letters;
}

function getDisplayedDistanceChartSizes(state, chartParam) {
  const normalized = (chartParam || '').replace(/[-\/]/g, '_').replace(/_+/g, '_');
  if (!normalized) return null;
  const parts = normalized.split('_').filter(Boolean);
  if (!parts.length) return null;
  const labels = parts.map(part => `20/${part}`);
  if (['B', 'C', 'D', 'L', 'S', 'T'].includes(state)) {
    return [labels[labels.length - 1]];
  }
  return labels;
}

// ── Stimulus descriptions per state ──
const STIMULUS_DESCRIPTIONS = {
  'B': 'Distance letter chart',
  'C': 'Distance vision confirmation',
  'D': 'Distance letter chart',
  'E': 'Dot chart for axis comparison',
  'F': 'Dot chart for power comparison',
  'G': 'Red-green chart',
  'H': 'Dot chart for axis comparison',
  'I': 'Dot chart for power comparison',
  'J': 'Red-green chart',
  'L': 'Distance vision confirmation',
  'K': 'Top-bottom balance chart',
  'P': 'Near text chart',
  'Q': 'Near text chart',
  'R': 'Near text with both eyes',
  'S': 'Final prescription comparison first option achieved Rx',
  'T': 'Final prescription comparison second option PGP',
  'U': 'Final prescription comparison',
};

const STIMULUS_DESCRIPTIONS_HI = {
  'B': 'दूरी का अक्षर चार्ट',
  'C': 'दूरी की दृष्टि की पुष्टि',
  'D': 'दूरी का अक्षर चार्ट',
  'E': 'अक्ष तुलना के लिए डॉट चार्ट',
  'F': 'पावर तुलना के लिए डॉट चार्ट',
  'G': 'लाल-हरा चार्ट',
  'H': 'अक्ष तुलना के लिए डॉट चार्ट',
  'I': 'पावर तुलना के लिए डॉट चार्ट',
  'J': 'लाल-हरा चार्ट',
  'L': 'दूरी की दृष्टि की पुष्टि',
  'K': 'ऊपर-नीचे बैलेंस चार्ट',
  'P': 'पास का टेक्स्ट चार्ट',
  'Q': 'पास का टेक्स्ट चार्ट',
  'R': 'दोनों आँखों से पास का टेक्स्ट',
  'S': 'अंतिम प्रिस्क्रिप्शन तुलना पहला विकल्प प्राप्त Rx',
  'T': 'अंतिम प्रिस्क्रिप्शन तुलना दूसरा विकल्प PGP',
  'U': 'अंतिम प्रिस्क्रिप्शन तुलना',
};

function localizeStimulusDescription(state, description, lang) {
  const activeLang = lang || sessionLanguage || 'en';
  if (activeLang === 'hi') return STIMULUS_DESCRIPTIONS_HI[state] || description || '';
  return description || '';
}

const STATIC_MOTIVATION_PROMPTS = Object.freeze({
  en: Object.freeze({
    'You are doing great. Please blink a few times. About 9 minutes left.':
      'You are doing great. Please blink a few times. About 9 minutes left.',
    'You are doing great. Please blink a few times. About 8 minutes left.':
      'You are doing great. Please blink a few times. About 8 minutes left.',
    'You are doing great. Please blink a few times. About 6 minutes left.':
      'You are doing great. Please blink a few times. About 6 minutes left.',
    'You are doing great. Please blink a few times. About 4 minutes left.':
      'You are doing great. Please blink a few times. About 4 minutes left.',
    'You are doing great. Please blink a few times. About 3 minutes left.':
      'You are doing great. Please blink a few times. About 3 minutes left.',
    'You are doing great. Please blink a few times. About 1 minute left.':
      'You are doing great. Please blink a few times. About 1 minute left.',
  }),
  hi: Object.freeze({
    'You are doing great. Please blink a few times. About 9 minutes left.':
      'आप बहुत अच्छा कर रहे हैं। कृपया कुछ बार पलक झपकाइए। लगभग 9 मिनट बाकी हैं।',
    'You are doing great. Please blink a few times. About 8 minutes left.':
      'आप बहुत अच्छा कर रहे हैं। कृपया कुछ बार पलक झपकाइए। लगभग 8 मिनट बाकी हैं।',
    'You are doing great. Please blink a few times. About 6 minutes left.':
      'आप बहुत अच्छा कर रहे हैं। कृपया कुछ बार पलक झपकाइए। लगभग 6 मिनट बाकी हैं।',
    'You are doing great. Please blink a few times. About 4 minutes left.':
      'आप बहुत अच्छा कर रहे हैं। कृपया कुछ बार पलक झपकाइए। लगभग 4 मिनट बाकी हैं।',
    'You are doing great. Please blink a few times. About 3 minutes left.':
      'आप बहुत अच्छा कर रहे हैं। कृपया कुछ बार पलक झपकाइए। लगभग 3 मिनट बाकी हैं।',
    'You are doing great. Please blink a few times. About 1 minute left.':
      'आप बहुत अच्छा कर रहे हैं। कृपया कुछ बार पलक झपकाइए। लगभग 1 मिनट बाकी है।',
  }),
});

const STATIC_PREFACE_PROMPTS = Object.freeze({
  en: Object.freeze({
    'Your eye test is about to begin. Please rest your forehead gently against the forehead bar and look straight ahead.':
      'Your eye test is about to begin. Please rest your forehead gently against the forehead bar and look straight ahead.',
  }),
  hi: Object.freeze({
    'Your eye test is about to begin. Please rest your forehead gently against the forehead bar and look straight ahead.':
      'आपका आई टेस्ट अब शुरू होने वाला है। कृपया अपना माथा धीरे से फोरहेड बार पर टिकाएँ और सामने देखें।',
  }),
});

function localizePrefacePrompt(prefacePrompt, lang) {
  const activeLang = lang || sessionLanguage || 'en';
  const text = (prefacePrompt || '').trim();
  if (!text) return text;

  const staticPreface = STATIC_PREFACE_PROMPTS[activeLang]?.[text];
  if (staticPreface) return staticPreface;
  const staticPrompt = STATIC_MOTIVATION_PROMPTS[activeLang]?.[text];
  if (staticPrompt) return staticPrompt;
  if (activeLang !== 'hi') return text;

  const match = text.match(/^You are doing great(?:\s+(.+?))?\. Please blink a few times\. About (\d+) (minute|minutes) left\.$/i);
  if (match) {
    const name = (match[1] || '').trim();
    const minutes = match[2];
    return `आप बहुत अच्छा कर रहे हैं${name ? ` ${name}` : ''}। कृपया कुछ बार पलक झपकाइए। लगभग ${minutes} मिनट बाकी हैं।`;
  }
  return text;
}

function getMotivationSpeech(prefacePrompt, lang) {
  const activeLang = lang || sessionLanguage || 'en';
  const text = (prefacePrompt || '').trim();
  if (!text) return '';

  const staticPreface = STATIC_PREFACE_PROMPTS[activeLang]?.[text];
  if (staticPreface) return staticPreface;
  const staticPrompt = STATIC_MOTIVATION_PROMPTS[activeLang]?.[text];
  if (staticPrompt) return staticPrompt;

  const match = text.match(/^You are doing great(?:\s+.+?)?\. Please blink a few times\. About (\d+) (minute|minutes) left\.$/i);
  if (!match) return '';

  const minutes = match[1];
  if (activeLang === 'hi') {
    return `आप बहुत अच्छा कर रहे हैं। कृपया कुछ बार पलक झपकाइए। लगभग ${minutes} मिनट बाकी हैं।`;
  }
  return `You are doing great. Please blink a few times. About ${minutes} minutes left.`;
}

// ── Voice input state ──
let voiceEnabled = true; // ON by default, like FSMv3.1_R2
let voiceRecording = false;
let recognition = null;
let voiceSubmitting = false; // Prevent double-submit during async match
let responseSubmitting = false; // Prevent duplicate submit/transition races
let voiceAttemptCount = 0; // Per-question attempt counter
const VOICE_REPROMPT_LIMIT = 2; // After this many failed attempts, show keyboard fallback msg
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let failedVoiceAttempts = []; // Structured log of failed voice attempts
let _observeAdvanceTimer = null;
let _ttsSessionId = 0;
let _voiceStartGeneration = 0;
let _voiceStartTimer = null;
let _questionTurnToken = 0;
let _duplexBargeInTimer = null;
let _duplexBargeInStream = null;
let _duplexBargeInCtx = null;
let _duplexBargeInSource = null;
let _duplexBargeInAnalyser = null;
let _duplexSpeechDetectedToken = 0;
let _duplexRecognitionFlush = null;

const DUPLEX_BARGE_IN_GRACE_MS = 200;
const DUPLEX_BARGE_IN_HOLD_MS = 100;
const DUPLEX_BARGE_IN_POLL_MS = 50;
const DUPLEX_BARGE_IN_LEVEL = 0.018;

// ── Faster-whisper backend state ──
let whisperAvailable = false; // Set true if backend has faster-whisper loaded
let mediaRecorder = null;
let audioChunks = [];
let micStream = null;

// Parallel mic capture while Web Speech API runs (browser mode → Supabase webm, same as Whisper)
let browserCaptureRecorder = null;
let browserCaptureStream = null;
let browserCaptureChunks = [];
let browserCaptureMimeType = 'audio/webm';

function discardBrowserParallelCapture() {
  if (browserCaptureRecorder) {
    try {
      browserCaptureRecorder.ondataavailable = null;
      browserCaptureRecorder.onstop = null;
      if (browserCaptureRecorder.state === 'recording') {
        browserCaptureRecorder.stop();
      }
    } catch (e) { /* ignore */ }
    browserCaptureRecorder = null;
  }
  browserCaptureChunks = [];
  if (browserCaptureStream) {
    try {
      browserCaptureStream.getTracks().forEach((t) => t.stop());
    } catch (e) { /* ignore */ }
    browserCaptureStream = null;
  }
}

async function startBrowserParallelCaptureForSession() {
  if (voiceMode !== 'browser') return;
  discardBrowserParallelCapture();
  try {
    browserCaptureStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    browserCaptureMimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';
    browserCaptureRecorder = new MediaRecorder(browserCaptureStream, {
      mimeType: browserCaptureMimeType,
    });
    browserCaptureChunks = [];
    browserCaptureRecorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) browserCaptureChunks.push(ev.data);
    };
    browserCaptureRecorder.start();
  } catch (e) {
    console.warn('[Voice] Browser parallel mic capture failed:', e);
    discardBrowserParallelCapture();
  }
}

function stopBrowserParallelCaptureAndEncodeBase64() {
  return new Promise((resolve) => {
    const finishEmpty = () => {
      discardBrowserParallelCapture();
      resolve('');
    };
    if (!browserCaptureRecorder || browserCaptureRecorder.state === 'inactive') {
      finishEmpty();
      return;
    }
    const rec = browserCaptureRecorder;
    const mime = browserCaptureMimeType;
    rec.onstop = async () => {
      const chunks = browserCaptureChunks.slice();
      browserCaptureRecorder = null;
      browserCaptureChunks = [];
      if (browserCaptureStream) {
        try {
          browserCaptureStream.getTracks().forEach((t) => t.stop());
        } catch (e) { /* ignore */ }
        browserCaptureStream = null;
      }
      if (!chunks.length) {
        resolve('');
        return;
      }
      try {
        const blob = new Blob(chunks, { type: mime });
        const arrayBuf = await blob.arrayBuffer();
        const bytes = new Uint8Array(arrayBuf);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        resolve(btoa(binary));
      } catch (e) {
        resolve('');
      }
    };
    try {
      rec.stop();
    } catch (e) {
      finishEmpty();
    }
  });
}

function invalidatePendingVoiceStart() {
  _voiceStartGeneration += 1;
  if (_voiceStartTimer) {
    clearTimeout(_voiceStartTimer);
    _voiceStartTimer = null;
  }
}

function nextQuestionTurnToken() {
  _questionTurnToken += 1;
  return _questionTurnToken;
}

function invalidateQuestionTurn() {
  _questionTurnToken += 1;
  stopDuplexBargeInMonitor();
  _duplexSpeechDetectedToken = 0;
  _duplexRecognitionFlush = null;
  return _questionTurnToken;
}

function isCurrentQuestionTurn(token) {
  return token === _questionTurnToken;
}

function isTtsActive() {
  return !!('speechSynthesis' in window && (speechSynthesis.speaking || speechSynthesis.pending));
}

function stopDuplexBargeInMonitor() {
  if (_duplexBargeInTimer) {
    clearInterval(_duplexBargeInTimer);
    _duplexBargeInTimer = null;
  }
  if (_duplexBargeInSource) {
    try { _duplexBargeInSource.disconnect(); } catch (e) {}
    _duplexBargeInSource = null;
  }
  _duplexBargeInAnalyser = null;
  if (_duplexBargeInCtx) {
    try { _duplexBargeInCtx.close(); } catch (e) {}
    _duplexBargeInCtx = null;
  }
  if (_duplexBargeInStream) {
    try { _duplexBargeInStream.getTracks().forEach(track => track.stop()); } catch (e) {}
    _duplexBargeInStream = null;
  }
}

function isDuplexSpeechDetected(questionToken) {
  return questionToken === _duplexSpeechDetectedToken;
}

function markDuplexSpeechDetected(questionToken) {
  if (!isCurrentQuestionTurn(questionToken)) return;
  _duplexSpeechDetectedToken = questionToken;
  if (typeof _duplexRecognitionFlush === 'function') {
    _duplexRecognitionFlush(questionToken);
  }
}

function canEnableEarlyControllerInput(data) {
  return isDuplexTurnMode()
    && !!data
    && !data.is_terminal
    && !data.auto_flip
    && Number(data.auto_advance_seconds || 0) <= 0;
}

function isVoiceBargeInEligible(state, questionText, data = {}) {
  if (!EARLY_VOICE_BARGE_IN_ENABLED) return false;
  if (!isDuplexTurnMode()) return false;
  if (!voiceEnabled || voiceMode !== 'browser' || !SpeechRecognition) return false;
  if (!state || state === 'LANG_SELECT') return false;
  if (data.auto_flip || Number(data.auto_advance_seconds || 0) > 0) return false;

  if (['E', 'F', 'G', 'H', 'I', 'J', 'K', 'P', 'Q', 'R', 'U'].includes(state)) {
    return true;
  }
  if ((state === 'B' || state === 'D') && !questionExpectsLineReading(state, questionText)) {
    return true;
  }
  return false;
}

async function startDuplexBargeInMonitor({ state, options, step, questionText, questionToken, data }) {
  if (!isVoiceBargeInEligible(state, questionText, data)) return;
  if (!navigator.mediaDevices?.getUserMedia) return;

  stopDuplexBargeInMonitor();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });

    if (!isCurrentQuestionTurn(questionToken) || responseSubmitting || voiceSubmitting || !isTtsActive()) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }

    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }

    _duplexBargeInStream = stream;
    _duplexBargeInCtx = new Ctx();
    _duplexBargeInSource = _duplexBargeInCtx.createMediaStreamSource(stream);
    _duplexBargeInAnalyser = _duplexBargeInCtx.createAnalyser();
    _duplexBargeInAnalyser.fftSize = 2048;
    _duplexBargeInAnalyser.smoothingTimeConstant = 0.25;
    _duplexBargeInSource.connect(_duplexBargeInAnalyser);

    const buffer = new Float32Array(_duplexBargeInAnalyser.fftSize);
    const startedAt = performance.now();
    let hotMs = 0;

    _duplexBargeInTimer = setInterval(() => {
      if (!isCurrentQuestionTurn(questionToken) || responseSubmitting || voiceSubmitting || !voiceEnabled) {
        stopDuplexBargeInMonitor();
        return;
      }
      if (!isTtsActive()) {
        stopDuplexBargeInMonitor();
        return;
      }

      const elapsedMs = performance.now() - startedAt;
      if (elapsedMs < DUPLEX_BARGE_IN_GRACE_MS) {
        return;
      }

      _duplexBargeInAnalyser.getFloatTimeDomainData(buffer);
      let sumSquares = 0;
      let peak = 0;
      for (let i = 0; i < buffer.length; i++) {
        const v = Math.abs(buffer[i]);
        sumSquares += buffer[i] * buffer[i];
        if (v > peak) peak = v;
      }
      const rms = Math.sqrt(sumSquares / buffer.length);
      const level = Math.max(rms, peak * 0.5);

      if (level >= DUPLEX_BARGE_IN_LEVEL) {
        hotMs += DUPLEX_BARGE_IN_POLL_MS;
      } else {
        hotMs = 0;
      }

      if (hotMs < DUPLEX_BARGE_IN_HOLD_MS) return;

      console.log(`[Duplex] Voice barge-in detected for ${state}`);
      stopDuplexBargeInMonitor();
      markDuplexSpeechDetected(questionToken);
      invalidatePendingVoiceStart();
      if (recognition) {
        try { recognition.abort(); } catch (e) {}
        recognition = null;
      }
      discardBrowserParallelCapture();
      _duplexRecognitionFlush = null;
      _inputEnabled = true;
      if (isTtsActive()) {
        _ttsSessionId += 1;
        try { speechSynthesis.cancel(); } catch (e) {}
      }
      updateVoiceStatus(sessionLanguage === 'hi' ? '🎙 बोलिए...' : '🎙 Listening...');
      setTimeout(() => {
        if (!isCurrentQuestionTurn(questionToken) || responseSubmitting || voiceSubmitting) return;
        cueAndStartVoiceCapture(state, options, step, {
          delayMs: 0,
          enableInput: true,
          questionToken,
          skipBeep: true,
          setupDelayMs: 0,
        });
      }, 20);
    }, DUPLEX_BARGE_IN_POLL_MS);
  } catch (e) {
    console.warn('[Duplex] Barge-in monitor unavailable:', e);
    stopDuplexBargeInMonitor();
  }
}

function stopActiveVoiceCapture({ resetVoiceSubmitting = false } = {}) {
  invalidatePendingVoiceStart();
  invalidateSpokenFlow();
  stopDuplexBargeInMonitor();
  _duplexRecognitionFlush = null;
  _duplexSpeechDetectedToken = 0;
  if (recognition) {
    try { recognition.abort(); } catch (e) {}
    recognition = null;
  }
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    try { mediaRecorder.stop(); } catch (e) {}
  }
  if (micStream) {
    try { micStream.getTracks().forEach(t => t.stop()); } catch (e) {}
    micStream = null;
  }
  mediaRecorder = null;
  discardBrowserParallelCapture();
  voiceRecording = false;
  if (resetVoiceSubmitting) voiceSubmitting = false;
}

// Check whisper availability on load
async function checkWhisperAvailability() {
  try {
    const resp = await fetch(`${API}/api/voice/status`);
    if (resp.ok) {
      const data = await resp.json();
      whisperAvailable = data.whisper_available === true;
      console.log(`[Voice] faster-whisper available: ${whisperAvailable}`);
    }
  } catch (e) {
    whisperAvailable = false;
  }
}

// ── Beep (audio cue before listening) ──
const beepCtx = new (window.AudioContext || window.webkitAudioContext)();
function playBeep() {
  return new Promise(resolve => {
    const osc = beepCtx.createOscillator();
    const gain = beepCtx.createGain();
    osc.connect(gain);
    gain.connect(beepCtx.destination);
    osc.frequency.value = 880;
    osc.type = 'sine';
    gain.gain.setValueAtTime(0.3, beepCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, beepCtx.currentTime + 0.15);
    osc.start(beepCtx.currentTime);
    osc.stop(beepCtx.currentTime + 0.15);
    osc.onended = resolve;
    // Fallback in case onended doesn't fire
    setTimeout(resolve, 200);
  });
}

function cueAndStartVoiceCapture(state, options, step, { delayMs = null, enableInput = true, statusText = null, questionToken = _questionTurnToken, skipBeep = false, setupDelayMs = 100 } = {}) {
  if (!voiceEnabled) return;
  invalidatePendingVoiceStart();
  const generation = _voiceStartGeneration;
  const resolvedDelayMs = delayMs == null ? (sessionLanguage === 'hi' ? 50 : 200) : delayMs;

  const scheduleActualStart = () => {
    _voiceStartTimer = setTimeout(() => {
      _voiceStartTimer = null;
      if (generation !== _voiceStartGeneration) return;
      if (!isCurrentQuestionTurn(questionToken)) return;
      if (!voiceEnabled || voiceSubmitting || responseSubmitting) return;
      if ('speechSynthesis' in window && (speechSynthesis.speaking || speechSynthesis.pending)) {
        waitForSpeechAndStart();
        return;
      }
      if (enableInput) _inputEnabled = true;
      startVoiceCapture(state, options, step, { questionToken, setupDelayMs });
    }, resolvedDelayMs);
  };

  const waitForSpeechAndStart = () => {
    if (generation !== _voiceStartGeneration) return;
    if (!isCurrentQuestionTurn(questionToken)) return;
    if (!voiceEnabled || voiceSubmitting || responseSubmitting) return;
    if ('speechSynthesis' in window && (speechSynthesis.speaking || speechSynthesis.pending)) {
      _voiceStartTimer = setTimeout(waitForSpeechAndStart, 100);
      return;
    }
    if (statusText) updateVoiceStatus(statusText);
    if (skipBeep) {
      scheduleActualStart();
      return;
    }
    playBeep().then(() => {
      if (generation !== _voiceStartGeneration) return;
      if (!isCurrentQuestionTurn(questionToken)) return;
      scheduleActualStart();
    });
  };

  waitForSpeechAndStart();
}

// ── Phase definitions ──
const ALL_PHASES = [
  { state: 'B', name: 'Coarse Sphere RE', eye: 'RE' },
  { state: 'E', name: 'JCC Axis RE', eye: 'RE' },
  { state: 'F', name: 'JCC Power RE', eye: 'RE' },
  { state: 'G', name: 'Duochrome RE', eye: 'RE' },
  { state: 'C', name: 'Distance VA Confirm RE', eye: 'RE' },
  { state: 'D', name: 'Coarse Sphere LE', eye: 'LE' },
  { state: 'H', name: 'JCC Axis LE', eye: 'LE' },
  { state: 'I', name: 'JCC Power LE', eye: 'LE' },
  { state: 'J', name: 'Duochrome LE', eye: 'LE' },
  { state: 'L', name: 'Distance VA Confirm LE', eye: 'LE' },
  { state: 'K', name: 'Binocular Balance', eye: 'BIN' },
  { state: 'P', name: 'Near Add RE', eye: 'RE' },
  { state: 'Q', name: 'Near Add LE', eye: 'LE' },
  { state: 'R', name: 'Near Binocular', eye: 'BIN' },
  { state: 'S', name: 'Final Compare First Option Achieved Rx', eye: 'BIN' },
  { state: 'T', name: 'Final Compare Second Option PGP', eye: 'BIN' },
  { state: 'U', name: 'Final Compare Decision', eye: 'BIN' },
];

// ── Option styling map ──
const OPTION_STYLES = {
  'CLEAR': 'clear', 'READABLE': 'clear',
  'BLURRY': 'blurry', 'NOT_READABLE': 'blurry',
  'REPEAT': 'repeat',
  'SAME': 'same', 'CANT_TELL': 'same',
  'ONE': '', 'TWO': '', 'BETTER_1': '', 'BETTER_2': '',
  'RED': 'red', 'RED_CLEARER': 'red',
  'GREEN': 'green', 'GREEN_CLEARER': 'green',
  'EQUAL': 'same',
  'TOP': '', 'BOTTOM': '', 'TOP_CLEARER': '', 'BOTTOM_CLEARER': '',
  'TARGET_OK': 'clear', 'NOT_CLEAR': 'blurry',
};

const SEMANTIC_SLOT_LAYOUT = Object.freeze({
  1: { column: '1', row: '1' },
  2: { column: '2', row: '1' },
  3: { column: '1', row: '2' },
  4: { column: '2', row: '2' },
});

const STATE_OPTION_SLOT_PREFERENCE = Object.freeze({
  B: ['CLEAR', 'BLURRY', null, 'REPEAT'],
  C: ['CLEAR', 'BLURRY', null, 'REPEAT'],
  D: ['CLEAR', 'BLURRY', null, 'REPEAT'],
  E: ['ONE', 'TWO', 'SAME', 'REPEAT'],
  F: ['ONE', 'TWO', 'SAME', 'REPEAT'],
  G: ['GREEN', 'RED', 'SAME', 'REPEAT'],
  H: ['ONE', 'TWO', 'SAME', 'REPEAT'],
  I: ['ONE', 'TWO', 'SAME', 'REPEAT'],
  J: ['GREEN', 'RED', 'SAME', 'REPEAT'],
  K: ['BOTTOM', 'TOP', 'SAME', 'REPEAT'],
  L: ['CLEAR', 'BLURRY', null, 'REPEAT'],
  P: ['CLEAR', 'BLURRY', null, 'REPEAT'],
  Q: ['CLEAR', 'BLURRY', null, 'REPEAT'],
  R: ['CLEAR', 'BLURRY', null, 'REPEAT'],
  U: ['ONE', 'TWO', null, 'REPEAT'],
});

// Change these four values later if you want a different physical controller
// button to act as semantic button 1/2/3/4.
// These are raw Gamepad API button indices for the attached controller.
const GAMEPAD_SLOT_BINDINGS = Object.freeze({
  1: 11,
  2: 10,
  3: 4,
  4: 3,
});

function positionButtonInSemanticGrid(btn, slot) {
  const layout = SEMANTIC_SLOT_LAYOUT[slot];
  if (!layout) return;
  btn.dataset.slot = String(slot);
  btn.style.gridColumn = layout.column;
  btn.style.gridRow = layout.row;
}

function buildSemanticOptionSlots(state, options = []) {
  const preferred = STATE_OPTION_SLOT_PREFERENCE[state] || [];
  const slots = [null, null, null, null];
  const available = new Set(options || []);
  const used = new Set();

  preferred.forEach((option, index) => {
    if (option && available.has(option)) {
      slots[index] = option;
      used.add(option);
    }
  });

  for (const option of options || []) {
    if (used.has(option)) continue;
    const emptyIndex = slots.findIndex(slot => slot === null);
    if (emptyIndex === -1) break;
    slots[emptyIndex] = option;
    used.add(option);
  }

  return slots;
}

function renderSemanticOptionButtons({ state, options, localizedLabels, onSelect }) {
  const grid = document.getElementById('optionsGrid');
  grid.innerHTML = '';

  const localizedByInternal = new Map();
  (localizedLabels || []).forEach(label => localizedByInternal.set(label.internal, label));

  buildSemanticOptionSlots(state, options).forEach((internalOption, slotIndex) => {
    if (!internalOption) return;

    const slot = slotIndex + 1;
    const localized = localizedByInternal.get(internalOption);
    const displayText = localized?.localized || localized?.display || internalOption;
    const internalHint = localized && localized.internal !== displayText ? localized.internal : '';

    const btn = document.createElement('button');
    btn.className = 'option-btn';
    const style = OPTION_STYLES[internalOption] || '';
    if (style) btn.classList.add(style);
    btn.innerHTML = `${displayText}${internalHint ? '<span class="opt-internal">[' + internalHint + ']</span>' : ''}<span class="key-hint">${slot}</span>`;
    positionButtonInSemanticGrid(btn, slot);
    btn.onclick = () => onSelect(internalOption);
    grid.appendChild(btn);
  });
}

// ── State ──
let sessionId = null;
let currentState = null;
let logsUnlocked = false;
let sessionLanguage = 'en'; // Default, set by language selection
let activeLogTab = 'conversation';
let completedPhases = new Set();
let heartbeatInterval = null;
let _examTimerIntervalId = null;
let _examClockStartMs = null;
let _langSelectPendingData = null; // Stores first-question data during language selection
let _autoFlipTimer = null; // Timer for JCC auto-flip
let _flipState = null; // 'flip1', 'flip2', or null
let _jccPromptExposureCount = 0; // Counts JCC prompt exposures within the current session
let _inputEnabled = false; // Global gate: voice, gamepad, keyboard only after beep

// ── Gamepad ──
let gamepadEnabled = true;
let gamepadConnected = false;
let gamepadIndex = null;
let _gamepadPrevButtons = [false, false, false, false, false];
let _gamepadPollId = null;

// ── Initialization ──
document.addEventListener('DOMContentLoaded', () => {
  _jccPromptExposureCount = 0;
  // Check for session from URL or sessionStorage
  const params = new URLSearchParams(window.location.search);
  sessionId = params.get('session_id') || sessionStorage.getItem('session_id');

  if (sessionId) {
    restoreSession();
  } else {
    document.getElementById('questionText').textContent = 'No session active. Go to /intake to start.';
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', handleKeyboard);

  // Initialize voice mode select
  setVoiceMode('browser'); // Default to browser STT
  if (!SpeechRecognition) {
    console.warn('SpeechRecognition not available — browser mode disabled');
  }

  // Check whisper availability and update select options
  checkWhisperAvailability().then(() => {
    const sel = document.getElementById('voiceModeSelect');
    if (sel) {
      const whisperOpt = sel.querySelector('option[value="whisper"]');
      if (whisperOpt) {
        whisperOpt.textContent = whisperAvailable ? 'Mic: Whisper' : 'Mic: Whisper (model not found)';
      }
    }
  });

  // Check if logs were previously unlocked
  if (localStorage.getItem('logs_unlocked_until')) {
    const until = parseInt(localStorage.getItem('logs_unlocked_until'));
    if (Date.now() < until) {
      logsUnlocked = true;
    }
  }

  renderPhaseList();
  updateGamepadStatus();
});

// ── Session restore ──
async function restoreSession() {
  try {
    const resp = await fetch(`${API}/api/session/${sessionId}/status`);
    if (!resp.ok) throw new Error('Session not found');
    const data = await resp.json();

    // Prefer the locally saved selection, but fall back to the session-stored language
    const savedLang = sessionStorage.getItem('session_language');
    const statusLang = (data.language === 'hi' || data.language === 'en') ? data.language : '';
    const resolvedLang = savedLang || statusLang;
    if (resolvedLang) {
      sessionLanguage = resolvedLang;
      sessionStorage.setItem('session_language', resolvedLang);
      restoreCachedConversation();

      // Auto-resume: start immediately, enable TTS via a one-time user interaction listener
      await handleSessionUpdate(data);
      schedulePipRefreshSequence({ forceFresh: true, reason: 'Syncing phoropter...' });
      document.getElementById('endBtn').style.display = '';
      updatePhoropBtn();
      startHeartbeat();
      const convEl = document.getElementById('conversationLog');
      if ((!convEl || !convEl.innerHTML.trim()) && data.question && !data.is_terminal) {
        addToConversation('optometrist', data.question, null, `${data.state}`);
      }
    } else {
      // Show language selection first (like FSMv3.1_R2)
      showLanguageSelection(data);
    }
  } catch (e) {
    document.getElementById('questionText').textContent = `Session error: ${e.message}`;
  }
}

function showLanguageSelection(pendingData) {
  // Store pendingData at module level so voice callback can access it
  _langSelectPendingData = pendingData;
  syncExamClockFromSessionData(pendingData);

  document.getElementById('questionCard').style.display = '';
  document.getElementById('questionStep').textContent = 'LANGUAGE SELECTION';
  document.getElementById('questionState').textContent = 'Setup';
  document.getElementById('questionState').className = 'question-state bin';
  document.getElementById('questionText').textContent = 'Please select your preferred language / कृपया अपनी भाषा चुनें';

  const stimEl = document.getElementById('stimulusDescription');
  if (stimEl) stimEl.textContent = '';
  const chartEl = document.getElementById('letterChart');
  if (chartEl) chartEl.style.display = 'none';

  const grid = document.getElementById('optionsGrid');
  grid.innerHTML = '';

  const enBtn = document.createElement('button');
  enBtn.className = 'option-btn clear';
  enBtn.innerHTML = 'English<span class="key-hint">1</span>';
  positionButtonInSemanticGrid(enBtn, 1);
  enBtn.onclick = () => selectLanguage('en', _langSelectPendingData);
  grid.appendChild(enBtn);

  const hiBtn = document.createElement('button');
  hiBtn.className = 'option-btn';
  hiBtn.style.borderColor = '#f97316';
  hiBtn.style.background = '#fff7ed';
  hiBtn.style.color = '#9a3412';
  hiBtn.innerHTML = 'हिन्दी (Hindi)<span class="key-hint">2</span>';
  positionButtonInSemanticGrid(hiBtn, 2);
  hiBtn.onclick = () => selectLanguage('hi', _langSelectPendingData);
  grid.appendChild(hiBtn);

  // Speak first, then beep + listen only after TTS finishes (prevents mic from picking up TTS)
  speakQuestionWithStableFollowup('Please select your preferred language. English or Hindi?', 'en', () => {
    if (voiceEnabled && SpeechRecognition) {
      cueAndStartVoiceCapture('LANG_SELECT', ['ENGLISH', 'HINDI'], 0, {
        statusText: 'Say "English" or "Hindi"',
        enableInput: false,
      });
    }
  }, 'default');
}

function selectLanguage(lang, pendingData) {
  sessionLanguage = lang;
  sessionStorage.setItem('session_language', lang);
  // Update recognition language
  if (lang === 'hi') {
    // Will use hi-IN for Hindi recognition
  }
  addToConversation('system', `Language selected: ${lang === 'en' ? 'English' : 'Hindi'}`, null);

  // Now show the actual first question
  handleSessionUpdate(pendingData);
  schedulePipRefreshSequence({ forceFresh: true, reason: 'Syncing phoropter...' });
  document.getElementById('endBtn').style.display = '';
  updatePhoropBtn();
  startHeartbeat();
  if (pendingData.question && !pendingData.is_terminal) {
    addToConversation('optometrist', pendingData.question, null, `${pendingData.state}`);
  }
}

// ── Heartbeat ──
function getSessionBrainId() {
  return sessionStorage.getItem('acquired_brain_id')
    || sessionStorage.getItem('operator_name')
    || 'brain_01';
}

function startHeartbeat() {
  const phoropterId = sessionStorage.getItem('phoropter_id');
  if (!phoropterId) return;
  if (heartbeatInterval) clearInterval(heartbeatInterval);
  heartbeatInterval = setInterval(async () => {
    try {
      const brainId = getSessionBrainId();
      await fetch(`${API}/api/devices/${phoropterId}/heartbeat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ brain_id: brainId }),
      });
    } catch (e) { /* ignore */ }
  }, 15000);
}

function formatExamElapsed(ms) {
  if (ms < 0) ms = 0;
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function updateExamTimerDisplay() {
  const el = document.getElementById('examTimer');
  if (!el) return;
  if (_examClockStartMs == null) {
    el.textContent = '—';
    return;
  }
  el.textContent = `Exam ${formatExamElapsed(Date.now() - _examClockStartMs)}`;
}

function startExamTimer() {
  if (_examClockStartMs == null) return;
  if (_examTimerIntervalId) clearInterval(_examTimerIntervalId);
  updateExamTimerDisplay();
  _examTimerIntervalId = setInterval(updateExamTimerDisplay, 1000);
}

function stopExamTimer() {
  if (_examTimerIntervalId) {
    clearInterval(_examTimerIntervalId);
    _examTimerIntervalId = null;
  }
}

function syncExamClockFromSessionData(data) {
  if (!data || !data.exam_clock_start_iso) return;
  const ms = Date.parse(data.exam_clock_start_iso);
  if (Number.isNaN(ms)) return;
  _examClockStartMs = ms;
  startExamTimer();
}

function resetExamTimer() {
  stopExamTimer();
  _examClockStartMs = null;
  updateExamTimerDisplay();
}

// ── Core: handle session update from backend ──
async function handleSessionUpdate(data) {
  if (data.error) {
    document.getElementById('questionText').textContent = data.error;
    return;
  }

  stopActiveVoiceCapture({ resetVoiceSubmitting: true });
  if (_observeAdvanceTimer) {
    clearTimeout(_observeAdvanceTimer);
    _observeAdvanceTimer = null;
  }

  currentState = data;
  syncExamClockFromSessionData(data);

  // Update topbar
  updateTopbar(data);

  // Update Rx table
  updateRxTable(data.prescription);

  // Update fog badge
  const fogBadge = document.getElementById('fogBadge');
  fogBadge.classList.toggle('active', !!data.fog_active);

  // Update session info
  updateSessionInfo(data);

  // Terminal states
  if (data.is_terminal) {
    showTerminal(data);
    return;
  }

  // Show question (await so localized text is ready before TTS speaks)
  await showQuestion(data);

  // Auto-flip for JCC states
  handleAutoFlip(data);

  // Update phase progress
  if (data.state) {
    updatePhaseProgress(data.state);
  }
}

// ── Topbar ──
function updateTopbar(data) {
  const phaseEl = document.getElementById('topbarPhase');
  phaseEl.textContent = data.phase_name || data.state;

  // Eye-based coloring
  const eye = data.eye || '';
  phaseEl.className = 'topbar-phase';
  if (data.state === 'END') phaseEl.classList.add('end');
  else if (data.state === 'ESCALATE') phaseEl.classList.add('escalate');
  else if (eye === 'RE') phaseEl.classList.add('re');
  else if (eye === 'LE') phaseEl.classList.add('le');
  else phaseEl.classList.add('bin');

  document.getElementById('topbarInfo').textContent =
    `Step ${data.step || 0} · ${sessionId || ''}`;
}

// ── Question display (async — fetches localized text BEFORE TTS) ──
async function showQuestion(data) {
  const questionToken = nextQuestionTurnToken();
  document.getElementById('questionCard').style.display = '';
  document.getElementById('endCard').classList.remove('active');

  document.getElementById('questionStep').textContent = `STEP ${data.step}`;

  const stateEl = document.getElementById('questionState');
  stateEl.textContent = `${data.state} — ${data.phase_name}`;
  stateEl.className = 'question-state';
  const eye = data.eye || '';
  if (eye === 'RE') stateEl.classList.add('re');
  else if (eye === 'LE') stateEl.classList.add('le');
  else stateEl.classList.add('bin');

  // Display stimulus description
  const stimDesc = STIMULUS_DESCRIPTIONS[data.state] || '';
  const prefacePrompt = localizePrefacePrompt(data.preface_prompt || '');
  const stimEl = document.getElementById('stimulusDescription');
  const localizedStimDesc = localizeStimulusDescription(data.state, stimDesc);
  if (stimEl) stimEl.textContent = prefacePrompt ? `${prefacePrompt} ${localizedStimDesc}`.trim() : localizedStimDesc;

  // Display letter chart for coarse sphere states (B, D)
  const chartEl = document.getElementById('letterChart');
  if (chartEl) {
    const letters = getDisplayedDistanceChartLines(data.state, data.chart_param);
    const sizes = getDisplayedDistanceChartSizes(data.state, data.chart_param);
    if ((data.state === 'B' || data.state === 'C' || data.state === 'D' || data.state === 'L') && letters) {
      chartEl.style.display = '';
      chartEl.innerHTML = letters.map((line, i) => {
        const fontSize = Math.max(1.0, 2.4 - i * 0.4);
        const sizeLabel = sizes && sizes[i] ? `<span style="display:inline-block;min-width:70px;margin-right:14px;font-size:0.95rem;letter-spacing:normal;color:var(--ink-secondary);font-weight:700;">${sizes[i]}</span>` : '';
        return `<div class="chart-line" style="font-size:${fontSize}rem;letter-spacing:${fontSize * 0.4}rem">${sizeLabel}${line.join(' ')}</div>`;
      }).join('');
    } else {
      chartEl.style.display = 'none';
      chartEl.innerHTML = '';
    }
  }

  // Reset voice attempt counter for new question
  voiceAttemptCount = 0;
  // Disable all input until beep (voice, gamepad, keyboard)
  _inputEnabled = false;

  // 1. Fetch localized labels + question FIRST (before TTS speaks)
  let localizedQuestion = data.question;
  let localizedLabels = null;
  try {
    const resp = await fetch(`${API}/api/voice/labels`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: data.state, language: sessionLanguage, question: data.question, options: data.options || [] }),
    });
    if (resp.ok) {
      const locData = await resp.json();
      if (locData.question) localizedQuestion = locData.question;
      if (locData.labels && locData.labels.length > 0) localizedLabels = locData.labels;
    }
  } catch (e) {
    console.log('[Localization] Fetch failed, using raw English:', e);
  }

  // 2. Display the localized question text
  document.getElementById('questionText').textContent = localizedQuestion;

  // 3. Render option buttons (localized if available)
  renderSemanticOptionButtons({
    state: data.state,
    options: data.options || [],
    localizedLabels,
    onSelect: submitResponse,
  });

  // 4. Speak the LOCALIZED question, then beep, then listen
  //    For JCC states with auto_flip: ALL TTS is handled by handleAutoFlip (Flip 1 → Flip 2)
  //    For all other states: TTS → beep → listen immediately
  const canListen = voiceEnabled && (voiceMode === 'whisper' || SpeechRecognition);
  const isAutoFlip = data.auto_flip;
  const isObserveOnly = Number(data.auto_advance_seconds || 0) > 0;
  const motivationSpeech = getMotivationSpeech(data.preface_prompt || '', sessionLanguage);
  const spokenPrompt = [motivationSpeech, getStaticQuestionSpeech(localizedQuestion)].filter(Boolean).join(' ').trim();
  const startObserveAdvance = () => {
    if (!isCurrentQuestionTurn(questionToken)) return;
    const delayMs = Math.max(0, Number(data.auto_advance_seconds || 0)) * 1000;
    updateVoiceStatus(sessionLanguage === 'hi' ? 'ध्यान से देखिए...' : 'Observe carefully...');
    _observeAdvanceTimer = setTimeout(() => {
      if (!isCurrentQuestionTurn(questionToken)) return;
      _observeAdvanceTimer = null;
      submitResponse(
        data.auto_advance_response || 'AUTO_ADVANCE',
        null,
        { skipConversationLog: true, inputMethodOverride: 'System_Auto' },
      );
    }, delayMs);
  };

  if (isObserveOnly) {
    if (ttsEnabled && !isAutoFlip) {
      speakQuestionWithStableFollowup(spokenPrompt, null, startObserveAdvance, getQuestionTTSProfile(data));
    } else {
      startObserveAdvance();
    }
    return;
  }

  if (canEnableEarlyControllerInput(data)) {
    _inputEnabled = true;
  }

  if (ttsEnabled && !isAutoFlip) {
    if (isVoiceBargeInEligible(data.state, data.question || localizedQuestion, data)) {
      startDuplexBargeInMonitor({
        state: data.state,
        options: data.options || [],
        step: data.step,
        questionText: data.question || localizedQuestion,
        questionToken,
        data,
      });
    }
    speakQuestionWithStableFollowup(spokenPrompt, null, () => {
      if (!isCurrentQuestionTurn(questionToken)) return;
      stopDuplexBargeInMonitor();
      if (isDuplexSpeechDetected(questionToken)) return;
      if (canListen) {
        cueAndStartVoiceCapture(data.state, data.options || [], data.step, {
          enableInput: true,
          questionToken,
        });
      } else {
        playBeep().then(() => {
          if (!isCurrentQuestionTurn(questionToken)) return;
          _inputEnabled = true;
        });
      }
    }, getQuestionTTSProfile(data));
  } else if (!isAutoFlip) {
    stopDuplexBargeInMonitor();
    if (canListen) {
      cueAndStartVoiceCapture(data.state, data.options || [], data.step, {
        enableInput: true,
        questionToken,
      });
    } else {
      playBeep().then(() => {
        if (!isCurrentQuestionTurn(questionToken)) return;
        _inputEnabled = true;
      });
    }
  }
  // For JCC auto-flip states: _inputEnabled is set in handleAutoFlip after Flip 2 beep
}

// fetchLocalizedLabels removed — logic is now inline in showQuestion() to ensure
// localized text is available BEFORE TTS speaks

// ── Voice input pipeline (Browser SpeechRecognition) ──

let voiceMode = 'browser'; // 'off', 'browser', 'whisper'

function setVoiceMode(mode) {
  // Stop any active recording
  stopActiveVoiceCapture({ resetVoiceSubmitting: true });

  voiceMode = mode;
  voiceEnabled = mode !== 'off';

  updateVoiceModeSelect();
  updateVoiceStatus(voiceEnabled ? `Ready (${mode})` : '—');

  // If turning on and we have an active question, start listening
  if (voiceEnabled && currentState && !currentState.is_terminal && _flipState !== 'flip1') {
    cueAndStartVoiceCapture(currentState.state, currentState.options || [], currentState.step, { enableInput: true });
  }
}

function updateVoiceModeSelect() {
  const sel = document.getElementById('voiceModeSelect');
  if (sel) sel.value = voiceMode;
}

// Legacy compatibility
function toggleVoice() { setVoiceMode(voiceMode === 'off' ? 'browser' : 'off'); }
function updateVoiceButton() { updateVoiceModeSelect(); }

function startVoiceCapture(state, options, step, runtime = {}) {
  const { skipTtsCheck = false, questionToken = _questionTurnToken, duplexArmOnly = false, setupDelayMs = 100 } = runtime;
  if (!voiceEnabled) return;
  if (voiceSubmitting) return;
  if (responseSubmitting) return;
  if (_flipState === 'flip1') return;
  if (!isCurrentQuestionTurn(questionToken)) return;
  // Don't start listening while TTS is speaking
  if (!skipTtsCheck && isTtsActive()) {
    return;
  }

  // Route based on user's explicit voiceMode selection
  if (voiceMode === 'whisper') {
    startWhisperCapture(state, options, step, questionToken);
    return;
  }

  // Browser SpeechRecognition mode
  if (!SpeechRecognition) return;

  // Stop any previous recognition
  if (recognition) {
    try { recognition.abort(); } catch (e) {}
    recognition = null;
  }
  discardBrowserParallelCapture();

  voiceRecording = false;

  // Small delay to let previous recognition clean up
  setTimeout(() => {
    if (!isCurrentQuestionTurn(questionToken)) return;
    if (!voiceEnabled || voiceSubmitting || responseSubmitting) return;

    recognition = new SpeechRecognition();
    const currentQuestionText = getCurrentQuestionText();
    recognition.lang = getBrowserRecognitionLang(state, currentQuestionText);
    const isEnglishLineReadingCapture = sessionLanguage !== 'hi'
      && questionExpectsEnglishLetterReading(state, getCurrentMatcherQuestionText());
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 5;

    // Fix 5: SpeechGrammarList constrains recognition to expected vocabulary
    try {
      const SpeechGrammarList = window.SpeechGrammarList || window.webkitSpeechGrammarList;
      if (SpeechGrammarList && !recognition.lang.startsWith('hi')) {
        const grammar = '#JSGF V1.0; grammar r; public <r> = clear | blurry | repeat | first option | second option | both same | red side | green side | top line | bottom line | clearer | more blurry ;';
        const list = new SpeechGrammarList();
        list.addFromString(grammar, 1);
        recognition.grammars = list;
      }
    } catch (e) { /* grammar not supported — ok */ }

    const capturedState = state;
    const capturedOptions = options;
    const capturedStep = step;
    const capturedQuestionToken = questionToken;
    const capturedDuplexArmOnly = duplexArmOnly;
    let lastInterimTranscript = ''; // store interim for fallback
    let quickMatchTimer = null; // 1s timer to force-process short words
    let lineReadingFinalizeTimer = null;
    let accumulatedFinalTranscript = '';
    let deferredFinalTranscript = '';
    let deferredAlternatives = [];
    let alreadyProcessed = false; // prevent double-processing

    const isDeferredDuplexMode = () => capturedDuplexArmOnly && !isDuplexSpeechDetected(capturedQuestionToken);
    const processFinalTranscript = (trimmed, alts = []) => {
      alreadyProcessed = true;
      try { recognition.stop(); } catch (e) {}
      voiceRecording = false;
      console.log(`[Voice] Final: "${trimmed}" | Alternatives: ${JSON.stringify(alts)}`);
      voiceSubmitting = true;
      void (async () => {
        let audioB64 = '';
        if (capturedState !== 'LANG_SELECT') {
          audioB64 = await stopBrowserParallelCaptureAndEncodeBase64();
        } else {
          discardBrowserParallelCapture();
        }
        if (!isCurrentQuestionTurn(capturedQuestionToken)) {
          voiceSubmitting = false;
          return;
        }
        updateVoiceStatus(`Processing: "${trimmed}"`);
        await matchVoiceResponseWithAlternatives(
          trimmed, alts, capturedState, capturedOptions, capturedQuestionToken, audioB64,
        );
      })();
    };
    const flushDeferredDuplexRecognition = () => {
      if (!isCurrentQuestionTurn(capturedQuestionToken)) return;
      if (!isDuplexSpeechDetected(capturedQuestionToken)) return;
      if (alreadyProcessed) return;
      if (deferredFinalTranscript) {
        processFinalTranscript(deferredFinalTranscript, deferredAlternatives);
        return;
      }
      if (lastInterimTranscript) {
        forceProcessInterim();
      }
    };
    _duplexRecognitionFlush = flushDeferredDuplexRecognition;

    recognition.onstart = () => {
      if (!isCurrentQuestionTurn(capturedQuestionToken)) {
        try { recognition.abort(); } catch (e) {}
        return;
      }
      voiceRecording = true;
      if (capturedState !== 'LANG_SELECT') {
        void startBrowserParallelCaptureForSession();
      }
      if (!capturedDuplexArmOnly) {
        updateVoiceStatus('🎙 Listening...');
      }
      console.log(`[Voice] Listening for step ${capturedStep}, state ${capturedState}, options: ${capturedOptions.join(', ')}`);
    };

    // Quick-match check: does this interim text match a known response?
    function interimMatchesOption(text) {
      if (isEnglishLineReadingCapture) return false;
      const t = text.toLowerCase().trim();
      if (!t) return false;
      // Also try with digits stripped of punctuation (Chrome may add "." or spaces)
      const cleaned = t.replace(/[^a-z0-9]/g, '');
      return clientSideMatch(t, capturedOptions) !== null
          || clientSideMatch(cleaned, capturedOptions) !== null;
    }

    function clearLineReadingFinalizeTimer() {
      if (lineReadingFinalizeTimer) {
        clearTimeout(lineReadingFinalizeTimer);
        lineReadingFinalizeTimer = null;
      }
    }

    function buildLineReadingTranscript() {
      const parts = [];
      const finalText = accumulatedFinalTranscript.trim();
      const interimText = lastInterimTranscript.trim();
      if (finalText) parts.push(finalText);
      if (interimText) {
        const finalNorm = finalText.toLowerCase();
        const interimNorm = interimText.toLowerCase();
        if (!finalNorm || (!finalNorm.includes(interimNorm) && !interimNorm.includes(finalNorm))) {
          parts.push(interimText);
        }
      }
      return parts.join(' ').trim();
    }

    function processLineReadingBuffer() {
      if (!isCurrentQuestionTurn(capturedQuestionToken)) return;
      if (alreadyProcessed) return;
      const combinedTranscript = buildLineReadingTranscript();
      if (!combinedTranscript) return;
      alreadyProcessed = true;
      clearLineReadingFinalizeTimer();
      try { recognition.stop(); } catch (e) {}
      voiceRecording = false;
      console.log(`[Voice] Line-reading final: "${combinedTranscript}"`);
      voiceSubmitting = true;
      void (async () => {
        const audioB64 = capturedState === 'LANG_SELECT'
          ? (discardBrowserParallelCapture(), '')
          : await stopBrowserParallelCaptureAndEncodeBase64();
        if (!isCurrentQuestionTurn(capturedQuestionToken)) {
          voiceSubmitting = false;
          return;
        }
        updateVoiceStatus(`Processing: "${combinedTranscript}"`);
        const altSet = new Set();
        if (accumulatedFinalTranscript.trim()) altSet.add(accumulatedFinalTranscript.trim());
        if (lastInterimTranscript.trim()) altSet.add(lastInterimTranscript.trim());
        await matchVoiceResponseWithAlternatives(
          combinedTranscript,
          Array.from(altSet).filter(t => t && t !== combinedTranscript),
          capturedState,
          capturedOptions,
          capturedQuestionToken,
          audioB64,
        );
      })();
    }

    function scheduleLineReadingFinalize(delayMs = 1600) {
      if (!isEnglishLineReadingCapture) return;
      clearLineReadingFinalizeTimer();
      lineReadingFinalizeTimer = setTimeout(processLineReadingBuffer, delayMs);
    }

    function forceProcessInterim() {
      if (!isCurrentQuestionTurn(capturedQuestionToken)) return;
      if (isDeferredDuplexMode()) return;
      if (alreadyProcessed || !lastInterimTranscript) return;
      alreadyProcessed = true;
      try { recognition.stop(); } catch (e) {}
      voiceRecording = false;
      // Try both raw and cleaned versions
      const cleaned = lastInterimTranscript.replace(/[^a-z0-9]/gi, '').toLowerCase();
      const alts = cleaned !== lastInterimTranscript.toLowerCase() ? [cleaned] : [];
      console.log(`[Voice] Quick-match: forcing "${lastInterimTranscript}" (cleaned: "${cleaned}")`);
      voiceSubmitting = true;
      void (async () => {
        const audioB64 = capturedState === 'LANG_SELECT'
          ? (discardBrowserParallelCapture(), '')
          : await stopBrowserParallelCaptureAndEncodeBase64();
        if (!isCurrentQuestionTurn(capturedQuestionToken)) {
          voiceSubmitting = false;
          return;
        }
        updateVoiceStatus(`Processing: "${lastInterimTranscript}"`);
        await matchVoiceResponseWithAlternatives(
          lastInterimTranscript, alts, capturedState, capturedOptions, capturedQuestionToken, audioB64,
        );
      })();
    }

    recognition.onresult = (event) => {
      if (!isCurrentQuestionTurn(capturedQuestionToken)) return;
      if (alreadyProcessed) return;

      let finalTranscript = '';
      let finalAlternatives = [];
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalTranscript = event.results[i][0].transcript;
          for (let j = 0; j < event.results[i].length; j++) {
            finalAlternatives.push(event.results[i][j].transcript);
          }
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      // Store interim and check for quick match
      if (interimTranscript && !finalTranscript) {
        lastInterimTranscript = interimTranscript.trim();
        if (!capturedDuplexArmOnly || isDuplexSpeechDetected(capturedQuestionToken)) {
          updateVoiceStatus(`🎙 "${interimTranscript}"...`);
        }

        if (isEnglishLineReadingCapture) {
          scheduleLineReadingFinalize();
          return;
        }

        if (isDeferredDuplexMode()) {
          return;
        }

        // If interim matches a valid option, force-process it
        // Single digits/chars (1, 2): process IMMEDIATELY (Chrome keeps appending)
        // Short words (≤3 chars: "to", "red"): 300ms
        // Longer words: 800ms
        if (interimMatchesOption(lastInterimTranscript)) {
          if (!quickMatchTimer) {
            const len = lastInterimTranscript.length;
            const delay = len <= 1 ? 0 : len <= 3 ? 300 : 800;
            console.log(`[Voice] Interim "${lastInterimTranscript}" matches — processing in ${delay}ms`);
            if (delay === 0) {
              forceProcessInterim();
            } else {
              quickMatchTimer = setTimeout(forceProcessInterim, delay);
            }
          }
        } else {
          // New interim doesn't match — cancel any pending quick-match
          if (quickMatchTimer) { clearTimeout(quickMatchTimer); quickMatchTimer = null; }
        }
      }

      // Process final result (overrides any pending quick-match)
      if (finalTranscript) {
        if (quickMatchTimer) { clearTimeout(quickMatchTimer); quickMatchTimer = null; }
        const trimmed = finalTranscript.trim();
        const alts = finalAlternatives.map(a => a.trim()).filter(a => a);
        if (isEnglishLineReadingCapture) {
          if (trimmed) {
            accumulatedFinalTranscript = [accumulatedFinalTranscript, trimmed].filter(Boolean).join(' ').trim();
          }
          lastInterimTranscript = '';
          console.log(`[Voice] Line-reading chunk: "${trimmed}" | Combined: "${accumulatedFinalTranscript}" | Alternatives: ${JSON.stringify(alts)}`);
          updateVoiceStatus(`🎙 "${accumulatedFinalTranscript || trimmed}"...`);
          scheduleLineReadingFinalize();
          return;
        }
        if (isDeferredDuplexMode()) {
          deferredFinalTranscript = trimmed;
          deferredAlternatives = alts;
          return;
        }
        processFinalTranscript(trimmed, alts);
      }
    };

    recognition.onerror = (event) => {
      if (!isCurrentQuestionTurn(capturedQuestionToken)) return;
      if (quickMatchTimer) { clearTimeout(quickMatchTimer); quickMatchTimer = null; }
      clearLineReadingFinalizeTimer();
      voiceRecording = false;
      if (_duplexRecognitionFlush === flushDeferredDuplexRecognition) {
        _duplexRecognitionFlush = null;
      }
      console.log(`[Voice] Error: ${event.error}`);
      if (capturedDuplexArmOnly && !isDuplexSpeechDetected(capturedQuestionToken)) {
        return;
      }
      if (event.error === 'no-speech') {
        updateVoiceStatus('No speech detected. Repeating question...');
        void stopBrowserParallelCaptureAndEncodeBase64().then((b64) => {
          if (!b64 || !sessionId) return;
          postFailedVoiceAudioToSupabase({
            audio: b64,
            audio_format: 'webm',
            state: capturedState,
            step: capturedStep,
            transcript: '',
            reason: 'browser_no_speech',
            attempt_number: voiceAttemptCount,
            language: sessionLanguage,
          });
        });
        // Re-speak the question (like FSMv3.1_R2 retry=True reprompt)
        if (voiceEnabled && !voiceSubmitting) {
          const questionEl = document.getElementById('questionText');
          const questionText = questionEl ? questionEl.textContent : '';
          const retryPrompt = sessionLanguage === 'hi'
            ? `फिर से सुनिए। ${questionText}`
            : `Let me repeat. ${questionText}`;
          speakQuestionWithStableFollowup(retryPrompt, null, () => {
            if (!isCurrentQuestionTurn(capturedQuestionToken)) return;
            cueAndStartVoiceCapture(capturedState, capturedOptions, capturedStep, { questionToken: capturedQuestionToken });
          }, 'retry');
        }
      } else if (event.error === 'aborted') {
        discardBrowserParallelCapture();
      } else {
        updateVoiceStatus(`Mic error: ${event.error}`);
        discardBrowserParallelCapture();
      }
    };

    let gotFinalResult = false;
    let gotError = false;

    // Patch: track whether we got a result or error
    const origOnResult = recognition.onresult;
    recognition.onresult = (event) => {
      if (!isCurrentQuestionTurn(capturedQuestionToken)) return;
      // Check if any result is final
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) gotFinalResult = true;
      }
      origOnResult(event);
    };
    const origOnError = recognition.onerror;
    recognition.onerror = (event) => {
      if (!isCurrentQuestionTurn(capturedQuestionToken)) return;
      gotError = true;
      origOnError(event);
    };

    recognition.onend = () => {
      if (!isCurrentQuestionTurn(capturedQuestionToken)) return;
      if (quickMatchTimer) { clearTimeout(quickMatchTimer); quickMatchTimer = null; }
      clearLineReadingFinalizeTimer();
      voiceRecording = false;
      if (_duplexRecognitionFlush === flushDeferredDuplexRecognition) {
        _duplexRecognitionFlush = null;
      }

      if (isEnglishLineReadingCapture) {
        if (alreadyProcessed || gotError || voiceSubmitting) return;
        const combinedTranscript = buildLineReadingTranscript();
        if (combinedTranscript) {
          alreadyProcessed = true;
          console.log(`[Voice] Using line-reading buffer on end: "${combinedTranscript}"`);
          voiceSubmitting = true;
          void (async () => {
            const audioB64 = await stopBrowserParallelCaptureAndEncodeBase64();
            if (!isCurrentQuestionTurn(capturedQuestionToken)) {
              voiceSubmitting = false;
              return;
            }
            updateVoiceStatus(`Processing: "${combinedTranscript}"`);
            await matchVoiceResponseWithAlternatives(
              combinedTranscript, [], capturedState, capturedOptions, capturedQuestionToken, audioB64,
            );
          })();
          return;
        }
      }

      // If already processed (final, quick-match, or error), do nothing
      if (alreadyProcessed || gotFinalResult || gotError || voiceSubmitting) return;

      if (capturedDuplexArmOnly && !isDuplexSpeechDetected(capturedQuestionToken)) {
        return;
      }

      // Fix 2: Try interim transcript as fallback for single-syllable words
      if (lastInterimTranscript) {
        alreadyProcessed = true;
        console.log(`[Voice] Using interim as fallback: "${lastInterimTranscript}"`);
        voiceSubmitting = true;
        void (async () => {
          const audioB64 = await stopBrowserParallelCaptureAndEncodeBase64();
          if (!isCurrentQuestionTurn(capturedQuestionToken)) {
            voiceSubmitting = false;
            return;
          }
          updateVoiceStatus(`Processing interim: "${lastInterimTranscript}"`);
          await matchVoiceResponseWithAlternatives(
            lastInterimTranscript, [], capturedState, capturedOptions, capturedQuestionToken, audioB64,
          );
        })();
        return;
      }

      // No interim either — speech wasn't picked up
      console.log('[Voice] Recognition ended without result — repeating question');
      updateVoiceStatus('Could not hear clearly. Repeating...');
      void stopBrowserParallelCaptureAndEncodeBase64().then((b64) => {
        if (!b64 || !sessionId) return;
        postFailedVoiceAudioToSupabase({
          audio: b64,
          audio_format: 'webm',
          state: capturedState,
          step: capturedStep,
          transcript: '',
          reason: 'browser_recognition_end_empty',
          attempt_number: voiceAttemptCount,
          language: sessionLanguage,
        });
      });

      if (voiceEnabled && !voiceSubmitting) {
        const questionEl = document.getElementById('questionText');
        const questionText = questionEl ? questionEl.textContent : '';
        const retryPrompt = sessionLanguage === 'hi'
          ? `सुनाई नहीं दिया। ${questionText}`
          : `I could not hear you. ${questionText}`;
        speakQuestionWithStableFollowup(retryPrompt, null, () => {
          if (!isCurrentQuestionTurn(capturedQuestionToken)) return;
          cueAndStartVoiceCapture(capturedState, capturedOptions, capturedStep, { questionToken: capturedQuestionToken });
        }, 'retry');
      }
    };

    try {
      recognition.start();
    } catch (e) {
      console.warn('[Voice] Could not start recognition:', e);
      updateVoiceStatus('Mic start failed. Click Mic: ON to retry.');
    }
  }, setupDelayMs);
}

/** Best-effort upload of failed / non-matching voice audio to Supabase (Whisper or browser parallel capture). */
function postFailedVoiceAudioToSupabase(payload) {
  if (!sessionId || !payload || !payload.audio) return;
  fetch(`${API}/api/session/${sessionId}/voice-failed-audio`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).catch(() => {});
}

// ── Faster-whisper recording pipeline ──
async function startWhisperCapture(state, options, step, questionToken = _questionTurnToken) {
  if (!voiceEnabled || voiceSubmitting || responseSubmitting) return;
  if (!isCurrentQuestionTurn(questionToken)) return;
  stopActiveVoiceCapture();

  // Request mic access
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
  } catch (e) {
    console.warn('[Whisper] Mic access denied:', e);
    updateVoiceStatus('Mic access denied. Using buttons.');
    return;
  }

  audioChunks = [];
  voiceRecording = true;
  updateVoiceStatus('🎙 Waiting for speech...');

  // ── VAD parameters (matching FSMv3.1_R2 record_audio_until_silence) ──
  const VAD_SILENCE_THRESHOLD = 0.015;  // RMS level to consider as speech
  const isDistanceReadingState = ['B', 'C', 'D', 'L'].includes(state);
  const VAD_START_TIMEOUT = isDistanceReadingState ? 5.0 : 2.5; // seconds to wait for first speech
  const VAD_END_SILENCE = isDistanceReadingState ? 2.0 : 0.8;   // trailing silence to stop
  const VAD_MIN_SPEECH = 0.25;  // minimum speech duration before silence can end
  const VAD_MAX_DURATION = isDistanceReadingState ? 15 : 5;     // hard max

  // Use MediaRecorder to capture audio
  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus' : 'audio/webm';
  mediaRecorder = new MediaRecorder(micStream, { mimeType });

  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) audioChunks.push(event.data);
  };

  // ── Web Audio VAD (amplitude-based, matching FSMv3.1_R2) ──
  const vadCtx = new (window.AudioContext || window.webkitAudioContext)();
  const vadSource = vadCtx.createMediaStreamSource(micStream);
  const vadAnalyser = vadCtx.createAnalyser();
  vadAnalyser.fftSize = 2048;
  vadAnalyser.smoothingTimeConstant = 0.3;
  vadSource.connect(vadAnalyser);

  let speechDetected = false;
  let speechDuration = 0;
  let trailingSilence = 0;
  let vadStartTime = performance.now();
  let vadStopReason = 'max_duration';
  const vadBuffer = new Float32Array(vadAnalyser.fftSize);
  const FRAME_MS = 50; // check every 50ms

  const vadInterval = setInterval(() => {
    if (!voiceRecording || !mediaRecorder || mediaRecorder.state !== 'recording') {
      clearInterval(vadInterval);
      return;
    }

    vadAnalyser.getFloatTimeDomainData(vadBuffer);
    // Compute RMS (matching FSMv3.1_R2's chunk_rms)
    let sumSquares = 0;
    let peak = 0;
    for (let i = 0; i < vadBuffer.length; i++) {
      const v = Math.abs(vadBuffer[i]);
      sumSquares += vadBuffer[i] * vadBuffer[i];
      if (v > peak) peak = v;
    }
    const rms = Math.sqrt(sumSquares / vadBuffer.length);
    const level = Math.max(rms, peak * 0.5); // same as FSMv3.1_R2
    const isSpeech = level >= VAD_SILENCE_THRESHOLD;

    const elapsed = (performance.now() - vadStartTime) / 1000;
    const frameSec = FRAME_MS / 1000;

    if (speechDetected) {
      if (isSpeech) {
        speechDuration += frameSec;
        trailingSilence = 0;
      } else {
        trailingSilence += frameSec;
        if (speechDuration >= VAD_MIN_SPEECH && trailingSilence >= VAD_END_SILENCE) {
          vadStopReason = 'silence_after_speech';
          console.log(`[VAD] Silence after speech (${speechDuration.toFixed(1)}s speech, ${trailingSilence.toFixed(1)}s silence)`);
          clearInterval(vadInterval);
          mediaRecorder.stop();
          return;
        }
      }
      updateVoiceStatus(`🎙 Recording... ${speechDuration.toFixed(1)}s`);
    } else {
      if (isSpeech) {
        speechDetected = true;
        speechDuration = frameSec;
        trailingSilence = 0;
        updateVoiceStatus('🎙 Speech detected...');
        console.log(`[VAD] Speech detected at ${elapsed.toFixed(1)}s`);
      } else if (elapsed >= VAD_START_TIMEOUT) {
        vadStopReason = 'start_timeout';
        console.log(`[VAD] Start timeout (${VAD_START_TIMEOUT}s, no speech)`);
        clearInterval(vadInterval);
        mediaRecorder.stop();
        return;
      }
    }

    // Hard max duration
    if (elapsed >= VAD_MAX_DURATION) {
      vadStopReason = 'max_duration';
      console.log(`[VAD] Max duration reached (${VAD_MAX_DURATION}s)`);
      clearInterval(vadInterval);
      mediaRecorder.stop();
      return;
    }
  }, FRAME_MS);

  mediaRecorder.onstop = async () => {
    if (!isCurrentQuestionTurn(questionToken)) return;
    clearInterval(vadInterval);
    vadCtx.close().catch(() => {});
    voiceRecording = false;
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    console.log(`[VAD] Stop reason: ${vadStopReason}, speech: ${speechDetected}, duration: ${speechDuration.toFixed(1)}s`);
    if (audioChunks.length === 0) {
      updateVoiceStatus('No audio captured');
      repeatAndListen(state, options, step, questionToken);
      return;
    }

    updateVoiceStatus('Processing with whisper...');
    voiceSubmitting = true;

    // Send raw WebM blob directly — backend decodes it via ffmpeg/faster-whisper
    const blob = new Blob(audioChunks, { type: mimeType });
    let audioBase64 = '';
    try {
      const arrayBuf = await blob.arrayBuffer();
      const bytes = new Uint8Array(arrayBuf);
      let binary = '';
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      audioBase64 = btoa(binary);

      console.log(`[Whisper] Sending ${(bytes.length / 1024).toFixed(1)}KB of ${mimeType} audio`);

      // Send to backend for transcription + matching
      const resp = await fetch(`${API}/api/voice/transcribe-and-match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio: audioBase64,
          audio_format: 'webm',
          state: state,
          options: options,
          language: sessionLanguage,
          stt_language: getWhisperSTTLanguageHint(state, getCurrentMatcherQuestionText()),
          stimulus_letters: getCurrentStimulusLetters(),
          question: getCurrentMatcherQuestionText(),
        }),
      });

      const result = resp.ok ? await resp.json() : { error: `Server error ${resp.status}`, accepted: false };
      console.log(`[Whisper] Result:`, result);
      if (!isCurrentQuestionTurn(questionToken)) {
        voiceSubmitting = false;
        return;
      }

      // Case 1: Matched — submit response (like FSMv3.1_R2 accepted path)
      if (result.accepted && result.response_value) {
        updateVoiceStatus(`✓ "${result.transcript}" → ${result.response_value} (${(result.confidence * 100).toFixed(0)}%, ${result.backend})`);
        addVoiceToConversation(result.transcript, result.response_value, result.confidence);
        const voiceMeta = {
          transcript: result.transcript,
          match_confidence: result.confidence,
          match_method: result.method,
          canonical_label: result.canonical_label,
          input_mode: `voice_${result.backend}`,
          detected_language: result.detected_language,
          inferred_language: result.inferred_language,
          stt_seconds: result.stt_seconds,
          response_attempt_count: voiceAttemptCount + 1,
          stimulus_letters: getCurrentStimulusLetters(),
          session_language: sessionLanguage,
          audio_base64: audioBase64,
        };
        voiceSubmitting = false;
        await submitResponse(result.response_value, voiceMeta);
        return;
      }

      // Case 2: No speech / STT error — repeat question (don't count as failed attempt)
      const errMsg = result.error || '';
      if (errMsg.includes('No speech') || errMsg.includes('too short') || errMsg.includes('too small') || !result.transcript) {
        voiceSubmitting = false;
        updateVoiceStatus('No speech detected. Repeating...');
        postFailedVoiceAudioToSupabase({
          audio: audioBase64,
          audio_format: 'webm',
          state,
          step,
          transcript: result.transcript || '',
          reason: result.error || 'no_speech',
          attempt_number: voiceAttemptCount,
          language: sessionLanguage,
        });
        repeatAndListen(state, options, step, questionToken);
        return;
      }

      // Case 3: Speech heard but not matched — count as failed attempt
      voiceSubmitting = false;
      voiceAttemptCount++;
      const transcript = result.transcript || 'unknown';
      updateVoiceStatus(`✗ "${transcript}" (attempt ${voiceAttemptCount}/${VOICE_REPROMPT_LIMIT})`);
      addVoiceToConversation(transcript, null, 0, result.reason || result.error || 'no match');

      failedVoiceAttempts.push({
        timestamp: new Date().toISOString(),
        session_id: sessionId,
        step: step,
        state: state,
        transcript: transcript,
        available_options: options,
        attempt_number: voiceAttemptCount,
        language: sessionLanguage,
        backend: result.backend,
        stt_seconds: result.stt_seconds,
      });

      postFailedVoiceAudioToSupabase({
        audio: audioBase64,
        audio_format: 'webm',
        state,
        step,
        transcript,
        reason: result.reason || result.error || 'no_match',
        attempt_number: voiceAttemptCount,
        language: sessionLanguage,
      });

      if (voiceAttemptCount >= VOICE_REPROMPT_LIMIT) {
        updateVoiceStatus(`✗ Failed ${voiceAttemptCount}x. Use buttons below.`);
      } else {
        repeatAndListen(state, options, step, questionToken);
      }
      return;
    } catch (e) {
      console.error('[Whisper] Processing error:', e);
      voiceSubmitting = false;
      updateVoiceStatus(`Whisper error: ${e.message}`);
      if (audioBase64) {
        postFailedVoiceAudioToSupabase({
          audio: audioBase64,
          audio_format: 'webm',
          state,
          step,
          transcript: '',
          reason: `client_error:${e.message || e}`,
          attempt_number: voiceAttemptCount,
          language: sessionLanguage,
        });
      }
      repeatAndListen(state, options, step, questionToken);
    }
  };

  mediaRecorder.start();
  // VAD interval handles stopping — no fixed timeout needed
}

function repeatAndListen(state, options, step, questionToken = _questionTurnToken) {
  if (!voiceEnabled) return;
  if (!isCurrentQuestionTurn(questionToken)) return;
  const questionEl = document.getElementById('questionText');
  const questionText = questionEl ? questionEl.textContent : '';
  const retryPrompt = sessionLanguage === 'hi'
    ? `समझ नहीं आया। ${questionText}`
    : `I didn't catch that. ${questionText}`;
  speakQuestionWithStableFollowup(retryPrompt, null, () => {
    if (!isCurrentQuestionTurn(questionToken)) return;
    cueAndStartVoiceCapture(state, options, step, { questionToken });
  }, 'retry');
}

function getCurrentStimulusLetters() {
  // Get the displayed chart letters for the current coarse sphere state
  if (!currentState) return null;
  const state = currentState.state;
  if (!['B', 'C', 'D', 'L'].includes(state)) return null;
  const letters = getDisplayedDistanceChartLines(state, currentState.chart_param);
  if (!letters) return null;
  // Format as space-separated letters per line (matching FSMv3.1_R2 format)
  return letters.map(line => line.join(' ')).join('\n');
}

function getCurrentQuestionText() {
  const questionEl = document.getElementById('questionText');
  return questionEl ? (questionEl.textContent || '') : '';
}

function getCurrentMatcherQuestionText() {
  if (currentState && typeof currentState.question === 'string' && currentState.question.trim()) {
    return currentState.question;
  }
  return getCurrentQuestionText();
}

function questionExpectsEnglishLetterReading(state, questionText) {
  const question = (questionText || '').toLowerCase();
  if (state === 'C' || state === 'L') return true;
  if (state === 'B' || state === 'D') {
    if (
      question.includes('better now')
      || question.includes('better than before')
      || question.includes('yes or no')
      || question.includes('हाँ या नहीं')
      || question.includes('बेहतर')
    ) {
      return false;
    }
    return true;
  }
  return false;
}

function getBrowserRecognitionLang(state, questionText) {
  if (state === 'LANG_SELECT') {
    return 'en-US';
  }
  if (sessionLanguage === 'hi') {
    return isSafariBrowser() ? 'en-US' : 'hi-IN';
  }
  return 'en-US';
}

function getWhisperSTTLanguageHint(state, questionText) {
  if (state === 'LANG_SELECT') {
    return 'en';
  }
  return sessionLanguage === 'hi' ? 'hi' : 'en';
}

let _currentVoiceAlternatives = []; // Stored for inclusion in voiceMeta

async function matchVoiceResponseWithAlternatives(
  primary,
  alternatives,
  state,
  options,
  questionToken = _questionTurnToken,
  browserAudioBase64 = '',
) {
  if (!isCurrentQuestionTurn(questionToken)) return;
  _currentVoiceAlternatives = alternatives || [];
  // Special handling for language selection
  if (state === 'LANG_SELECT') {
    const t = primary.toLowerCase();
    if (t.includes('english') || t.includes('en') || t === '1' || t === 'one') {
      selectLanguage('en', _langSelectPendingData || {});
      _langSelectPendingData = null;
      voiceSubmitting = false;
      return;
    }
    if (t.includes('hindi') || t.includes('हिन्दी') || t.includes('हिंदी') || t === '2' || t === 'two') {
      selectLanguage('hi', _langSelectPendingData || {});
      _langSelectPendingData = null;
      voiceSubmitting = false;
      return;
    }
    voiceSubmitting = false;
    updateVoiceStatus('Say "English" or "Hindi"');
    if (voiceEnabled) {
      setTimeout(() => {
        if (!isCurrentQuestionTurn(questionToken)) return;
        cueAndStartVoiceCapture('LANG_SELECT', ['ENGLISH', 'HINDI'], 0, {
          statusText: 'Say "English" or "Hindi"',
          enableInput: false,
          questionToken,
        });
      }, 1000);
    }
    return;
  }

  // Try the primary transcript first, then each alternative
  const transcriptsToTry = [primary, ...alternatives.filter(a => a !== primary)];
  let lastFailure = null;

  for (const transcript of transcriptsToTry) {
    const result = await matchVoiceResponse(
      transcript, state, options, questionToken, browserAudioBase64,
    );
    if (!isCurrentQuestionTurn(questionToken)) return;
    if (result && result.matched) return; // Matched and submitted
    if (result && result.stale) return;
    if (result && !result.matched) lastFailure = result;
  }

  // None matched — track failed attempt
  if (!isCurrentQuestionTurn(questionToken)) return;
  voiceSubmitting = false;
  voiceAttemptCount++;

  if (browserAudioBase64 && sessionId) {
    postFailedVoiceAudioToSupabase({
      audio: browserAudioBase64,
      audio_format: 'webm',
      state,
      step: currentState ? currentState.step : 0,
      transcript: primary,
      reason: lastFailure ? (lastFailure.reason || 'no_match') : 'no_match',
      attempt_number: voiceAttemptCount,
      language: sessionLanguage,
    });
  }

  // Log structured failed attempt
  failedVoiceAttempts.push({
    timestamp: new Date().toISOString(),
    session_id: sessionId,
    step: currentState ? currentState.step : 0,
    state: state,
    phase_name: currentState ? currentState.phase_name : '',
    transcript: primary,
    alternatives: alternatives,
    available_options: options,
    attempt_number: voiceAttemptCount,
    language: sessionLanguage,
    match_confidence: lastFailure ? (lastFailure.confidence ?? '') : '',
    match_method: lastFailure ? (lastFailure.method || '') : '',
    canonical_label: lastFailure ? (lastFailure.canonical_label || '') : '',
    reason: lastFailure ? (lastFailure.reason || '') : '',
    stimulus_letters: getCurrentStimulusLetters(),
  });

  addVoiceToConversation(primary, null, 0, `no match (attempt ${voiceAttemptCount}/${VOICE_REPROMPT_LIMIT})`);

  if (voiceAttemptCount >= VOICE_REPROMPT_LIMIT) {
    // Reached reprompt limit — show keyboard fallback message
    updateVoiceStatus(`✗ Voice failed ${voiceAttemptCount}x. Use the buttons below.`);
    // Stop voice for this question
    if (recognition) { try { recognition.abort(); } catch (e) {} }
    discardBrowserParallelCapture();
    voiceRecording = false;
  } else {
    // Retry with voice — re-speak the question (like FSMv3.1_R2 reprompt with retry=True)
    updateVoiceStatus(`✗ "${primary}" (${voiceAttemptCount}/${VOICE_REPROMPT_LIMIT}) — repeating...`);
    if (voiceEnabled) {
      const questionEl = document.getElementById('questionText');
      const questionText = questionEl ? questionEl.textContent : '';
      const retryPrompt = sessionLanguage === 'hi'
        ? `समझ नहीं आया। ${questionText}`
        : `I didn't catch that. ${questionText}`;
      speakQuestionWithStableFollowup(retryPrompt, null, () => {
        if (!isCurrentQuestionTurn(questionToken)) return;
        cueAndStartVoiceCapture(state, options, 0, { questionToken });
      }, 'retry');
    }
  }
}

async function matchVoiceResponse(
  transcript,
  state,
  options,
  questionToken = _questionTurnToken,
  browserAudioBase64 = '',
) {
  if (!isCurrentQuestionTurn(questionToken)) {
    voiceSubmitting = false;
    return { matched: false, stale: true, reason: 'stale_question' };
  }
  // Stop recognition while we process
  if (recognition) {
    try { recognition.abort(); } catch (e) {}
  }

  voiceSubmitting = true;

  // Get stimulus letters for chart-reading matching (coarse sphere states)
  const stimulusLetters = getCurrentStimulusLetters();

  // Try server-side matching first (uses fsm/audio/response_matching.py)
  let matched = null;
  let voiceMeta = null;
  let lastFailure = {
    matched: false,
    confidence: 0,
    method: '',
    canonical_label: '',
    reason: 'no_match',
  };
  try {
    const resp = await fetch(`${API}/api/voice/match`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transcript: transcript,
        state: state,
        options: options,
        session_id: sessionId,
        stimulus_letters: stimulusLetters,
        language: sessionLanguage,
        question: getCurrentMatcherQuestionText(),
      }),
    });
    if (resp.ok) {
      const result = await resp.json();
      if (!isCurrentQuestionTurn(questionToken)) {
        voiceSubmitting = false;
        return { matched: false, stale: true, reason: 'stale_question' };
      }
      if (result.accepted && result.response_value) {
        matched = result.response_value;
        voiceMeta = {
          transcript: transcript,
          alternatives: _currentVoiceAlternatives,
          match_confidence: result.confidence || 0.8,
          match_method: result.method || 'server_side',
          canonical_label: result.canonical_label || matched,
          input_mode: 'voice_browser_speech_recognition',
          response_attempt_count: voiceAttemptCount + 1,
          stimulus_letters: getCurrentStimulusLetters(),
          session_language: sessionLanguage,
        };
        if (browserAudioBase64) {
          voiceMeta.audio_base64 = browserAudioBase64;
        }
        updateVoiceStatus(`✓ "${transcript}" → ${matched} (${(result.confidence * 100).toFixed(0)}%)`);
        addVoiceToConversation(transcript, matched, result.confidence);
      } else {
        console.log(`[Voice] Server no match: ${result.reason}`);
        lastFailure = {
          matched: false,
          confidence: result.confidence || 0,
          method: result.method || 'server_side',
          canonical_label: result.canonical_label || '',
          reason: result.reason || 'server_no_match',
        };
      }
    }
  } catch (e) {
    console.log('[Voice] Server match unavailable, using client-side fallback');
    lastFailure = {
      matched: false,
      confidence: 0,
      method: 'server_error',
      canonical_label: '',
      reason: 'server_match_unavailable',
    };
  }

  // Client-side fallback if server didn't match
  if (!matched) {
    matched = clientSideMatch(transcript, options);
    if (matched) {
      voiceMeta = {
        transcript: transcript,
        match_confidence: 0.8,
        match_method: 'client_side',
        canonical_label: matched,
        input_mode: 'voice_browser_speech_recognition',
        response_attempt_count: voiceAttemptCount + 1,
        stimulus_letters: getCurrentStimulusLetters(),
        session_language: sessionLanguage,
      };
      if (browserAudioBase64) {
        voiceMeta.audio_base64 = browserAudioBase64;
      }
      updateVoiceStatus(`✓ "${transcript}" → ${matched}`);
      addVoiceToConversation(transcript, matched, 0.8);
    }
  }

  if (matched) {
    if (!isCurrentQuestionTurn(questionToken)) {
      voiceSubmitting = false;
      return { matched: false, stale: true, reason: 'stale_question' };
    }
    voiceSubmitting = false;
    await submitResponse(matched, voiceMeta);
    return { matched: true }; // Successfully matched
  }

  voiceSubmitting = false;
  return lastFailure; // Not matched — caller will try alternatives
}

function clientSideMatch(transcript, options) {
  const raw = transcript.toLowerCase().trim();
  let t = raw
    .replace(/\bदोनोंसमान है\b/g, 'दोनों समान')
    .replace(/\bदोनोंसमान\b/g, 'दोनों समान')
    .replace(/\bdo\s+no\s+saman\b/g, 'dono saman')
    .replace(/\bdo\s+no\s+same\b/g, 'dono same')
    .replace(/\bdo\s+no\s+barabar\b/g, 'dono barabar')
    .replace(/\bdono\s*saman\b/g, 'dono saman')
    .replace(/\bdono\s*same\b/g, 'dono same')
    .replace(/\bdono\s*barabar\b/g, 'dono barabar')
    .replace(/\bdono\s*equal\b/g, 'dono equal')
    .replace(/\bsaif nahi hai\b/g, 'dhundla hai')
    .replace(/\bsafi nahi hai\b/g, 'dhundla hai')
    .replace(/\bsaaf nahi hai\b/g, 'dhundla hai')
    .replace(/\bsaf nahi hai\b/g, 'dhundla hai')
    .replace(/\bsaath nahi hai\b/g, 'dhundla hai');

  if (
    (t.includes('nahi') || t.includes('nahin') || t.includes('not')) &&
    (t.includes('saaf') || t.includes('saf') || t.includes('safi') || t.includes('saif') || t.includes('clear'))
  ) {
    t = 'dhundla hai';
  }

  if (/^(option|options|vikalp|विकल्प)(\s+(hai|hey|is|the))?$/.test(t)) {
    return null;
  }

  // Direct keyword map + common Chrome misrecognitions for single-syllable words
  const KEYWORD_MAP = {
    // Clarity + misrecognitions
    'clear': 'CLEAR', 'clearly': 'CLEAR', 'yes': 'CLEAR', 'readable': 'CLEAR',
    'clearer': 'CLEAR', 'got clearer': 'CLEAR', 'read line': 'CLEAR', 'read last line': 'CLEAR',
    'better now': 'CLEAR', 'better than before': 'CLEAR', 'got better': 'CLEAR', 'it got better': 'CLEAR',
    'here': 'CLEAR', 'beer': 'CLEAR', 'cheer': 'CLEAR', 'dear': 'CLEAR', 'near': 'CLEAR',
    'saaf': 'CLEAR', 'saaf hai': 'CLEAR', 'saf': 'CLEAR', 'saf hai': 'CLEAR', 'safi': 'CLEAR', 'safi hai': 'CLEAR', 'saif': 'CLEAR', 'saif hai': 'CLEAR', 'saath hai': 'CLEAR', 'haan': 'CLEAR', 'haan ji': 'CLEAR', 'हाँ': 'CLEAR', 'साफ': 'CLEAR', 'साफ है': 'CLEAR', 'क्लियर': 'CLEAR', 'क्लियर है': 'CLEAR', 'हां बेहतर है': 'CLEAR', 'हाँ बेहतर है': 'CLEAR', 'clear enough': 'CLEAR', 'all clear': 'CLEAR', 'allclear': 'CLEAR',
    'blurry': 'BLURRY', 'blurred': 'BLURRY', 'blur': 'BLURRY', 'not clear': 'BLURRY',
    'more blurry': 'BLURRY', 'got more blurry': 'BLURRY',
    'no': 'BLURRY', 'not better': 'BLURRY', 'did not get better': 'BLURRY', "didn't get better": 'BLURRY',
    'blare': 'BLURRY', 'blaring': 'BLURRY', 'glory': 'BLURRY',
    'dhundhla': 'BLURRY', 'dhundla': 'BLURRY', 'dhundhla hai': 'BLURRY', 'dhundla hai': 'BLURRY', 'dundle hai': 'BLURRY', 'jule hai': 'BLURRY', 'jule hain': 'BLURRY', 'dunai hai': 'BLURRY', 'nahi dikh raha': 'BLURRY', 'धुंधला': 'BLURRY', 'धुंधला है': 'BLURRY', 'धुंदला': 'BLURRY', 'धुन्दला': 'BLURRY', 'धुंधली': 'BLURRY', 'नहीं': 'BLURRY', 'अभी भी धुंधला': 'BLURRY',
    'thula': 'BLURRY', 'thula hai': 'BLURRY', 'hulahula': 'BLURRY', 'thula thula': 'BLURRY',
    'repeat': 'REPEAT', 'again': 'REPEAT', 'dobara': 'REPEAT', 'dubara': 'REPEAT', 'phir se': 'REPEAT', 'phirse': 'REPEAT', 'फिर से': 'REPEAT', 'फिरसे': 'REPEAT', 'दोबारा': 'REPEAT',
    'fir se': 'REPEAT', 'firse': 'REPEAT', 'phirse': 'REPEAT', 'फिर से कहिए': 'REPEAT', 'दोबारा बोलिए': 'REPEAT', 'say again': 'REPEAT', 'repeat please': 'REPEAT', 'please repeat': 'REPEAT',
    // Comparison + misrecognitions
    'one': 'ONE', 'first': 'ONE', 'first option': 'ONE', 'option 1': 'ONE', 'ek': 'ONE', 'pehla': 'ONE', 'pehla vikalp': 'ONE', '1': 'ONE', 'firstoption': 'ONE', 'optionone': 'ONE',
    'एक': 'ONE', 'पहला': 'ONE', 'पहला विकल्प': 'ONE',
    'won': 'ONE', 'want': 'ONE', 'on': 'ONE', 'wan': 'ONE', 'wand': 'ONE',
    'two': 'TWO', 'second': 'TWO', 'second option': 'TWO', 'option 2': 'TWO', 'do': 'TWO', 'doosra': 'TWO', 'dusra': 'TWO', 'dusra vikalp': 'TWO', 'doosra vikalp': 'TWO', '2': 'TWO', 'secondoption': 'TWO', 'optiontwo': 'TWO',
    'दो': 'TWO', 'दूसरा': 'TWO', 'दूसरा विकल्प': 'TWO',
    'to': 'TWO', 'too': 'TWO', 'tu': 'TWO', 'who': 'TWO', 'through': 'TWO',
    'same': 'SAME', 'both same': 'SAME', 'both are same': 'SAME', 'equal': 'SAME', 'barabar': 'SAME', 'dono same': 'SAME', 'dono saman': 'SAME', 'donosaman': 'SAME', 'dono barabar': 'SAME', 'dono equal': 'SAME', 'saman': 'SAME',
    'both': 'SAME', 'दोनों': 'SAME', 'दोनों समान': 'SAME', 'दोनोंसमान': 'SAME', 'दोनोंसमान है': 'SAME', 'बराबर': 'SAME', 'bothsame': 'SAME', 'botharesame': 'SAME', 'but same': 'SAME', 'bootsame': 'SAME',
    "can't tell": 'SAME', 'cant tell': 'SAME',
    'sane': 'SAME', 'saint': 'SAME', 'shame': 'SAME', 'came': 'SAME',
    // Duochrome + misrecognitions
    'red': 'RED', 'red one': 'RED', 'red side': 'RED', 'laal': 'RED', 'लाल': 'RED', 'लाल साइड': 'RED', 'redside': 'RED', 'readside': 'RED', 'lal side': 'RED',
    'read': 'RED', 'bread': 'RED', 'wed': 'RED', 'said': 'RED', 'bed': 'RED', 'dead': 'RED',
    'green': 'GREEN', 'green one': 'GREEN', 'green side': 'GREEN', 'hara': 'GREEN', 'har': 'GREEN', 'hari': 'GREEN', 'हरा': 'GREEN', 'हरा साइड': 'GREEN', 'greenside': 'GREEN', 'hara side': 'GREEN',
    'queen': 'GREEN', 'cream': 'GREEN', 'gene': 'GREEN', 'lean': 'GREEN', 'mean': 'GREEN',
    // Binocular + misrecognitions
    'top': 'TOP', 'top one': 'TOP', 'top line': 'TOP', 'upar': 'TOP', 'ऊपर': 'TOP', 'ऊपर की लाइन': 'TOP', 'topline': 'TOP', 'upper line': 'TOP',
    'talk': 'TOP', 'tall': 'TOP', 'stop': 'TOP',
    'bottom': 'BOTTOM', 'bottom one': 'BOTTOM', 'bottom line': 'BOTTOM', 'neeche': 'BOTTOM', 'नीचे': 'BOTTOM', 'नीचे की लाइन': 'BOTTOM', 'bottomline': 'BOTTOM', 'neeche line': 'BOTTOM',
    'button': 'BOTTOM', 'bought him': 'BOTTOM',
    // Near
    'target ok': 'TARGET_OK', 'ok': 'TARGET_OK', 'fine': 'TARGET_OK',
    'not clear': 'NOT_CLEAR',
  };

  // Try exact match first
  if (KEYWORD_MAP[t] && options.includes(KEYWORD_MAP[t])) {
    return KEYWORD_MAP[t];
  }

  const letterMatch = clientSideLetterReadingMatch(transcript, options);
  if (letterMatch) {
    return letterMatch;
  }

  // Try partial match
  const sortedKeywordEntries = Object.entries(KEYWORD_MAP).sort((a, b) => b[0].length - a[0].length);
  for (const [keyword, value] of sortedKeywordEntries) {
    if (t.includes(keyword) && options.includes(value)) {
      return value;
    }
  }

  // Try matching option names directly
  for (const opt of options) {
    if (t.includes(opt.toLowerCase())) {
      return opt;
    }
  }

  return null;
}

const CHART_LETTER_ALIASES_JS = Object.freeze({
  A: new Set(['a', 'ay', 'eh', 'ए']),
  B: new Set(['b', 'bee', 'बी']),
  C: new Set(['c', 'see', 'सी']),
  D: new Set(['d', 'dee', 'डी']),
  E: new Set(['e', 'ee', 'ई']),
  F: new Set(['f', 'eff', 'एफ']),
  G: new Set(['g', 'gee', 'जी']),
  H: new Set(['h', 'aitch', 'etch', 'एच']),
  L: new Set(['l', 'ell', 'el', 'एल']),
  N: new Set(['n', 'en', 'एन']),
  O: new Set(['o', 'oh', 'ओ']),
  P: new Set(['p', 'pee', 'पी']),
  S: new Set(['s', 'ess', 'एस']),
  T: new Set(['t', 'tee', 'टी']),
  U: new Set(['u', 'you', 'यू']),
  V: new Set(['v', 'vee', 'वी']),
  Z: new Set(['z', 'zee', 'zed', 'जेड', 'ज़ेड']),
});

function questionExpectsLineReading(state, questionText) {
  const question = (questionText || '').toLowerCase();
  if (state === 'C' || state === 'L') return true;
  if (state === 'B' || state === 'D') {
    if (
      question.includes('better now')
      || question.includes('better than before')
      || question.includes('yes or no')
      || question.includes('हाँ या नहीं')
      || question.includes('बेहतर')
      || question.includes('still blurry')
      || question.includes('अभी भी धुंधला')
    ) {
      return false;
    }
    return true;
  }
  return false;
}

function extractChartLetterTokensJS(text) {
  const normalized = (text || '').replace(/[^\p{L}\p{N}\s']/gu, ' ').trim();
  const tokens = normalized.split(/\s+/).filter(Boolean);
  const extracted = [];
  const allowedLetters = new Set(Object.keys(CHART_LETTER_ALIASES_JS));

  for (const token of tokens) {
    const upper = token.toUpperCase();
    if (allowedLetters.has(upper)) {
      extracted.push(upper);
      continue;
    }
    if (/^[A-Z]+$/.test(upper) && upper.length > 1) {
      upper.split('').forEach(ch => {
        if (allowedLetters.has(ch)) extracted.push(ch);
      });
      continue;
    }
    for (const [letter, aliases] of Object.entries(CHART_LETTER_ALIASES_JS)) {
      if (aliases.has(token.toLowerCase()) || aliases.has(token)) {
        extracted.push(letter);
        break;
      }
    }
  }
  return extracted;
}

function lcsLengthJS(a, b) {
  if (!a.length || !b.length) return 0;
  const dp = Array.from({ length: a.length + 1 }, () => Array(b.length + 1).fill(0));
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }
  return dp[a.length][b.length];
}

function clientSideLetterReadingMatch(transcript, options) {
  if (!currentState) return null;
  if (sessionLanguage === 'hi') return null;
  const state = currentState.state;
  if (!['B', 'C', 'D', 'L'].includes(state)) return null;
  if (!questionExpectsLineReading(state, getCurrentMatcherQuestionText())) return null;
  if (!options.includes('CLEAR')) return null;

  const stimulus = getCurrentStimulusLetters();
  if (!stimulus) return null;

  const targetLetters = stimulus
    .split(/\s+/)
    .map(token => token.trim().toUpperCase())
    .filter(Boolean)
    .flatMap(token => token.split(''));
  const spokenLetters = extractChartLetterTokensJS(transcript);

  if (targetLetters.length < 2 || spokenLetters.length < 2) return null;

  const accuracy = lcsLengthJS(spokenLetters, targetLetters) / targetLetters.length;
  if (accuracy >= 0.8 && options.includes('CLEAR')) return 'CLEAR';
  if (accuracy >= 0.3 && options.includes('BLURRY')) return 'BLURRY';
  if (options.includes('REPEAT')) return 'REPEAT';
  return null;
}

function updateVoiceStatus(status) {
  const el = document.getElementById('voiceStatus');
  if (el) {
    el.textContent = status;
    // Color coding
    if (status.startsWith('🎙')) el.style.color = '#22c55e';
    else if (status.startsWith('✓')) el.style.color = '#2563eb';
    else if (status.startsWith('✗')) el.style.color = '#dc2626';
    else el.style.color = '';
  }
  console.log(`[Voice] ${status}`);
}

function fmtAxisDisplay(axis) {
  if (axis == null) return '—';
  const rounded = Math.round(Number(axis) / 5) * 5;
  const wrapped = ((rounded % 180) + 180) % 180;
  return String(wrapped === 0 ? 180 : wrapped);
}

// ── Terminal display ──
function showTerminal(data) {
  invalidateQuestionTurn();
  stopActiveVoiceCapture({ resetVoiceSubmitting: true });
  if (_autoFlipTimer) {
    clearTimeout(_autoFlipTimer);
    _autoFlipTimer = null;
  }
  _flipState = null;
  updateFlipIndicator(null);
  stopExamTimer();
  updateExamTimerDisplay();

  document.getElementById('questionCard').style.display = 'none';
  const card = document.getElementById('endCard');
  card.classList.add('active');

  if (data.state === 'END') {
    const rx = data.prescription || {};
    const achievedRx = data.achieved_prescription || {};
    const currentRx = data.pgp_rx || data.current_rx || {};
    const finalCompare = data.final_compare || {};
    const va = data.distance_va || {};
    const r = rx.right || {};
    const l = rx.left || {};
    const rVa = (va.right || {}).line || '—';
    const lVa = (va.left || {}).line || '—';
    const fmt = (v) => v != null ? (v >= 0 ? '+' : '') + parseFloat(v).toFixed(2) : '—';
    const fmtRx = (eye) => `${fmt(eye.sph)} / ${fmt(eye.cyl)} x ${fmtAxisDisplay(eye.axis ?? 180)}${eye.add ? ` ADD ${fmt(eye.add)}` : ''}`;
    const acceptedAchieved = finalCompare.accepted_achieved_over_current_rx === 'Yes';
    const compareRan = !!finalCompare.enabled;
    const compareMessage = compareRan
      ? acceptedAchieved
        ? 'Patient accepted the achieved Rx over the PGP.'
        : 'Patient accepted the PGP over the achieved Rx.'
      : 'Here is your final prescription:';
    const secondaryBlock = compareRan
      ? acceptedAchieved
        ? `<div style="margin-top:12px;font-size:0.84rem;color:var(--ink-secondary)">PGP kept for comparison</div>` +
          `<div style="display:flex;gap:24px;justify-content:center;margin-top:8px;">` +
            `<div style="text-align:center"><div style="font-size:0.72rem;color:var(--re-color);font-weight:700;margin-bottom:4px;">PGP RE</div><div style="font:500 0.98rem var(--font-mono)">${fmtRx((currentRx || {}).right || {})}</div></div>` +
            `<div style="text-align:center"><div style="font-size:0.72rem;color:var(--le-color);font-weight:700;margin-bottom:4px;">PGP LE</div><div style="font:500 0.98rem var(--font-mono)">${fmtRx((currentRx || {}).left || {})}</div></div>` +
          `</div>`
        : `<div style="margin-top:12px;font-size:0.84rem;color:var(--ink-secondary)">Achieved Rx during the eye test</div>` +
          `<div style="display:flex;gap:24px;justify-content:center;margin-top:8px;">` +
            `<div style="text-align:center"><div style="font-size:0.72rem;color:var(--re-color);font-weight:700;margin-bottom:4px;">ACHIEVED RE</div><div style="font:500 0.98rem var(--font-mono)">${fmtRx((achievedRx || {}).right || {})}</div></div>` +
            `<div style="text-align:center"><div style="font-size:0.72rem;color:var(--le-color);font-weight:700;margin-bottom:4px;">ACHIEVED LE</div><div style="font:500 0.98rem var(--font-mono)">${fmtRx((achievedRx || {}).left || {})}</div></div>` +
          `</div>`
      : '';

    document.getElementById('terminalIcon').textContent = '✅';
    document.getElementById('terminalTitle').textContent = 'Congratulations! Your eye test is complete.';
    document.getElementById('terminalSubtitle').innerHTML =
      `<div style="margin-bottom:12px">${compareMessage}</div>` +
      `<div style="display:flex;gap:24px;justify-content:center;margin-bottom:16px;">` +
        `<div style="text-align:center"><div style="font-size:0.75rem;color:var(--re-color);font-weight:700;margin-bottom:4px;">RIGHT EYE (RE)</div><div style="font:600 1.1rem var(--font-mono)">${fmtRx(r)}</div><div style="margin-top:6px;font-size:0.84rem;color:var(--ink-secondary)">Distance VA: ${rVa}</div></div>` +
        `<div style="text-align:center"><div style="font-size:0.75rem;color:var(--le-color);font-weight:700;margin-bottom:4px;">LEFT EYE (LE)</div><div style="font:600 1.1rem var(--font-mono)">${fmtRx(l)}</div><div style="margin-top:6px;font-size:0.84rem;color:var(--ink-secondary)">Distance VA: ${lVa}</div></div>` +
      `</div>` +
      secondaryBlock +
      `<div style="height:12px"></div>` +
      `<div style="font-size:0.85rem;color:var(--ink-secondary)">Please review and sign off below.</div>`;

    const terminalSpeech = getStaticTerminalSpeech({
      isEscalate: false,
      compareRan,
      acceptedAchieved,
    });
    speakQuestion(terminalSpeech, null, null, 'terminal');
  } else {
    document.getElementById('terminalIcon').textContent = '⚠️';
    document.getElementById('terminalTitle').textContent = 'Escalation Required';
    document.getElementById('terminalSubtitle').textContent = 'This test requires optometrist review. Please consult with a qualified optometrist.';
    speakQuestion(
      getStaticTerminalSpeech({ isEscalate: true, compareRan: false, acceptedAchieved: false }),
      null,
      null,
      'terminal',
    );
  }
}

// ── Rx Table ──
function updateRxTable(rx) {
  if (!rx) return;
  const r = rx.right || {};
  const l = rx.left || {};
  document.getElementById('rxReSph').textContent = fmtD(r.sph);
  document.getElementById('rxReCyl').textContent = fmtD(r.cyl);
  document.getElementById('rxReAxis').textContent = fmtAxisDisplay(r.axis);
  document.getElementById('rxReAdd').textContent = r.add ? fmtD(r.add) : '—';
  document.getElementById('rxLeSph').textContent = fmtD(l.sph);
  document.getElementById('rxLeCyl').textContent = fmtD(l.cyl);
  document.getElementById('rxLeAxis').textContent = fmtAxisDisplay(l.axis);
  document.getElementById('rxLeAdd').textContent = l.add ? fmtD(l.add) : '—';
}

function fmtD(val) {
  if (val == null) return '—';
  const n = parseFloat(val);
  return isNaN(n) ? '—' : (n >= 0 ? '+' : '') + n.toFixed(2);
}

// ── Session Info ──
function updateSessionInfo(data) {
  const el = document.getElementById('sessionInfo');
  const op = sessionStorage.getItem('operator_name') || '';
  const pid = sessionStorage.getItem('phoropter_id') || '';
  el.innerHTML = `
    <div>Session: <strong>${sessionId}</strong></div>
    <div>Phoropter: <strong>${pid}</strong></div>
    <div>Operator: <strong>${op}</strong></div>
    <div>Step: <strong>${data.step || 0}</strong></div>
    <div>Chart: <strong>${data.chart_param || '—'}</strong></div>
    <div>Response Type: <strong>${data.response_type || '—'}</strong></div>
  `;
}

// ── Phase Progress ──
function renderPhaseList() {
  const list = document.getElementById('phaseList');
  list.innerHTML = '';
  ALL_PHASES.forEach(p => {
    const item = document.createElement('div');
    item.className = 'phase-item pending';
    item.id = `phase-${p.state}`;
    item.innerHTML = `<span class="phase-dot pending"></span>${p.state}: ${p.name}`;
    list.appendChild(item);
  });
}

function updatePhaseProgress(activeState) {
  ALL_PHASES.forEach(p => {
    const item = document.getElementById(`phase-${p.state}`);
    const dot = item.querySelector('.phase-dot');
    if (completedPhases.has(p.state)) {
      item.className = 'phase-item completed';
      dot.className = 'phase-dot completed';
    } else if (p.state === activeState) {
      item.className = 'phase-item active';
      dot.className = 'phase-dot active';
    } else {
      item.className = 'phase-item pending';
      dot.className = 'phase-dot pending';
    }
  });
}

// ── Input method tracking ──
let _lastInputMethod = 'Button'; // Default; overridden by keyboard/gamepad/voice paths

function prescriptionsEqual(a, b) {
  return JSON.stringify(a || null) === JSON.stringify(b || null);
}

// ── Submit response ──
async function submitResponse(responseValue, voiceMeta, overrides = {}) {
  if (!sessionId) return;
  if (responseSubmitting) return;
  invalidateQuestionTurn();
  responseSubmitting = true;
  _inputEnabled = false;
  if (_observeAdvanceTimer) {
    clearTimeout(_observeAdvanceTimer);
    _observeAdvanceTimer = null;
  }
  stopActiveVoiceCapture({ resetVoiceSubmitting: true });
  if ('speechSynthesis' in window && (speechSynthesis.speaking || speechSynthesis.pending)) {
    _ttsSessionId += 1;
    try { speechSynthesis.cancel(); } catch (e) {}
  }

  const prevStateId = currentState ? currentState.state : '';
  const prevPrescription = currentState ? currentState.prescription : null;
	
  // Track completed phase before it potentially changes
  if (currentState && currentState.state) {
    const prevState = currentState.state;
    // We'll mark it completed if the state changes after response
    setTimeout(() => {
      if (currentState && currentState.state !== prevState) {
        completedPhases.add(prevState);
      }
    }, 100);
  }

  // Log the response to conversation (skip if voice already logged it via addVoiceToConversation)
  if (!voiceMeta && !overrides.skipConversationLog) {
    addToConversation('patient', responseValue, responseValue,
      currentState ? `${currentState.state}:${currentState.step}` : '');
  }

  try {
    const resp = await fetch(`${API}/api/session/${sessionId}/respond`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        response: responseValue,
        voice_meta: voiceMeta || null,
        input_method: overrides.inputMethodOverride || (voiceMeta ? (voiceMeta.input_mode === 'voice_browser_speech_recognition' ? 'Voice_Browser' : 'Voice_Whisper') : _lastInputMethod),
        language: sessionLanguage,
      }),
    });
    const data = await resp.json();

    // Track phase completion
    if (currentState && data.state !== currentState.state) {
      completedPhases.add(currentState.state);
    }

    // Log the question to conversation
    if (data.question && !data.is_terminal) {
      addToConversation('optometrist', data.question, null, `${data.state}`);
    }

    handleSessionUpdate(data);

    // Cache state for refresh recovery
    cacheSessionState();

    // Auto-refresh logs if the panel is open
    if (document.getElementById('logsDrawer')?.classList.contains('open') && logsUnlocked) {
      loadLogs();
    }

    // Auto-update screenshot PIP
    autoUpdatePip({
      powerChanged: !prescriptionsEqual(prevPrescription, data.prescription),
      phaseChanged: !!prevStateId && data.state !== prevStateId,
    });
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    responseSubmitting = false;
    _lastInputMethod = 'Button'; // Reset to default after each submission
  }
}

// ── Keyboard shortcuts ──
function handleKeyboard(e) {
  if (!_inputEnabled || (!isDuplexTurnMode() && isTtsActive())) return;
  if (e.key >= '1' && e.key <= '4') {
    const slot = parseInt(e.key, 10);
    const btn = document.querySelector(`#optionsGrid .option-btn[data-slot="${slot}"]`);
    if (btn) {
      e.preventDefault();
      _lastInputMethod = 'Keyboard';
      btn.click();
    }
  }
}

// ── Gamepad input (Xbox controller via Chrome Gamepad API) ──
// Standard indices: A=0, B=1, X=2, Y=3.
// Physical buttons are translated into semantic buttons 1-4 through GAMEPAD_SLOT_BINDINGS.

window.addEventListener('gamepadconnected', (e) => {
  console.log(`[Gamepad] Connected: ${e.gamepad.id}`);
  gamepadIndex = e.gamepad.index;
  gamepadConnected = true;
  _gamepadPrevButtons = [false, false, false, false, false];
  updateGamepadStatus();
  if (gamepadEnabled && !_gamepadPollId) startGamepadPoll();
});

window.addEventListener('gamepaddisconnected', (e) => {
  console.log(`[Gamepad] Disconnected: ${e.gamepad.id}`);
  gamepadConnected = false;
  gamepadIndex = null;
  updateGamepadStatus();
  if (_gamepadPollId) { cancelAnimationFrame(_gamepadPollId); _gamepadPollId = null; }
});

function startGamepadPoll() {
  function poll() {
    if (!gamepadEnabled || !gamepadConnected) {
      _gamepadPollId = null;
      return;
    }
    const gp = navigator.getGamepads()[gamepadIndex];
    if (gp) {
      Object.entries(GAMEPAD_SLOT_BINDINGS).forEach(([slotKey, btnIdx]) => {
        const slot = Number(slotKey);
        const pressed = gp.buttons[btnIdx]?.pressed || false;
        if (pressed && !_gamepadPrevButtons[slot]) {
          handleGamepadOptionSlot(slot);
        }
        _gamepadPrevButtons[slot] = pressed;
      });
    }
    _gamepadPollId = requestAnimationFrame(poll);
  }
  _gamepadPollId = requestAnimationFrame(poll);
}

function handleGamepadOptionSlot(slot) {
  if (!_inputEnabled || _flipState === 'flip1' || (!isDuplexTurnMode() && isTtsActive())) return;
  const btn = document.querySelector(`#optionsGrid .option-btn[data-slot="${slot}"]`);
  if (btn && !btn.disabled) {
    console.log(`[Gamepad] Semantic button ${slot} → "${btn.textContent.trim()}"`);
    _lastInputMethod = 'Gamepad';
    btn.click();
  }
}

function toggleGamepad() {
  gamepadEnabled = !gamepadEnabled;
  const btn = document.getElementById('gamepadBtn');
  if (btn) {
    btn.textContent = `Gamepad: ${gamepadEnabled ? 'ON' : 'OFF'}`;
    btn.style.background = gamepadEnabled ? 'rgba(34,197,94,0.3)' : '';
  }
  updateGamepadStatus();
  if (gamepadEnabled && gamepadConnected && !_gamepadPollId) startGamepadPoll();
  if (!gamepadEnabled && _gamepadPollId) {
    cancelAnimationFrame(_gamepadPollId);
    _gamepadPollId = null;
  }
}

function updateGamepadStatus() {
  const el = document.getElementById('gamepadStatus');
  if (!el) return;
  if (!gamepadEnabled) {
    el.textContent = 'GP: OFF';
    el.style.color = 'rgba(255,255,255,0.4)';
  } else if (gamepadConnected) {
    el.textContent = 'GP: Connected';
    el.style.color = '#4ade80';
  } else {
    el.textContent = 'GP: No pad';
    el.style.color = 'rgba(255,255,255,0.5)';
  }
}

// ── End session ──
async function endSession() {
  if (!confirm('End the current session?')) return;
  showTerminal({ state: 'END' });
  document.getElementById('questionCard').style.display = 'none';
}

async function signOff() {
  if (!sessionId) return;
  try {
    // Flush failed voice attempts to server
    if (failedVoiceAttempts.length > 0) {
      try {
        await fetch(`${API}/api/session/${sessionId}/failed-voice-attempts`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ attempts: failedVoiceAttempts }),
        });
      } catch (e) { console.warn('Could not save failed voice attempts:', e); }
    }

    const resp = await fetch(`${API}/api/session/${sessionId}/end`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        operator_name: sessionStorage.getItem('operator_name') || '',
        qualitative_feedback: document.getElementById('feedbackArea').value,
        language: sessionLanguage,
      }),
    });
    const data = await resp.json();
    alert(`Session stored: ${data.session_id}${data.remote_upload_error ? '\nRemote upload error: ' + data.remote_upload_error : ''}`);
    cleanup();
  } catch (e) {
    alert('Error: ' + e.message);
  }
}

async function discardSession() {
  if (!confirm('Discard this session? No data will be saved.')) return;
  try {
    await fetch(`${API}/api/session/${sessionId}/discard`, { method: 'POST' });
  } catch (e) { /* ignore */ }
  cleanup();
}

async function releaseDevice() {
  const deviceId = sessionStorage.getItem('acquired_device_id');
  if (!deviceId) return;
  const body = JSON.stringify({ brain_id: getSessionBrainId() });
  try {
    // keepalive: true ensures the request completes even if the page navigates away
    await fetch(`${API}/api/devices/${deviceId}/release`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      keepalive: true,
    });
    console.log(`[Device] Released: ${deviceId}`);
  } catch (e) {
    // Fallback: sendBeacon (fire-and-forget, survives unload)
    try {
      navigator.sendBeacon(`${API}/api/devices/${deviceId}/release`, body);
      console.log(`[Device] Released via beacon: ${deviceId}`);
    } catch (_) {
      console.warn('[Device] Release failed:', e);
    }
  }
}

// Also release on page unload as a safety net
window.addEventListener('pagehide', () => {
  const deviceId = sessionStorage.getItem('acquired_device_id');
  if (!deviceId) return;
  try {
    navigator.sendBeacon(`${API}/api/devices/${deviceId}/release`,
      JSON.stringify({ brain_id: getSessionBrainId() }));
  } catch (_) {}
});

async function cleanup() {
  if (heartbeatInterval) clearInterval(heartbeatInterval);
  if (_observeAdvanceTimer) {
    clearTimeout(_observeAdvanceTimer);
    _observeAdvanceTimer = null;
  }
  resetExamTimer();
  await releaseDevice();
  sessionStorage.removeItem('session_id');
  sessionStorage.removeItem('session_language');
  sessionStorage.removeItem('cached_state');
  sessionStorage.removeItem('acquired_device_id');
  sessionStorage.removeItem('acquired_brain_id');
  window.location.href = '/intake';
}

// ── Logs ──
function toggleLogs() {
  const drawer = document.getElementById('logsDrawer');
  drawer.classList.toggle('open');
  if (drawer.classList.contains('open') && logsUnlocked) {
    loadLogs();
  }
}

function unlockLogs() {
  const pwd = document.getElementById('logsPassword').value;
  if (pwd === LOGS_PASSWORD) {
    logsUnlocked = true;
    localStorage.setItem('logs_unlocked_until', Date.now() + 86400000); // 24h
    document.getElementById('logsGate').style.display = 'none';
    const unlocked = document.getElementById('logsUnlocked');
    unlocked.style.display = 'flex';
    loadLogs();
  } else {
    alert('Incorrect password');
  }
}

function switchLogTab(btn) {
  document.querySelectorAll('.logs-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  activeLogTab = btn.dataset.tab;
  loadLogs();
}

async function loadLogs() {
  if (!sessionId || !logsUnlocked) return;
  const content = document.getElementById('logsContent');
  content.textContent = 'Loading...';

  try {
    const endpoint = {
      conversation: 'conversation',
      curl: 'curl',
      responses: 'responses',
    }[activeLogTab] || 'conversation';

    const resp = await fetch(`${API}/api/session/${sessionId}/logs/${endpoint}`);
    const data = await resp.json();

    // Guard: API may return an error object instead of an array
    if (!Array.isArray(data)) {
      content.textContent = data.error || 'No data';
      return;
    }

    if (activeLogTab === 'conversation') {
      content.innerHTML = data.map(entry => `
        <div class="log-entry">
          <span class="log-time">${new Date(entry.timestamp).toLocaleTimeString()}</span>
          <span class="log-role ${entry.role}">${entry.role}</span>:
          ${escapeHtml(entry.message)}
          ${entry.state ? `<span style="color:var(--ink-muted)"> [${entry.state}]</span>` : ''}
        </div>
      `).join('');
    } else if (activeLogTab === 'curl') {
      const hideHb = document.getElementById('hideHeartbeat')?.checked;
      const hideScreens = document.getElementById('hideScreenshots')?.checked;
      let filtered = data;
      if (hideHb) filtered = filtered.filter(e => !e.url || !e.url.includes('/heartbeat'));
      content.innerHTML = (filtered.length ? '' : '<div>No CURL commands recorded yet.</div>') +
        filtered.map(entry => {
          let html = `<div class="log-entry">
            <span class="log-time">${new Date(entry.timestamp).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}</span>
            <code>curl -X ${entry.method} ${escapeHtml(entry.url)}${entry.body ? ` -d '${escapeHtml(JSON.stringify(entry.body))}'` : ''}</code>`;
          if (entry.screenshot && !hideScreens) {
            html += `<a href="#" onclick="openScreenshot(this); return false;" class="screenshot-trigger"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="3" width="12" height="10" rx="1.5"/><circle cx="8" cy="8" r="2.5"/></svg> View</a>`;
            html += `<input type="hidden" class="screenshot-data" value="${entry.screenshot}">`;
          } else if (entry.screenshot) {
            html += `<a href="#" onclick="openScreenshot(this); return false;" class="screenshot-trigger mini">[img]</a>`;
            html += `<input type="hidden" class="screenshot-data" value="${entry.screenshot}">`;
          }
          html += `</div>`;
          return html;
        }).join('');
    } else {
      content.innerHTML = data.map(entry => `
        <div class="log-entry">
          <span class="log-time">${entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : ''}</span>
          Step ${entry.step || ''} · <strong>${entry.state || ''}</strong> ·
          Q: ${escapeHtml(entry.question || '')} →
          <strong>${escapeHtml(entry.response_value || '')}</strong>
        </div>
      `).join('') || '<div>No responses recorded yet.</div>';
    }
  } catch (e) {
    content.textContent = 'Error loading logs: ' + e.message;
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML;
}

// ── WhatsApp-style conversation log ──
function addToConversation(role, text, intent, extra) {
  const container = document.getElementById('conversationLog');
  if (!container) return;

  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;

  const time = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});

  if (role === 'patient' && intent) {
    // Compact patient bubble: "transcript" → INTENT (extra)  time
    bubble.innerHTML = `<span class="chat-text">${escapeHtml(text)}</span> <span class="chat-intent">${escapeHtml(intent)}</span>${extra ? ` <span class="chat-extra">${escapeHtml(extra)}</span>` : ''}<span class="chat-time">${time}</span>`;
  } else if (role === 'optometrist') {
    // Compact optometrist bubble: question  (state info)  time
    bubble.innerHTML = `<span class="chat-text">${escapeHtml(text)}</span>${extra ? `<span class="chat-extra">${escapeHtml(extra)}</span>` : ''}<span class="chat-time">${time}</span>`;
  } else {
    bubble.innerHTML = `<span class="chat-text">${escapeHtml(text)}</span><span class="chat-time">${time}</span>`;
  }

  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
}

function addVoiceToConversation(transcript, matchedResponse, confidence, reason) {
  if (matchedResponse) {
    addToConversation('patient', `"${transcript}"`, matchedResponse, `${(confidence * 100).toFixed(0)}%`);
  } else {
    addToConversation('patient', `"${transcript}"`, null, `no match: ${reason || '?'}`);
  }
}

// Auto-log questions and responses
const _origShowQuestion = showQuestion;
// Wrap handleSessionUpdate to log conversation
const _origHandleSessionUpdate = handleSessionUpdate;

// ── DV Summary panel ──
function toggleDvPanel() {
  const drawer = document.getElementById('dvDrawer');
  drawer.classList.toggle('open');
  if (drawer.classList.contains('open')) loadDvSummary();
}

async function loadDvSummary() {
  if (!sessionId) return;
  const content = document.getElementById('dvContent');
  content.textContent = 'Loading...';
  try {
    const resp = await fetch(`${API}/api/session/${sessionId}/derived-variables`);
    const dv = await resp.json();
    if (dv.error) { content.textContent = dv.error; return; }

    const fD = (v) => v != null ? (v >= 0 ? '+' : '') + parseFloat(v).toFixed(2) : '—';
    const fA = (v) => v != null ? Math.round(v) : '—';
    const rxRow = (cls, eye) => eye
      ? `<tr><td class="${cls}">${cls === 'eye-r' ? 'RE' : 'LE'}</td><td>${fD(eye.sph)}</td><td>${fD(eye.cyl)}</td><td>${fA(eye.axis)}</td></tr>`
      : `<tr><td class="${cls}">${cls === 'eye-r' ? 'RE' : 'LE'}</td><td colspan="3" style="color:var(--ink-muted);text-align:center">No data</td></tr>`;
    const rxCard = (title, re, le) => `<div class="dv-rx-card"><div class="dv-rx-title">${title}</div>
      <table class="dv-rx-table"><tr><th></th><th>SPH</th><th>CYL</th><th>AXIS</th></tr>${rxRow('eye-r',re)}${rxRow('eye-l',le)}</table></div>`;

    let html = '';

    // AR & Lenso power cards
    if (dv._ar || dv._lenso) {
      const ar = dv._ar || {}, lo = dv._lenso || {};
      html += rxCard('Autorefractor (AR)', ar.re, ar.le);
      html += rxCard('Lensometry', lo.re, lo.le);
    }

    // Badge formatter for risk/level/boolean values
    const badgeMap = { 'Low': 'low', 'Medium': 'medium', 'High': 'high' };
    const fmtVal = (k, v) => {
      if (v === true) return `<span class="dv-badge yes">Yes</span>`;
      if (v === false) return `<span class="dv-badge no">No</span>`;
      const s = String(v ?? '—');
      if (badgeMap[s]) return `<span class="dv-badge ${badgeMap[s]}">${s}</span>`;
      return `<strong>${s}</strong>`;
    };
    const label = (k) => k.replace(/^dv_/,'').replace(/_/g, ' ');

    const cats = {
      'Patient Profile': ['dv_age_bucket', 'dv_distance_priority', 'dv_near_priority'],
      'Risk Assessment': ['dv_symptom_risk_level', 'dv_medical_risk_level', 'dv_stability_level', 'dv_anomaly_watch', 'dv_requires_optom_review'],
      'AR / Lenso Mismatch': ['dv_ar_lenso_mismatch_level_RE', 'dv_ar_lenso_mismatch_level_LE', 'dv_start_source_policy'],
      'Starting Rx': ['dv_start_rx_RE_sph', 'dv_start_rx_RE_cyl', 'dv_start_rx_RE_axis', 'dv_start_rx_LE_sph', 'dv_start_rx_LE_cyl', 'dv_start_rx_LE_axis'],
      'Test Config': ['dv_target_distance_va', 'dv_endpoint_bias_policy', 'dv_step_size_policy', 'dv_confidence_requirement', 'dv_expected_convergence_time', 'dv_branching_guardrails'],
      'Fogging': ['dv_fogging_policy', 'dv_fogging_amount_D', 'dv_fogging_clearance_mode', 'dv_fogging_required'],
      'Axis / Cylinder': [
        'dv_jcc_axis_same_required',
        'dv_jcc_axis_max_flips',
        'dv_axis_source_used_RE',
        'dv_axis_source_used_LE',
        'dv_axis_cyl_magnitude_for_lane_RE',
        'dv_axis_cyl_magnitude_for_lane_LE',
        'dv_axis_is_near_cardinal_RE',
        'dv_axis_is_near_cardinal_LE',
        'dv_axis_lane_id_RE',
        'dv_axis_lane_id_LE',
        'dv_axis_lane_name_RE',
        'dv_axis_lane_name_LE',
        'dv_axis_step_sequence_RE',
        'dv_axis_step_sequence_LE',
        'dv_axis_confidence_label_RE',
        'dv_axis_confidence_label_LE',
        'dv_axis_selection_reason_RE',
        'dv_axis_selection_reason_LE',
      ],
      'Duochrome': ['dv_duochrome_max_flips'],
      'Near Vision': ['dv_add_expected', 'dv_near_test_required', 'dv_near_binoc_step_D', 'dv_near_binoc_max_plus_steps', 'dv_near_binoc_max_minus_steps'],
      'Safety / Escalation': ['dv_max_delta_from_start_sph', 'dv_max_delta_from_ar_sph'],
    };
    const listed = new Set(Object.values(cats).flat());
    const otherKeys = Object.keys(dv).filter(k => k.startsWith('dv_') && !listed.has(k));
    if (otherKeys.length) cats['Other'] = otherKeys;

    for (const [cat, keys] of Object.entries(cats)) {
      html += `<div class="dv-section">
        <div class="dv-section-head" onclick="this.parentElement.classList.toggle('collapsed')">
          <span class="dv-section-label">${cat}</span>
          <span class="dv-section-chevron">▼</span>
        </div><div class="dv-section-body">`;
      for (const k of keys) {
        html += `<div class="dv-row"><span class="dv-key">${label(k)}</span><span class="dv-val">${fmtVal(k, dv[k])}</span></div>`;
      }
      html += `</div></div>`;
    }

    content.innerHTML = html || 'No DV data.';
  } catch (e) { content.textContent = 'Error: ' + e.message; }
}

// ── Session cache for refresh recovery (Item 7) ──
function cacheSessionState() {
  if (!sessionId || !currentState) return;
  try {
    const convEl = document.getElementById('conversationLog');
    sessionStorage.setItem('cached_state', JSON.stringify({
      sessionId, completedPhases: [...completedPhases],
      conversationHtml: convEl ? convEl.innerHTML : '',
      timestamp: Date.now(),
    }));
  } catch (e) {}
}

function restoreCachedConversation() {
  try {
    const raw = sessionStorage.getItem('cached_state');
    if (!raw) return;
    const cached = JSON.parse(raw);
    if (cached.sessionId !== sessionId) return;
    if (Date.now() - cached.timestamp > 30 * 60 * 1000) return;
    if (cached.completedPhases) completedPhases = new Set(cached.completedPhases);
    if (cached.conversationHtml) {
      const el = document.getElementById('conversationLog');
      if (el) el.innerHTML = cached.conversationHtml;
    }
  } catch (e) {}
}

// ── TTS toggle ──
function toggleTTS() {
  ttsEnabled = !ttsEnabled;
  document.getElementById('ttsBtn').textContent = `TTS: ${ttsEnabled ? 'ON' : 'OFF'}`;
  if (!ttsEnabled) {
    _ttsSessionId += 1;
    speechSynthesis.cancel();
  }
}

// ── Phoropter auto-dispatch toggle ──
let phoropterEnabled = true;

function updatePhoropBtn() {
  const btn = document.getElementById('phoropBtn');
  if (!btn) return;
  const deviceId = sessionStorage.getItem('acquired_device_id');
  if (deviceId) {
    btn.textContent = phoropterEnabled ? `Device: ${deviceId}` : `Device: ${deviceId} (OFF)`;
    btn.style.background = phoropterEnabled ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.2)';
  } else {
    btn.textContent = phoropterEnabled ? 'Device: Test' : 'Device: OFF';
    btn.style.background = phoropterEnabled ? 'rgba(34,197,94,0.2)' : '';
  }
}

async function togglePhoropter() {
  phoropterEnabled = !phoropterEnabled;
  updatePhoropBtn();
  // Tell backend to enable/disable auto-dispatch
  if (sessionId) {
    try {
      await fetch(`${API}/api/session/${sessionId}/phoropter-dispatch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: phoropterEnabled }),
      });
    } catch (e) { console.warn('Could not toggle phoropter dispatch:', e); }
  }
}

// ── Loading ──
// ── Screenshot lightbox ──
let _currentScreenshotB64 = '';

function openScreenshot(clickedEl) {
  const hiddenInput = clickedEl.parentElement.querySelector('.screenshot-data');
  if (!hiddenInput || !hiddenInput.value) return;
  _currentScreenshotB64 = hiddenInput.value;

  const img = document.getElementById('screenshotImg');
  img.src = 'data:image/jpeg;base64,' + _currentScreenshotB64;
  img.classList.remove('zoomed');

  const time = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  document.getElementById('lightboxTitle').textContent = `PHOROPTER CAPTURE  ${time}`;
  document.getElementById('screenshotLightbox').classList.add('open');
}

function closeScreenshot() {
  document.getElementById('screenshotLightbox').classList.remove('open');
  setTimeout(() => { document.getElementById('screenshotImg').src = ''; }, 200);
  _currentScreenshotB64 = '';
}

function toggleZoom() {
  document.getElementById('screenshotImg').classList.toggle('zoomed');
}

function downloadScreenshot() {
  if (!_currentScreenshotB64) return;
  const a = document.createElement('a');
  a.href = 'data:image/jpeg;base64,' + _currentScreenshotB64;
  a.download = `phoropter_${Date.now()}.jpg`;
  a.click();
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeScreenshot();
});

// ── Auto-screenshot toggle + PIP ──
let autoScreenshot = true; // ON by default
let _pipRefreshToken = 0;
let _pipRefreshTimers = [];

function clearPendingPipRefreshes() {
  _pipRefreshTimers.forEach(timerId => clearTimeout(timerId));
  _pipRefreshTimers = [];
}

function setPipLoadingState(message) {
  const loading = document.getElementById('pipLoading');
  const img = document.getElementById('pipImg');
  const footer = document.getElementById('pipFooter');
  if (loading) {
    loading.style.display = 'flex';
    loading.textContent = message;
  }
  if (img) {
    img.style.opacity = '0.45';
  }
  if (footer && message) {
    footer.textContent = message;
  }
}

async function toggleAutoScreenshot() {
  autoScreenshot = !autoScreenshot;
  const btn = document.getElementById('screenshotBtn');
  if (btn) {
    btn.textContent = `Screenshot: ${autoScreenshot ? 'ON' : 'OFF'}`;
    btn.style.background = autoScreenshot ? 'rgba(34,197,94,0.3)' : '';
  }

  // Show/hide PIP
  const pip = document.getElementById('screenshotPip');
  if (pip) pip.style.display = autoScreenshot ? '' : 'none';

  if (sessionId) {
    try {
      await fetch(`${API}/api/session/${sessionId}/phoropter-dispatch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: phoropterEnabled, auto_screenshot: autoScreenshot }),
      });
    } catch (e) { console.warn('Could not toggle screenshot:', e); }
  }

  // Take initial screenshot
  if (autoScreenshot) schedulePipRefreshSequence({ forceFresh: true, reason: 'Initial capture' });
}

async function refreshPipScreenshot(refreshToken = _pipRefreshToken) {
  if (!sessionId) return;
  const loading = document.getElementById('pipLoading');
  const img = document.getElementById('pipImg');
  const footer = document.getElementById('pipFooter');
  if (loading) {
    loading.style.display = 'flex';
    loading.textContent = 'Capturing...';
  }

  try {
    const resp = await fetch(`${API}/api/session/${sessionId}/screenshot`, {
      method: 'POST',
      signal: AbortSignal.timeout(8000), // 8s timeout
    });
    if (refreshToken !== _pipRefreshToken) return;
    if (resp.ok) {
      const data = await resp.json();
      if (data.screenshot) {
        img.src = 'data:image/jpeg;base64,' + data.screenshot;
        img.style.display = '';
        img.style.opacity = '1';
        if (loading) loading.style.display = 'none';
        if (footer) footer.textContent = new Date().toLocaleTimeString();
        return;
      }
    }
    // Non-OK or no screenshot
    if (loading) loading.textContent = 'Device not connected';
    if (footer) footer.textContent = 'Phoropter offline';
    if (img) img.style.opacity = '0.45';
  } catch (e) {
    console.warn('PIP screenshot failed:', e);
    if (refreshToken !== _pipRefreshToken) return;
    if (loading) loading.textContent = 'Device not reachable';
    if (footer) footer.textContent = 'Connection timeout';
    if (img) img.style.opacity = '0.45';
  }
}

function schedulePipRefreshSequence({ powerChanged = false, phaseChanged = false, forceFresh = false, reason = '' } = {}) {
  if (!autoScreenshot) return;
  _pipRefreshToken += 1;
  const token = _pipRefreshToken;
  clearPendingPipRefreshes();

  const delays = forceFresh
    ? [700, 1700]
    : powerChanged
      ? [900, 1900, 3200]
      : phaseChanged
        ? [800, 1800]
        : [900];

  setPipLoadingState(reason || (powerChanged ? 'Syncing phoropter...' : 'Refreshing...'));
  delays.forEach((delayMs) => {
    const timerId = setTimeout(() => refreshPipScreenshot(token), delayMs);
    _pipRefreshTimers.push(timerId);
  });
}

// Auto-update PIP after each response (called from submitResponse)
async function autoUpdatePip({ powerChanged = false, phaseChanged = false } = {}) {
  schedulePipRefreshSequence({
    powerChanged,
    phaseChanged,
    reason: powerChanged ? 'Syncing phoropter...' : 'Refreshing phoropter...',
  });
}

function expandPip() {
  const img = document.getElementById('pipImg');
  if (!img || !img.src) return;
  // Re-use the lightbox
  document.getElementById('screenshotImg').src = img.src;
  document.getElementById('lightboxTitle').textContent = 'PHOROPTER CAPTURE  ' + new Date().toLocaleTimeString();
  document.getElementById('screenshotLightbox').classList.add('open');
}

function closePip() {
  const pip = document.getElementById('screenshotPip');
  if (pip) pip.style.display = 'none';
  autoScreenshot = false;
  const btn = document.getElementById('screenshotBtn');
  if (btn) { btn.textContent = 'Screenshot: OFF'; btn.style.background = ''; }
}

// ── Draggable PIP ──
(function() {
  const pip = document.getElementById('screenshotPip');
  const header = document.getElementById('pipHeader');
  if (!pip || !header) return;
  let isDragging = false, startX, startY, startLeft, startTop;

  header.addEventListener('mousedown', (e) => {
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
    const rect = pip.getBoundingClientRect();
    startLeft = rect.left;
    startTop = rect.top;
    pip.style.transition = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    pip.style.left = (startLeft + e.clientX - startX) + 'px';
    pip.style.top = (startTop + e.clientY - startY) + 'px';
    pip.style.right = 'auto';
    pip.style.bottom = 'auto';
  });

  document.addEventListener('mouseup', () => { isDragging = false; });
})();

// ── PIP Zoom & Pan ──
let _pipZoom = 1;
let _pipPanX = 0, _pipPanY = 0;
let _pipPanning = false, _pipPanStartX = 0, _pipPanStartY = 0;

function pipZoom(direction) {
  const steps = [0.5, 0.75, 1, 1.5, 2, 3, 4, 5];
  const idx = steps.indexOf(_pipZoom);
  const newIdx = Math.max(0, Math.min(steps.length - 1, (idx >= 0 ? idx : 2) + direction));
  _pipZoom = steps[newIdx];
  applyPipTransform();
}

function pipResetZoom() {
  _pipZoom = 1;
  _pipPanX = 0;
  _pipPanY = 0;
  applyPipTransform();
}

function applyPipTransform() {
  const img = document.getElementById('pipImg');
  if (img) img.style.transform = `scale(${_pipZoom}) translate(${_pipPanX}px, ${_pipPanY}px)`;
  const label = document.getElementById('pipZoomLevel');
  if (label) label.textContent = `${_pipZoom}x`;
}

// Mouse wheel zoom
(function() {
  const body = document.getElementById('pipBody');
  if (!body) return;

  body.addEventListener('wheel', (e) => {
    e.preventDefault();
    pipZoom(e.deltaY < 0 ? 1 : -1);
  }, { passive: false });

  // Pan with mouse drag inside pip-body
  body.addEventListener('mousedown', (e) => {
    if (_pipZoom <= 1) return;
    _pipPanning = true;
    _pipPanStartX = e.clientX - _pipPanX * _pipZoom;
    _pipPanStartY = e.clientY - _pipPanY * _pipZoom;
    body.classList.add('grabbing');
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!_pipPanning) return;
    _pipPanX = (e.clientX - _pipPanStartX) / _pipZoom;
    _pipPanY = (e.clientY - _pipPanStartY) / _pipZoom;
    applyPipTransform();
  });

  document.addEventListener('mouseup', () => {
    _pipPanning = false;
    const body = document.getElementById('pipBody');
    if (body) body.classList.remove('grabbing');
  });
})();

// ── JCC Auto-Flip ──
function handleAutoFlip(data) {
  // Clear any pending flip timer
  if (_autoFlipTimer) { clearTimeout(_autoFlipTimer); _autoFlipTimer = null; }
  stopDuplexBargeInMonitor();

  if (!data.auto_flip) {
    _flipState = null;
    updateFlipIndicator(null);
    setOptionsEnabled(true);
    return;
  }

  _jccPromptExposureCount += 1;

  const questionToken = _questionTurnToken;
  const promptQuestion = (document.getElementById('questionText')?.textContent || data.question || '').trim();

  // ── Flip 1: Show + speak "This is Flip 1", WAIT for TTS, THEN start 2s timer ──
  _flipState = 'flip1';
  updateFlipIndicator('flip1');
  setOptionsEnabled(false);
  _inputEnabled = false;

  const flip1Text = getStaticFlipPrompt('flip1', data.question || promptQuestion);
  document.getElementById('questionText').textContent = flip1Text;
  const waitSeconds = data.flip_wait_seconds || 2;

  // Wait for Flip 1 TTS to finish via onEnd callback, THEN wait the observation period
  speakQuestionWithStableFollowup(flip1Text, null, () => {
    if (!isCurrentQuestionTurn(questionToken)) return;
    _autoFlipTimer = setTimeout(() => {
      if (!isCurrentQuestionTurn(questionToken)) return;
      doFlip2();
    }, waitSeconds * 1000);
  }, 'flip1');

  async function doFlip2() {
    if (!isCurrentQuestionTurn(questionToken)) return;
    // Send handle command to flip to position 2
    if (sessionId) {
      try {
        await fetch(`${API}/api/session/${sessionId}/jcc-flip`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        });
      } catch (e) { console.warn('JCC flip failed:', e); }
    }
    if (!isCurrentQuestionTurn(questionToken)) return;

    // ── Flip 2: Show + speak "This is Flip 2. Which is better?" ──
    _flipState = 'flip2';
    updateFlipIndicator('flip2');

    const flip2Text = getStaticFlipPrompt('flip2', data.question || promptQuestion);
    const flip2Prompt = promptQuestion;
    document.getElementById('questionText').textContent = flip2Prompt;
    if (isDuplexTurnMode()) {
      setOptionsEnabled(true);
      _inputEnabled = true;
      if (currentState && isVoiceBargeInEligible(currentState.state, currentState.question || flip2Prompt, currentState)) {
        startDuplexBargeInMonitor({
          state: currentState.state,
          options: currentState.options || [],
          step: currentState.step,
          questionText: currentState.question || flip2Prompt,
          questionToken,
          data: currentState,
        });
      }
    }
    // Wait for Flip 2 TTS to finish via onEnd callback, then beep + enable buttons + listen
    speakQuestionWithStableFollowup(flip2Text, null, () => {
      if (!isCurrentQuestionTurn(questionToken)) return;
      stopDuplexBargeInMonitor();
      if (isDuplexSpeechDetected(questionToken)) return;
      setOptionsEnabled(true);
      if (voiceEnabled && currentState) {
        cueAndStartVoiceCapture(currentState.state, currentState.options || [], currentState.step, {
          enableInput: true,
          questionToken,
        });
      } else {
        playBeep().then(() => {
          if (!isCurrentQuestionTurn(questionToken)) return;
          _inputEnabled = true;
        });
      }
    }, 'flip2');
  }
}

function updateFlipIndicator(state) {
  const el = document.getElementById('flipIndicator');
  if (!el) return;
  if (!state) {
    el.style.display = 'none';
    el.textContent = '';
    return;
  }
  el.style.display = '';
  if (state === 'flip1') {
    el.textContent = 'First Option — Observing...';
    el.className = 'flip-indicator flip1';
  } else {
    el.textContent = 'Second Option — Which is better?';
    el.className = 'flip-indicator flip2';
  }
}

function setOptionsEnabled(enabled) {
  const btns = document.querySelectorAll('#optionsGrid .option-btn');
  btns.forEach(btn => {
    btn.disabled = !enabled;
    btn.style.opacity = enabled ? '' : '0.4';
    btn.style.pointerEvents = enabled ? '' : 'none';
  });
}

// ── Collapsible sidebar sections ──
// ── Sidebar overlay (iPad) ──
function toggleSidebarOverlay() {
  const sidebar = document.querySelector('.sidebar');
  const backdrop = document.getElementById('sidebarBackdrop');
  const fab = document.getElementById('sidebarFab');
  if (!sidebar) return;
  const open = sidebar.classList.toggle('overlay-open');
  backdrop.classList.toggle('show', open);
  fab.classList.toggle('active', open);
}

function toggleSidebarSection(sectionId) {
  const section = document.getElementById(sectionId);
  if (!section) return;
  section.classList.toggle('collapsed');
  const icon = section.querySelector('.collapse-icon');
  if (icon) icon.textContent = section.classList.contains('collapsed') ? '▶' : '▼';
}

// ── Logs auto-show if unlocked ──
if (localStorage.getItem('logs_unlocked_until')) {
  const until = parseInt(localStorage.getItem('logs_unlocked_until'));
  if (Date.now() < until) {
    document.addEventListener('DOMContentLoaded', () => {
      document.getElementById('logsGate').style.display = 'none';
      document.getElementById('logsUnlocked').style.display = 'flex';
    });
  }
}
