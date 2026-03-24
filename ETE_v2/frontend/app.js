/**
 * ETE v2 — Main Test Screen JavaScript
 *
 * Drives the eye test session: receives questions from the FSM backend,
 * displays options, sends responses, updates prescription table and phase progress.
 */

const API = window.BACKEND_URL || '';
const LOGS_PASSWORD = 'Shantanu';

// ── TTS (Browser SpeechSynthesis) ──
let ttsEnabled = true;

function speakQuestion(text, langOverride) {
  if (!ttsEnabled || !('speechSynthesis' in window)) {
    console.log('[TTS] Disabled or unavailable');
    return;
  }

  // Chrome fix: cancel + resume to clear any stuck state
  speechSynthesis.cancel();
  speechSynthesis.resume();

  const doSpeak = () => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    const lang = langOverride || sessionLanguage || 'en';
    const voices = speechSynthesis.getVoices();
    console.log(`[TTS] Speaking (${lang}): "${text.substring(0, 50)}..." [${voices.length} voices available]`);

    if (lang === 'hi') {
      utterance.lang = 'hi-IN';
      const hiVoice = voices.find(v => v.lang.startsWith('hi'))
        || voices.find(v => v.lang === 'hi-IN');
      if (hiVoice) { utterance.voice = hiVoice; console.log(`[TTS] Hindi voice: ${hiVoice.name}`); }
    } else {
      utterance.lang = 'en-IN';
      const enVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Samantha'))
        || voices.find(v => v.lang.startsWith('en-IN'))
        || voices.find(v => v.lang.startsWith('en-') && !v.name.includes('Compact'))
        || voices.find(v => v.lang.startsWith('en'));
      if (enVoice) { utterance.voice = enVoice; console.log(`[TTS] English voice: ${enVoice.name}`); }
    }

    utterance.onstart = () => console.log('[TTS] Started speaking');
    utterance.onend = () => console.log('[TTS] Finished speaking');
    utterance.onerror = (e) => console.error('[TTS] Error:', e.error);

    speechSynthesis.speak(utterance);

    // Chrome workaround: Chrome sometimes pauses speech after 15s.
    // Periodic resume keeps it going.
    const keepAlive = setInterval(() => {
      if (!speechSynthesis.speaking) { clearInterval(keepAlive); return; }
      speechSynthesis.pause();
      speechSynthesis.resume();
    }, 10000);
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

// Preload voices (needed on some browsers)
if ('speechSynthesis' in window) {
  speechSynthesis.getVoices();
  speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
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

// ── Stimulus descriptions per state ──
const STIMULUS_DESCRIPTIONS = {
  'B': 'Distance letter chart',
  'D': 'Distance letter chart',
  'E': 'Dot chart for axis comparison',
  'F': 'Dot chart for power comparison',
  'G': 'Red-green chart',
  'H': 'Dot chart for axis comparison',
  'I': 'Dot chart for power comparison',
  'J': 'Red-green chart',
  'K': 'Top-bottom balance chart',
  'P': 'Near text chart',
  'Q': 'Near text chart',
  'R': 'Near text with both eyes',
};

// ── Voice input state ──
let voiceEnabled = true; // ON by default, like FSMv3.1_R2
let voiceRecording = false;
let recognition = null;
let voiceSubmitting = false; // Prevent double-submit during async match
let voiceAttemptCount = 0; // Per-question attempt counter
const VOICE_REPROMPT_LIMIT = 2; // After this many failed attempts, show keyboard fallback msg
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let failedVoiceAttempts = []; // Structured log of failed voice attempts

// ── Faster-whisper backend state ──
let whisperAvailable = false; // Set true if backend has faster-whisper loaded
let mediaRecorder = null;
let audioChunks = [];
let micStream = null;

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

// ── Phase definitions ──
const ALL_PHASES = [
  { state: 'B', name: 'Coarse Sphere RE', eye: 'RE' },
  { state: 'E', name: 'JCC Axis RE', eye: 'RE' },
  { state: 'F', name: 'JCC Power RE', eye: 'RE' },
  { state: 'G', name: 'Duochrome RE', eye: 'RE' },
  { state: 'D', name: 'Coarse Sphere LE', eye: 'LE' },
  { state: 'H', name: 'JCC Axis LE', eye: 'LE' },
  { state: 'I', name: 'JCC Power LE', eye: 'LE' },
  { state: 'J', name: 'Duochrome LE', eye: 'LE' },
  { state: 'K', name: 'Binocular Balance', eye: 'BIN' },
  { state: 'P', name: 'Near Add RE', eye: 'RE' },
  { state: 'Q', name: 'Near Add LE', eye: 'LE' },
  { state: 'R', name: 'Near Binocular', eye: 'BIN' },
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
  'TOP_CLEARER': '', 'BOTTOM_CLEARER': '',
  'TARGET_OK': 'clear', 'NOT_CLEAR': 'blurry',
};

// ── State ──
let sessionId = null;
let currentState = null;
let logsUnlocked = false;
let sessionLanguage = 'en'; // Default, set by language selection
let activeLogTab = 'conversation';
let completedPhases = new Set();
let heartbeatInterval = null;
let _langSelectPendingData = null; // Stores first-question data during language selection
let _autoFlipTimer = null; // Timer for JCC auto-flip
let _flipState = null; // 'flip1', 'flip2', or null
let _inputEnabled = false; // Global gate: voice, gamepad, keyboard only after beep

// ── Gamepad ──
let gamepadEnabled = true;
let gamepadConnected = false;
let gamepadIndex = null;
let _gamepadPrevButtons = [false, false, false, false];
let _gamepadPollId = null;

// ── Initialization ──
document.addEventListener('DOMContentLoaded', () => {
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

    // Check if language was already selected (page refresh)
    const savedLang = sessionStorage.getItem('session_language');
    if (savedLang) {
      sessionLanguage = savedLang;
      restoreCachedConversation();

      // Auto-resume: start immediately, enable TTS via a one-time user interaction listener
      await handleSessionUpdate(data);
      document.getElementById('endBtn').style.display = '';
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

  document.getElementById('questionCard').style.display = '';
  document.getElementById('questionStep').textContent = 'LANGUAGE SELECTION';
  document.getElementById('questionState').textContent = 'Setup';
  document.getElementById('questionState').className = 'question-state bin';
  document.getElementById('questionText').textContent = 'Please select your preferred language / कृपया अपनी भाषा चुनें';

  const stimEl = document.getElementById('stimulusDescription');
  if (stimEl) stimEl.textContent = '';
  const chartEl = document.getElementById('letterChart');
  if (chartEl) chartEl.style.display = 'none';

  // Speak in English (neutral, before language is chosen)
  speakQuestion('Please select your preferred language. English or Hindi?', 'en');

  const grid = document.getElementById('optionsGrid');
  grid.innerHTML = '';

  const enBtn = document.createElement('button');
  enBtn.className = 'option-btn clear';
  enBtn.innerHTML = 'English<span class="key-hint">1</span>';
  enBtn.onclick = () => selectLanguage('en', _langSelectPendingData);
  grid.appendChild(enBtn);

  const hiBtn = document.createElement('button');
  hiBtn.className = 'option-btn';
  hiBtn.style.borderColor = '#f97316';
  hiBtn.style.background = '#fff7ed';
  hiBtn.style.color = '#9a3412';
  hiBtn.innerHTML = 'हिन्दी (Hindi)<span class="key-hint">2</span>';
  hiBtn.onclick = () => selectLanguage('hi', _langSelectPendingData);
  grid.appendChild(hiBtn);

  // If voice is enabled, listen for language choice
  if (voiceEnabled && SpeechRecognition) {
    updateVoiceStatus('Say "English" or "Hindi"');
    playBeep();
    setTimeout(() => {
      startVoiceCapture('LANG_SELECT', ['ENGLISH', 'HINDI'], 0);
    }, 200);
  }
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
  document.getElementById('endBtn').style.display = '';
  startHeartbeat();
  if (pendingData.question && !pendingData.is_terminal) {
    addToConversation('optometrist', pendingData.question, null, `${pendingData.state}`);
  }
}

// ── Heartbeat ──
function startHeartbeat() {
  const phoropterId = sessionStorage.getItem('phoropter_id');
  if (!phoropterId) return;
  if (heartbeatInterval) clearInterval(heartbeatInterval);
  heartbeatInterval = setInterval(async () => {
    try {
      await fetch(`${API}/api/devices/${phoropterId}/heartbeat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ brain_id: 'ete_v2' }),
      });
    } catch (e) { /* ignore */ }
  }, 15000);
}

// ── Core: handle session update from backend ──
async function handleSessionUpdate(data) {
  if (data.error) {
    document.getElementById('questionText').textContent = data.error;
    return;
  }

  currentState = data;

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
  const stimEl = document.getElementById('stimulusDescription');
  if (stimEl) stimEl.textContent = stimDesc;

  // Display letter chart for coarse sphere states (B, D)
  const chartEl = document.getElementById('letterChart');
  if (chartEl) {
    const chartParam = (data.chart_param || '').replace(/[-\/]/g, '_').replace(/_+/g, '_');
    const letters = DISTANCE_CHART_STIMULI[chartParam] || DISTANCE_CHART_STIMULI[chartParam.replace(/_/g, '')];
    if ((data.state === 'B' || data.state === 'D') && letters) {
      chartEl.style.display = '';
      chartEl.innerHTML = letters.map((line, i) => {
        const fontSize = Math.max(1.0, 2.4 - i * 0.4);
        return `<div class="chart-line" style="font-size:${fontSize}rem;letter-spacing:${fontSize * 0.4}rem">${line.join(' ')}</div>`;
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
      body: JSON.stringify({ state: data.state, language: sessionLanguage, question: data.question }),
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
  const grid = document.getElementById('optionsGrid');
  grid.innerHTML = '';
  if (localizedLabels) {
    localizedLabels.forEach((label, i) => {
      const btn = document.createElement('button');
      btn.className = 'option-btn';
      const styleKey = (data.options || []).find(o => o === label.internal) || label.internal;
      const style = OPTION_STYLES[styleKey] || '';
      if (style) btn.classList.add(style);
      const displayText = label.localized || label.display || label.internal;
      const internalHint = label.internal !== displayText ? label.internal : '';
      btn.innerHTML = `${displayText}${internalHint ? '<span class="opt-internal">[' + internalHint + ']</span>' : ''}<span class="key-hint">${i + 1}</span>`;
      btn.onclick = () => submitResponse(label.internal);
      grid.appendChild(btn);
    });
  } else {
    (data.options || []).forEach((opt, i) => {
      const btn = document.createElement('button');
      btn.className = 'option-btn';
      const style = OPTION_STYLES[opt] || '';
      if (style) btn.classList.add(style);
      btn.innerHTML = `${opt}<span class="key-hint">${i + 1}</span>`;
      btn.onclick = () => submitResponse(opt);
      grid.appendChild(btn);
    });
  }

  // 4. Speak the LOCALIZED question, then beep, then listen
  //    For JCC states with auto_flip: ALL TTS is handled by handleAutoFlip (Flip 1 → Flip 2)
  //    For all other states: TTS → beep → listen immediately
  const canListen = voiceEnabled && (voiceMode === 'whisper' || SpeechRecognition);
  const isAutoFlip = data.auto_flip;

  if (ttsEnabled && !isAutoFlip) {
    speakQuestion(localizedQuestion);
    const waitForSpeech = () => {
      if (speechSynthesis.speaking) {
        setTimeout(waitForSpeech, 100);
      } else {
        setTimeout(async () => {
          await playBeep();
          _inputEnabled = true; // Enable ALL input (voice, gamepad, keyboard) after beep
          if (canListen) startVoiceCapture(data.state, data.options || [], data.step);
        }, 300);
      }
    };
    setTimeout(waitForSpeech, 200);
  } else if (!isAutoFlip) {
    playBeep().then(() => {
      _inputEnabled = true;
      if (canListen) startVoiceCapture(data.state, data.options || [], data.step);
    });
  }
  // For JCC auto-flip states: _inputEnabled is set in handleAutoFlip after Flip 2 beep
}

// fetchLocalizedLabels removed — logic is now inline in showQuestion() to ensure
// localized text is available BEFORE TTS speaks

// ── Voice input pipeline (Browser SpeechRecognition) ──

let voiceMode = 'browser'; // 'off', 'browser', 'whisper'

function setVoiceMode(mode) {
  // Stop any active recording
  if (recognition) { try { recognition.abort(); } catch(e) {} recognition = null; }
  if (mediaRecorder && mediaRecorder.state === 'recording') { mediaRecorder.stop(); }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  voiceRecording = false;
  voiceSubmitting = false;

  voiceMode = mode;
  voiceEnabled = mode !== 'off';

  updateVoiceModeSelect();
  updateVoiceStatus(voiceEnabled ? `Ready (${mode})` : '—');

  // If turning on and we have an active question, start listening
  if (voiceEnabled && currentState && !currentState.is_terminal) {
    playBeep();
    setTimeout(() => startVoiceCapture(currentState.state, currentState.options || [], currentState.step), 200);
  }
}

function updateVoiceModeSelect() {
  const sel = document.getElementById('voiceModeSelect');
  if (sel) sel.value = voiceMode;
}

// Legacy compatibility
function toggleVoice() { setVoiceMode(voiceMode === 'off' ? 'browser' : 'off'); }
function updateVoiceButton() { updateVoiceModeSelect(); }

function startVoiceCapture(state, options, step) {
  if (!voiceEnabled) return;
  if (voiceSubmitting) return;
  // Don't start listening while TTS is speaking
  if (speechSynthesis.speaking) {
    const waitForTTS = () => {
      if (speechSynthesis.speaking) { setTimeout(waitForTTS, 100); return; }
      startVoiceCapture(state, options, step);
    };
    setTimeout(waitForTTS, 100);
    return;
  }

  // Route based on user's explicit voiceMode selection
  if (voiceMode === 'whisper') {
    startWhisperCapture(state, options, step);
    return;
  }

  // Browser SpeechRecognition mode
  if (!SpeechRecognition) return;

  // Stop any previous recognition
  if (recognition) {
    try { recognition.abort(); } catch (e) {}
    recognition = null;
  }

  voiceRecording = false;

  // Small delay to let previous recognition clean up
  setTimeout(() => {
    if (!voiceEnabled || voiceSubmitting) return;

    recognition = new SpeechRecognition();
    // Fix 4: en-US better for short words than en-IN
    recognition.lang = sessionLanguage === 'hi' ? 'hi-IN' : 'en-US';
    // Fix 1: continuous=true prevents premature termination on single-syllable words
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 5;

    // Fix 5: SpeechGrammarList constrains recognition to expected vocabulary
    try {
      const SpeechGrammarList = window.SpeechGrammarList || window.webkitSpeechGrammarList;
      if (SpeechGrammarList) {
        const grammar = '#JSGF V1.0; grammar r; public <r> = clear | blurry | repeat | one | two | same | red | green | top | bottom | first | second ;';
        const list = new SpeechGrammarList();
        list.addFromString(grammar, 1);
        recognition.grammars = list;
      }
    } catch (e) { /* grammar not supported — ok */ }

    const capturedState = state;
    const capturedOptions = options;
    const capturedStep = step;
    let lastInterimTranscript = ''; // store interim for fallback
    let quickMatchTimer = null; // 1s timer to force-process short words
    let alreadyProcessed = false; // prevent double-processing

    recognition.onstart = () => {
      voiceRecording = true;
      updateVoiceStatus('🎙 Listening...');
      console.log(`[Voice] Listening for step ${capturedStep}, state ${capturedState}, options: ${capturedOptions.join(', ')}`);
    };

    // Quick-match check: does this interim text match a known response?
    function interimMatchesOption(text) {
      const t = text.toLowerCase().trim();
      if (!t) return false;
      // Also try with digits stripped of punctuation (Chrome may add "." or spaces)
      const cleaned = t.replace(/[^a-z0-9]/g, '');
      return clientSideMatch(t, capturedOptions) !== null
          || clientSideMatch(cleaned, capturedOptions) !== null;
    }

    function forceProcessInterim() {
      if (alreadyProcessed || !lastInterimTranscript) return;
      alreadyProcessed = true;
      try { recognition.stop(); } catch(e) {}
      voiceRecording = false;
      // Try both raw and cleaned versions
      const cleaned = lastInterimTranscript.replace(/[^a-z0-9]/gi, '').toLowerCase();
      const alts = cleaned !== lastInterimTranscript.toLowerCase() ? [cleaned] : [];
      console.log(`[Voice] Quick-match: forcing "${lastInterimTranscript}" (cleaned: "${cleaned}")`);
      updateVoiceStatus(`Processing: "${lastInterimTranscript}"`);
      matchVoiceResponseWithAlternatives(lastInterimTranscript, alts, capturedState, capturedOptions);
    }

    recognition.onresult = (event) => {
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
        updateVoiceStatus(`🎙 "${interimTranscript}"...`);

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
        alreadyProcessed = true;
        try { recognition.stop(); } catch(e) {}
        voiceRecording = false;
        const trimmed = finalTranscript.trim();
        const alts = finalAlternatives.map(a => a.trim()).filter(a => a);
        console.log(`[Voice] Final: "${trimmed}" | Alternatives: ${JSON.stringify(alts)}`);
        updateVoiceStatus(`Processing: "${trimmed}"`);
        matchVoiceResponseWithAlternatives(trimmed, alts, capturedState, capturedOptions);
      }
    };

    recognition.onerror = (event) => {
      if (quickMatchTimer) { clearTimeout(quickMatchTimer); quickMatchTimer = null; }
      voiceRecording = false;
      console.log(`[Voice] Error: ${event.error}`);
      if (event.error === 'no-speech') {
        updateVoiceStatus('No speech detected. Repeating question...');
        // Re-speak the question (like FSMv3.1_R2 retry=True reprompt)
        if (voiceEnabled && !voiceSubmitting) {
          const questionEl = document.getElementById('questionText');
          const questionText = questionEl ? questionEl.textContent : '';
          const retryPrompt = sessionLanguage === 'hi'
            ? `फिर से सुनिए। ${questionText}`
            : `Let me repeat. ${questionText}`;
          speakQuestion(retryPrompt);
          // Wait for TTS, then beep + listen
          const waitAndListen = () => {
            if (speechSynthesis.speaking) {
              setTimeout(waitAndListen, 100);
            } else {
              setTimeout(() => {
                playBeep();
                setTimeout(() => startVoiceCapture(capturedState, capturedOptions, capturedStep), 200);
              }, 300);
            }
          };
          setTimeout(waitAndListen, 200);
        }
      } else if (event.error === 'aborted') {
        // Intentional abort — don't restart
      } else {
        updateVoiceStatus(`Mic error: ${event.error}`);
      }
    };

    let gotFinalResult = false;
    let gotError = false;

    // Patch: track whether we got a result or error
    const origOnResult = recognition.onresult;
    recognition.onresult = (event) => {
      // Check if any result is final
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) gotFinalResult = true;
      }
      origOnResult(event);
    };
    const origOnError = recognition.onerror;
    recognition.onerror = (event) => {
      gotError = true;
      origOnError(event);
    };

    recognition.onend = () => {
      if (quickMatchTimer) { clearTimeout(quickMatchTimer); quickMatchTimer = null; }
      voiceRecording = false;

      // If already processed (final, quick-match, or error), do nothing
      if (alreadyProcessed || gotFinalResult || gotError || voiceSubmitting) return;

      // Fix 2: Try interim transcript as fallback for single-syllable words
      if (lastInterimTranscript) {
        console.log(`[Voice] Using interim as fallback: "${lastInterimTranscript}"`);
        updateVoiceStatus(`Processing interim: "${lastInterimTranscript}"`);
        matchVoiceResponseWithAlternatives(lastInterimTranscript, [], capturedState, capturedOptions);
        return;
      }

      // No interim either — speech wasn't picked up
      console.log('[Voice] Recognition ended without result — repeating question');
      updateVoiceStatus('Could not hear clearly. Repeating...');

      if (voiceEnabled && !voiceSubmitting) {
        const questionEl = document.getElementById('questionText');
        const questionText = questionEl ? questionEl.textContent : '';
        const retryPrompt = sessionLanguage === 'hi'
          ? `सुनाई नहीं दिया। ${questionText}`
          : `I could not hear you. ${questionText}`;
        speakQuestion(retryPrompt);
        const waitAndListen = () => {
          if (speechSynthesis.speaking) {
            setTimeout(waitAndListen, 100);
          } else {
            setTimeout(() => {
              playBeep();
              setTimeout(() => startVoiceCapture(capturedState, capturedOptions, capturedStep), 200);
            }, 300);
          }
        };
        setTimeout(waitAndListen, 200);
      }
    };

    try {
      recognition.start();
    } catch (e) {
      console.warn('[Voice] Could not start recognition:', e);
      updateVoiceStatus('Mic start failed. Click Mic: ON to retry.');
    }
  }, 100);
}

// ── Faster-whisper recording pipeline ──
async function startWhisperCapture(state, options, step) {
  if (!voiceEnabled || voiceSubmitting) return;

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
  const VAD_START_TIMEOUT = (state === 'B' || state === 'D') ? 5.0 : 2.5; // seconds to wait for first speech
  const VAD_END_SILENCE = (state === 'B' || state === 'D') ? 2.0 : 0.8;   // trailing silence to stop
  const VAD_MIN_SPEECH = 0.25;  // minimum speech duration before silence can end
  const VAD_MAX_DURATION = (state === 'B' || state === 'D') ? 15 : 5;     // hard max

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
    clearInterval(vadInterval);
    vadCtx.close().catch(() => {});
    voiceRecording = false;
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    console.log(`[VAD] Stop reason: ${vadStopReason}, speech: ${speechDetected}, duration: ${speechDuration.toFixed(1)}s`);
    if (audioChunks.length === 0) {
      updateVoiceStatus('No audio captured');
      repeatAndListen(state, options, step);
      return;
    }

    updateVoiceStatus('Processing with whisper...');
    voiceSubmitting = true;

    // Send raw WebM blob directly — backend decodes it via ffmpeg/faster-whisper
    const blob = new Blob(audioChunks, { type: mimeType });
    try {
      const arrayBuf = await blob.arrayBuffer();
      const bytes = new Uint8Array(arrayBuf);
      let binary = '';
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      const audioBase64 = btoa(binary);

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
          stimulus_letters: getCurrentStimulusLetters(),
        }),
      });

      const result = resp.ok ? await resp.json() : { error: `Server error ${resp.status}`, accepted: false };
      console.log(`[Whisper] Result:`, result);

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
        repeatAndListen(state, options, step);
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

      if (voiceAttemptCount >= VOICE_REPROMPT_LIMIT) {
        updateVoiceStatus(`✗ Failed ${voiceAttemptCount}x. Use buttons below.`);
      } else {
        repeatAndListen(state, options, step);
      }
      return;
    } catch (e) {
      console.error('[Whisper] Processing error:', e);
      voiceSubmitting = false;
      updateVoiceStatus(`Whisper error: ${e.message}`);
      repeatAndListen(state, options, step);
    }
  };

  mediaRecorder.start();
  // VAD interval handles stopping — no fixed timeout needed
}

function repeatAndListen(state, options, step) {
  if (!voiceEnabled) return;
  const questionEl = document.getElementById('questionText');
  const questionText = questionEl ? questionEl.textContent : '';
  const retryPrompt = sessionLanguage === 'hi'
    ? `समझ नहीं आया। ${questionText}`
    : `I didn't catch that. ${questionText}`;
  speakQuestion(retryPrompt);
  const waitAndListen = () => {
    if (speechSynthesis.speaking) {
      setTimeout(waitAndListen, 100);
    } else {
      setTimeout(() => {
        playBeep();
        setTimeout(() => startVoiceCapture(state, options, step), 200);
      }, 300);
    }
  };
  setTimeout(waitAndListen, 200);
}

function getCurrentStimulusLetters() {
  // Get the displayed chart letters for the current coarse sphere state
  if (!currentState) return null;
  const state = currentState.state;
  if (state !== 'B' && state !== 'D') return null;
  const chartParam = (currentState.chart_param || '').replace(/[-\/]/g, '_').replace(/_+/g, '_');
  const letters = DISTANCE_CHART_STIMULI[chartParam] || DISTANCE_CHART_STIMULI[chartParam.replace(/_/g, '')];
  if (!letters) return null;
  // Format as space-separated letters per line (matching FSMv3.1_R2 format)
  return letters.map(line => line.join(' ')).join('\n');
}

async function matchVoiceResponseWithAlternatives(primary, alternatives, state, options) {
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
        playBeep();
        setTimeout(() => startVoiceCapture('LANG_SELECT', ['ENGLISH', 'HINDI'], 0), 200);
      }, 1000);
    }
    return;
  }

  // Try the primary transcript first, then each alternative
  const transcriptsToTry = [primary, ...alternatives.filter(a => a !== primary)];

  for (const transcript of transcriptsToTry) {
    const result = await matchVoiceResponse(transcript, state, options);
    if (result) return; // Matched and submitted
  }

  // None matched — track failed attempt
  voiceSubmitting = false;
  voiceAttemptCount++;

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
  });

  addVoiceToConversation(primary, null, 0, `no match (attempt ${voiceAttemptCount}/${VOICE_REPROMPT_LIMIT})`);

  if (voiceAttemptCount >= VOICE_REPROMPT_LIMIT) {
    // Reached reprompt limit — show keyboard fallback message
    updateVoiceStatus(`✗ Voice failed ${voiceAttemptCount}x. Use the buttons below.`);
    // Stop voice for this question
    if (recognition) { try { recognition.abort(); } catch(e) {} }
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
      speakQuestion(retryPrompt);
      const waitAndListen = () => {
        if (speechSynthesis.speaking) {
          setTimeout(waitAndListen, 100);
        } else {
          setTimeout(() => {
            playBeep();
            setTimeout(() => startVoiceCapture(state, options, 0), 200);
          }, 300);
        }
      };
      setTimeout(waitAndListen, 200);
    }
  }
}

async function matchVoiceResponse(transcript, state, options) {
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
      }),
    });
    if (resp.ok) {
      const result = await resp.json();
      if (result.accepted && result.response_value) {
        matched = result.response_value;
        voiceMeta = {
          transcript: transcript,
          match_confidence: result.confidence || 0.8,
          match_method: result.method || 'server_side',
          canonical_label: result.canonical_label || matched,
          input_mode: 'voice_browser_speech_recognition',
          response_attempt_count: voiceAttemptCount + 1,
          stimulus_letters: getCurrentStimulusLetters(),
          session_language: sessionLanguage,
        };
        updateVoiceStatus(`✓ "${transcript}" → ${matched} (${(result.confidence * 100).toFixed(0)}%)`);
        addVoiceToConversation(transcript, matched, result.confidence);
      } else {
        console.log(`[Voice] Server no match: ${result.reason}`);
      }
    }
  } catch (e) {
    console.log('[Voice] Server match unavailable, using client-side fallback');
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
      updateVoiceStatus(`✓ "${transcript}" → ${matched}`);
      addVoiceToConversation(transcript, matched, 0.8);
    }
  }

  if (matched) {
    await submitResponse(matched, voiceMeta);
    voiceSubmitting = false;
    return true; // Successfully matched
  }

  voiceSubmitting = false;
  return false; // Not matched — caller will try alternatives
}

function clientSideMatch(transcript, options) {
  const t = transcript.toLowerCase().trim();

  // Direct keyword map + common Chrome misrecognitions for single-syllable words
  const KEYWORD_MAP = {
    // Clarity + misrecognitions
    'clear': 'CLEAR', 'clearly': 'CLEAR', 'yes': 'CLEAR', 'readable': 'CLEAR',
    'here': 'CLEAR', 'beer': 'CLEAR', 'cheer': 'CLEAR', 'dear': 'CLEAR', 'near': 'CLEAR',
    'saaf': 'CLEAR', 'saaf hai': 'CLEAR', 'haan': 'CLEAR',
    'blurry': 'BLURRY', 'blurred': 'BLURRY', 'blur': 'BLURRY', 'not clear': 'BLURRY',
    'blare': 'BLURRY', 'blaring': 'BLURRY', 'glory': 'BLURRY',
    'dhundhla': 'BLURRY', 'nahi dikh raha': 'BLURRY',
    'repeat': 'REPEAT', 'again': 'REPEAT', 'dobara': 'REPEAT', 'phir se': 'REPEAT',
    // Comparison + misrecognitions
    'one': 'ONE', 'first': 'ONE', 'option 1': 'ONE', 'ek': 'ONE', 'pehla': 'ONE', '1': 'ONE',
    'won': 'ONE', 'want': 'ONE', 'on': 'ONE', 'wan': 'ONE', 'wand': 'ONE',
    'two': 'TWO', 'second': 'TWO', 'option 2': 'TWO', 'do': 'TWO', 'doosra': 'TWO', '2': 'TWO',
    'to': 'TWO', 'too': 'TWO', 'tu': 'TWO', 'who': 'TWO', 'through': 'TWO',
    'same': 'SAME', 'both same': 'SAME', 'equal': 'SAME', 'barabar': 'SAME', 'dono same': 'SAME',
    "can't tell": 'SAME', 'cant tell': 'SAME',
    'sane': 'SAME', 'saint': 'SAME', 'shame': 'SAME', 'came': 'SAME',
    // Duochrome + misrecognitions
    'red': 'RED', 'red one': 'RED', 'laal': 'RED',
    'read': 'RED', 'bread': 'RED', 'wed': 'RED', 'said': 'RED', 'bed': 'RED', 'dead': 'RED',
    'green': 'GREEN', 'green one': 'GREEN', 'hara': 'GREEN',
    'queen': 'GREEN', 'cream': 'GREEN', 'gene': 'GREEN', 'lean': 'GREEN', 'mean': 'GREEN',
    // Binocular + misrecognitions
    'top': 'TOP_CLEARER', 'top one': 'TOP_CLEARER', 'upar': 'TOP_CLEARER',
    'talk': 'TOP_CLEARER', 'tall': 'TOP_CLEARER', 'stop': 'TOP_CLEARER',
    'bottom': 'BOTTOM_CLEARER', 'bottom one': 'BOTTOM_CLEARER', 'neeche': 'BOTTOM_CLEARER',
    'button': 'BOTTOM_CLEARER', 'bought him': 'BOTTOM_CLEARER',
    // Near
    'target ok': 'TARGET_OK', 'ok': 'TARGET_OK', 'fine': 'TARGET_OK',
    'not clear': 'NOT_CLEAR',
  };

  // Try exact match first
  if (KEYWORD_MAP[t] && options.includes(KEYWORD_MAP[t])) {
    return KEYWORD_MAP[t];
  }

  // Try partial match
  for (const [keyword, value] of Object.entries(KEYWORD_MAP)) {
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

// ── Terminal display ──
function showTerminal(data) {
  document.getElementById('questionCard').style.display = 'none';
  const card = document.getElementById('endCard');
  card.classList.add('active');

  if (data.state === 'END') {
    const rx = data.prescription || {};
    const r = rx.right || {};
    const l = rx.left || {};
    const fmt = (v) => v != null ? (v >= 0 ? '+' : '') + parseFloat(v).toFixed(2) : '—';
    const fmtRx = (eye) => `${fmt(eye.sph)} / ${fmt(eye.cyl)} x ${Math.round(eye.axis||180)}${eye.add ? ` ADD ${fmt(eye.add)}` : ''}`;

    document.getElementById('terminalIcon').textContent = '✅';
    document.getElementById('terminalTitle').textContent = 'Congratulations! Your eye test is complete.';
    document.getElementById('terminalSubtitle').innerHTML =
      `<div style="margin-bottom:12px">Here is your final prescription:</div>` +
      `<div style="display:flex;gap:24px;justify-content:center;margin-bottom:16px;">` +
        `<div style="text-align:center"><div style="font-size:0.75rem;color:var(--re-color);font-weight:700;margin-bottom:4px;">RIGHT EYE (RE)</div><div style="font:600 1.1rem var(--font-mono)">${fmtRx(r)}</div></div>` +
        `<div style="text-align:center"><div style="font-size:0.75rem;color:var(--le-color);font-weight:700;margin-bottom:4px;">LEFT EYE (LE)</div><div style="font:600 1.1rem var(--font-mono)">${fmtRx(l)}</div></div>` +
      `</div>` +
      `<div style="font-size:0.85rem;color:var(--ink-secondary)">Please review and sign off below.</div>`;

    // Speak the final prescription
    const sph = (v) => parseFloat(v||0).toFixed(2);
    const ax = (v) => Math.round(v||180);
    const speechText = sessionLanguage === 'hi'
      ? `बधाई हो! आपका आई टेस्ट पूरा हो गया है। दाईं आँख का पावर: sphere ${sph(r.sph)}, cylinder ${sph(r.cyl)}, axis ${ax(r.axis)}। बाईं आँख का पावर: sphere ${sph(l.sph)}, cylinder ${sph(l.cyl)}, axis ${ax(l.axis)}।`
      : `Congratulations! Your eye test is complete. Your right eye power is: sphere ${sph(r.sph)}, cylinder ${sph(r.cyl)}, axis ${ax(r.axis)}. Your left eye power is: sphere ${sph(l.sph)}, cylinder ${sph(l.cyl)}, axis ${ax(l.axis)}.`;
    speakQuestion(speechText);
  } else {
    document.getElementById('terminalIcon').textContent = '⚠️';
    document.getElementById('terminalTitle').textContent = 'Escalation Required';
    document.getElementById('terminalSubtitle').textContent = 'This test requires optometrist review. Please consult with a qualified optometrist.';
    speakQuestion(sessionLanguage === 'hi'
      ? 'इस टेस्ट को ऑप्टोमेट्रिस्ट की समीक्षा की आवश्यकता है।'
      : 'This test requires optometrist review.');
  }
}

// ── Rx Table ──
function updateRxTable(rx) {
  if (!rx) return;
  const r = rx.right || {};
  const l = rx.left || {};
  document.getElementById('rxReSph').textContent = fmtD(r.sph);
  document.getElementById('rxReCyl').textContent = fmtD(r.cyl);
  document.getElementById('rxReAxis').textContent = r.axis != null ? Math.round(r.axis) : '—';
  document.getElementById('rxReAdd').textContent = r.add ? fmtD(r.add) : '—';
  document.getElementById('rxLeSph').textContent = fmtD(l.sph);
  document.getElementById('rxLeCyl').textContent = fmtD(l.cyl);
  document.getElementById('rxLeAxis').textContent = l.axis != null ? Math.round(l.axis) : '—';
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

// ── Submit response ──
async function submitResponse(responseValue, voiceMeta) {
  if (!sessionId) return;

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
  if (!voiceMeta) {
    addToConversation('patient', responseValue, responseValue,
      currentState ? `${currentState.state}:${currentState.step}` : '');
  }

  showLoading(true);
  try {
    const resp = await fetch(`${API}/api/session/${sessionId}/respond`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ response: responseValue, voice_meta: voiceMeta || null, language: sessionLanguage }),
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
    autoUpdatePip();
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    showLoading(false);
  }
}

// ── Keyboard shortcuts ──
function handleKeyboard(e) {
  if (!_inputEnabled || speechSynthesis.speaking) return;
  // Number keys 1-9 for options
  if (e.key >= '1' && e.key <= '9') {
    const idx = parseInt(e.key) - 1;
    const btns = document.querySelectorAll('#optionsGrid .option-btn');
    if (btns[idx]) {
      e.preventDefault();
      btns[idx].click();
    }
  }
}

// ── Gamepad input (Xbox controller via Chrome Gamepad API) ──
// B=option1, A=option2, X=option3, Y=REPEAT (always)
// Standard indices: A=0, B=1, X=2, Y=3

const GAMEPAD_FACE_BUTTONS = [1, 0, 2]; // B, A, X → option indices 0, 1, 2
const GAMEPAD_REPEAT_BUTTON = 3; // Y → always REPEAT

window.addEventListener('gamepadconnected', (e) => {
  console.log(`[Gamepad] Connected: ${e.gamepad.id}`);
  gamepadIndex = e.gamepad.index;
  gamepadConnected = true;
  _gamepadPrevButtons = [false, false, false, false];
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
      // Check face buttons B, A, X (options 0, 1, 2)
      GAMEPAD_FACE_BUTTONS.forEach((btnIdx, optionIdx) => {
        const pressed = gp.buttons[btnIdx]?.pressed || false;
        if (pressed && !_gamepadPrevButtons[optionIdx]) {
          handleGamepadOption(optionIdx);
        }
        _gamepadPrevButtons[optionIdx] = pressed;
      });
      // Check Y button (always REPEAT)
      const yPressed = gp.buttons[GAMEPAD_REPEAT_BUTTON]?.pressed || false;
      if (yPressed && !_gamepadPrevButtons[3]) {
        handleGamepadRepeat();
      }
      _gamepadPrevButtons[3] = yPressed;
    }
    _gamepadPollId = requestAnimationFrame(poll);
  }
  _gamepadPollId = requestAnimationFrame(poll);
}

function handleGamepadOption(optionIdx) {
  if (!_inputEnabled || _flipState === 'flip1' || speechSynthesis.speaking) return;
  // Get non-REPEAT option buttons from DOM
  const allBtns = document.querySelectorAll('#optionsGrid .option-btn');
  const nonRepeatBtns = [];
  for (const btn of allBtns) {
    // Check if this button's response value is REPEAT
    const text = btn.textContent.trim().toUpperCase();
    if (!text.startsWith('REPEAT') && !text.startsWith('फिर से')) {
      nonRepeatBtns.push(btn);
    }
  }
  if (optionIdx < nonRepeatBtns.length && !nonRepeatBtns[optionIdx].disabled) {
    console.log(`[Gamepad] Button ${optionIdx} → "${nonRepeatBtns[optionIdx].textContent.trim()}"`);
    nonRepeatBtns[optionIdx].click();
  }
}

function handleGamepadRepeat() {
  if (!_inputEnabled || _flipState === 'flip1' || speechSynthesis.speaking) return;
  // Find the REPEAT button in the DOM
  const allBtns = document.querySelectorAll('#optionsGrid .option-btn');
  for (const btn of allBtns) {
    const text = btn.textContent.trim().toUpperCase();
    if (text.startsWith('REPEAT') || text.startsWith('फिर से')) {
      if (!btn.disabled) {
        console.log('[Gamepad] Y → REPEAT');
        btn.click();
      }
      return;
    }
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
  showLoading(true);
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
  } finally {
    showLoading(false);
  }
}

async function discardSession() {
  if (!confirm('Discard this session? No data will be saved.')) return;
  try {
    await fetch(`${API}/api/session/${sessionId}/discard`, { method: 'POST' });
  } catch (e) { /* ignore */ }
  cleanup();
}

function cleanup() {
  if (heartbeatInterval) clearInterval(heartbeatInterval);
  sessionStorage.removeItem('session_id');
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
    const cats = {
      'Patient Profile': ['dv_age_bucket', 'dv_distance_priority', 'dv_near_priority'],
      'Risk Assessment': ['dv_symptom_risk_level', 'dv_medical_risk_level', 'dv_stability_level', 'dv_anomaly_watch', 'dv_requires_optom_review'],
      'AR/Lenso': ['dv_ar_lenso_mismatch_level_RE', 'dv_ar_lenso_mismatch_level_LE', 'dv_start_source_policy'],
      'Starting Rx': ['dv_start_rx_RE_sph', 'dv_start_rx_RE_cyl', 'dv_start_rx_RE_axis', 'dv_start_rx_LE_sph', 'dv_start_rx_LE_cyl', 'dv_start_rx_LE_axis'],
      'Test Config': ['dv_target_distance_va', 'dv_endpoint_bias_policy', 'dv_step_size_policy', 'dv_confidence_requirement', 'dv_expected_convergence_time'],
      'Fogging': ['dv_fogging_policy', 'dv_fogging_amount_D', 'dv_fogging_clearance_mode', 'dv_fogging_required'],
      'Near Vision': ['dv_add_expected', 'dv_near_test_required'],
      'Safety': ['dv_max_delta_from_start_sph', 'dv_max_delta_from_ar_sph', 'dv_axis_tolerance_deg', 'dv_cyl_tolerance_D'],
    };
    let html = '';
    for (const [cat, keys] of Object.entries(cats)) {
      html += `<div style="font-weight:700;color:var(--accent);margin-top:10px;font-size:0.8rem">${cat}</div>`;
      for (const k of keys) {
        const v = dv[k];
        if (v !== undefined && v !== null) {
          html += `<div><span style="color:var(--ink-muted)">${k.replace(/^dv_/,'').replace(/_/g,' ')}:</span> <strong>${v}</strong></div>`;
        }
      }
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
  if (!ttsEnabled) speechSynthesis.cancel();
}

// ── Phoropter auto-dispatch toggle ──
let phoropterEnabled = true;

async function togglePhoropter() {
  phoropterEnabled = !phoropterEnabled;
  const btn = document.getElementById('phoropBtn');
  if (btn) {
    btn.textContent = `Phoropter: ${phoropterEnabled ? 'ON' : 'OFF'}`;
    btn.style.background = phoropterEnabled ? 'rgba(34,197,94,0.3)' : '';
  }
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
  if (autoScreenshot) refreshPipScreenshot();
}

async function refreshPipScreenshot() {
  if (!sessionId) return;
  const loading = document.getElementById('pipLoading');
  const img = document.getElementById('pipImg');
  const footer = document.getElementById('pipFooter');
  if (loading) { loading.style.display = 'flex'; loading.textContent = 'Capturing...'; }

  try {
    const resp = await fetch(`${API}/api/session/${sessionId}/screenshot`, {
      method: 'POST',
      signal: AbortSignal.timeout(8000), // 8s timeout
    });
    if (resp.ok) {
      const data = await resp.json();
      if (data.screenshot) {
        img.src = 'data:image/jpeg;base64,' + data.screenshot;
        img.style.display = '';
        if (loading) loading.style.display = 'none';
        if (footer) footer.textContent = new Date().toLocaleTimeString();
        return;
      }
    }
    // Non-OK or no screenshot
    if (loading) loading.textContent = 'Device not connected';
    if (footer) footer.textContent = 'Phoropter offline';
  } catch (e) {
    console.warn('PIP screenshot failed:', e);
    if (loading) loading.textContent = 'Device not reachable';
    if (footer) footer.textContent = 'Connection timeout';
  }
}

// Auto-update PIP after each response (called from submitResponse)
async function autoUpdatePip() {
  if (!autoScreenshot) return;
  // Small delay to let phoropter finish physical movement
  setTimeout(refreshPipScreenshot, 500);
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

  if (!data.auto_flip) {
    _flipState = null;
    updateFlipIndicator(null);
    setOptionsEnabled(true);
    return;
  }

  const isAxis = data.state === 'E' || data.state === 'H';
  const phaseLabel = isAxis ? 'axis' : 'power';
  const eyeLabel = (data.state === 'E' || data.state === 'F') ? 'right eye' : 'left eye';
  const eyeLabelHi = (data.state === 'E' || data.state === 'F') ? 'दाईं आँख' : 'बाईं आँख';

  // ── Flip 1: Show + speak "This is Flip 1", WAIT for TTS, THEN start 2s timer ──
  _flipState = 'flip1';
  updateFlipIndicator('flip1');
  setOptionsEnabled(false);

  const flip1Text = sessionLanguage === 'hi'
    ? `यह विकल्प 1 है। ध्यान से देखिए।`
    : `This is Dot Chart. First option. Look carefully.`;
  document.getElementById('questionText').textContent = flip1Text;
  speakQuestion(flip1Text);

  const waitSeconds = data.flip_wait_seconds || 2;

  // Wait for Flip 1 TTS to finish, THEN wait 2s, THEN flip
  const waitForFlip1Speech = () => {
    if (speechSynthesis.speaking) {
      setTimeout(waitForFlip1Speech, 100);
      return;
    }
    // Flip 1 speech done — now wait the observation period
    _autoFlipTimer = setTimeout(doFlip2, waitSeconds * 1000);
  };
  setTimeout(waitForFlip1Speech, 200);

  async function doFlip2() {
    // Send handle command to flip to position 2
    if (sessionId) {
      try {
        await fetch(`${API}/api/session/${sessionId}/jcc-flip`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        });
      } catch (e) { console.warn('JCC flip failed:', e); }
    }

    // ── Flip 2: Show + speak "This is Flip 2. Which is better?" ──
    _flipState = 'flip2';
    updateFlipIndicator('flip2');

    const flip2Text = sessionLanguage === 'hi'
      ? `यह विकल्प 2 है। कौन सा बेहतर है? पहला, दूसरा, समान, या फिर से कहिए।`
      : `This is second option. Which is better? Say first, second, same, or repeat.`;
    document.getElementById('questionText').textContent = flip2Text;
    speakQuestion(flip2Text);

    // Wait for TTS to finish, then beep + enable buttons + listen
    const waitAndEnable = () => {
      if (speechSynthesis.speaking) {
        setTimeout(waitAndEnable, 100);
      } else {
        setOptionsEnabled(true);
        playBeep().then(() => {
          _inputEnabled = true; // Enable ALL input after Flip 2 beep
          if (voiceEnabled && currentState) {
            startVoiceCapture(currentState.state, currentState.options || [], currentState.step);
          }
        });
      }
    };
    setTimeout(waitAndEnable, 200);
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
function toggleSidebarSection(sectionId) {
  const section = document.getElementById(sectionId);
  if (!section) return;
  section.classList.toggle('collapsed');
  const icon = section.querySelector('.collapse-icon');
  if (icon) icon.textContent = section.classList.contains('collapsed') ? '▶' : '▼';
}

function showLoading(show) {
  document.getElementById('loadingOverlay').classList.toggle('active', show);
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
