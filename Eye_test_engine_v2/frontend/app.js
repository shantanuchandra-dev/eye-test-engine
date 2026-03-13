// Eye Test Engine v2 — Frontend Application
// Integrates with Flask API backend and Phoropter broker

const CONFIG = {
    backendUrl: (typeof window !== 'undefined' && window.BACKEND_URL)
        ? window.BACKEND_URL
        : ((typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'))
            ? 'http://localhost:5050' : ''),
    phoropterUrl: 'https://rajasthan-royals.preprod.lenskart.com',
    get phoropterId() {
        const el = document.getElementById('phoropterIdInput');
        return (el && el.value.trim()) ? el.value.trim() : 'phoropter-1';
    }
};

// ── Request logging (curl commands) ─────────────────────────────────────
function buildCurlFromFetch(url, options) {
    const u = typeof url === 'string' ? url : (url.url || '');
    const absUrl = u.startsWith('http') ? u : new URL(u, window.location.origin).href;
    const method = (options && options.method) ? String(options.method).toUpperCase() : 'GET';
    const headers = options && options.headers ? (options.headers instanceof Headers
        ? Object.fromEntries(options.headers.entries())
        : options.headers) : {};
    let body = options && options.body !== undefined ? options.body : null;
    if (typeof body === 'object' && body !== null && !(body instanceof String)) {
        try { body = JSON.stringify(body); } catch (_) { body = String(body); }
    }
    const escapeShell = (s) => (s || '').replace(/'/g, "'\\''");
    const parts = ['curl -X ' + method + " '" + escapeShell(absUrl) + "'"];
    Object.keys(headers).forEach((k) => {
        const v = headers[k];
        if (v !== undefined && v !== null) {
            parts.push("-H '" + escapeShell(k + ': ' + v) + "'");
        }
    });
    if (body != null && body !== '' && method !== 'GET') {
        parts.push("-d '" + escapeShell(body) + "'");
    }
    return parts.join(' \\\n  ');
}

function addRequestLog(curlString, method, url) {
    if (typeof document === 'undefined') return;
    const container = document.getElementById('requestLogsContent');
    if (!container) return;
    const placeholder = document.getElementById('requestLogsPlaceholder');
    if (placeholder) placeholder.style.display = 'none';
    const absUrl = url.startsWith('http') ? url : new URL(url, window.location.origin).href;
    const time = new Date().toLocaleTimeString(undefined, { hour12: false });
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <div class="log-entry-header">
            <span class="log-entry-method">${method}</span>
            <span>${time}</span>
            <span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${absUrl}">${absUrl}</span>
        </div>
        <pre class="log-entry-curl">${(curlString || '').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>
    `;
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
}

function clearRequestLogs() {
    const container = document.getElementById('requestLogsContent');
    const placeholder = document.getElementById('requestLogsPlaceholder');
    if (container) {
        container.innerHTML = '';
        if (placeholder) {
            container.appendChild(placeholder);
            placeholder.style.display = 'block';
            placeholder.textContent = 'No requests yet.';
        }
    }
}

// ── Logs panel: password gate, minimize by default ──────────────────────
const LOGS_PASSWORD = 'Shantanu';
const LOGS_UNLOCK_HOURS = 24;
const LOGS_STORAGE_DENIED = 'eyeTest_logsUnlockDenied';
const LOGS_STORAGE_UNTIL = 'eyeTest_logsUnlockUntil';

function isLogsAccessDenied() {
    try { return localStorage.getItem(LOGS_STORAGE_DENIED) === 'true'; } catch (_) { return false; }
}
function getLogsUnlockUntil() {
    try { const v = localStorage.getItem(LOGS_STORAGE_UNTIL); return v ? parseInt(v, 10) : 0; } catch (_) { return 0; }
}
function setLogsUnlockUntil(untilMs) {
    try { localStorage.setItem(LOGS_STORAGE_UNTIL, String(untilMs)); } catch (_) {}
}
function setLogsAccessDenied() {
    try { localStorage.setItem(LOGS_STORAGE_DENIED, 'true'); } catch (_) {}
}
function isLogsUnlocked() {
    if (isLogsAccessDenied()) return false;
    return getLogsUnlockUntil() > Date.now();
}

function openLogsPasswordModal() {
    const modal = document.getElementById('logsPasswordModal');
    const input = document.getElementById('logsPasswordInput');
    const err = document.getElementById('logsPasswordError');
    if (modal) modal.classList.add('active');
    if (input) { input.value = ''; input.focus(); }
    if (err) err.classList.remove('visible');
}

function closeLogsPasswordModal() {
    const modal = document.getElementById('logsPasswordModal');
    if (modal) modal.classList.remove('active');
}

function submitLogsPassword() {
    const input = document.getElementById('logsPasswordInput');
    const err = document.getElementById('logsPasswordError');
    const value = (input && input.value) ? input.value.trim() : '';
    if (value !== LOGS_PASSWORD) {
        setLogsAccessDenied();
        if (err) { err.textContent = 'Incorrect. Logs access disabled.'; err.classList.add('visible'); }
        closeLogsPasswordModal();
        const tab = document.getElementById('logsTabBtn');
        if (tab) tab.classList.add('hidden');
        return;
    }
    setLogsUnlockUntil(Date.now() + LOGS_UNLOCK_HOURS * 3600000);
    closeLogsPasswordModal();
    openLogsPanel();
}

function requestShowLogsPanel() {
    if (isLogsAccessDenied()) return;
    if (isLogsUnlocked()) { openLogsPanel(); return; }
    openLogsPasswordModal();
}

function openLogsPanel() {
    const main = document.querySelector('.main-content');
    const tab = document.getElementById('logsTabBtn');
    if (main) main.classList.add('logs-expanded');
    if (tab) tab.classList.add('hidden');
}

function closeLogsPanel() {
    const main = document.querySelector('.main-content');
    const tab = document.getElementById('logsTabBtn');
    if (main) main.classList.remove('logs-expanded');
    if (tab && !isLogsAccessDenied()) tab.classList.remove('hidden');
}

// ── Debug (Derived Variables) panel ─────────────────────────────────────
function requestShowDebugPanel() {
    if (isLogsAccessDenied()) return;
    if (isLogsUnlocked()) { openDebugPanel(); return; }
    openLogsPasswordModal();
}

function openDebugPanel() {
    const main = document.querySelector('.main-content');
    const tab = document.getElementById('debugTabBtn');
    if (main) main.classList.add('debug-expanded');
    if (tab) tab.classList.add('hidden');
    refreshDerivedVariables();
}

function closeDebugPanel() {
    const main = document.querySelector('.main-content');
    const tab = document.getElementById('debugTabBtn');
    if (main) main.classList.remove('debug-expanded');
    if (tab && !isLogsAccessDenied()) tab.classList.remove('hidden');
}

async function refreshDerivedVariables() {
    if (!sessionState.sessionId) return;
    try {
        const resp = await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/derived-variables`);
        if (!resp.ok) return;
        const data = await resp.json();
        renderDerivedVariables(data);
    } catch (e) {
        console.warn('Failed to fetch derived variables:', e);
    }
}

function renderDerivedVariables(data) {
    const container = document.getElementById('debugContent');
    if (!container) return;

    let html = '';

    if (data.derived_variables && Object.keys(data.derived_variables).length > 0) {
        html += '<div class="debug-section"><div class="debug-section-title">Derived Variables</div>';
        for (const [key, val] of Object.entries(data.derived_variables)) {
            html += `<div class="debug-row"><span class="debug-key">${key}</span><span class="debug-val">${val}</span></div>`;
        }
        html += '</div>';
    }

    if (data.working_variables && Object.keys(data.working_variables).length > 0) {
        html += '<div class="debug-section"><div class="debug-section-title">Working Variables</div>';
        for (const [key, val] of Object.entries(data.working_variables)) {
            html += `<div class="debug-row"><span class="debug-key">${key}</span><span class="debug-val">${val}</span></div>`;
        }
        html += '</div>';
    }

    if (data.calibration && Object.keys(data.calibration).length > 0) {
        html += '<div class="debug-section"><div class="debug-section-title">Calibration</div>';
        for (const [key, val] of Object.entries(data.calibration)) {
            html += `<div class="debug-row"><span class="debug-key">${key}</span><span class="debug-val">${val}</span></div>`;
        }
        html += '</div>';
    }

    container.innerHTML = html || '<div style="padding:10px;color:#888;">No data yet.</div>';
}

// ── Fetch interception for request logs ─────────────────────────────────
const _originalFetch = window.fetch;
window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    const options = init || (input && typeof input === 'object' ? {
        method: input.method, headers: input.headers, body: input.body
    } : {});
    const method = (options.method || 'GET').toUpperCase();
    const absUrl = url.startsWith('http') ? url : new URL(url, window.location.origin).href;
    try {
        const curl = buildCurlFromFetch(input, options);
        addRequestLog(curl, method, absUrl);
    } catch (e) { console.warn('Log build failed:', e); }
    return _originalFetch.call(this, input, init);
};

// ── Session State ───────────────────────────────────────────────────────
let sessionState = {
    sessionId: null,
    currentPhase: null,
    currentChart: null,
    intentsLocked: false,
    responseCount: 0,
    history: [],
    lastResponse: null,
};

let _deviceAcquired = false;
let _configReady = false;
let operatorName = '';
let _cachedClientIp = null;

// ── Phase progress tracker ──────────────────────────────────────────────
const FSM_STATES_ORDER = ['A','B','E','F','G','D','H','I','J','K','P','Q','R','END'];
const FSM_STATE_NAMES = {
    'A': 'Distance Baseline',
    'B': 'Coarse Sphere RE',
    'E': 'JCC Axis RE',
    'F': 'JCC Power RE',
    'G': 'Duochrome RE',
    'D': 'Coarse Sphere LE',
    'H': 'JCC Axis LE',
    'I': 'JCC Power LE',
    'J': 'Duochrome LE',
    'K': 'Binocular Balance',
    'P': 'Near Add RE',
    'Q': 'Near Add LE',
    'R': 'Near Binocular',
    'END': 'Complete',
    'ESCALATE': 'Escalation',
};

// Response type color mapping
const RESPONSE_COLORS = {
    'READABLE': '#4caf50',
    'NOT_READABLE': '#f44336',
    'BLURRY': '#ff9800',
    'BETTER_1': '#2196f3',
    'BETTER_2': '#2196f3',
    'SAME': '#9e9e9e',
    'CANT_TELL': '#9e9e9e',
    'RED_CLEARER': '#f44336',
    'GREEN_CLEARER': '#4caf50',
    'EQUAL': '#9e9e9e',
    'TOP_CLEARER': '#2196f3',
    'BOTTOM_CLEARER': '#2196f3',
    'TARGET_OK': '#4caf50',
    'NOT_CLEAR': '#f44336',
};

// ── Config & Device Management ──────────────────────────────────────────
const TEST_DEVICE_ID = 'Test';

function isTestDeviceId(deviceId) {
    return String(deviceId || CONFIG.phoropterId || '').trim().toLowerCase() === TEST_DEVICE_ID.toLowerCase();
}

async function fetchConfig() {
    try {
        const resp = await fetch('/api/config');
        if (resp.ok) {
            const data = await resp.json();
            if (data.backend_url) CONFIG.backendUrl = data.backend_url;
            if (data.phoropter_base_url) CONFIG.phoropterUrl = data.phoropter_base_url;
        }
    } catch (err) {
        console.warn('[CONFIG] Could not fetch:', err);
    } finally {
        _configReady = true;
    }
}

async function fetchDevices() {
    const select = document.getElementById('phoropterIdInput');
    if (!select) return;

    if (_deviceAcquired) {
        const id = localStorage.getItem('phoropterId') || '';
        if (id) {
            select.innerHTML = '';
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = isTestDeviceId(id) ? `${id} (test mode)` : `${id} (connected)`;
            opt.selected = true;
            select.appendChild(opt);
            select.disabled = true;
        }
        return;
    }

    try {
        const resp = await fetch(`${CONFIG.backendUrl}/api/devices`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const devices = Array.isArray(data) ? data : (data.devices || []);
        select.innerHTML = '';
        const savedId = localStorage.getItem('phoropterId');
        const deviceIds = devices.map(d => d.device_id || d.id || d.name || '').filter(Boolean);
        if (!deviceIds.some(id => id.toLowerCase() === TEST_DEVICE_ID.toLowerCase())) {
            deviceIds.unshift(TEST_DEVICE_ID);
        }
        if (deviceIds.length === 0) deviceIds.push(TEST_DEVICE_ID);
        deviceIds.forEach(id => {
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = isTestDeviceId(id) ? `${id} (safe mode)` : id;
            if (id === savedId) opt.selected = true;
            select.appendChild(opt);
        });
        if (!select.value && deviceIds.length > 0) select.value = deviceIds[0];
        if (select.value) localStorage.setItem('phoropterId', select.value);
        onDeviceSelectionChanged();
    } catch (err) {
        console.warn('Could not fetch devices:', err);
        select.innerHTML = `<option value="${TEST_DEVICE_ID}">${TEST_DEVICE_ID} (safe mode)</option><option value="phoropter-1">phoropter-1</option>`;
        const savedId = localStorage.getItem('phoropterId');
        select.value = savedId || TEST_DEVICE_ID;
        localStorage.setItem('phoropterId', select.value);
        onDeviceSelectionChanged();
    }
}

function onDeviceSelectionChanged() {
    const select = document.getElementById('phoropterIdInput');
    const acquireBtn = document.getElementById('acquireDeviceBtn');
    if (!select) return;
    const id = select.value;
    if (id) {
        localStorage.setItem('phoropterId', id);
        if (acquireBtn && !_deviceAcquired && !isTestDeviceId(id)) {
            acquireBtn.style.display = 'inline-block';
        } else if (acquireBtn) {
            acquireBtn.style.display = 'none';
        }
    }
}

async function getClientIp() {
    if (_cachedClientIp) return _cachedClientIp;
    try {
        const resp = await fetch('https://api.ipify.org?format=json');
        const data = await resp.json();
        _cachedClientIp = data.ip || 'unknown';
    } catch { _cachedClientIp = 'unknown'; }
    return _cachedClientIp;
}

async function getBrainId() {
    const ip = await getClientIp();
    const name = (operatorName || 'unknown').replace(/\s+/g, '_');
    return `${name}@${ip}`;
}

async function acquireSelectedDevice() {
    const deviceId = CONFIG.phoropterId;
    if (!deviceId) return;
    if (isTestDeviceId(deviceId)) {
        _deviceAcquired = true;
        const select = document.getElementById('phoropterIdInput');
        const btn = document.getElementById('acquireDeviceBtn');
        if (select) select.disabled = true;
        if (btn) btn.style.display = 'none';
        addToHistory('Test mode active: phoropter commands skipped', 'info');
        return;
    }
    const btn = document.getElementById('acquireDeviceBtn');
    if (btn) { btn.disabled = true; btn.textContent = 'Acquiring...'; }
    try {
        const brainId = await getBrainId();
        const resp = await fetch(`${CONFIG.backendUrl}/api/devices/${deviceId}/acquire`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ brain_id: brainId, name: operatorName || 'Eye Test UI' })
        });
        const data = await resp.json();
        if (resp.ok) {
            _deviceAcquired = true;
            if (btn) btn.style.display = 'none';
            document.getElementById('phoropterIdInput').disabled = true;
        } else {
            alert(`Could not acquire ${deviceId}: ${data.error || data.reason || resp.status}`);
            if (btn) { btn.disabled = false; btn.textContent = 'Acquire'; }
        }
    } catch (err) {
        alert(`Failed to acquire device: ${err.message}`);
        if (btn) { btn.disabled = false; btn.textContent = 'Acquire'; }
    }
}

async function releaseDevice() {
    const deviceId = CONFIG.phoropterId;
    if (!deviceId) return;
    if (isTestDeviceId(deviceId)) {
        _deviceAcquired = false;
        const select = document.getElementById('phoropterIdInput');
        if (select) select.disabled = false;
        return;
    }
    try {
        const brainId = await getBrainId();
        await fetch(`${CONFIG.backendUrl}/api/devices/${deviceId}/release`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ brain_id: brainId })
        });
        addToHistory('Device released', 'success');
    } catch (err) {
        addToHistory('Release failed: ' + err.message, 'warning');
    }
    _deviceAcquired = false;
    const select = document.getElementById('phoropterIdInput');
    if (select) select.disabled = false;
    const btn = document.getElementById('acquireDeviceBtn');
    if (btn) { btn.style.display = 'inline-block'; btn.disabled = false; btn.textContent = 'Acquire'; }
}

// ── Session Persistence ─────────────────────────────────────────────────
const SESSION_STORAGE_KEY = 'eyeTestV2Session';

function _saveSessionToStorage() {
    if (!sessionState.sessionId) return;
    try {
        sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify({
            sessionId: sessionState.sessionId,
            responseCount: sessionState.responseCount,
            deviceAcquired: _deviceAcquired,
            deviceId: CONFIG.phoropterId,
        }));
    } catch (e) { console.warn('sessionStorage write failed:', e); }
}

function _clearSessionStorage() {
    try { sessionStorage.removeItem(SESSION_STORAGE_KEY); } catch (_) {}
}

async function _tryRestoreSession() {
    let saved;
    try { saved = JSON.parse(sessionStorage.getItem(SESSION_STORAGE_KEY)); } catch (_) { return false; }
    if (!saved || !saved.sessionId) return false;

    try {
        const statusResp = await fetch(`${CONFIG.backendUrl}/api/session/${saved.sessionId}/status`);
        if (!statusResp.ok) {
            _clearSessionStorage();
            return false;
        }

        if (saved.deviceAcquired && saved.deviceId && !isTestDeviceId(saved.deviceId)) {
            const brainId = await getBrainId();
            const hbResp = await fetch(`${CONFIG.backendUrl}/api/devices/${saved.deviceId}/heartbeat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ brain_id: brainId })
            });
            if (hbResp.status !== 200 && hbResp.status !== 202) {
                _clearSessionStorage();
                return false;
            }
        }

        const data = await statusResp.json();
        sessionState.sessionId = saved.sessionId;
        sessionState.responseCount = saved.responseCount || data.total_rows || 0;

        document.getElementById('testScreen').style.display = 'block';
        updateSessionInfo(data);
        displayQuestion(data);
        updateStatusIndicator(true);
        updatePhaseProgress(data.state);

        if (saved.deviceAcquired && saved.deviceId) {
            _deviceAcquired = true;
            const select = document.getElementById('phoropterIdInput');
            if (select) { select.value = saved.deviceId; select.disabled = true; }
            const acqBtn = document.getElementById('acquireDeviceBtn');
            if (acqBtn) acqBtn.style.display = 'none';
        }

        addToHistory('Session restored after refresh', 'info');
        return true;
    } catch (err) {
        console.warn('Session restore failed:', err);
        _clearSessionStorage();
        return false;
    }
}

// ── Optometrist Name ────────────────────────────────────────────────────
const OPTOMETRIST_CACHE_KEY = 'optometristName';
const OPTOMETRIST_TS_KEY = 'optometristNameTimestamp';
const OPTOMETRIST_TTL_KEY = 'optometristTTL';

function checkOptometristName() {
    const cached = localStorage.getItem(OPTOMETRIST_CACHE_KEY);
    const ts = parseInt(localStorage.getItem(OPTOMETRIST_TS_KEY) || '0', 10);
    const ttlHours = parseInt(localStorage.getItem(OPTOMETRIST_TTL_KEY) || '0', 10);
    const ttlMs = ttlHours * 3600000;
    const expired = ttlHours === 0 || (Date.now() - ts) > ttlMs;

    if (cached && !expired) {
        operatorName = cached;
        return true;
    }
    operatorName = '';
    const modal = document.getElementById('optometristModal');
    if (modal) {
        modal.classList.add('active');
        const input = document.getElementById('optometristNameInput');
        if (input) setTimeout(() => input.focus(), 200);
    }
    return false;
}

function saveOptometristName() {
    const input = document.getElementById('optometristNameInput');
    const name = (input ? input.value.trim() : '');
    const persistenceSelect = document.getElementById('optometristPersistenceSelect');
    const ttlHours = persistenceSelect ? parseInt(persistenceSelect.value, 10) : 0;
    if (!name) {
        input.style.borderColor = '#f44336';
        input.placeholder = 'Name is required';
        return;
    }
    operatorName = name;
    localStorage.setItem(OPTOMETRIST_CACHE_KEY, name);
    localStorage.setItem(OPTOMETRIST_TS_KEY, String(Date.now()));
    localStorage.setItem(OPTOMETRIST_TTL_KEY, String(ttlHours));
    const modal = document.getElementById('optometristModal');
    if (modal) modal.classList.remove('active');
}

// ── Manual Refraction Controls ──────────────────────────────────────────
let manualControlsLocked = false;
let _manualAutoUnlockTimer = null;
let typeModeActive = false;
let _typeModeEditing = false;
let _phoropterBusy = false;

function _setPhoropterBusy(busy) {
    _phoropterBusy = busy;
    const cells = document.querySelectorAll('.rt-val');
    if (busy) cells.forEach(c => c.classList.add('locked'));
    else if (!manualControlsLocked) cells.forEach(c => c.classList.remove('locked'));
}

function _setIntentsDisabled(disabled) {
    document.querySelectorAll('.intent-button').forEach(btn => {
        btn.disabled = disabled;
        btn.style.opacity = disabled ? '0.45' : '';
    });
}

function _setManualLock(locked) {
    manualControlsLocked = locked;
    const btn = document.getElementById('manualLockBtn');
    const cells = document.querySelectorAll('.rt-val');
    if (locked) {
        if (btn) { btn.textContent = '\u{1F512}'; btn.classList.add('locked'); }
        cells.forEach(c => c.classList.add('locked'));
    } else {
        if (btn) { btn.textContent = '\u{1F513}'; btn.classList.remove('locked'); }
        cells.forEach(c => c.classList.remove('locked'));
    }
}

function toggleManualLock() {
    if (_manualAutoUnlockTimer) { clearTimeout(_manualAutoUnlockTimer); _manualAutoUnlockTimer = null; }
    if (!manualControlsLocked && typeModeActive) _exitTypeMode();
    _setManualLock(!manualControlsLocked);
}

// ── Type Mode ───────────────────────────────────────────────────────────
function toggleTypeMode() {
    if (typeModeActive) _exitTypeMode();
    else _enterTypeMode();
}

function _enterTypeMode() {
    typeModeActive = true;
    if (manualControlsLocked) _setManualLock(false);
    const btn = document.getElementById('typeModeBtn');
    if (btn) btn.classList.add('active');
    document.querySelectorAll('.rt-val[data-param]').forEach(el => {
        el.classList.add('type-mode');
        el.setAttribute('data-tip', 'Click to type value');
    });
}

function _exitTypeMode() {
    _commitActiveInput(false);
    typeModeActive = false;
    _typeModeEditing = false;
    const btn = document.getElementById('typeModeBtn');
    if (btn) btn.classList.remove('active');
    document.querySelectorAll('.rt-val[data-param]').forEach(el => {
        el.classList.remove('type-mode');
        el.setAttribute('data-tip', 'Right-click = +  |  Left-click = \u2212');
    });
}

function _getTypableFields() {
    return Array.from(document.querySelectorAll('.rt-val[data-param]'));
}

function _openInputInCell(cell) {
    if (!cell || cell.querySelector('.rt-type-input')) return;
    _typeModeEditing = true;
    const originalText = cell.textContent.trim();
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'rt-type-input';
    input.value = originalText.replace(/[+\u00b0]/g, '');
    input.dataset.originalValue = originalText;
    if (cell.dataset.param === 'axis') { input.inputMode = 'numeric'; input.pattern = '[0-9]*'; }
    else { input.inputMode = 'decimal'; }
    input.addEventListener('keydown', (e) => _handleTypeInputKey(e, cell, input));
    input.addEventListener('blur', () => {
        setTimeout(() => {
            if (cell.contains(input)) { _restoreCell(cell, input.dataset.originalValue); _typeModeEditing = false; }
        }, 50);
    });
    cell.textContent = '';
    cell.appendChild(input);
    input.focus();
    input.select();
}

function _restoreCell(cell, text) {
    const input = cell.querySelector('.rt-type-input');
    if (input) input.remove();
    cell.textContent = text;
    _typeModeEditing = false;
}

function _handleTypeInputKey(e, cell, input) {
    if (e.key === 'Enter') { e.preventDefault(); _commitActiveInput(true); }
    else if (e.key === 'Escape') { e.preventDefault(); _restoreCell(cell, input.dataset.originalValue); }
    else if (e.key === 'Tab') {
        e.preventDefault();
        const parsed = _parseTypedValue(cell.dataset.param, input.value);
        _restoreCell(cell, parsed !== null ? _formatCellValue(cell.dataset.param, parsed) : input.dataset.originalValue);
        const fields = _getTypableFields();
        const idx = fields.indexOf(cell);
        _openInputInCell(fields[(idx + 1) % fields.length]);
    }
}

function _parseTypedValue(param, raw) {
    const s = raw.replace(/[+\u00b0\s]/g, '').trim();
    if (s === '') return null;
    const n = parseFloat(s);
    if (isNaN(n)) return null;
    if (param === 'axis') { const r = Math.round(n / 5) * 5; return (r <= 0 || r > 180) ? 180 : r; }
    if (param === 'cyl') { if (n > 0) return 0; return Math.round(n * 4) / 4; }
    if (param === 'add') { if (n < 0) return 0; return Math.round(n * 4) / 4; }
    return Math.round(n * 4) / 4;
}

function _normalizeAxisDisplay(val) {
    const n = Math.round(parseFloat(val) || 0);
    return (n === 0 || n === 180) ? '180' : String(n);
}

function _formatCellValue(param, val) {
    if (param === 'axis') return _normalizeAxisDisplay(val);
    if (param === 'add') return '+' + val.toFixed(2);
    return (val >= 0 ? '+' : '') + val.toFixed(2);
}

async function _commitActiveInput(submit) {
    if (!submit) {
        document.querySelectorAll('.rt-type-input').forEach(input => {
            const cell = input.parentElement;
            if (cell) _restoreCell(cell, input.dataset.originalValue);
        });
        _typeModeEditing = false;
        return;
    }

    const fields = _getTypableFields();
    const power = { right: { sph: 0, cyl: 0, axis: 180, add: 0 }, left: { sph: 0, cyl: 0, axis: 180, add: 0 } };
    const oldPower = sessionState.lastResponse?.power || { right: { sph: 0, cyl: 0, axis: 180, add: 0 }, left: { sph: 0, cyl: 0, axis: 180, add: 0 } };
    let anyChanged = false;

    for (const cell of fields) {
        const eye = cell.dataset.eye === 'R' ? 'right' : 'left';
        const param = cell.dataset.param;
        const input = cell.querySelector('.rt-type-input');
        const rawText = input ? input.value : cell.textContent;
        const parsed = _parseTypedValue(param, rawText);
        power[eye][param] = parsed !== null ? parsed : (oldPower[eye]?.[param] || (param === 'axis' ? 180 : 0));
        const orig = oldPower[eye]?.[param] || (param === 'axis' ? 180 : 0);
        if (Math.abs(power[eye][param] - orig) > 0.001) anyChanged = true;
        _restoreCell(cell, _formatCellValue(param, power[eye][param]));
    }
    _typeModeEditing = false;

    if (!anyChanged || !sessionState.sessionId) return;

    _setPhoropterBusy(true);
    _setIntentsDisabled(true);
    try {
        showLoading(true);
        if (sessionState.sessionId) {
            await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/sync-power`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(power)
            }).catch(e => console.warn('Sync failed:', e));
        }
        if (!sessionState.lastResponse) sessionState.lastResponse = {};
        sessionState.lastResponse.power = power;
        updateSessionInfo({ ...sessionState.lastResponse, power });
        sessionState.responseCount++;
        document.getElementById('responseCount').textContent = sessionState.responseCount;
        addToHistory('Typed power applied', 'adjust');
        _saveSessionToStorage();
        refreshScreenshotIfModalOpen();
    } catch (error) {
        console.error('Error applying typed power:', error);
    } finally {
        showLoading(false);
        _setPhoropterBusy(false);
        _setIntentsDisabled(false);
    }
}

function bindTableInteractions() {
    const table = document.getElementById('refractionTable');
    if (table) {
        table.oncontextmenu = (e) => { e.preventDefault(); e.stopPropagation(); return false; };
        table.querySelectorAll('.rt-val').forEach(el => {
            el.setAttribute('data-tip', 'R = +  |  L = \u2212');
            el.addEventListener('mousedown', (event) => handleTableMousedown(event, el));
        });
    }
}

async function handleTableMousedown(event, el) {
    if (_phoropterBusy || !sessionState.sessionId) return;
    if (typeModeActive) { event.preventDefault(); event.stopPropagation(); if (el.dataset.param) _openInputInCell(el); return; }
    if (manualControlsLocked) return;

    const action = (event.button === 0) ? 'subtract' : (event.button === 2 ? 'add' : null);
    if (!action) return;
    event.preventDefault();
    event.stopPropagation();

    const param = el.dataset.param;
    let delta = param === 'axis' ? 5 : 0.25;
    if (action === 'subtract') delta = -delta;

    _setManualLock(true);
    if (_manualAutoUnlockTimer) clearTimeout(_manualAutoUnlockTimer);
    _manualAutoUnlockTimer = setTimeout(() => { _manualAutoUnlockTimer = null; _setManualLock(false); }, 1000);

    await applyManualPowerChange(el.dataset.eye, param, delta);
}

async function applyManualPowerChange(eye, param, delta) {
    if (!sessionState.lastResponse || !sessionState.lastResponse.power) return;
    _setPhoropterBusy(true);
    _setIntentsDisabled(true);

    const p = sessionState.lastResponse.power;
    const eyeKey = eye === 'R' ? 'right' : 'left';
    if (!p[eyeKey]) p[eyeKey] = { sph: 0, cyl: 0, axis: 180 };

    let current = parseFloat(p[eyeKey][param]) || 0;
    let newVal = current + delta;
    if (param === 'axis') {
        newVal = Math.round(newVal);
        newVal = ((newVal - 1) % 180 + 180) % 180 + 1;
        if (newVal === 0) newVal = 180;
    }
    p[eyeKey][param] = newVal;
    updateSessionInfo(sessionState.lastResponse);

    try {
        showLoading(true);
        if (sessionState.sessionId) {
            await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/sync-power`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(p)
            }).catch(e => console.warn('Sync failed:', e));
        }
        sessionState.responseCount++;
        document.getElementById('responseCount').textContent = sessionState.responseCount;
        addToHistory(`Manual: ${param.toUpperCase()} ${delta > 0 ? '+' : ''}${delta} [${eye}]`, 'adjust');
        _saveSessionToStorage();
        refreshScreenshotIfModalOpen();
    } catch (error) {
        console.error('Error applying manual power:', error);
    } finally {
        showLoading(false);
        _setPhoropterBusy(false);
        _setIntentsDisabled(false);
    }
}

// ── Screenshot Modal ────────────────────────────────────────────────────
let _screenshotDragInited = false;
let _screenshotZoom = 1;
let _screenshotHistory = [];
let _screenshotHistoryIndex = -1;

function isScreenshotModalOpen() {
    const backdrop = document.getElementById('screenshotModalBackdrop');
    return backdrop && backdrop.classList.contains('active');
}

async function fetchScreenshot() {
    if (isTestDeviceId()) return null;
    try {
        const brainId = await getBrainId();
        const resp = await fetch(`${CONFIG.phoropterUrl}/phoropter/${CONFIG.phoropterId}/screenshot`, {
            method: 'POST',
            headers: { 'x-brain-id': brainId }
        });
        if (!resp.ok) return null;
        let rawText = await resp.text();
        let base64 = rawText.trim();
        if (base64.startsWith('"') && base64.endsWith('"')) base64 = base64.slice(1, -1);
        if (base64.startsWith('{')) {
            try { const parsed = JSON.parse(rawText); base64 = parsed.image || parsed.screenshot || parsed.data || rawText; } catch (_) {}
        }
        base64 = base64.replace(/\s+/g, '');
        return base64 && base64.length > 50 ? base64 : null;
    } catch (e) { return null; }
}

function setScreenshotImage(base64) {
    const img = document.getElementById('screenshotImage');
    const wrap = document.getElementById('screenshotImgWrap');
    const loading = document.getElementById('screenshotLoading');
    const err = document.getElementById('screenshotError');
    if (!img || !loading || !err) return;
    if (base64) {
        _screenshotHistory.push({ base64, ts: Date.now() });
        if (_screenshotHistory.length > 50) _screenshotHistory = _screenshotHistory.slice(-50);
        _screenshotHistoryIndex = _screenshotHistory.length - 1;
        img.src = 'data:image/jpeg;base64,' + base64;
        img.classList.add('loaded');
        loading.style.display = 'none';
        err.style.display = 'none';
        if (wrap) wrap.style.transform = `scale(${_screenshotZoom})`;
    } else {
        img.removeAttribute('src');
        img.classList.remove('loaded');
        loading.style.display = 'none';
        err.style.display = 'block';
    }
}

function refreshScreenshotIfModalOpen() {
    if (!isScreenshotModalOpen()) return;
    (async () => {
        const loading = document.getElementById('screenshotLoading');
        const err = document.getElementById('screenshotError');
        if (loading) loading.style.display = 'block';
        if (err) err.style.display = 'none';
        const base64 = await fetchScreenshot();
        if (isScreenshotModalOpen()) setScreenshotImage(base64);
    })();
}

function openScreenshotModal() {
    const backdrop = document.getElementById('screenshotModalBackdrop');
    if (!backdrop) return;
    _screenshotZoom = 1;
    backdrop.classList.add('active');
    refreshScreenshotIfModalOpen();
}

function closeScreenshotModal() {
    const backdrop = document.getElementById('screenshotModalBackdrop');
    if (backdrop) backdrop.classList.remove('active');
}

function toggleScreenshotModal() {
    if (isScreenshotModalOpen()) closeScreenshotModal();
    else openScreenshotModal();
}

// ── Session Initialization (from intake page) ───────────────────────────
// Check if we arrived from the intake form with a session ID
async function checkIntakeRedirect() {
    const params = new URLSearchParams(window.location.search);
    const sid = params.get('session');
    if (sid) {
        sessionState.sessionId = sid;

        // Restore device state immediately so the dropdown shows the
        // acquired phoropter (intake already acquired it).
        const savedDevice = localStorage.getItem('phoropterId');
        if (savedDevice && !isTestDeviceId(savedDevice)) {
            _deviceAcquired = true;
            const select = document.getElementById('phoropterIdInput');
            if (select) {
                select.innerHTML = '';
                const opt = document.createElement('option');
                opt.value = savedDevice;
                opt.textContent = savedDevice + ' (connected)';
                opt.selected = true;
                select.appendChild(opt);
                select.disabled = true;
            }
            const acqBtn = document.getElementById('acquireDeviceBtn');
            if (acqBtn) acqBtn.style.display = 'none';
        }

        // Fetch session status, send reset, then show live view
        await fetchSessionStatus(sid);
        // Clean URL
        window.history.replaceState({}, '', window.location.pathname);
        return true;
    }
    return false;
}

async function fetchSessionStatus(sessionId) {
    try {
        const resp = await fetch(`${CONFIG.backendUrl}/api/session/${sessionId}/status`);
        if (!resp.ok) throw new Error('Session not found');
        const data = await resp.json();

        document.getElementById('testScreen').style.display = 'block';
        document.getElementById('noSessionScreen').style.display = 'none';
        updateSessionInfo(data);
        displayQuestion(data);
        updateStatusIndicator(true);
        updatePhaseProgress(data.state);
        addToHistory('Session started from intake', 'success');
        _saveSessionToStorage();

        // Restore device state from localStorage (intake already acquired)
        const savedDevice = localStorage.getItem('phoropterId');
        if (savedDevice) {
            _deviceAcquired = true;
            const select = document.getElementById('phoropterIdInput');
            if (select) { select.value = savedDevice; select.disabled = true; }
            const acqBtn = document.getElementById('acquireDeviceBtn');
            if (acqBtn) acqBtn.style.display = 'none';
        }

        // Reset phoropter to 0/0/180 before the first question.
        // Wait for the reset to complete, then open live view so the
        // operator sees the zeroed-out phoropter display.
        if (_deviceAcquired && !isTestDeviceId(savedDevice)) {
            addToHistory('Resetting phoropter...', 'info');
            try {
                const resetResp = await fetch(
                    `${CONFIG.backendUrl}/api/phoropter/${CONFIG.phoropterId}/reset`,
                    { method: 'POST' }
                );
                if (resetResp.ok) {
                    addToHistory('Phoropter reset to 0/0/180', 'success');
                } else {
                    addToHistory('Warning: reset returned ' + resetResp.status, 'warning');
                }
            } catch (resetErr) {
                addToHistory('Warning: Could not reset phoropter', 'warning');
            }
        }

        // Open live view after reset so the screenshot shows the clean state
        openScreenshotModal();

    } catch (err) {
        console.error('Failed to load session:', err);
        alert('Session not found. Please start from the intake form.');
    }
}

// ── Submit Response ─────────────────────────────────────────────────────
async function submitResponse(responseValue) {
    if (sessionState.intentsLocked || _phoropterBusy) return;
    try {
        showLoading(true);
        sessionState.intentsLocked = true;
        _setPhoropterBusy(true);

        const intentButtonsContainer = document.getElementById('intentButtons');
        intentButtonsContainer.innerHTML = '<div class="alert alert-info">Processing...</div>';

        sessionState.responseCount++;
        addToHistory(`Response: ${responseValue}`, 'info');

        const response = await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/respond`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ response: responseValue })
        });

        if (!response.ok) throw new Error('Failed to submit response');
        const data = await response.json();

        // Check terminal states
        if (data.is_terminal) {
            if (data.terminal_state === 'ESCALATE') {
                handleEscalation(data);
            } else {
                await completeTest(data);
            }
            return;
        }

        updateSessionInfo(data);
        displayQuestion(data);
        updatePhaseProgress(data.state);
        _saveSessionToStorage();
        refreshDerivedVariables();

    } catch (error) {
        console.error('Error submitting response:', error);
        alert('Failed to submit response. Please try again.');
        sessionState.intentsLocked = false;
    } finally {
        _setPhoropterBusy(false);
        showLoading(false);
        refreshScreenshotIfModalOpen();
    }
}

// ── Display Question and Response Buttons ───────────────────────────────
function displayQuestion(data) {
    const phaseBadge = document.getElementById('phaseBadge');
    const questionText = document.getElementById('questionText');
    const intentButtons = document.getElementById('intentButtons');
    const eyeIndicator = document.getElementById('eyeIndicator');

    if (phaseBadge) phaseBadge.textContent = data.phase_name || data.state || 'Unknown';
    if (questionText) questionText.textContent = data.question || 'Waiting for response...';

    // Eye indicator
    if (eyeIndicator) {
        const eye = data.eye || '';
        if (eye === 'RE') { eyeIndicator.textContent = 'RIGHT EYE'; eyeIndicator.className = 'eye-indicator eye-right'; }
        else if (eye === 'LE') { eyeIndicator.textContent = 'LEFT EYE'; eyeIndicator.className = 'eye-indicator eye-left'; }
        else if (eye === 'BIN') { eyeIndicator.textContent = 'BOTH EYES'; eyeIndicator.className = 'eye-indicator eye-both'; }
        else { eyeIndicator.textContent = ''; eyeIndicator.className = 'eye-indicator'; }
    }

    // Build response buttons
    if (intentButtons) {
        intentButtons.innerHTML = '';
        sessionState.intentsLocked = false;
        const options = data.options || [];
        options.forEach((opt, index) => {
            const button = document.createElement('button');
            button.className = 'intent-button';
            const color = RESPONSE_COLORS[opt] || '#607d8b';
            button.style.borderLeft = `4px solid ${color}`;
            button.textContent = `${index + 1}. ${opt.replace(/_/g, ' ')}`;
            button.onclick = () => submitResponse(opt);
            // Keyboard shortcut
            button.dataset.shortcut = String(index + 1);
            intentButtons.appendChild(button);
        });
    }

    // Update step info
    if (data.step_info) {
        const stepEl = document.getElementById('stepInfo');
        if (stepEl) {
            stepEl.textContent = `Step ${data.step_info.step || 0} | Phase step ${data.step_info.phase_step_count || 0}`;
        }
    }
}

// ── Update Session Info (Power Table, Status) ───────────────────────────
function updateSessionInfo(data) {
    const sessionIdEl = document.getElementById('sessionId');
    if (sessionIdEl) sessionIdEl.textContent = sessionState.sessionId || '-';

    const sessionStatusEl = document.getElementById('sessionStatus');
    if (sessionStatusEl) sessionStatusEl.textContent = 'Active';

    const responseCountEl = document.getElementById('responseCount');
    if (responseCountEl) responseCountEl.textContent = sessionState.responseCount;

    const currentPhaseEl = document.getElementById('currentPhase');
    if (currentPhaseEl) currentPhaseEl.textContent = data.phase_name || data.state || '-';

    // Power table
    if (data.power) {
        const right = data.power.right || { sph: 0, cyl: 0, axis: 180, add: 0 };
        const left = data.power.left || { sph: 0, cyl: 0, axis: 180, add: 0 };

        const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        set('rt-r-sph', (right.sph >= 0 ? '+' : '') + right.sph.toFixed(2));
        set('rt-r-cyl', (right.cyl >= 0 ? '+' : '') + right.cyl.toFixed(2));
        set('rt-r-axis', _normalizeAxisDisplay(right.axis));
        set('rt-l-sph', (left.sph >= 0 ? '+' : '') + left.sph.toFixed(2));
        set('rt-l-cyl', (left.cyl >= 0 ? '+' : '') + left.cyl.toFixed(2));
        set('rt-l-axis', _normalizeAxisDisplay(left.axis));

        // ADD columns for near phases
        const state = data.state || '';
        const isNear = ['P', 'Q', 'R'].includes(state);
        const addColHeader = document.getElementById('addColHeader');
        const rAddCell = document.getElementById('rt-r-add');
        const lAddCell = document.getElementById('rt-l-add');
        const addDisplay = isNear ? '' : 'none';
        if (addColHeader) addColHeader.style.display = addDisplay;
        if (rAddCell) rAddCell.style.display = addDisplay;
        if (lAddCell) lAddCell.style.display = addDisplay;
        if (isNear) {
            if (rAddCell) rAddCell.textContent = '+' + (right.add || 0).toFixed(2);
            if (lAddCell) lAddCell.textContent = '+' + (left.add || 0).toFixed(2);
        }
    }

    // Chart type indicator
    const chartEl = document.getElementById('chartDisplay');
    if (chartEl) chartEl.textContent = data.chart_type || '-';

    sessionState.currentPhase = data.state;
    sessionState.lastResponse = data;
}

// ── Phase Progress Sidebar ──────────────────────────────────────────────
function updatePhaseProgress(currentState) {
    const container = document.getElementById('phaseProgress');
    if (!container) return;

    container.innerHTML = '';
    const currentIdx = FSM_STATES_ORDER.indexOf(currentState);

    FSM_STATES_ORDER.forEach((state, idx) => {
        const div = document.createElement('div');
        div.className = 'phase-step';
        if (state === currentState) div.classList.add('current');
        else if (idx < currentIdx) div.classList.add('completed');

        const label = document.createElement('span');
        label.className = 'phase-label';
        label.textContent = state;

        const name = document.createElement('span');
        name.className = 'phase-name';
        name.textContent = FSM_STATE_NAMES[state] || state;

        div.appendChild(label);
        div.appendChild(name);
        container.appendChild(div);
    });
}

// ── Escalation ──────────────────────────────────────────────────────────
function handleEscalation(data) {
    const questionText = document.getElementById('questionText');
    const intentButtons = document.getElementById('intentButtons');

    if (questionText) {
        questionText.innerHTML = `
            <div style="color:#c62828;font-weight:600;font-size:1.1em;">ESCALATION REQUIRED</div>
            <div style="margin-top:8px;">This test requires optometrist review.</div>
            <div style="margin-top:4px;color:#666;">Current state: ${data.state || 'Unknown'}</div>
        `;
    }

    if (intentButtons) {
        intentButtons.innerHTML = `
            <button class="intent-button" style="background:#c62828;color:#fff;" onclick="endTestWithStatus('escalated')">End Test (Escalated)</button>
            <button class="intent-button" style="background:#ff9800;color:#fff;" onclick="alert('Manual override not yet implemented')">Override & Continue</button>
        `;
    }

    updatePhaseProgress('ESCALATE');
}

// ── Test Completion ─────────────────────────────────────────────────────
async function completeTest(data) {
    const questionText = document.getElementById('questionText');
    const intentButtons = document.getElementById('intentButtons');

    if (questionText) {
        const power = data?.power || sessionState.lastResponse?.power || {};
        const r = power.right || { sph: 0, cyl: 0, axis: 180 };
        const l = power.left || { sph: 0, cyl: 0, axis: 180 };
        questionText.innerHTML = `
            <div style="color:#2e7d32;font-weight:600;font-size:1.2em;">TEST COMPLETE</div>
            <div style="margin-top:12px;">
                <table style="width:100%;border-collapse:collapse;text-align:center;">
                    <tr style="font-weight:600;"><td></td><td>SPH</td><td>CYL</td><td>AXIS</td><td>ADD</td></tr>
                    <tr><td style="font-weight:600;">RE</td><td>${r.sph?.toFixed(2)}</td><td>${r.cyl?.toFixed(2)}</td><td>${Math.round(r.axis||180)}\u00b0</td><td>${(r.add||0).toFixed(2)}</td></tr>
                    <tr><td style="font-weight:600;">LE</td><td>${l.sph?.toFixed(2)}</td><td>${l.cyl?.toFixed(2)}</td><td>${Math.round(l.axis||180)}\u00b0</td><td>${(l.add||0).toFixed(2)}</td></tr>
                </table>
            </div>
        `;
    }

    if (intentButtons) {
        intentButtons.innerHTML = `
            <button class="intent-button" style="background:#2e7d32;color:#fff;" onclick="endTestWithStatus('completed')">Save & End</button>
            <button class="intent-button" style="background:#f44336;color:#fff;" onclick="discardTest()">Discard</button>
        `;
    }

    updatePhaseProgress('END');
    addToHistory('Test complete', 'success');
}

async function endTestWithStatus(status) {
    try {
        showLoading(true);
        const resp = await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/end`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                store: true,
                operator_name: operatorName,
                qualitative_feedback: status === 'escalated' ? 'Escalated to optometrist' : '',
            })
        });
        const data = await resp.json();
        addToHistory(`Session ended: ${status}`, 'success');
        _clearSessionStorage();
        updateStatusIndicator(false);

        // Release device
        await releaseDevice();

        // Show completion message
        setTimeout(() => {
            if (confirm('Test saved. Start a new test?')) {
                window.location.href = 'intake.html';
            }
        }, 500);
    } catch (err) {
        console.error('End session failed:', err);
        alert('Failed to save session.');
    } finally {
        showLoading(false);
    }
}

async function discardTest() {
    if (!confirm('Discard this test? Data will not be saved.')) return;
    try {
        await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/discard`, { method: 'POST' });
        _clearSessionStorage();
        await releaseDevice();
        window.location.href = 'intake.html';
    } catch (err) {
        console.error('Discard failed:', err);
    }
}

// ── Phoropter Direct Controls ───────────────────────────────────────────
async function resetPhoropter() {
    try {
        if (isTestDeviceId()) { addToHistory('Test mode: reset skipped', 'info'); return; }
        await fetch(`${CONFIG.backendUrl}/api/phoropter/${CONFIG.phoropterId}/reset`, { method: 'POST' });
        addToHistory('Phoropter reset to 0/0/180', 'success');
    } catch (error) {
        addToHistory('Warning: Could not reset phoropter', 'warning');
    }
    refreshScreenshotIfModalOpen();
}

// ── UI Helpers ──────────────────────────────────────────────────────────
function showLoading(show) {
    const el = document.getElementById('loadingIndicator');
    if (el) el.style.display = show ? 'block' : 'none';
}

function updateStatusIndicator(active) {
    const el = document.getElementById('statusDot');
    if (el) {
        el.className = active ? 'status-dot active' : 'status-dot';
        el.title = active ? 'Session Active' : 'No Active Session';
    }
}

function addToHistory(text, type) {
    sessionState.history.push({ text, type, time: new Date() });
    const container = document.getElementById('historyContent');
    if (!container) return;
    const entry = document.createElement('div');
    entry.className = `history-entry history-${type || 'info'}`;
    const time = new Date().toLocaleTimeString(undefined, { hour12: false });
    entry.innerHTML = `<span class="history-time">${time}</span> ${text}`;
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
}

function toggleSection(sectionId) {
    const section = document.getElementById('section-' + sectionId);
    const arrowEl = document.getElementById(sectionId + 'Arrow');
    if (!section || !arrowEl) return;
    section.classList.toggle('collapsed');
    arrowEl.textContent = section.classList.contains('collapsed') ? '\u25B6' : '\u25BC';
}

// ── Keyboard Shortcuts ──────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    // Number keys 1-6 for response buttons
    if (e.key >= '1' && e.key <= '6' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (_typeModeEditing) return;
        const buttons = document.querySelectorAll('.intent-button');
        const idx = parseInt(e.key) - 1;
        if (idx < buttons.length && !buttons[idx].disabled) {
            buttons[idx].click();
        }
    }
});

// ── Heartbeat ───────────────────────────────────────────────────────────
let _heartbeatInterval = null;

function startHeartbeat() {
    if (_heartbeatInterval) return;
    _heartbeatInterval = setInterval(async () => {
        if (!_deviceAcquired || isTestDeviceId()) return;
        try {
            const brainId = await getBrainId();
            await fetch(`${CONFIG.backendUrl}/api/devices/${CONFIG.phoropterId}/heartbeat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ brain_id: brainId })
            });
        } catch (e) { console.warn('Heartbeat failed:', e); }
    }, 30000);
}

// ── DOMContentLoaded ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Eye Test Engine v2 Frontend Loaded');

    // Start sections collapsed
    ['history', 'commands'].forEach(id => {
        const section = document.getElementById('section-' + id);
        const arrow = document.getElementById(id + 'Arrow');
        if (section) section.classList.add('collapsed');
        if (arrow) arrow.textContent = '\u25B6';
    });

    // Logs panel: hide tab if access denied
    const logsTab = document.getElementById('logsTabBtn');
    if (logsTab && isLogsAccessDenied()) logsTab.classList.add('hidden');
    const debugTab = document.getElementById('debugTabBtn');
    if (debugTab && isLogsAccessDenied()) debugTab.classList.add('hidden');

    await fetchConfig();
    updateStatusIndicator(false);
    bindTableInteractions();
    checkOptometristName();

    // Check for intake redirect first — this sets _deviceAcquired before
    // fetchDevices() runs, so the dropdown shows the locked device.
    const cameFromIntake = await checkIntakeRedirect();

    // fetchDevices() respects _deviceAcquired: if already acquired it
    // locks the dropdown to the acquired device instead of re-fetching.
    await fetchDevices();

    // If we didn't come from intake, try restoring a previous session
    if (!cameFromIntake) {
        const restored = await _tryRestoreSession();
        // Open live view for restored sessions (intake path opens it
        // after reset inside fetchSessionStatus)
        if (restored) openScreenshotModal();
    }

    // Start heartbeat
    startHeartbeat();
});
