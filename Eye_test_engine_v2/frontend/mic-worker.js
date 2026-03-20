// Web Worker for mic polling timer.
// Unlike setInterval in the main thread, Worker timers are NOT throttled
// by Chrome's power-saving / background-tab policies on Windows.

let intervalId = null;

self.onmessage = (e) => {
    const { command, intervalMs } = e.data;
    if (command === 'start') {
        if (intervalId) clearInterval(intervalId);
        intervalId = setInterval(() => {
            self.postMessage({ type: 'tick' });
        }, intervalMs || 33);
    } else if (command === 'stop') {
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
        }
    }
};
