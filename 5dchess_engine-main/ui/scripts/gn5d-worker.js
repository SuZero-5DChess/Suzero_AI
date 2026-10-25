// Local GN5D transport for the engine's UI. Rules stay in the native backend;
// the upstream WASM worker remains available for separately built analyzers.
import { NativeReplay } from './gn5d-replay.js';

let replay = null;
let pending = null;
let loading = null;
let positionId = 0;

function publish(changed = false) {
    if (!replay) return;
    self.postMessage({ type: 'update_buttons',
        undo: replay.cursor > 0, prev: replay.cursor > 0,
        redo: replay.cursor < replay.payload.actions.length,
        next: replay.cursor < replay.payload.actions.length,
        submit: false, hint: false, readOnlyReplay: true,
        allowSubmitWithChecks: false, allowSubmitWithChecksLocked: true,
    });
    self.postMessage({ type: 'update_select', positionId, options: replay.options() });
    self.postMessage({ type: 'update_hud_status', ...replay.hud() });
    self.postMessage({ type: 'data', positionChanged: changed, data: replay.view() });
}

async function load(data) {
    loading?.controller.abort();
    pending = null;
    const transaction = { requestId: data.requestId, controller: new AbortController() };
    loading = transaction;
    try {
        const initial = data.requestId === undefined && data.pgn === '[Board "Standard - Turn Zero"]';
        if (!initial && !/^\s*\{/.test(data.pgn)) {
            throw new Error('This local entrypoint reads .gn5d and .5dck JSON records. Use the separately built WASM analyzer for PGN.');
        }
        const response = await fetch(initial ? '/api/initial' : '/api/replay', {
            method: initial ? 'GET' : 'POST',
            headers: initial ? {} : { 'Content-Type': 'application/json' },
            body: initial ? undefined : data.pgn,
            signal: transaction.controller.signal,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `Native replay failed: HTTP ${response.status}`);
        if (loading !== transaction) return;
        let source = data.pgn;
        if (initial) {
            const { frames, ...record } = payload;
            source = JSON.stringify(record);
        }
        const next = new NativeReplay(payload, source);
        if (initial) {
            replay = next;
            positionId++;
            publish(true);
        } else {
            pending = { requestId: data.requestId, replay: next };
            self.postMessage({ type: 'load_result', requestId: data.requestId, success: true });
        }
    } catch (error) {
        if (loading !== transaction || error.name === 'AbortError') return;
        if (data.requestId === undefined) {
            self.postMessage({ type: 'alert', message: error.message });
        } else {
            self.postMessage({ type: 'load_result', requestId: data.requestId,
                success: false, diagnostic: { message: error.message } });
        }
    } finally {
        if (loading === transaction) loading = null;
    }
}

self.onmessage = ({ data }) => {
    if (data.type === 'load') {
        load(data);
        return;
    }
    if (data.type === 'cancel_load') {
        if (loading?.requestId === data.requestId) {
            loading.controller.abort();
            loading = null;
        }
        if (pending?.requestId === data.requestId) pending = null;
        return;
    }
    if (data.type === 'commit_load') {
        if (!pending || pending.requestId !== data.requestId) return;
        replay = pending.replay;
        pending = null;
        positionId++;
        publish(true);
        self.postMessage({ type: 'load_committed', requestId: data.requestId });
        return;
    }
    if (!replay) return;
    if (data.positionId !== undefined && data.positionId !== positionId) return;
    if (['undo', 'redo', 'prev', 'next'].includes(data.type)) {
        if (data.type === 'undo' || data.type === 'redo') replay.step(data.type === 'undo' ? -1 : 1);
        else replay.jump(data.type === 'prev' ? -1 : 1);
        positionId++;
        self.postMessage({ type: 'navigated' });
        publish(true);
    } else if (data.type === 'view' || data.type === 'update_settings') {
        publish();
    } else if (data.type === 'gen_move_if_playable') {
        self.postMessage({ type: 'moves', moves: [], requestId: data.requestId });
    } else if (data.type === 'open_export') {
        self.postMessage({ type: 'open_export' });
    } else if (data.type === 'export') {
        self.postMessage({ type: 'update_pgn', pgn: replay.source,
            extension: replay.extension, requestId: data.requestId });
    }
};

self.postMessage({ type: 'engine_version', version: 'Native GN5D replay' });
self.postMessage({ type: 'ready' });
