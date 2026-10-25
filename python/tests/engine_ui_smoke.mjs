// Manual browser smoke test. Requires Node >= 22 (global WebSocket) and Chrome/Edge.
// node engine_ui_smoke.mjs URL OUTPUT_DIR RECORD.gn5d [RECORD.5dck ...]
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { readFile, writeFile, mkdtemp, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [url, outputDir, ...records] = process.argv.slice(2);
assert(url && outputDir && records.length, 'Usage: URL OUTPUT_DIR RECORD [RECORD ...]');
const browserPath = process.env.BROWSER || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const profile = await mkdtemp(join(tmpdir(), 'suzero-engine-ui-'));
const browser = spawn(browserPath, ['--headless=new', '--remote-debugging-port=0',
    `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check',
    '--window-size=1440,1000', 'about:blank'], { stdio: 'ignore' });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
let socket;
try {
    let port;
    for (let i = 0; i < 100; i++) {
        try { port = Number((await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]); break; }
        catch { await delay(100); }
    }
    assert(port, 'Browser did not start');
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    socket = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
    await once(socket, 'open');
    let id = 0;
    const pending = new Map(), errors = [];
    socket.addEventListener('message', event => {
        const message = JSON.parse(event.data);
        if (message.id) {
            const callback = pending.get(message.id);
            pending.delete(message.id);
            if (message.error) callback.reject(new Error(JSON.stringify(message.error)));
            else callback.resolve(message.result);
        } else if (message.method === 'Runtime.exceptionThrown') {
            errors.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
        } else if (message.method === 'Page.javascriptDialogOpening') {
            errors.push(message.params.message);
            send('Page.handleJavaScriptDialog', { accept: true });
        }
    });
    function send(method, params = {}) {
        return new Promise((resolve, reject) => {
            pending.set(++id, { resolve, reject });
            socket.send(JSON.stringify({ id, method, params }));
        });
    }
    async function evaluate(expression) {
        const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
        assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
        return result.result.value;
    }
    async function wait(expression) {
        for (let i = 0; i < 200; i++) {
            if (await evaluate(expression)) return;
            if (errors.length) throw new Error(errors.join('\n'));
            await delay(100);
        }
        throw new Error(`Timed out: ${expression}`);
    }
    async function screenshot(name) {
        await delay(1000); // Let the original UI finish its camera animation.
        await mkdir(outputDir, { recursive: true });
        const result = await send('Page.captureScreenshot', { format: 'png' });
        await writeFile(join(outputDir, name), Buffer.from(result.data, 'base64'));
    }
    await send('Page.enable');
    await send('Runtime.enable');
    await send('Page.navigate', { url });
    await wait('!document.getElementById("loadingScreen") && window.chessBoardCanvas?.data?.boards.length === 2');
    await wait('document.getElementById("popupInfo").classList.contains("show")');
    await evaluate('document.getElementById("popupOverlay").click()');
    await wait('!document.getElementById("popupOverlay").classList.contains("show")');
    assert.equal(await evaluate('window.nativeReplay'), true);
    await screenshot('initial.png');

    for (const [index, recordPath] of records.entries()) {
        const source = await readFile(recordPath, 'utf8');
        const record = JSON.parse(source);
        const expected = await (await fetch(`${url}api/replay`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: source,
        })).json();
        assert(expected.frames, JSON.stringify(expected));
        await evaluate(`document.getElementById('btnImport').click();
            document.getElementById('importTextarea').value = ${JSON.stringify(source)};
            document.getElementById('importTextarea').dispatchEvent(new Event('input', { bubbles: true }));
            document.getElementById('loadButton').click();`);
        await wait(`!document.getElementById('popupImport').classList.contains('show')
            && document.querySelector('.hud-title').textContent === 'GN5D replay · 0/${record.actions.length}'`);
        assert.deepEqual(await evaluate('window.chessBoardCanvas.data.boards.map(({parsed, ...b}) => b)'), expected.frames[0].boards);
        assert.equal(await evaluate('document.querySelector(".remote-btn.ok").disabled'), true);
        assert.equal(await evaluate('document.querySelector(".hud-text").isContentEditable'), false);
        for (let cursor = 1; cursor <= record.actions.length; cursor++) {
            await evaluate('document.querySelector(".remote-btn.right").click()');
            await wait(`document.querySelector('.hud-title').textContent === 'GN5D replay · ${cursor}/${record.actions.length}'`);
            const frame = await evaluate('({ boards: window.chessBoardCanvas.data.boards.map(({parsed, ...b}) => b), present: window.chessBoardCanvas.data.present })');
            assert.deepEqual(frame.boards, expected.frames[cursor].boards);
            assert.deepEqual(frame.present, expected.frames[cursor].present);
        }
        await evaluate('document.getElementById("btnFocus").click()');
        await screenshot(`record-${index}-final.png`);
        await evaluate('document.getElementById("btnExport").click()');
        await wait('!document.getElementById("saveButton").disabled');
        assert.equal(await evaluate('document.getElementById("exportTextarea").value'), source);
        assert.equal(await evaluate('document.getElementById("exportExtension").value'), record.replay ? '.5dck' : '.gn5d');
        await evaluate('document.getElementById("popupOverlay").click()');

        // Big-step navigation must preserve submit boundaries and pending tails.
        const ends = record.actions.flatMap((action, i) => action === 'submit' || action.type === 'submit' ? [i + 1] : []);
        if (ends.at(-1) !== record.actions.length) ends.push(record.actions.length);
        for (const cursor of [...ends.slice(0, -1).reverse(), 0]) {
            await evaluate('document.querySelector(".remote-btn.up").click()');
            await wait(`document.querySelector('.hud-title').textContent === 'GN5D replay · ${cursor}/${record.actions.length}'`);
        }
        for (const cursor of ends.filter(end => end > 0)) {
            await evaluate('document.querySelector(".remote-btn.down").click()');
            await wait(`document.querySelector('.hud-title').textContent === 'GN5D replay · ${cursor}/${record.actions.length}'`);
        }
        console.log(`PASS ${recordPath}: ${record.actions.length} actions, native boards/present at every frame, export and big-step navigation`);
    }
    // A rejected import must keep the previous game and render the native UI error.
    const oldTitle = await evaluate('document.querySelector(".hud-title").textContent');
    await evaluate(`document.getElementById('btnImport').click();
        document.getElementById('importTextarea').value = '{"format":"invalid"}';
        document.getElementById('importTextarea').dispatchEvent(new Event('input', { bubbles: true }));
        document.getElementById('loadButton').click();`);
    await wait('document.getElementById("importTextarea").getAttribute("aria-invalid") === "true"');
    assert.equal(await evaluate('document.querySelector(".hud-title").textContent'), oldTitle);
    await screenshot('rejected-import.png');
    await evaluate('document.getElementById("popupOverlay").click()');
    await send('Emulation.setDeviceMetricsOverride', {
        width: 390, height: 844, deviceScaleFactor: 1, mobile: true,
    });
    await evaluate('document.getElementById("btnFocus").click()');
    await screenshot('mobile.png');
    await evaluate('document.getElementById("btnExport").click()');
    await wait('!document.getElementById("saveButton").disabled');
    await evaluate(`HTMLAnchorElement.prototype.click = function () { window.savedFilename = this.download; };
        document.getElementById('saveButton').click();`);
    const lastRecord = JSON.parse(await readFile(records.at(-1), 'utf8'));
    assert((await evaluate('window.savedFilename')).endsWith(lastRecord.replay ? '.5dck' : '.gn5d'));
    assert.deepEqual(errors, []);
    console.log('PASS startup, read-only controls, rejected import preserves current replay, mobile export extension, no browser exceptions');
} finally {
    socket?.close();
    browser.kill();
    await once(browser, 'exit').catch(() => {});
    await delay(300);
    await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
