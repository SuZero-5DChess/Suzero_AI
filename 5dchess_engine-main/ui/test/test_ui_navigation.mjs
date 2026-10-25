import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const load = async name => import(`data:text/javascript;base64,${Buffer.from(await readFile(new URL(`../scripts/${name}.js`, import.meta.url), 'utf8')).toString('base64')}`);
const { nearestBoard, directionalTarget, focusViewBoard, heldNavigationSteps } = await load('navigation');
const { createKeyboardShortcuts } = await load('keyboard');
const drawSource = (await readFile(new URL('../scripts/draw.js', import.meta.url), 'utf8'))
    .replace(/^import .*;\n/gm, '');
const { default: ChessBoardCanvas } = await import(`data:text/javascript;base64,${Buffer.from(
    `const nearestBoard = ${nearestBoard}; const focusViewBoard = ${focusViewBoard};\n` + drawSource
).toString('base64')}`);
const directions = { up: { x: 0, y: -1 }, left: { x: -1, y: 0 }, down: { x: 0, y: 1 }, right: { x: 1, y: 0 } };

test('nearest board uses visual distance and handles empty data and ties deterministically', () => {
    const boards = [{ x: 0, y: 2 }, { x: 3, y: 0 }];
    assert.equal(nearestBoard(boards, { x: 0, y: 0 }, { x: 100, y: 200 }), boards[1]);
    assert.equal(nearestBoard(boards, { x: 0, y: 0 }, { x: 100, y: 100 }), boards[0]);
    assert.equal(nearestBoard([], { x: 0, y: 0 }, { x: 1, y: 1 }), null);
    assert.deepEqual(nearestBoard([{ x: -1, y: 0 }, { x: 1, y: 0 }], { x: 0, y: 0 }, { x: 1, y: 1 }), { x: -1, y: 0 });
});

test('directional navigation skips gaps only in the same row or column', () => {
    const boards = [{ x: 0, y: 0 }, { x: 2, y: 0 }, { x: 7, y: 0 }, { x: 0, y: -3 }, { x: 1, y: -1 }];
    const origin = { x: 0, y: 0 };
    assert.deepEqual(directionalTarget(boards, origin, directions.right), { x: 2, y: 0 });
    assert.deepEqual(directionalTarget(boards, origin, directions.up), { x: 0, y: -3 });
    assert.deepEqual(directionalTarget(boards, origin, directions.left), { x: -1, y: 0 });
    assert.deepEqual(directionalTarget(boards, origin, directions.down), { x: 0, y: 1 });
    assert.deepEqual(directionalTarget(boards, { x: 7, y: 0 }, directions.left), { x: 2, y: 0 });
});

test('empty space supports continued steps and returning to a board', () => {
    const boards = [{ x: -2, y: -1 }];
    let target = { x: -2.1, y: -0.9 };
    target = directionalTarget(boards, target, directions.left);
    assert.deepEqual(target, { x: -3, y: -1 });
    target = directionalTarget(boards, target, directions.left);
    assert.deepEqual(target, { x: -4, y: -1 });
    assert.deepEqual(directionalTarget(boards, target, directions.right), boards[0]);
});

function keyboard() {
    const calls = [];
    let time = 0;
    let blocked = false;
    let scheduled = null;
    const handler = createKeyboardShortcuts({ requestFrame: callback => { scheduled = callback; return 1; }, cancelFrame: () => { scheduled = null; }, onAction: (...args) => calls.push(args), isBlocked: () => blocked, now: () => time });
    const press = (key, overrides = {}) => {
        const event = { key, target: { closest: () => null }, preventDefault() { this.defaultPrevented = true; }, ...overrides };
        handler.handleKeyDown(event);
        return event;
    };
    return { calls, press, reset: handler.reset, tick: ms => { time += ms; const callback = scheduled; scheduled = null; callback?.(); }, release: key => handler.handleKeyUp({ key }), block: value => { blocked = value; } };
}

test('Space delegates camera-state decisions without a timing window; Shift+Space is direct', () => {
    const k = keyboard();
    assert(k.press(' ').defaultPrevented);
    assert.deepEqual(k.calls.pop(), ['nearby', { wide: false }]);
    k.tick(20000); k.press(' ', { repeat: true });
    assert.equal(k.calls.length, 0);
    k.press(' ');
    assert.deepEqual(k.calls.pop(), ['nearby', { wide: false }]);
    k.press(' ', { shiftKey: true });
    assert.deepEqual(k.calls.pop(), ['nearby', { wide: true }]);
});

test('focus view recognition uses screen-position and relative-zoom tolerance', () => {
    const boards = [{ x: 3, y: -2 }, { x: 0, y: 0 }];
    const spacing = { x: 100, y: 112 }, size = { x: 80, y: 80 };
    const camera = { x: -340, y: 184, scale: 2 };
    const match = value => focusViewBoard(boards, value, spacing, size, 2);
    assert.equal(match(camera), boards[0]);
    assert.equal(match({ ...camera, x: camera.x + 0.1 }), boards[0]);
    assert.equal(match({ ...camera, x: camera.x + 0.2 }), null);
    assert.equal(match({ ...camera, scale: 2 + Math.log2(1.0005) }), boards[0]);
    assert.equal(match({ ...camera, scale: 1 }), null);
    assert.equal(focusViewBoard([], camera, spacing, size, 2), null);
});

test('touch focus targets the tapped visible board and toggles normal/wide destinations during motion', () => {
    const previousDocument = globalThis.document;
    globalThis.document = { body: { classList: { contains: () => true } } };
    try {
        for (const flipped of [false, true]) {
            const canvas = Object.assign(Object.create(ChessBoardCanvas.prototype), {
                boardLengthX: 8, boardLengthY: 8, squareSize: 10,
                boardSkipX: 100, boardSkipY: 112, flipped,
                data: { boards: [{ l: 1, t: 1, c: false, fen: 'board' }, { l: 0, t: 0, c: false, fen: 'board' },
                    { l: 0, t: 2, c: false, fen: null }] },
                infCanvas: {
                    viewportSize: { width: 390, height: 600 },
                    cameraCurrent: { x: -40, y: -40, scale: 0 },
                    cameraTarget: { x: -40, y: -40, scale: 0 },
                    moveTo(x, y, scale) { this.cameraTarget = { x: -x, y: -y, scale }; }
                }
            });
            const tap = () => canvas.zoomTappedBoard(220, (flipped ? -112 : 112) + 20);
            const normal = canvas._focusScale(), wide = canvas._focusScale(true);
            tap();
            assert.deepEqual(canvas.infCanvas.cameraTarget, { x: -240, y: flipped ? 72 : -152, scale: normal });
            tap(); // Rendered camera is still at the original board.
            assert.equal(canvas.infCanvas.cameraTarget.scale, wide);
            tap();
            assert.equal(canvas.infCanvas.cameraTarget.scale, normal);
            canvas.infCanvas.cameraCurrent = { ...canvas.infCanvas.cameraTarget };
            tap(); // Same toggle after settling.
            assert.equal(canvas.infCanvas.cameraTarget.scale, wide);
            canvas.zoomTappedBoard(20, 20); // Another board always gets normal zoom.
            assert.deepEqual(canvas.infCanvas.cameraTarget, { x: -40, y: -40, scale: normal });
            const before = { ...canvas.infCanvas.cameraTarget };
            for (const x of [90, 320, 420]) canvas.zoomTappedBoard(x, 20); // Gap, absent board, hidden preview.
            assert.deepEqual(canvas.infCanvas.cameraTarget, before);
            const original = canvas.data;
            canvas.data = { boards: [] }; // A temporary position can hide an original phantom destination.
            assert.equal(canvas._boardAtWorldPoint(20, 20), null);
            assert.deepEqual(canvas._boardAtWorldPoint(20, 20, original), { x: 0, y: canvas._displayL(0) });
            canvas.data = original;
            canvas.infCanvas.cameraCurrent = { ...before, scale: normal - 1 };
            canvas.infCanvas.cameraTarget = { ...canvas.infCanvas.cameraCurrent };
            canvas.zoomTappedBoard(20, 20);
            assert.equal(canvas.infCanvas.cameraTarget.scale, normal); // Center alone does not widen.
        }
    } finally {
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
    }
});

test('editing, composition, browser modifiers, dialogs and native activation are untouched', () => {
    const k = keyboard();
    for (const overrides of [
        { target: { isContentEditable: true } },
        { target: { closest: selector => selector.includes('input') ? {} : null } },
        { isComposing: true }, { keyCode: 229 }, { metaKey: true }, { ctrlKey: true }, { altKey: true }
    ]) {
        for (const key of ['f', 'w', ' ', 'Enter', 'ArrowLeft']) assert(!k.press(key, overrides).defaultPrevented);
    }
    const button = { closest: selector => selector.includes('button') ? {} : null };
    assert(!k.press(' ', { target: button }).defaultPrevented);
    assert(!k.press('Enter', { target: button }).defaultPrevented);
    k.block(true); assert(!k.press('w').defaultPrevented); k.block(false);
    assert.equal(k.calls.length, 0);
    k.press('w', { target: button });
    k.tick(40);
    k.press('F', { target: button });
    assert.deepEqual(k.calls.map(([action]) => action), ['navigate', 'focus']);
});

test('distinct taps move once, OS repeat is ignored, and focus does not repeat', () => {
    const k = keyboard();
    k.press('d'); k.release('d'); k.press('d'); k.release('d');
    assert.equal(k.calls.length, 2);
    k.press('d'); k.press('d', { repeat: true }); k.tick(40);
    k.press('f'); k.press('f', { repeat: true });
    assert.deepEqual(k.calls.map(([action]) => action), ['navigate', 'navigate', 'navigate', 'focus']);
});

test('existing history-arrow shortcuts retain key-repeat behavior', () => {
    const k = keyboard();
    k.press('ArrowUp'); k.press('ArrowUp', { repeat: true });
    assert.deepEqual(k.calls.map(([action]) => action), ['prev', 'prev']);
});


test('held directions accelerate independently of OS repeat and stop on release or reset', () => {
    const k = keyboard();
    k.press('w'); k.tick(249); assert.equal(k.calls.length, 1);
    k.tick(1); const slow = k.calls.at(-1)[1].rate;
    for (let i = 0; i < 100; i++) k.tick(16);
    assert(k.calls.at(-1)[1].rate > slow * 2);
    k.release('w'); const count = k.calls.length; k.tick(1000);
    assert.equal(k.calls.length, count);
    k.press('d'); k.reset(); k.tick(1000); assert.equal(k.calls.length, count);
    k.press('a'); k.block(true); k.tick(300); assert.equal(k.calls.length, count);
});

test('diagonal chords ignore key order and remaining held keys continue after release', () => {
    for (const keys of [['w', 'a'], ['a', 'w']]) {
        const k = keyboard();
        k.press(keys[0]); k.tick(10); k.press(keys[1]); k.tick(30);
        assert.deepEqual(k.calls[0], ['navigate', { direction: { x: -1, y: -1 }, held: false }]);
        k.tick(300); assert.deepEqual(k.calls.at(-1)[1].direction, { x: -1, y: -1 });
        k.release('w'); k.tick(16);
        assert.deepEqual(k.calls.at(-1)[1].direction, { x: -1, y: 0 });
        k.release('a'); const count = k.calls.length; k.tick(1000);
        assert.equal(k.calls.length, count);
    }
    const k = keyboard(); k.press('w'); k.press('a'); k.release('a'); k.release('w');
    assert.deepEqual(k.calls, [['navigate', { direction: { x: -1, y: -1 }, held: false }]]);
});

test('opposite keys cancel per axis without resetting hold acceleration', () => {
    const k = keyboard(); k.press('w'); k.press('s'); k.tick(300);
    assert.equal(k.calls.length, 0);
    k.press('d'); k.tick(1500);
    assert.deepEqual(k.calls.at(-1)[1].direction, { x: 1, y: 0 });
    assert.equal(k.calls.at(-1)[1].rate, 10);
});

test('diagonal rays skip gaps, ignore other quadrants, and batch repeated steps', () => {
    const boards = [{ x: -3, y: -3 }, { x: -8, y: -8 }, { x: -1, y: -2 }];
    const direction = { x: -1, y: -1 };
    assert.deepEqual(directionalTarget(boards, { x: 0, y: 0 }, direction), boards[0]);
    assert.deepEqual(directionalTarget(boards, { x: 0, y: 0 }, direction, 2), boards[1]);
    assert.deepEqual(directionalTarget(boards, { x: 0, y: 0 }, direction, 4), { x: -10, y: -10 });
    assert.deepEqual(directionalTarget([], { x: 0, y: 0 }, direction), { x: -1, y: -1 });
});

test('held screen-speed budget scales with zoom and normalizes diagonal travel', () => {
    const options = { spacing: { x: 100, y: 112 }, referenceZoom: 1, rate: 10, elapsed: 1000 };
    const normal = heldNavigationSteps({ ...options, direction: { x: 1, y: 0 }, zoom: 1 });
    const far = heldNavigationSteps({ ...options, direction: { x: 1, y: 0 }, zoom: 0.25 });
    const diagonal = heldNavigationSteps({ ...options, direction: { x: 1, y: 1 }, zoom: 1 });
    assert.equal(normal.steps, 10); assert.equal(far.steps, 40);
    assert(Math.abs((diagonal.steps + diagonal.remainder) * Math.hypot(100,112) - 1000) < 1e-9);
});
