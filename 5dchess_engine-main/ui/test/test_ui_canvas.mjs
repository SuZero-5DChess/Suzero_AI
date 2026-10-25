// UI gesture regressions. Run with: node --test ui/test/test_ui_canvas.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

// The browser sources are ES modules; load without changing ui/'s local Node configuration.
const source = await readFile(new URL('../scripts/canvas.js', import.meta.url), 'utf8');
const { InfiniteScrollableCanvas, Camera, AnimationManager } = await import(`data:text/javascript;base64,${Buffer.from(source + '\nexport { Camera };').toString('base64')}`);
const selectionSource = await readFile(new URL('../scripts/selection.js', import.meta.url), 'utf8');
const { createSelection } = await import(`data:text/javascript;base64,${Buffer.from(selectionSource).toString('base64')}`);

function fixture() {
    const clicks = [];
    const canvas = Object.create(InfiniteScrollableCanvas.prototype);
    Object.assign(canvas, {
        config: { dragThreshold: 5, dragSpeed: 1, minZoom: -5, maxZoom: 7 },
        cameraCurrent: { scale: 0, getZoomLevel: () => 1 },
        cameraTarget: { x: 0, y: 0, scale: 0 },
        dragStart: {}, touchStartPos: {},
        startAnimation() {},
        _getTouchWorldPosition: touch => ({ x: touch.clientX, y: touch.clientY }),
        onClick: (x, y) => clicks.push([x, y])
    });
    return { canvas, clicks };
}
const point = (x, y = 20) => ({ clientX: x, clientY: y });
const event = (touches, changedTouches = []) => ({ touches, changedTouches, preventDefault() {} });

test('a stationary tap selects exactly once', () => {
    const { canvas, clicks } = fixture();
    canvas._handleTouchStart(event([point(20)]));
    canvas._handleTouchEnd(event([], [point(20)]));
    assert.deepEqual(clicks, [[20, 20]]);
});

function doubleTapFixture(t) {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const result = fixture();
    result.doubles = [];
    result.canvas.getDoubleTapTarget = x => x < 200 ? Math.floor(x / 100) : null;
    result.canvas.onDoubleTap = (x, y) => result.doubles.push([x, y]);
    result.tap = (x, y = 20) => {
        result.canvas._handleTouchStart(event([point(x, y)]));
        result.canvas._handleTouchEnd(event([], [point(x, y)]));
    };
    return result;
}

test('single board taps wait 300 ms; nearby double taps focus without any selection', t => {
    const { tap, clicks, doubles } = doubleTapFixture(t);
    tap(20);
    t.mock.timers.tick(299);
    assert.deepEqual(clicks, []);
    t.mock.timers.tick(1);
    assert.deepEqual(clicks, [[20, 20]]);
    tap(20);
    t.mock.timers.tick(100);
    tap(24);
    t.mock.timers.tick(300);
    assert.deepEqual(clicks, [[20, 20]]);
    assert.deepEqual(doubles, [[24, 20]]);
    tap(24); tap(24);
    assert.equal(doubles.length, 2);
});

test('taps on different boards or far-apart squares remain individual selections', t => {
    const { tap, clicks, doubles } = doubleTapFixture(t);
    tap(95); tap(105); // Nearby screen positions on distinct boards.
    t.mock.timers.tick(300);
    tap(20); tap(60); // Same board, but beyond the 24 CSS pixel radius.
    t.mock.timers.tick(300);
    tap(220); tap(220); // Empty space never focuses.
    assert.deepEqual(clicks, [[95, 20], [105, 20], [20, 20], [60, 20], [220, 20], [220, 20]]);
    assert.deepEqual(doubles, []);
});

test('tap previews are immediate; recognition cancels them and distinct taps commit in order', t => {
    const { canvas, tap, clicks, doubles } = doubleTapFixture(t);
    const previews = [], commits = [], cancellations = [];
    canvas.onTapPreview = (x, y) => {
        const token = { x, y };
        previews.push(token);
        return token;
    };
    canvas.onTapCommit = token => commits.push(token);
    canvas.onTapCancel = token => cancellations.push(token);
    tap(20);
    assert.equal(previews.length, 1);
    assert.equal(commits.length, 0);
    tap(24);
    assert.deepEqual(cancellations, [previews[0]]);
    assert.deepEqual(doubles, [[24, 20]]);
    t.mock.timers.tick(300);
    assert.equal(commits.length, 0);
    tap(95); tap(105);
    assert.deepEqual(commits, [previews[1]]);
    t.mock.timers.tick(300);
    assert.deepEqual(commits, [previews[1], previews[2]]);
    tap(20);
    canvas._handleTouchCancel();
    t.mock.timers.tick(300);
    assert.deepEqual(cancellations, [previews[0], previews[3]]);
    assert.deepEqual(clicks, []);
});

test('real gesture flow requests a new position immediately and discards it on a destination double tap', t => {
    const { canvas, tap, doubles } = doubleTapFixture(t);
    const messages = [];
    let visual;
    const selection = createSelection({ send: message => messages.push(message),
        onChange: value => { visual = value; }, presentColor: () => false });
    canvas.onTapPreview = (x, y) => selection.preview({ l: 0, t: 0, c: false, x, y });
    canvas.onTapCommit = token => selection.commit(token);
    canvas.onTapCancel = token => selection.cancel(token);
    tap(20);
    selection.receiveMoves({ requestId: messages[0].requestId, moves: [{ l: 0, t: 0, x: 60, y: 20 }] });
    tap(60);
    assert.equal(messages[1].type, 'preview_move');
    assert.equal(visual.selected, undefined);
    assert.equal(visual.tap, undefined); // No temporary destination-square highlight.
    tap(60);
    t.mock.timers.tick(300);
    assert.equal(messages[2].type, 'discard_preview');
    assert.equal(doubles.length, 1);
    assert.equal(visual.selected.x, 20);
    assert.equal(visual.moves.length, 1);
    assert(!messages.some(message => message.type === 'apply_move'));
    tap(60);
    assert.equal(messages[3].type, 'preview_move');
    t.mock.timers.tick(299);
    assert.equal(messages.length, 4);
    t.mock.timers.tick(1);
    assert.equal(messages[4].type, 'apply_move');
});

test('drag, fast swipe, pinch and cancellation discard pending selection and double taps', t => {
    const { canvas, tap, clicks, doubles } = doubleTapFixture(t);
    for (const cancel of [
        () => canvas._handleTouchMove(event([point(40)])),
        () => canvas._handleTouchEnd(event([], [point(40)])),
        () => canvas._handleTouchStart(event([point(20), point(60)])),
        () => canvas._handleTouchCancel()
    ]) {
        tap(20);
        canvas._handleTouchStart(event([point(20)]));
        cancel();
        t.mock.timers.tick(300);
        assert.deepEqual(clicks, []);
        assert.deepEqual(doubles, []);
    }
    tap(20);
    t.mock.timers.tick(300);
    assert.deepEqual(clicks, [[20, 20]]);
});

test('a fast swipe cannot become a tap even without an intermediate move event', () => {
    const { canvas, clicks } = fixture();
    canvas._handleTouchStart(event([point(20)]));
    canvas._handleTouchEnd(event([], [point(40)]));
    assert.deepEqual(clicks, []);
});

test('dragging out and back to the starting square does not select it', () => {
    const { canvas, clicks } = fixture();
    canvas._handleTouchStart(event([point(20)]));
    canvas._handleTouchMove(event([point(40)]));
    canvas._handleTouchMove(event([point(20)]));
    canvas._handleTouchEnd(event([], [point(20)]));
    assert.deepEqual(clicks, []);
});

test('lifting fingers after a pinch does not select a piece; the next tap still works', () => {
    const { canvas, clicks } = fixture();
    canvas._handleTouchStart(event([point(20)]));
    canvas._handleTouchStart(event([point(20), point(60)]));
    canvas._handleTouchMove(event([point(20), point(80)]));
    canvas._handleTouchEnd(event([point(20)], [point(80)]));
    canvas._handleTouchEnd(event([], [point(20)]));
    assert.deepEqual(clicks, []);
    canvas._handleTouchStart(event([point(30)]));
    canvas._handleTouchEnd(event([], [point(30)]));
    assert.deepEqual(clicks, [[30, 20]]);
});

test('resizing uses the reserved canvas area and preserves relative mobile zoom', () => {
    const { canvas } = fixture();
    const previousDocument = globalThis.document;
    globalThis.document = { body: { classList: { contains: () => true } } };
    try {
        let rect = { width: 390, height: 600 };
        canvas.canvas = { getBoundingClientRect: () => rect };
        canvas._render = () => {};
        canvas.resize();
        assert.equal(canvas.canvas.width, 390);
        assert.equal(canvas.canvas.height, 600);
        rect = { width: 390, height: 195 };
        canvas.resize();
        assert.equal(canvas.canvas.height, 195);
        assert.equal(canvas.cameraTarget.scale, -1);
        assert.equal(canvas.cameraCurrent.scale, -1);
        rect = { width: 390, height: 600 };
        canvas.resize();
        assert.equal(canvas.cameraTarget.scale, 0);
    } finally {
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
    }
});

test('display density changes preserve camera bounds and mouse/touch positions', () => {
    const previous = Object.fromEntries(['document', 'window', 'ResizeObserver', 'devicePixelRatio'].map(key => [key, globalThis[key]]));
    const calls = [];
    const ctx = Object.fromEntries(['save', 'restore', 'setTransform', 'clearRect', 'scale', 'translate'].map(name => [name, (...args) => calls.push([name, ...args])]));
    const element = {
        getContext: () => ctx,
        getBoundingClientRect: () => ({ left: 10, top: 20, right: 400, bottom: 620, width: 390, height: 600 }),
        addEventListener() {}
    };
    let densityChanged;
    globalThis.document = { getElementById: () => element, body: { classList: { contains: () => true } } };
    globalThis.window = {
        addEventListener() {},
        matchMedia: () => ({ addEventListener: (_, listener) => { densityChanged = listener; }, removeEventListener() {} })
    };
    globalThis.ResizeObserver = class { observe() {} };
    globalThis.devicePixelRatio = 1;
    try {
        const canvas = new InfiniteScrollableCanvas('canvas', { initialZoom: 1, initialX: 7, initialY: -3 });
        const bounds = canvas.getVisibleBounds();
        const pointer = { clientX: 245, clientY: 360 };
        const expected = canvas._getMouseWorldPosition(pointer);
        assert.deepEqual(expected, { x: 13, y: 23 });
        for (const ratio of [1.25, 1.5, 2, 3, 1]) {
            globalThis.devicePixelRatio = ratio;
            calls.length = 0;
            densityChanged();
            assert.equal(element.width, Math.round(390 * ratio));
            assert.equal(element.height, Math.round(600 * ratio));
            assert.deepEqual(canvas.getVisibleBounds(), bounds);
            assert.deepEqual(canvas._getMouseWorldPosition(pointer), expected);
            assert.deepEqual(canvas._getTouchWorldPosition(pointer), expected);
            assert.equal(canvas.cameraCurrent.scale, 1);
            assert.equal(canvas.cameraTarget.scale, 1);
            assert.deepEqual(calls.find(([name]) => name === 'clearRect'), ['clearRect', 0, 0, element.width, element.height]);
            assert.deepEqual(calls.find(([name]) => name === 'scale'), ['scale', element.width / 390, element.height / 600]);
        }
    } finally {
        for (const [key, value] of Object.entries(previous)) {
            if (value === undefined) delete globalThis[key];
            else globalThis[key] = value;
        }
    }
});

test('camera animation settles at unit zoom and negative logarithmic scales', () => {
    for (const scale of [0, -2, 2]) {
        const current = new Camera(0, 0, scale - 1);
        const target = new Camera(300, -200, scale);
        for (let frame = 0; frame < 200 && !current.isCloseTo(target); frame++) {
            current.lerpTowards(target, 16, 0.01);
        }
        assert(current.isCloseTo(target));
        assert(Math.abs(current.getZoomLevel() / target.getZoomLevel() - 1) < 0.02);
    }
});


test('finite camera motion retargets from the current position and lands exactly', () => {
    const camera = new Camera(0, 0, 0);
    const target = new Camera(300, -200, -1);
    camera.lerpTowards(target, 100);
    const position = camera.clone();
    const next = new Camera(600, -400, -2);
    camera.lerpTowards(next, 0);
    assert.equal(camera.x, position.x);
    camera.lerpTowards(next, camera.motion.duration);
    for (const axis of ['x', 'y', 'scale']) {
        assert.equal(camera[axis], next[axis]);
        assert.equal(camera.velocity[axis], 0);
    }
});

test('camera trajectory is independent of frame subdivision', () => {
    const target = new Camera(100, 200, 2);
    const a = new Camera(), b = new Camera();
    a.lerpTowards(target, 160);
    for (let i = 0; i < 10; i++) b.lerpTowards(target, 16);
    assert.equal(a.x, b.x); assert.equal(a.scale, b.scale);
});


test('duration uses original positional and zoom gauges independent of starting zoom', () => {
    for (const scale of [-2, 0, 3]) {
        const camera = new Camera(0, 0, scale);
        camera.lerpTowards(new Camera(100, 40, scale), 0);
        assert.equal(camera.motion.duration, Math.log1p(100 / 0.1) / camera.motion.decay);
    }
    const camera = new Camera();
    camera.lerpTowards(new Camera(0, 0, 2), 0);
    assert.equal(camera.motion.duration, Math.log1p(2 * Math.LN2 / Math.log1p(0.02)) / camera.motion.decay);
});

test('corrected exponential keeps the original early response and finishes smoothly at its deadline', () => {
    const camera = new Camera(), target = new Camera(100, -100, 1);
    camera.lerpTowards(target, 100);
    assert(camera.x > 55 && camera.x < 70);
    camera.lerpTowards(target, camera.motion.duration - 100 - 0.1);
    assert(Math.abs(camera.x - target.x) < 1e-6);
    assert(Math.abs(camera.velocity.x) < 1e-5);
    camera.lerpTowards(target, 0.1);
    assert.equal(camera.x, target.x); assert.equal(camera.velocity.x, 0);
});

test('zero and tiny distances remain finite and land exactly', () => {
    for (const distance of [0, 1e-12, 1e-8, 0.0001]) {
        const camera = new Camera(), target = new Camera(distance, 0, 0);
        camera.lerpTowards(target, 0);
        assert(Number.isFinite(camera.x));
        camera.lerpTowards(target, camera.motion.duration);
        assert.equal(camera.x, target.x);
        assert.equal(camera.velocity.x, 0);
    }
});


test('delayed animation frames consume their full elapsed time', () => {
    let elapsed;
    const manager = new AnimationManager(delta => { elapsed = delta; return false; });
    manager.timeRecord = 100;
    manager._animate(600);
    assert.equal(elapsed, 500);
});

test('steadily advancing drag targets never reverse, including the settling tail', () => {
    for (const axis of ['x', 'y', 'scale']) {
        for (const sign of [-1, 1]) {
            const camera = new Camera();
            const target = new Camera();
            for (let i = 1; i <= 100; i++) {
                const before = camera[axis];
                target[axis] = sign * i * (axis === 'scale' ? 0.01 : 2);
                camera.lerpTowards(target, 16);
                assert(sign * (camera[axis] - before) >= -1e-12);
                assert(sign * (target[axis] - camera[axis]) >= -1e-12);
            }
            while (camera.motion.elapsed < camera.motion.duration) {
                const before = camera[axis];
                camera.lerpTowards(target, 16);
                assert(sign * (camera[axis] - before) >= -1e-12);
            }
            assert.equal(camera[axis], target[axis]);
        }
    }
});

test('reversing a destination follows the new direction immediately without overshoot', () => {
    const camera = new Camera();
    camera.lerpTowards(new Camera(100, 0, 0), 100);
    const before = camera.x;
    const target = new Camera(-100, 0, 0);
    camera.lerpTowards(target, 16);
    assert(camera.x < before);
    assert(camera.x > target.x);
    camera.lerpTowards(target, camera.motion.duration);
    assert.equal(camera.x, target.x);
});

test('dragging takes over from the rendered camera rather than a pending opposite destination', () => {
    const { canvas } = fixture();
    canvas.cameraCurrent = new Camera(20, 30, 0);
    canvas.cameraTarget = new Camera(-100, -200, 1);
    canvas.mouseStartPos = {};
    canvas._getMouseWorldPosition = e => ({ x: e.clientX, y: e.clientY });
    canvas._handleMouseDown({ button: 0, clientX: 400, clientY: 300 });
    canvas._handleMouseMove({ clientX: 420, clientY: 300 });
    assert.equal(canvas.cameraTarget.x, 40);
    assert.equal(canvas.cameraTarget.y, 30);
    assert.equal(canvas.cameraTarget.scale, 0);
    assert.equal(canvas.cameraCurrent.motion, null);
});


test('zoom-only doubling takes about 388ms and lands exactly at the deadline', () => {
    for (const direction of [-1, 1]) {
        const camera = new Camera(), target = new Camera(0, 0, direction);
        camera.lerpTowards(target, 0);
        assert(Math.abs(camera.motion.duration - 388) < 1);
        camera.lerpTowards(target, camera.motion.duration);
        assert.equal(camera.scale, target.scale);
        assert.equal(camera.velocity.scale, 0);
    }
});


test('combined position and logarithmic zoom share progress and the remaining-error derivative after retargeting', () => {
    const camera = new Camera();
    camera.lerpTowards(new Camera(100, -80, 0), 100);
    const start = camera.clone(), target = new Camera(300, 150, 2);
    camera.lerpTowards(target, 0);
    for (let i = 0; i < 5; i++) {
        camera.lerpTowards(target, 30);
        const progress = ['x', 'y', 'scale'].map(axis =>
            (camera[axis] - start[axis]) / (target[axis] - start[axis]));
        assert(progress.every(value => Math.abs(value - progress[0]) < 1e-12));
        const { decay: k, duration, elapsed } = camera.motion;
        const q = k * (duration - elapsed);
        const rate = k * Math.expm1(q) / (Math.expm1(q) - q);
        for (const axis of ['x', 'y', 'scale']) {
            assert(Math.abs(camera.velocity[axis] / (target[axis] - camera[axis]) - rate) < 1e-12);
        }
    }
    camera.lerpTowards(target, camera.motion.duration);
    assert.equal(camera.x, target.x); assert.equal(camera.scale, target.scale);
});
