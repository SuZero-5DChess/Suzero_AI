// Route shortcuts without stealing text entry, native control activation, or browser commands.
export function createKeyboardShortcuts({ onAction, isBlocked, now = () => performance.now(),
    requestFrame = callback => requestAnimationFrame(callback),
    cancelFrame = id => cancelAnimationFrame(id) }) {
    const actions = {
        ArrowUp: 'prev', ArrowDown: 'next', ArrowLeft: 'undo', ArrowRight: 'redo',
        Enter: 'submit', h: 'hint', f: 'focus',
        w: 'up', a: 'left', s: 'down', d: 'right'
    };
    const directionKeys = new Set(['w', 'a', 's', 'd']);
    let held = null;
    let frame = null;

    function direction() {
        return { x: Number(held.keys.has('d')) - Number(held.keys.has('a')),
            y: Number(held.keys.has('s')) - Number(held.keys.has('w')) };
    }

    function initialStep() {
        if (!held || held.initial) return;
        held.initial = true;
        const vector = direction();
        if (vector.x || vector.y) onAction('navigate', { direction: vector, held: false });
    }

    function stopHolding() {
        held = null;
        if (frame !== null) cancelFrame(frame);
        frame = null;
    }

    function repeatDirection() {
        frame = null;
        if (!held) return;
        if (isBlocked()) { reset(); return; }
        const time = now();
        if (time >= held.started + 40) initialStep();
        if (time >= held.started + 250) {
            const vector = direction();
            const elapsed = Math.min(100, time - Math.max(held.lastFrame, held.started + 250));
            const rate = 4 + 6 * Math.min(1, (time - held.started - 250) / 1500);
            if (vector.x || vector.y) onAction('navigate', {
                direction: vector, held: true, rate, elapsed, session: held.started
            });
        }
        held.lastFrame = time;
        frame = requestFrame(repeatDirection);
    }

    function handleKeyUp(event) {
        const key = event.key.toLowerCase();
        if (!held?.keys.has(key)) return;
        if (isBlocked()) { reset(); return; }
        initialStep();
        held.keys.delete(key);
        if (!held.keys.size) stopHolding();
    }

    function reset() {
        stopHolding();
    }

    function handleKeyDown(event) {
        const target = event.target;
        const editing = target?.isContentEditable
            || target?.closest?.('input, textarea, select, [role="textbox"], [role="combobox"]');
        const nativeActivation = target?.closest?.('button, a[href], summary, [role="button"]')
            && (event.key === 'Enter' || event.key === ' ');
        if (event.defaultPrevented || event.isComposing || event.keyCode === 229
            || event.metaKey || event.ctrlKey || event.altKey || editing
            || nativeActivation || isBlocked()) {
            reset();
            return;
        }
        const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
        if (key === ' ') {
            stopHolding();
            event.preventDefault();
            if (event.repeat) return;
            onAction('nearby', { wide: Boolean(event.shiftKey) });
            return;
        }
        const action = actions[key];
        if (!action) return;
        event.preventDefault();
        if (event.repeat && action === 'focus') return;
        if (directionKeys.has(key)) {
            if (event.repeat || held?.keys.has(key)) return;
            const time = now();
            if (!held) {
                held = { keys: new Set(), started: time, lastFrame: time, initial: false };
                frame = requestFrame(repeatDirection);
            }
            held.keys.add(key);
            return;
        }
        stopHolding();
        onAction(action);
    }
    return { handleKeyDown, handleKeyUp, reset };
}
