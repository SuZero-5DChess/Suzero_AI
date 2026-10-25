// Board positions use displayed grid coordinates, including the flipped timeline order.
export function nearestBoard(boards, position, spacing) {
    let nearest = null;
    let distance = Infinity;
    for (const board of boards) {
        const candidate = ((board.x - position.x) * spacing.x) ** 2
            + ((board.y - position.y) * spacing.y) ** 2;
        if (candidate < distance) {
            nearest = board;
            distance = candidate;
        }
    }
    return nearest;
}

export function directionalTarget(boards, position, direction, steps = 1) {
    const origin = { x: Math.round(position.x), y: Math.round(position.y) };
    if (!direction.x && !direction.y) return origin;
    const distances = new Set();
    for (const board of boards) {
        const dx = board.x - origin.x, dy = board.y - origin.y;
        const distance = direction.x ? dx / direction.x : dy / direction.y;
        if (distance > 0 && dx === direction.x * distance && dy === direction.y * distance) {
            distances.add(distance);
        }
    }
    const ordered = [...distances].sort((a, b) => a - b);
    const distance = ordered.length >= steps ? ordered[steps - 1]
        : (ordered.at(-1) ?? 0) + steps - ordered.length;
    return { x: origin.x + direction.x * distance, y: origin.y + direction.y * distance };
}

// One screen-speed budget for cardinal and diagonal navigation. Fractional
// steps carry across frames; distant zoom levels batch whole steps together.
export function heldNavigationSteps({ direction, spacing, zoom, referenceZoom, rate, elapsed, remainder = 0 }) {
    const stepPixels = Math.hypot(direction.x * spacing.x, direction.y * spacing.y) * zoom;
    if (!stepPixels) return { steps: 0, remainder: 0 };
    const travel = remainder + rate * spacing.x * referenceZoom * elapsed / (1000 * stepPixels);
    const steps = Math.floor(travel);
    return { steps, remainder: travel - steps };
}

// Match a normal focus view in CSS pixels and relative zoom, including a
// destination still being approached by the rendered camera.
export function focusViewBoard(boards, camera, spacing, boardSize, scale) {
    const zoom = 2 ** camera.scale;
    if (Math.abs(2 ** (camera.scale - scale) - 1) > 0.001) return null;
    const position = {
        x: (-camera.x - boardSize.x / 2) / spacing.x,
        y: (-camera.y - boardSize.y / 2) / spacing.y
    };
    const board = nearestBoard(boards, position, spacing);
    return board && Math.max(Math.abs((board.x - position.x) * spacing.x * zoom),
        Math.abs((board.y - position.y) * spacing.y * zoom)) <= 0.5 ? board : null;
}
