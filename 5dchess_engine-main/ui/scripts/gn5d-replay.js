// Native replay snapshots adapted to the engine UI's existing view protocol.
import { parse_FEN } from './parse.js';

function isSubmit(action) {
    return action === 'submit' || action?.type === 'submit';
}

function moveCoordinates(action) {
    return Array.isArray(action) ? action : [...action.from, ...action.to];
}

function position(values, c) {
    const [x, y, t, l] = values;
    return { x, y, t, l, c };
}

function nextTurn(pos) {
    return { ...pos, t: pos.t + Number(pos.c), c: !pos.c };
}

export class NativeReplay {
    constructor(payload, source) {
        if (!Array.isArray(payload.frames) || !Array.isArray(payload.actions)
            || payload.frames.length !== payload.actions.length + 1) {
            throw new Error('Invalid native replay response.');
        }
        this.payload = payload;
        this.source = source;
        this.extension = Object.hasOwn(JSON.parse(source), 'replay') ? '.5dck' : '.gn5d';
        this.cursor = 0;
        this.ends = payload.actions.flatMap((action, i) => isSubmit(action) ? [i + 1] : []);
        if (this.ends.at(-1) !== payload.actions.length) this.ends.push(payload.actions.length);
        this.view();
    }

    step(delta) {
        this.cursor = Math.max(0, Math.min(this.payload.actions.length, this.cursor + delta));
    }

    jump(direction) {
        this.cursor = direction > 0
            ? this.ends.find(end => end > this.cursor) ?? this.cursor
            : this.ends.filter(end => end < this.cursor).at(-1) ?? 0;
    }

    options() {
        if (this.cursor === this.payload.actions.length) return [];
        const end = this.ends.find(value => value > this.cursor);
        const action = [];
        const labels = [];
        for (let i = this.cursor; i < end; i++) {
            const raw = this.payload.actions[i];
            if (isSubmit(raw)) {
                labels.push('submit');
            } else {
                const move = moveCoordinates(raw);
                const c = this.payload.frames[i].current_player === 1;
                action.push({ from: position(move.slice(0, 4), c), to: position(move.slice(4), c) });
                labels.push(`(${move.slice(0, 4)}) → (${move.slice(4)})`);
            }
        }
        return [{ pgn: labels.join('; '), action }];
    }

    view() {
        const frame = this.payload.frames[this.cursor];
        const highlights = [false, true].flatMap(c => {
            const coordinates = [], arrows = [];
            for (let i = 0; i < this.cursor; i++) {
                const action = this.payload.actions[i];
                if (isSubmit(action) || (this.payload.frames[i].current_player === 1) !== c) continue;
                const move = moveCoordinates(action);
                const from = position(move.slice(0, 4), c), to = position(move.slice(4), c);
                if (from.l === to.l && from.t === to.t) {
                    coordinates.push(nextTurn(from), nextTurn(to));
                } else {
                    arrows.push({ from, to });
                }
            }
            const side = c ? 'black' : 'white';
            return [
                { color: `--highlight-${side}-move`, coordinates },
                { color: `--highlight-${side}-move-arrow`, arrows },
            ];
        });
        return {
            size: frame.size, present: frame.present,
            boards: frame.boards.map(board => ({
                ...board, parsed: parse_FEN(board.fen, frame.size.y, frame.size.x),
            })),
            focus: frame.timeline_status.mandatory_timelines.map(l => ({ l, ...frame.present })),
            highlights, phantom: [], phantomChecks: [], readOnlyReplay: true,
        };
    }

    hud() {
        const { variant, result, meta, actions } = this.payload;
        return {
            hudTitle: `GN5D replay · ${this.cursor}/${actions.length}`,
            hudText: `${variant} · ${result.terminal_reason} · White outcome: ${result.outcome}`,
            metadata: { Event: 'GN5D native replay', Round: `${meta.iteration}/${meta.game_id}` },
        };
    }
}
