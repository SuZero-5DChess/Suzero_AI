// Destination previews use the engine's temporary state; commitment applies normally.
export function createSelection({ send, onChange, presentColor, onDiscardPreview = () => {} }) {
    let selected = null;
    let pending = null;
    let waitingMove = null;
    let busy = false;
    let requestId = 0;
    let appliedPreviewId = null;

    function redraw() {
        onChange({ selected: selected?.pos, moves: selected?.moves || [] });
    }

    function legal(move) {
        return move.from.moves?.some(destination => destination.l === move.to.l && destination.t === move.to.t
            && destination.x === move.to.x && destination.y === move.to.y);
    }

    function previewPosition(move) {
        if (!move || !move.previewAllowed || move.previewed || !legal(move)) return;
        move.previewed = true;
        send({ type: 'preview_move', from: move.from.pos, to: move.to, requestId: move.previewId });
    }

    function finishMove() {
        if (!waitingMove || waitingMove.from.moves === null) return;
        const { from, to } = waitingMove;
        waitingMove = null;
        if (legal({ from, to })) {
            busy = true;
            send({ type: 'apply_move', from: from.pos, to });
        }
    }

    function preview(pos, previewAllowed = true) {
        if (busy || waitingMove || pending) return null;
        const transaction = { before: selected, pos };
        pending = transaction;
        if (pos.c !== presentColor()) {
            selected = null;
        } else if (selected) {
            transaction.move = { from: selected, to: pos, previewId: ++requestId, previewAllowed };
            selected = null;
            previewPosition(transaction.move);
        } else {
            selected = { pos, moves: null, requestId: ++requestId };
            send({ type: 'gen_move_if_playable', pos, requestId: selected.requestId });
        }
        redraw();
        return transaction;
    }

    function commit(transaction) {
        if (!transaction || pending !== transaction) return;
        pending = null;
        waitingMove = transaction.move || null;
        appliedPreviewId = transaction.move?.previewed ? transaction.move.previewId : null;
        finishMove();
        redraw();
    }

    function cancel(transaction) {
        if (!transaction || pending !== transaction) return;
        if (transaction.move?.previewed) {
            send({ type: 'discard_preview', requestId: transaction.move.previewId });
            onDiscardPreview();
        }
        selected = transaction.before;
        if (selected?.moves?.length === 0) selected = null;
        pending = null;
        redraw();
    }

    return {
        preview, commit, cancel,
        acceptsPreview(id) {
            return id != null && (pending?.move?.previewId === id || appliedPreviewId === id);
        },
        click(pos) { commit(preview(pos, false)); },
        receiveMoves(message) {
            const selection = [selected, pending?.before, waitingMove?.from]
                .find(value => value && value.requestId === message.requestId);
            if (!selection) return;
            selection.moves = message.moves;
            if (selection === selected && !selection.moves.length) selected = null;
            previewPosition(pending?.move);
            finishMove();
            redraw();
        },
        reset() {
            if (pending) cancel(pending);
            selected = pending = waitingMove = null;
            busy = false;
            appliedPreviewId = null;
            redraw();
        }
    };
}
