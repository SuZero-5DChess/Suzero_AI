// Exact move identity, independent of object property order and display notation.
export function actionKey(action) {
    return JSON.stringify(action.map(move => [
        move.from.l, move.from.t, move.from.x, move.from.y,
        move.to.l, move.to.t, move.to.x, move.to.y, move.promote ?? 0,
    ]));
}

export class PendingBranches {
    constructor() { this.keys = new Set(); }
    clear() { this.keys.clear(); }
    has(action) { return this.keys.has(actionKey(action)); }
    toggle(action, children) {
        const key = actionKey(action);
        if (!children.some(child => actionKey(child.action) === key)) return false;
        if (this.keys.has(key)) this.keys.delete(key);
        else this.keys.add(key);
        return true;
    }
    // Resolve and validate every target before any C++ mutation. Submitting an
    // existing continuation preserves that branch, including comments/descendants.
    plan(children, protectedAction = null) {
        const byKey = new Map(children.map(child => [actionKey(child.action), child]));
        for (const key of this.keys) {
            if (!byKey.has(key)) throw new Error('The continuations changed. Review the pending deletions again.');
        }
        const protectedKey = protectedAction === null ? null : actionKey(protectedAction);
        return [...this.keys].filter(key => key !== protectedKey).map(key => byKey.get(key));
    }
}
