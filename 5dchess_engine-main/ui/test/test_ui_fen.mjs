import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const { parse_FEN } = await import('data:text/javascript,' + encodeURIComponent(
    await readFile(new URL('../scripts/parse.js', import.meta.url), 'utf8')));

test('FEN display preserves interior walls, empty runs, and fully walled rectangular ranks', () => {
    assert.deepEqual(parse_FEN('k1#1/####/R1K#', 4, 3), [
        ['k', '1', '#', '1'], ['#', '#', '#', '#'], ['R', '1', 'K', '#']
    ]);
    assert.deepEqual(parse_FEN('1#2#1', 6, 1), [['1', '#', '1', '1', '#', '1']]);
    assert.deepEqual(parse_FEN('#/#', 1, 2), [['#'], ['#']]);
    assert.throws(() => parse_FEN('4/4', 4, 3), /exactly 3 rows/);
    assert.throws(() => parse_FEN('3/####/4', 4, 3), /exactly 4 squares/);
});
