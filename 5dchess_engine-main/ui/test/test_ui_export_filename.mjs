import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../scripts/filename.js', import.meta.url), 'utf8');
const { getExportFilename } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const now = new Date('2026-10-09T00:30:00Z');
const filename = metadata => getExportFilename(metadata, now);

test('metadata combinations follow event, player, and site precedence', () => {
    const cases = [
        [{ Event: 'Cup', Round: '2', White: 'Alice', Black: 'Bob', Date: '2024.02.29' }, 'Cup_round_2_Alice_vs_Bob_20240229.5dpgn'],
        [{ Event: 'Cup', White: 'Alice', Black: 'Bob' }, 'Cup_Alice_vs_Bob_20261009.5dpgn'],
        [{ Round: '2', White: 'Alice', Black: 'Bob' }, 'Alice_vs_Bob_20261009.5dpgn'],
        [{ White: 'Alice' }, 'Alice_vs_anonymous_20261009.5dpgn'],
        [{ Black: 'Bob' }, 'anonymous_vs_Bob_20261009.5dpgn'],
        [{ Event: 'Cup', Round: '2', Site: 'Paris' }, 'Cup_round_2_20261009.5dpgn'],
        [{ Site: 'Paris', Round: '2', Date: '2024-01-01' }, 'Paris_20240101.5dpgn'],
        [{ Round: '2' }, 'game_20261009.5dpgn'],
        [{ Date: '20240101' }, 'game_20240101.5dpgn'],
        [{}, 'game_20261009.5dpgn']
    ];
    for (const [headers, expected] of cases) assert.equal(filename(headers), expected);
});

test('missing placeholders and values sanitized to nothing use the same fallbacks', () => {
    assert.equal(filename({ Event: '?', Round: '4', White: '   ', Black: '???', Site: ' /:* ' }), 'game_20261009.5dpgn');
    assert.equal(filename({ Event: '...', White: 'Alice', Black: '/' }), 'Alice_vs_anonymous_20261009.5dpgn');
    assert.equal(filename({ Event: 'Cup', Round: '?', White: '?', Black: '?' }), 'Cup_20261009.5dpgn');
});

test('complete dates are validated, including leap years and early years', () => {
    for (const Date of ['2025.02.29', '2024.04.31', '2024.13.01', '2024.01.00', '2024.??.??', '0000.01.01', '?']) {
        assert.equal(filename({ Date }), 'game_20261009.5dpgn');
    }
    assert.equal(filename({ Date: '0099.01.01' }), 'game_00990101.5dpgn');
});

test('unsafe characters and whitespace become underscores while Unicode is preserved', () => {
    assert.equal(filename({ Event: ' ..Cafe\u0301 / Cup:*?<>|"\\\n ', White: '李 小龍', Black: 'Zoë' }),
        'Café_Cup_李_小龍_vs_Zoë_20261009.5dpgn');
    assert.equal(filename({ Site: 'New\u0000\u007f York' }), 'New_York_20261009.5dpgn');
});

test('Windows device names are protected even before a dot', () => {
    for (const Site of ['CON', 'nul', 'COM1.notes', 'LPT9', 'COM¹']) {
        assert.equal(filename({ Site }), `game_${Site}_20261009.5dpgn`);
    }
    assert.equal(filename({ Site: 'CONtest' }), 'CONtest_20261009.5dpgn');
});

test('long ASCII and multibyte metadata fit the byte budget and retain structure', () => {
    for (const character of ['x', '棋', '😀']) {
        const result = filename({ Event: character.repeat(1000), Round: '123', White: character.repeat(1000), Black: character.repeat(1000) });
        assert.ok(Buffer.byteLength(result, 'utf8') <= 180);
        assert.match(result, /_round_123_/);
        assert.match(result, /_vs_/);
        assert.ok(result.endsWith('_20261009.5dpgn'));
        assert.ok(!result.includes('\ufffd'));
    }
});
