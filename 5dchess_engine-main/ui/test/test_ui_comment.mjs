import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
const source = await readFile(new URL('../scripts/comment.js', import.meta.url), 'utf8');
const { readCommentText } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const text = textContent => ({ nodeType: 3, textContent });
const el = (tagName, ...childNodes) => ({ nodeType: 1, tagName, childNodes });

test('Enter blocks preserve lines and empty lines when saving a comment', () => {
    const editor = el('DIV', text('This is a'), el('DIV', text('multi')),
        el('DIV', text('line')), el('DIV', el('BR')), el('DIV', text('comment')));
    assert.equal(readCommentText(editor), 'This is a\nmulti\nline\n\ncomment');
});

test('Shift+Enter breaks and trailing caret placeholders preserve intended lines', () => {
    assert.equal(readCommentText(el('DIV', text('a'), el('BR'), text('b'))), 'a\nb');
    assert.equal(readCommentText(el('DIV', text('a'), el('BR'), el('BR'))), 'a\n');
    assert.equal(readCommentText(el('DIV', el('BR'))), '');
    assert.equal(readCommentText(el('DIV', text('a'), el('DIV', el('BR')))), 'a\n');
});

test('saved plain text and inline formatting retain whitespace and characters', () => {
    const value = '  a\n\nb\n';
    assert.equal(readCommentText(el('DIV', text(value))), value);
    assert.equal(readCommentText(el('DIV', el('P', text('one')), el('P', el('B', text('two'))))), 'one\ntwo');
});
