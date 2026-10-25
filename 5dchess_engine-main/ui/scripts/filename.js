const EXTENSION = '.5dpgn';
const MAX_BYTES = 180;
const encoder = new TextEncoder();
// eslint-disable-next-line no-control-regex -- Filenames must exclude control characters.
const UNSAFE = /[<>:"/\\|?*\u0000-\u001f\u007f-\u009f\s]+/gu;

function clean(value) {
    const text = String(value ?? '').trim();
    if (!text || /^\?+$/.test(text)) return '';
    return text.normalize('NFC')
        .replace(UNSAFE, '_')
        .replace(/_+/g, '_').replace(/^[._ ]+|[._ ]+$/g, '');
}

function dateStamp(value, now) {
    const match = String(value ?? '').trim().match(/^(\d{4})([.-]?)(\d{2})\2(\d{2})$/);
    if (match) {
        const [, year, , month, day] = match;
        const date = new Date(0);
        date.setUTCFullYear(Number(year), Number(month) - 1, Number(day));
        if (Number(year) > 0 && date.getUTCFullYear() === Number(year) &&
            date.getUTCMonth() + 1 === Number(month) && date.getUTCDate() === Number(day)) {
            return year + month + day;
        }
    }
    return now.toISOString().slice(0, 10).replace(/-/g, '');
}

function shorten(value, budget) {
    let result = '';
    for (const character of value) {
        const bytes = encoder.encode(character).length;
        if (bytes > budget) break;
        result += character;
        budget -= bytes;
    }
    return result;
}

// Keep the date and extension intact; share the remaining space between headers.
export function getExportFilename(metadata = {}, now = new Date()) {
    const headers = Object.fromEntries(Object.entries(metadata).map(([key, value]) => [key.toLowerCase(), value]));
    const event = clean(headers.event);
    const white = clean(headers.white);
    const black = clean(headers.black);
    const parts = [];
    const add = (value, separator = '') => parts.push({ value, separator });
    if (event) {
        add(event);
        const round = clean(headers.round);
        if (round) add(round, '_round_');
    }
    if (white || black) {
        add(white || 'anonymous', event ? '_' : '');
        add(black || 'anonymous', '_vs_');
    }
    if (!parts.length) add(clean(headers.site) || 'game');
    // Windows reserves these names even when followed by a dot and more text.
    if (/^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)/i.test(parts[0].value)) {
        parts[0].value = 'game_' + parts[0].value;
    }
    const suffix = '_' + dateStamp(headers.date, now) + EXTENSION;
    const budget = MAX_BYTES - encoder.encode(suffix).length;
    for (const part of parts) part.value = shorten(part.value, budget);
    const size = () => parts.reduce((total, part) => total + encoder.encode(part.separator + part.value).length, 0);
    while (size() > budget) {
        const longest = parts.reduce((a, b) => encoder.encode(a.value).length >= encoder.encode(b.value).length ? a : b);
        longest.value = Array.from(longest.value).slice(0, -1).join('');
    }
    return parts.map(part => part.separator + part.value).join('') + suffix;
}
