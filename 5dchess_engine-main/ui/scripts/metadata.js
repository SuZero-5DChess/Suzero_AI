// Read-only game headers, independent of the position currently being viewed.
const panel = document.getElementById('gameMetadata');
const summary = document.getElementById('gameMetadataSummary');
const details = document.getElementById('gameMetadataDetails');
const toggle = document.getElementById('gameMetadataToggle');
const icon = toggle.querySelector('path');
const expandIcon = 'M6.41421 5H10V3H3V10H5V6.41421L9.29289 10.7071L10.7071 9.29289L6.41421 5ZM21 14H19V17.5858L14.7071 13.2929L13.2929 14.7071L17.5858 19H14V21H21V14Z';
const foldIcon = 'M9.00008 4.00008H11.0001V11.0001H4.00008V9.00008H7.58586L3.29297 4.70718L4.70718 3.29297L9.00008 7.58586V4.00008ZM20 15H16.4142L20.7071 19.2929L19.2929 20.7071L15 16.4142V20H13V13H20V15Z';
const headerElements = [...panel.querySelectorAll('[data-game-header]')];
let titleFrame = null;
function refreshMetadataTitles() {
    titleFrame = null;
    for (const element of headerElements) {
        const known = !element.classList.contains('game-metadata-unknown')
            && !element.classList.contains('game-metadata-result-empty')
            && element.textContent !== '—';
        const truncated = element.clientWidth > 0
            && getComputedStyle(element).textOverflow === 'ellipsis'
            && element.scrollWidth > element.clientWidth;
        if (known && truncated) element.title = element.textContent;
        else element.removeAttribute('title');
    }
}
function scheduleMetadataTitles() {
    if (titleFrame === null) titleFrame = requestAnimationFrame(refreshMetadataTitles);
}
const titleObserver = new ResizeObserver(scheduleMetadataTitles);
headerElements.forEach(element => titleObserver.observe(element));
window.addEventListener('resize', scheduleMetadataTitles);
document.fonts.ready.then(scheduleMetadataTitles);
document.fonts.addEventListener('loadingdone', scheduleMetadataTitles);

let expanded = false;
let hasMetadata = false;
const showPanelCheckbox = document.getElementById('showMetadataPanel');
const SHOW_PANEL_KEY = 'show-metadata-panel';
let showPanel = true;
try {
    showPanel = localStorage.getItem(SHOW_PANEL_KEY) !== 'false';
} catch {
    // Keep the default when browser storage is unavailable.
}
showPanelCheckbox.checked = showPanel;

function updateVisibility() {
    const visible = showPanel && hasMetadata;
    panel.hidden = !visible;
    scheduleMetadataTitles();
    document.querySelector('.mobile-game-button').hidden = !visible;
    document.dispatchEvent(new CustomEvent('metadata-visibility-change', { detail: { visible } }));
}

showPanelCheckbox.addEventListener('change', () => {
    showPanel = showPanelCheckbox.checked;
    try {
        localStorage.setItem(SHOW_PANEL_KEY, String(showPanel));
    } catch {
        // The setting still applies for this session when storage is unavailable.
    }
    updateVisibility();
});

function setExpanded(value) {
    expanded = value;
    panel.classList.toggle('expanded', expanded);
    summary.hidden = expanded;
    details.hidden = !expanded;
    toggle.setAttribute('aria-expanded', String(expanded));
    toggle.setAttribute('aria-label', expanded ? 'Fold game information' : 'Expand game information');
    toggle.title = toggle.getAttribute('aria-label');
    icon.setAttribute('d', expanded ? foldIcon : expandIcon);
    scheduleMetadataTitles();
}

toggle.addEventListener('click', () => setExpanded(!expanded));
// Native button activation must not also submit a turn or focus the board.
toggle.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') event.stopPropagation();
});
setExpanded(false);
updateVisibility();

export function setGameMetadata(metadata = {}) {
    const headers = Object.fromEntries(Object.entries(metadata).map(([key, value]) => [key.toLowerCase(), value]));
    hasMetadata = ['event', 'round', 'white', 'black', 'site', 'date', 'result'].some(key => {
        const value = String(headers[key] ?? '').trim();
        return value !== '' && value !== '?' && (key !== 'result' || value !== '*');
    });
    updateVisibility();
    for (const element of headerElements) {
        const key = element.dataset.gameHeader;
        const value = String(headers[key] ?? '').trim();
        const known = value !== '' && value !== '?';
        const isPlayer = key === 'white' || key === 'black';
        element.textContent = known ? value : (isPlayer ? 'Unknown Player' : '—');
        element.classList.toggle('game-metadata-unknown', isPlayer && !known);
        element.removeAttribute('title');
        if (key === 'result') {
            const missing = !known || value === '*';
            const showSeparator = missing && details.contains(element);
            element.hidden = missing && !showSeparator;
            element.classList.toggle('game-metadata-result-empty', showSeparator);
            if (showSeparator) element.textContent = '/';
        }
        if (key === 'event') element.hidden = !known;
    }
    scheduleMetadataTitles();
}
