// Reuse the compact shell unless the viewport meets every desktop requirement.
// `not all` negates the entire condition, including the minimum width/height ratio.
export const compactViewport = window.matchMedia('not all and (min-width: 1100px) and (min-height: 600px) and (min-aspect-ratio: 4/5)');

export function initMobileLayout() {
    const shell = document.getElementById('mobileLayout');
    const sheet = document.getElementById('mobileSheet');
    const heading = document.getElementById('mobileSheetTitle');
    const sheetHeader = document.getElementById('mobileSheetHeader');
    const continuationActions = document.getElementById('mobileContinuationActions');
    const editContinuations = document.getElementById('mobileEditContinuations');
    const restoreBranches = document.getElementById('restoreAllBranches');
    const desktopRestoreLabel = restoreBranches.textContent;
    const deleteBranches = document.getElementById('confirmBranches');
    const desktopDeleteLabel = deleteBranches.textContent;
    let editingContinuations = false;
    let pendingCount = 0;

    function updateContinuationEditor() {
        shell.classList.toggle('editing-continuations', editingContinuations || pendingCount > 0);
        editContinuations.hidden = pendingCount > 0;
        editContinuations.textContent = editingContinuations ? 'Done' : 'Edit';
        editContinuations.setAttribute('aria-pressed', String(editingContinuations));
    }
    editContinuations.onclick = () => {
        if (pendingCount > 0) return;
        editingContinuations = !editingContinuations;
        updateContinuationEditor();
    };
    document.addEventListener('pending-branches-change', event => {
        const hadPending = pendingCount > 0;
        pendingCount = event.detail.count;
        if (pendingCount > 0) editingContinuations = true;
        else if (hadPending) editingContinuations = false;
        updateContinuationEditor();
        if (hadPending && !pendingCount && compactViewport.matches
            && document.getElementById('pendingBranches').contains(document.activeElement)) {
            editContinuations.focus({ preventScroll: true });
        }
    });
    updateContinuationEditor();
    const panes = [...shell.querySelectorAll('[data-mobile-pane]')];
    const triggers = [...shell.querySelectorAll('[data-mobile-panel]')];
    const commentButton = document.getElementById('mobileCommentButton');
    const commentIcon = commentButton.querySelector('use');
    const commentText = document.querySelector('.hud-text');
    const metadataIcon = shell.querySelector('.mobile-game-button use');
    const metadata = document.getElementById('gameMetadata');
    const metadataToggle = document.getElementById('gameMetadataToggle');
    const focus = document.getElementById('btnFocus');
    const buttons = document.getElementById('buttonsPanel');
    const placements = [];
    let activePanel = null;
    let returnFocus = null;
    let metadataWasExpanded = false;

    function move(element, destination) {
        const marker = document.createComment('desktop placement');
        element.before(marker);
        placements.push({ element, marker });
        destination.appendChild(element);
    }

    function updateCommentIcon() {
        const hasComment = commentText.textContent.trim().length > 0;
        const icon = hasComment ? 'comment-message' : 'comment-empty';
        commentIcon.setAttribute('href', `./icons/remix.svg#icon-${icon}${activePanel === 'comments' ? '-open' : ''}`);
        commentButton.setAttribute('aria-label', hasComment ? 'Comment (has comment)' : 'Comment (empty)');
    }

    // Cover typing, discard, saving, and comments arriving from the game worker.
    const commentObserver = new MutationObserver(updateCommentIcon);
    commentObserver.observe(commentText, { childList: true, characterData: true, subtree: true });
    commentText.addEventListener('input', updateCommentIcon);
    updateCommentIcon();

    function showPanel(name, trigger = null) {
        activePanel = name;
        shell.classList.toggle('has-mobile-panel', Boolean(name));
        metadataIcon.setAttribute('href', name === 'game' ? './icons/remix.svg#icon-price-tag-3-fill' : './icons/remix.svg#icon-price-tag-3-line');
        updateCommentIcon();
        if (trigger) returnFocus = trigger;
        sheet.hidden = !name;
        panes.forEach(pane => { pane.hidden = pane.dataset.mobilePane !== name; });
        triggers.forEach(button => button.setAttribute('aria-expanded', String(button.dataset.mobilePanel === name)));
        sheetHeader.hidden = name === 'tools';
        continuationActions.hidden = name !== 'variations';
        if (name !== 'variations' && pendingCount === 0) {
            editingContinuations = false;
            updateContinuationEditor();
        }
        heading.textContent = { variations: 'Select Next Action', comments: 'Comment', game: 'Game details' }[name] || '';
        if (name === 'tools') sheet.removeAttribute('aria-labelledby');
        else sheet.setAttribute('aria-labelledby', 'mobileSheetTitle');
        if (name && trigger) document.getElementById('mobileSheetClose').focus();
    }

    function closePanel() {
        showPanel(null);
        if (returnFocus?.isConnected) returnFocus.focus();
    }

    triggers.forEach(button => button.addEventListener('click', () => {
        showPanel(activePanel === button.dataset.mobilePanel ? null : button.dataset.mobilePanel, button);
    }));
    document.getElementById('mobileSheetClose').addEventListener('click', closePanel);
    shell.addEventListener('keydown', event => {
        // Native controls must not also activate the application's game shortcuts.
        if ((event.key === 'Enter' || event.key === ' ') && event.target.closest('button')) event.stopPropagation();
        if (event.key === 'Escape' && activePanel) {
            closePanel();
            event.stopPropagation();
        }
    });

    document.addEventListener('metadata-visibility-change', event => {
        if (!event.detail.visible && activePanel === 'game') closePanel();
    });

    // Reuse the existing icon and title for a labeled, touch-sized tools menu.
    [...buttons.children].filter(button => button !== focus).forEach(button => {
        const label = document.createElement('span');
        label.className = 'mobile-tool-label';
        label.textContent = button.title;
        button.appendChild(label);
        button.setAttribute('aria-label', button.title);
    });
    focus.setAttribute('aria-label', 'Focus board');

    function updateLayout() {
        const compact = compactViewport.matches;
        document.body.classList.toggle('compact-layout', compact);
        shell.hidden = !compact;
        restoreBranches.textContent = compact ? 'Restore' : desktopRestoreLabel;
        deleteBranches.textContent = compact ? 'Delete' : desktopDeleteLabel;
        // Match visual order and keyboard traversal: the safe exit stays at the right.
        restoreBranches.parentElement.append(...(compact
            ? [deleteBranches, restoreBranches] : [restoreBranches, deleteBranches]));
        if (compact && placements.length === 0) {
            metadataWasExpanded = metadata.classList.contains('expanded');
            if (!metadataWasExpanded) metadataToggle.click();
            const destinations = [
                ['.hud-header', '#mobileStatus'],
                ['#canvas', '#mobileBoard'],
                ['#btnFocus', '#mobileBoard'],
                ['.remote-control', '#mobileNavigation'],
                ['#selectedDisplay', '#mobileContinuation'],
                ['#selectContainer', '[data-mobile-pane="variations"]'],
                ['#pendingBranches', '#mobileContinuationActions'],
                ['.hud-content', '[data-mobile-pane="comments"]'],
                ['#hudCommentActions', '[data-mobile-pane="comments"]'],
                ['#buttonsPanel', '[data-mobile-pane="tools"]'],
                ['#gameMetadata', '[data-mobile-pane="game"]']
            ];
            destinations.forEach(([source, target]) => move(document.querySelector(source), shell.querySelector(target)));
        } else if (!compact && placements.length) {
            showPanel(null);
            for (const { element, marker } of placements.reverse()) marker.replaceWith(element);
            placements.length = 0;
            if (!metadataWasExpanded) metadataToggle.click();
            // Desktop positioning depends on the restored elements' dimensions.
            window.dispatchEvent(new Event('resize'));
        }
    }
    compactViewport.addEventListener('change', updateLayout);
    updateLayout();
}
