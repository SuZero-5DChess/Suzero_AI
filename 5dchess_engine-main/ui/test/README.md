# Web UI tests

Startup tests cover ordered worker-message delivery when the engine becomes
ready before or after the UI, including replay without duplicate messages.
Loading-screen tests cover either asset group finishing first, both completing
together, and removal after the starting position is displayed.

The dependency-free Node tests cover phone taps, immediate selection and position previews, double-tap recognition and rollback, drags, pinch release, and canvas
resizing when panels reserve screen space, and display-density changes without
shifting camera bounds or pointer coordinates, plus export filename metadata
precedence, date validation, sanitization, and byte limits. Piece-cache tests
check independent raster LODs, cache reuse, and the SVG-to-raster zoom boundary.
Navigation tests cover sparse/empty grid movement, nearest-board distance,
Space dispatch and focus-view tolerance, hold acceleration and cancellation, key repeat,
and editing/native-control/modifier guards. Camera tests check exact settling
with zero velocity, synchronized position/logarithmic-zoom progress on retargeting, frame subdivision, and
animation convergence at unit zoom, duration calculation, exponential response,
and zero/tiny movements. Comment tests cover editable
block boundaries, blank lines, Shift+Enter, and browser caret placeholders.

Run from the repository root:

```sh
node --test ui/test/*.mjs
```

These tests exercise canvas, filename, and piece-cache behavior without a browser.
Flip tests cover coordinate and click agreement on rectangular boards, rendering
alignment of pieces, squares, labels, highlights and arrows, and changing the
file-order preference without moving boards or the camera.
Arrow tests cover endpoint-distance fading across canvas sizes, zoom levels and board
orientations, restoration on returning to an endpoint, skipping paths and
gradients at zero or nearly transparent opacity, restoration above the `0.002`
effective-alpha cutoff, and candidate filtering across board boundaries.
Opacity tests cover the inclusive `0.002` and `0.998` boundaries, invalid alpha,
applying the cutoff after arrow alpha scaling, and L/T and file/rank label fades
skipping near-zero draws and snapping near-one opacity to full opacity.
Piece-cache tests use a simulated canvas; actual raster pixels need browser checks. Layout and game-control
integration should also be checked in the built WebAssembly interface at phone
portrait, phone landscape, tablet, and desktop sizes.

From `ui/`, run `node --test test/*.mjs`.

For panel keyboard checks, build with `cd build-wasm && ./build.sh` and serve
`build-wasm/ui/`. With both desktop panels collapsed, repeatedly press Tab:
the comment editor, hidden Update/Discard actions, and Hint/continuation options
must be skipped, and the status title must not shift vertically. Expand each
panel and verify its controls become reachable. In compact mode, open Comment
and Continuations and verify those same controls remain accessible.

For camera shortcuts, check F, Space, double-Space, Shift+Space, and WASD in the
built UI, including flipped timelines, empty gaps, rapid presses during animation,
and typing in an expanded comment editor. Clicking Submit/Next should return
focus to the canvas; keyboard activation should retain button focus. Dialogs
and the compact tools menu must suppress camera shortcuts.

For board orientation, click Flip with **Do not reverse files when flipped** off:
files should run right to left and ranks from 1 at the top to 8 at the bottom on
an 8 × 8 board. Toggle the checkbox under Settings → Interface while flipped;
files should return to left-to-right order without moving boards or the camera.
Verify piece selection and arrows in both modes, then reload to confirm the
checkbox preference persists. Unflipped boards should always show files left to
right.

For arrow fading, zoom into the middle of a long arrow until both endpoints
leave the viewport. The shaft should fade smoothly and disappear once the nearer
endpoint is outside by 20% of the larger canvas dimension. Zoom out or pan back
to an endpoint and
verify that opacity returns smoothly. Repeat while flipped and with an endpoint
leaving through a viewport corner.

For comment saving, type several lines using Enter, including an empty line, then
click Update Comment. Verify the worker response retains each line and repeat with
Shift+Enter. Make another edit and discard it; the saved line breaks must remain.
Check both desktop and compact editors. While editing, the panel has an opaque
theme tint; clicking the canvas or a panel control restores its normal surface.

For held navigation, hold a direction for two seconds and verify that motion
accelerates. Release it, focus a comment editor, or blur the window; repeated
steps must stop without a queue of future destinations. Press Space, wait more than ten
seconds, then press it again: the same board must remain centered with surrounding
boards visible. Camera motion must land exactly without a final visible snap.

Space should widen whenever the camera is at a normal Focus view, including a
view reached through F or the Focus button. During the first Space animation,
press it again immediately to widen around the same destination; a third press
returns to normal Focus. Pan away or change zoom and verify Space focuses the
nearest visible board again. Check desktop and compact layouts.

For dragging, move slowly in each of the four directions, then release. The
camera must follow that direction throughout the drag and settling tail, without
reverse movement. Repeat with touch, and interrupt an opposite keyboard camera
movement with a drag. Regression tests cover repeated targets, direction changes,
and takeover from the rendered camera.

For combined movement and zoom, press Space or Focus during an existing camera
movement. Position and logarithmic zoom must start and slow together. Automated
camera tests verify shared progress and the remaining-error derivative after
retargeting, along with exact settling.

For diagonal navigation, press W+A in both key orders within 40 ms and verify
the same diagonal destination. Hold both keys, release one, then release the
other. Opposite keys should cancel per axis, and adding a key should preserve
acceleration. Repeat at distant zoom: the same hold should traverse more grid
steps while diagonal and cardinal movement have the same screen-speed budget.
Tests cover diagonal rays, sparse batched movement, chord order, release behavior,
opposite keys, and zoom/diagonal speed normalization.

For touch focusing, double tap a board to center it at normal Focus zoom; double
tap it again to show surrounding boards, and again to return. Repeat during the
focus animation and with flipped boards. Double tapping another board must focus
that board. Legal destinations must appear as soon as the worker responds. A
destination tap must display the calculated position, including new boards,
before the 300 ms window closes, without a temporary square highlight. Double tapping must
restore the previous selection and destinations without applying a move, even
when a preview response arrives late. Preview boards must disappear immediately
on cancellation and must not reappear from late replies. Normal application at
300 ms must keep the preview boards steady; rapid taps
on different boards or far-apart squares commit the earlier tap immediately.
Empty space must not focus a board. Dragging, pinching, cancellation, window blur,
and mouse interaction must roll back pending previews. Import, history navigation,
and changed positions must clear selection. Automated tests cover these selection
transitions, delayed and stale replies, gesture integration, merging local highlights,
and the main-thread/worker message protocol.

Check physical moves, captures, promotions, and moves branching into other times
or timelines. With an existing redo continuation, cancel a preview and verify
Redo still restores that continuation. Assess whether a board briefly appearing
and disappearing on a double tap feels distracting; appearance animation is not
implemented yet. Native preview tests cover replacement, invalid moves, unchanged
history and PGN, promotion, branching, and ordinary mutation semantics.

Branch-deletion tests cover exact promotion identity, target validation, cancel,
restore-all, stale confirmations, safe Next navigation, Submit preservation,
export gating, direct confirmation, and successful/failed imports. In the built
UI, verify that hovering reveals Delete/Restore, leaving a selected row hides it,
and keyboard focus reveals it. Pending rows must keep their height. Collapse
and reopen the panel, cancel a navigation/export confirmation, and restore all.
Confirm deletion and check exported PGN excludes the branch and descendants.
Check the expanded panel plus pending controls at 1100 × 600 and after switching
to compact layout and back.

For continuation icons, check desktop Previous/Next/Undo/Redo use the bundled
Lucide stroke icons and compact navigation retains its existing icons. Hover
Delete/Restore over both long and short move text: only overlapping text should
fade, with no bright patch in empty space. Leaving the row restores all text
without changing wrapping or height; keyboard focus should reveal the same
action and text mask.

For Hint, press H repeatedly with a continuation selected and verify selection
and keyboard focus survive, while the list scrolls to the newest suggestion.
Repeat by clicking Hint and in compact mode. The first hint from an empty list
should be selected; explicit deselection after staging branch deletion must
remain deselected. Ordinary refreshes must not scroll the list to the bottom.

For compact continuation editing, verify Select Next Action shows Edit and hides
row actions. Edit changes to Done; Done hides actions again. Stage a deletion:
Restore/Delete replace Done in the fixed header without changing its height.
Resolve pending deletions through either header action or restore the final row;
normal selection must return. Close/reopen and switch to desktop/back while
pending: deletions and controls must persist. Check 320px width and confirm the
header stays in place when scrolling the list or adding Hint suggestions.

## Illegal history

Worker tests in `test_branch_deletion.mjs` cover strict/permissive imports,
failed-load preservation, destination-summary permission checks before draft or
branch changes, rechecking permission after deletion confirmation, checkbox
locking/unlocking without restoring a preference, disabled Hint, illegal-history
status, and ordinary submission checks. `ui/wasm/test.js` exercises the real
binding's import option, continuation summaries, inherited flags, status and Hint.

In the built browser interface, import an illegal setup with the option off and
on. Check the red continuation rows in light and dark themes, selected and with
pending deletion. Enter illegal history, verify the checkbox is checked and
locked and Hint disabled, return to a legal ancestor, turn it off manually, and
press Next on an illegal continuation. The alert must leave drafts and pending
deletions unchanged. Repeat in the compact layout.

## Import diagnostics

`test_import_diagnostics.mjs` checks location labels, hover and pin behavior,
failed/loading/edited status transitions, retaining stale messages after edits,
suspending highlights during retries, cancellation and stale results, range-free
errors, missing-input markers and viewport synchronization. Worker tests verify
that prepared imports do not replace the open game before acceptance.
`ui/wasm/test.js` verifies real serialized syntax, move, setup and FEN diagnostics,
including UTF-8 to UTF-16 conversion and rebased board-metadata errors.

Browser checks should include wrapped text, tabs, Unicode comments, scrolling,
resizing and compact layouts. Verify that the yellow highlight follows the
native text, edits preserve the exact diagnostic wording with a gray state line and
disabled highlighting, and successful loads close the dialog without a browser alert. Check
short and long errors, missing ranges, and multiline ranges in light and dark
themes, including enlarged text. The editor height and bottom button positions
must stay unchanged across empty, failed, loading, and edited states; long
diagnostics must scroll within the reserved region. Check keyboard preview and
pinning, disabled stale locations, and live status announcements.
