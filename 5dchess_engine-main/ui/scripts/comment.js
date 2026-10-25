// Read editable HTML without losing block boundaries or counting the browser's
// trailing caret-placeholder <br> as an extra line. Works while hidden too.
export function readCommentText(element) {
    let text = '';
    let previousBlock = false;
    const children = Array.from(element.childNodes);
    children.forEach((node, index) => {
        if (node.nodeType === 3) {
            if (previousBlock) text += '\n';
            text += node.textContent;
            previousBlock = false;
            return;
        }
        if (node.nodeType !== 1) return;
        const block = node.tagName === 'DIV' || node.tagName === 'P';
        if (node.tagName === 'BR') {
            if (index !== children.length - 1) text += '\n';
            previousBlock = false;
        } else {
            if (index > 0 && (block || previousBlock)) text += '\n';
            text += readCommentText(node);
            previousBlock = block;
        }
    });
    return text;
}
