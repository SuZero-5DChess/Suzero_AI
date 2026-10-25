// canvas.js
// implements infinite scrollable canvas in general

export class AnimationManager {
    constructor(callback) {
        this.running = false;
        this.timeRecord = 0;
        this.callback = callback;
    }

    start() {
        if (this.running) return;
        this.running = true;
        this.timeRecord = 0;
        requestAnimationFrame((time) => this._animate(time));
    }

    stop() {
        this.running = false;
    }

    _animate(timeNow) {
        let timeDiff = 20;
        if (timeNow && this.timeRecord) {
            timeDiff = Math.max(timeNow - this.timeRecord, 0);
        }
        this.timeRecord = timeNow;

        const shouldContinue = this.callback(timeDiff);
        
        if (this.running && shouldContinue) {
            requestAnimationFrame((time) => this._animate(time));
        } else {
            this.running = false;
        }
    }
}

class Camera {
    constructor(x = 0, y = 0, scale = 0) {
        this.x = x;
        this.y = y;
        this.scale = scale; // logarithmic scale (base 2)
    }

    clone() {
        return new Camera(this.x, this.y, this.scale);
    }

    isCloseTo(other, positionThreshold = 0.1, scaleThreshold = 0.02) {
        return Math.abs(this.x - other.x) < positionThreshold 
            && Math.abs(this.y - other.y) < positionThreshold 
            && Math.abs(2 ** (this.scale - other.scale) - 1) < scaleThreshold;
    }

    lerpTowards(target, timeDelta, speed = 0.01) {
        const axes = ['x', 'y', 'scale'];
        if (!this.motion || axes.some(axis => this.motion.target[axis] !== target[axis])) {
            // Normalize remaining errors by the original stopping gauges.
            const gauge = Math.max(Math.abs(target.x - this.x) / 0.1,
                Math.abs(target.y - this.y) / 0.1,
                Math.abs(target.scale - this.scale) * Math.LN2 / Math.log1p(0.02));
            const decay = Math.log1p(speed * (1000 / 60)) / (1000 / 60);
            const duration = Math.log1p(gauge) / decay;
            this.motion = {
                start: this.clone(), target: target.clone(), elapsed: 0, duration, decay
            };
        }
        const motion = this.motion;
        motion.elapsed = Math.min(motion.duration, motion.elapsed + timeDelta);
        this.velocity = {};
        if (motion.elapsed === motion.duration) {
            for (const axis of axes) {
                this[axis] = motion.target[axis];
                this.velocity[axis] = 0;
            }
            return;
        }
        const t = motion.elapsed, duration = motion.duration, k = motion.decay;
        const u = t / duration, q = k * duration;
        const tail = Math.exp(-q);
        // Series avoid cancellation for nearly identical camera positions.
        const denominator = q < 0.001
            ? q*q * (0.5 - q/3 + q*q/8 - q*q*q/30)
            : -Math.expm1(-q) - q * tail;
        const numerator = q < 0.001
            ? q*q * (u - u*u/2 + q*(u*u*u/6 - u/2)
                + q*q*(u/6 - u**4/24) + q**3*(u**5/120 - u/24))
            : -Math.expm1(-k*t) - k*t * tail;
        const progress = numerator / denominator;
        const slope = k * (Math.expm1(-k*t) - Math.expm1(-q)) / denominator;
        for (const axis of axes) {
            const distance = motion.target[axis] - motion.start[axis];
            // Analytic solution of the shared remaining-error ODE. Position and
            // logarithmic zoom start and decelerate together, also on retargeting.
            this[axis] = motion.start[axis] + distance * progress;
            this.velocity[axis] = distance * slope;
        }
    }

    getZoomLevel() {
        return Math.pow(2, this.scale);
    }

    screenToWorld(screenX, screenY, canvasWidth, canvasHeight) {
        const centerX = canvasWidth / 2;
        const centerY = canvasHeight / 2;
        const zoomLevel = this.getZoomLevel();
        
        return {
            x: (screenX - centerX) / zoomLevel - this.x,
            y: (screenY - centerY) / zoomLevel - this.y
        };
    }

    worldToScreen(worldX, worldY, canvasWidth, canvasHeight) {
        const centerX = canvasWidth / 2;
        const centerY = canvasHeight / 2;
        const zoomLevel = this.getZoomLevel();
        
        return {
            x: (worldX + this.x) * zoomLevel + centerX,
            y: (worldY + this.y) * zoomLevel + centerY
        };
    }

    getVisibleWorldBounds(canvasWidth, canvasHeight) {
        const topLeft = this.screenToWorld(0, 0, canvasWidth, canvasHeight);
        const bottomRight = this.screenToWorld(canvasWidth, canvasHeight, canvasWidth, canvasHeight);
        
        return {
            left: topLeft.x,
            top: topLeft.y,
            right: bottomRight.x,
            bottom: bottomRight.y,
            width: bottomRight.x - topLeft.x,
            height: bottomRight.y - topLeft.y
        };
    }
}

export class InfiniteScrollableCanvas {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            throw new Error(`Canvas element with id "${canvasId}" not found`);
        }
        
        this.ctx = this.canvas.getContext('2d');
        
        // Configuration
        this.config = {
            minZoom: options.minZoom ?? -5.0,
            maxZoom: options.maxZoom ?? 7.0,
            zoomSpeed: options.zoomSpeed ?? 0.005,
            dragSpeed: options.dragSpeed ?? 1.0,
            lerpSpeed: options.lerpSpeed ?? 0.01,
            dragThreshold: options.dragThreshold ?? 10, // Minimum distance in pixels to consider it a drag
            ...options
        };
        
        // Camera state
        this.cameraTarget = new Camera(
            options.initialX ?? 0,
            options.initialY ?? 0,
            options.initialZoom ?? -1.0
        );
        this.cameraCurrent = this.cameraTarget.clone();
        
        // Input state
        this.isMouseDown = false;
        this.isDragging = false;
        this.dragStart = { x: 0, y: 0 };
        this.lastMousePos = { x: 0, y: 0 };
        this.mouseStartPos = { x: 0, y: 0 };
        
        // Touch state
        this.touchStartDistance = 0;
        this.touchStartScale = 0;
        this.touchMoved = false;
        this.touchStartPos = { x: 0, y: 0 };
        this.pendingTap = null;
        
        // Animation
        this.animationManager = new AnimationManager((timeDiff) => this._onAnimationFrame(timeDiff));
        
        // Callbacks (to be overridden by implementation)
        this.onRender = null;
        this.onClick = null;
        this.getDoubleTapTarget = null;
        this.onDoubleTap = null;
        this.onTapPreview = null;
        this.onTapCommit = null;
        this.onTapCancel = null;
        this.onRightClick = null;
        this.onHover = null;
        
        this._setupEventListeners();
        this.resize();
        this._watchPixelRatio();
    }

    _watchPixelRatio() {
        // Moving between displays can change density without changing CSS size.
        this.pixelRatioQuery?.removeEventListener('change', this.pixelRatioListener);
        this.pixelRatioQuery = window.matchMedia(`(resolution: ${globalThis.devicePixelRatio || 1}dppx)`);
        this.pixelRatioListener = () => {
            this.resize();
            this._watchPixelRatio();
        };
        this.pixelRatioQuery.addEventListener('change', this.pixelRatioListener);
    }

    _setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this._handleMouseDown(e));
        this.canvas.addEventListener('mouseup', (e) => this._handleMouseUp(e));
        this.canvas.addEventListener('mousemove', (e) => this._handleMouseMove(e));
        this.canvas.addEventListener('mouseover', () => this.isMouseDown = false);
        this.canvas.addEventListener('mouseout', () => this.isMouseDown = false);
        this.canvas.addEventListener('wheel', (e) => this._handleWheel(e));
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => this._handleTouchStart(e));
        this.canvas.addEventListener('touchmove', (e) => this._handleTouchMove(e));
        this.canvas.addEventListener('touchend', (e) => this._handleTouchEnd(e));
        this.canvas.addEventListener('touchcancel', () => this._handleTouchCancel());
        window.addEventListener('blur', () => this._handleTouchCancel());
        
        // Window resize
        window.addEventListener('resize', () => this.resize());
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this.canvas);
    }

    _handleMouseDown(e) {
        this._cancelPendingTap();
        if (e.button !== 0) return; // Only left mouse button for dragging
        
        this.isMouseDown = true;
        const dragSpeed = this.config.dragSpeed / this.cameraCurrent.getZoomLevel();
        this.dragStart.speed = dragSpeed;
        this.dragStart.x = dragSpeed * e.clientX - this.cameraCurrent.x;
        this.dragStart.y = dragSpeed * e.clientY - this.cameraCurrent.y;
        this.mouseStartPos.x = e.clientX;
        this.mouseStartPos.y = e.clientY;
        this.isDragging = false;
    }

    _handleMouseUp(e) {
        this.isMouseDown = false;
        //console.log(`_handleMouseUp ${e.button} ${this.isDragging}`);

        if (e.button === 0 && this.isDragging) {
            return; // Don't treat drag release as click
        }
        
        const worldPos = this._getMouseWorldPosition(e);

        if (e.button === 0 && this.onClick) {
            this.onClick(worldPos.x, worldPos.y, e);
        } else if (e.button === 2 && this.onRightClick) {
            this.onRightClick(worldPos.x, worldPos.y, e);
        }
    }

    _handleMouseMove(e) {
        const worldPos = this._getMouseWorldPosition(e);
        this.lastMousePos = worldPos;
        
        if (this.isMouseDown) {
            // Check if movement exceeds drag threshold
            const distance = this._getDistance(e.clientX, e.clientY, this.mouseStartPos.x, this.mouseStartPos.y);
            if (distance > this.config.dragThreshold) {
                this.isDragging = true;
            }
            
            // Only perform drag if we've exceeded the threshold
            if (this.isDragging) {
                const dragSpeed = this.dragStart.speed ?? this.config.dragSpeed / this.cameraCurrent.getZoomLevel();
                this.cameraCurrent.motion = null;
                this.cameraTarget.scale = this.cameraCurrent.scale;
                this.cameraTarget.x = dragSpeed * e.clientX - this.dragStart.x;
                this.cameraTarget.y = dragSpeed * e.clientY - this.dragStart.y;
                this.startAnimation();
            }
        }
        
        if (this.onHover) {
            this.onHover(worldPos.x, worldPos.y, e);
        }
    }

    _handleWheel(e) {
        e.preventDefault();
        
        this.cameraTarget.scale += e.deltaY * -this.config.zoomSpeed;
        this.cameraTarget.scale = Math.max(this.config.minZoom, 
                                           Math.min(this.config.maxZoom, this.cameraTarget.scale));
        this.startAnimation();
    }

    _getMouseWorldPosition(e) {
        const rect = this.canvas.getBoundingClientRect();
        const screenX = (e.clientX - rect.left) / (rect.right - rect.left) * this.viewportSize.width;
        const screenY = (e.clientY - rect.top) / (rect.bottom - rect.top) * this.viewportSize.height;
        
        return this.cameraCurrent.screenToWorld(screenX, screenY, this.viewportSize.width, this.viewportSize.height);
    }

    _getTouchWorldPosition(touch) {
        const rect = this.canvas.getBoundingClientRect();
        const screenX = (touch.clientX - rect.left) / (rect.right - rect.left) * this.viewportSize.width;
        const screenY = (touch.clientY - rect.top) / (rect.bottom - rect.top) * this.viewportSize.height;
        
        return this.cameraCurrent.screenToWorld(screenX, screenY, this.viewportSize.width, this.viewportSize.height);
    }

    _getTouchDistance(touch1, touch2) {
        const dx = touch1.clientX - touch2.clientX;
        const dy = touch1.clientY - touch2.clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    _getDistance(x1, y1, x2, y2) {
        const dx = x1 - x2;
        const dy = y1 - y2;
        return Math.sqrt(dx * dx + dy * dy);
    }

    _handleTouchStart(e) {
        e.preventDefault();
        
        if (e.touches.length === 1) {
            // Single finger touch - start potential drag or tap
            const touch = e.touches[0];
            const dragSpeed = this.config.dragSpeed / this.cameraCurrent.getZoomLevel();
            this.dragStart.speed = dragSpeed;
            this.dragStart.x = dragSpeed * touch.clientX - this.cameraCurrent.x;
            this.dragStart.y = dragSpeed * touch.clientY - this.cameraCurrent.y;
            this.touchStartPos.x = touch.clientX;
            this.touchStartPos.y = touch.clientY;
            this.isDragging = false;
            this.touchMoved = false;
        } else if (e.touches.length >= 2) {
            // Two finger pinch - start zoom
            this._cancelPendingTap();
            this.touchMoved = true;
            this.touchStartDistance = this._getTouchDistance(e.touches[0], e.touches[1]);
            this.touchStartScale = this.cameraTarget.scale;
        }
    }

    _handleTouchMove(e) {
        e.preventDefault();
        
        if (e.touches.length === 1) {
            const touch = e.touches[0];
            
            // Check if movement exceeds drag threshold
            const distance = this._getDistance(touch.clientX, touch.clientY, this.touchStartPos.x, this.touchStartPos.y);
            if (distance > this.config.dragThreshold) {
                this._cancelPendingTap();
                this.isDragging = true;
            }
            
            // Only perform drag if we've exceeded the threshold
            if (this.isDragging) {
                const dragSpeed = this.dragStart.speed ?? this.config.dragSpeed / this.cameraCurrent.getZoomLevel();
                this.cameraCurrent.motion = null;
                this.cameraTarget.scale = this.cameraCurrent.scale;
                this.cameraTarget.x = dragSpeed * touch.clientX - this.dragStart.x;
                this.cameraTarget.y = dragSpeed * touch.clientY - this.dragStart.y;
                this.startAnimation();
            }
            
            // Update hover position
            const worldPos = this._getTouchWorldPosition(touch);
            this.lastMousePos = worldPos;
            if (this.onHover) {
                this.onHover(worldPos.x, worldPos.y, e);
            }
        } else if (e.touches.length >= 2) {
            // Two finger pinch zoom
            const currentDistance = this._getTouchDistance(e.touches[0], e.touches[1]);
            const scaleFactor = currentDistance / this.touchStartDistance;
            
            // Use logarithmic scaling for zoom
            const zoomDelta = Math.log2(scaleFactor);
            this.cameraTarget.scale = this.touchStartScale + zoomDelta;
            this.cameraTarget.scale = Math.max(this.config.minZoom, 
                                               Math.min(this.config.maxZoom, this.cameraTarget.scale));
            this.startAnimation();
            this.isDragging = true;
        }
    }

    _handleTouchEnd(e) {
        e.preventDefault();
        
        if (e.touches.length === 0) {
            // A drag or pinch must never become a piece selection on release.
            if (!e.changedTouches.length) return;
            const touch = e.changedTouches[0];
            const distance = this._getDistance(touch.clientX, touch.clientY, this.touchStartPos.x, this.touchStartPos.y);
            const isTap = !this.touchMoved && !this.isDragging && distance <= this.config.dragThreshold;
            
            if (isTap && e.changedTouches.length > 0) {
                this._handleTap(touch, e);
            } else {
                this._cancelPendingTap();
            }
            
            this.isDragging = false;
        } else if (e.touches.length === 1) {
            // One finger released, but another still touching
            // Start drag with remaining finger
            const touch = e.touches[0];
            const dragSpeed = this.config.dragSpeed / this.cameraCurrent.getZoomLevel();
            this.dragStart.speed = dragSpeed;
            this.dragStart.x = dragSpeed * touch.clientX - this.cameraCurrent.x;
            this.dragStart.y = dragSpeed * touch.clientY - this.cameraCurrent.y;
            this.isDragging = false;
            this.touchMoved = true;
        }
    }

    _cancelPendingTap(commit = false) {
        if (!this.pendingTap) return;
        const pending = this.pendingTap;
        clearTimeout(pending.timer);
        this.pendingTap = null;
        if (pending.previewed) {
            if (commit) this.onTapCommit?.(pending.preview);
            else this.onTapCancel?.(pending.preview);
        } else if (commit) {
            this.onClick?.(pending.world.x, pending.world.y, pending.event);
        }
    }

    _handleTouchCancel() {
        this._cancelPendingTap();
        this.touchMoved = true;
        this.isDragging = false;
    }

    _handleTap(touch, event) {
        const world = this._getTouchWorldPosition(touch);
        const target = this.getDoubleTapTarget?.(world.x, world.y);
        const previous = this.pendingTap;
        const now = performance.now();
        if (previous && target != null && target === previous.target
            && now - previous.time <= 300
            && this._getDistance(touch.clientX, touch.clientY, previous.x, previous.y) <= 24) {
            this._cancelPendingTap();
            this.onDoubleTap?.(world.x, world.y, event);
            return;
        }
        if (previous) {
            this._cancelPendingTap(true);
        }
        if (target == null || !this.onDoubleTap) {
            this.onClick?.(world.x, world.y, event);
            return;
        }
        const pending = {
            target, world, event, time: now, x: touch.clientX, y: touch.clientY
        };
        this.pendingTap = pending;
        pending.previewed = Boolean(this.onTapPreview);
        if (pending.previewed) pending.preview = this.onTapPreview(world.x, world.y, event);
        pending.timer = setTimeout(() => this._cancelPendingTap(true), 300);
    }

    _onAnimationFrame(timeDiff) {
        // Update camera position
        this.cameraCurrent.lerpTowards(this.cameraTarget, timeDiff, this.config.lerpSpeed);
        
        // Render
        this._render();
        
        // Continue animating if camera hasn't reached target
        return ['x', 'y', 'scale'].some(axis => this.cameraCurrent[axis] !== this.cameraTarget[axis]);
    }

    _render() {
        this.ctx.save();
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.scale(
            this.canvas.width / this.viewportSize.width,
            this.canvas.height / this.viewportSize.height
        );
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';
        
        // Apply camera transform
        const centerX = this.viewportSize.width / 2;
        const centerY = this.viewportSize.height / 2;
        const zoom = this.cameraCurrent.getZoomLevel();
        
        this.ctx.translate(centerX, centerY);
        this.ctx.scale(zoom, zoom);
        this.ctx.translate(this.cameraCurrent.x, this.cameraCurrent.y);
        
        // Call user-defined render callback
        if (this.onRender) {
            const bounds = this.cameraCurrent.getVisibleWorldBounds(
                this.viewportSize.width,
                this.viewportSize.height
            );
            this.onRender(this.ctx, bounds);
        }
        
        this.ctx.restore();
    }

    startAnimation() {
        this.animationManager.start();
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width));
        const height = Math.max(1, Math.round(rect.height));
        // Preserve the relative zoom while a sheet or orientation changes the usable area.
        if (this.viewportSize && document.body.classList.contains('compact-layout')) {
            const ratio = Math.min(width, height) / Math.min(this.viewportSize.width, this.viewportSize.height);
            const delta = Math.log2(ratio);
            for (const camera of [this.cameraCurrent, this.cameraTarget]) {
                camera.scale = Math.max(this.config.minZoom, Math.min(this.config.maxZoom, camera.scale + delta));
            }
        }
        this.viewportSize = { width, height };
        this.pixelRatio = globalThis.devicePixelRatio || 1;
        this.canvas.width = Math.max(1, Math.round(width * this.pixelRatio));
        this.canvas.height = Math.max(1, Math.round(height * this.pixelRatio));
        this._render();
    }

    moveTo(x, y, zoom = null) {
        this.cameraTarget.x = -x;
        this.cameraTarget.y = -y;
        if (zoom !== null) {
            this.cameraTarget.scale = zoom;
        }
        this.startAnimation();
    }

    resetView() {
        this.moveTo(0, 0, this.config.initialZoom ?? -1.0);
    }

    getCamera() {
        return this.cameraCurrent.clone();
    }

    getVisibleBounds() {
        return this.cameraCurrent.getVisibleWorldBounds(
            this.viewportSize.width,
            this.viewportSize.height
        );
    }
}

