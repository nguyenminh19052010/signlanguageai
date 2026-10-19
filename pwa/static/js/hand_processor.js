

class LandmarkSmoother {
    constructor(alpha = 0.35) {
        this.alpha = alpha;
        this._prev = null;
    }

    update(vec) {
        if (!this._prev || this._prev.length !== vec.length) {
            this._prev = Float32Array.from(vec);
            return Float32Array.from(vec);
        }
        const blended = new Float32Array(vec.length);
        for (let i = 0; i < vec.length; i++) {
            blended[i] = this.alpha * vec[i] + (1 - this.alpha) * this._prev[i];
        }
        this._prev = blended;
        return blended;
    }

    reset() {
        this._prev = null;
    }
}

function normalizeHand(landmarks, width, height) {
    
    
    const pts = landmarks.map(lm => [lm.x * width, lm.y * height, lm.z * width]);

    
    const wrist = [pts[0][0], pts[0][1], pts[0][2]];
    for (let i = 0; i < pts.length; i++) {
        pts[i][0] -= wrist[0];
        pts[i][1] -= wrist[1];
        pts[i][2] -= wrist[2];
    }

    
    const dx = pts[9][0], dy = pts[9][1], dz = pts[9][2];
    const scale = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;

    const flat = new Float32Array(63); 
    for (let i = 0; i < 21; i++) {
        flat[i * 3 + 0] = pts[i][0] / scale;
        flat[i * 3 + 1] = pts[i][1] / scale;
        flat[i * 3 + 2] = pts[i][2] / scale;
    }
    return flat;
}

function buildFeatureVector(multiHandLandmarks, width, height) {
    const vectors = [];
    let handsVisible = false;

    if (multiHandLandmarks && multiHandLandmarks.length > 0) {
        handsVisible = true;
        for (let i = 0; i < Math.min(2, multiHandLandmarks.length); i++) {
            vectors.push(normalizeHand(multiHandLandmarks[i], width, height));
        }
    }

    
    const full = new Float32Array(126);
    if (vectors.length >= 1) {
        full.set(vectors[0], 0);
    }
    if (vectors.length >= 2) {
        full.set(vectors[1], 63);
    }

    return { vector: full, handsVisible };
}

function buildFeatureVectorHolistic(results, width, height) {
    const full = new Float32Array(1629);
    let handsVisible = false;
    let offset = 0;

    
    const addLandmarks = (lms, num) => {
        if (lms) {
            for (let i = 0; i < num; i++) {
                if (i < lms.length) {
                    full[offset++] = lms[i].x;
                    full[offset++] = lms[i].y;
                    full[offset++] = lms[i].z;
                } else {
                    full[offset++] = 0; full[offset++] = 0; full[offset++] = 0;
                }
            }
        } else {
            for (let i = 0; i < num * 3; i++) full[offset++] = 0;
        }
    };

    
    
    

    
    addLandmarks(results.poseLandmarks, 33);
    
    addLandmarks(results.faceLandmarks, 468);
    
    if (results.leftHandLandmarks) handsVisible = true;
    addLandmarks(results.leftHandLandmarks, 21);
    
    if (results.rightHandLandmarks) handsVisible = true;
    addLandmarks(results.rightHandLandmarks, 21);

    return { vector: full, handsVisible };
}

class HandProcessor {
    constructor(options = {}) {
        this.width = options.width || 640;
        this.height = options.height || 480;
        this.mode = options.mode || 'hands'; 
        this.smoother = new LandmarkSmoother(options.smoothAlpha || 0.35);
        this.hands = null;
        this.holistic = null;
        this.camera = null;
        this.onResult = options.onResult || (() => { });
        this.running = false;

        
        this._frameCount = 0;
        this._lastFpsTime = performance.now();
        this.fps = 0;
    }

    async initialize(videoElement, canvasElement) {
        this.videoEl = videoElement;
        this.canvasEl = canvasElement;
        this.canvasCtx = canvasElement.getContext('2d');

        if (this.mode === 'holistic') {
            this.holistic = new Holistic({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`,
            });
            this.holistic.setOptions({
                modelComplexity: 1,
                smoothLandmarks: true,
                minDetectionConfidence: 0.65,
                minTrackingConfidence: 0.55
            });
            this.holistic.onResults((results) => this._onResultsHolistic(results));
            await this.holistic.initialize();
        } else {
            this.hands = new Hands({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
            });
            this.hands.setOptions({
                maxNumHands: 2,
                modelComplexity: 1,
                minDetectionConfidence: 0.65,
                minTrackingConfidence: 0.55,
            });
            this.hands.onResults((results) => this._onResults(results));
            await this.hands.initialize();
        }
    }

    start() {
        if (this.running) return;
        this.running = true;

        this.camera = new Camera(this.videoEl, {
            onFrame: async () => {
                if (!this.running) return;
                if (this.mode === 'holistic' && this.holistic) {
                    await this.holistic.send({ image: this.videoEl });
                } else if (this.hands) {
                    await this.hands.send({ image: this.videoEl });
                }
            },
            width: this.width,
            height: this.height,
            facingMode: 'user',
        });

        this.camera.start();
    }

    stop() {
        this.running = false;
        if (this.camera) {
            this.camera.stop();
            this.camera = null;
        }
        this.smoother.reset();
    }

    _onResults(results) {
        const ctx = this.canvasCtx;

        
        const imgW = results.image.width || this.videoEl.videoWidth || this.width;
        const imgH = results.image.height || this.videoEl.videoHeight || this.height;

        const cw = this.canvasEl.width = imgW;
        const ch = this.canvasEl.height = imgH;

        
        ctx.save();
        ctx.clearRect(0, 0, cw, ch);
        ctx.drawImage(results.image, 0, 0, cw, ch);

        
        if (results.multiHandLandmarks) {
            for (const landmarks of results.multiHandLandmarks) {
                drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                    color: 'rgba(34, 197, 94, 0.7)',
                    lineWidth: 2,
                });
                drawLandmarks(ctx, landmarks, {
                    color: 'rgba(255, 255, 255, 0.85)',
                    fillColor: 'rgba(34, 197, 94, 0.5)',
                    radius: 3,
                });
            }
        }
        ctx.restore();

        
        const { vector, handsVisible } = buildFeatureVector(
            results.multiHandLandmarks,
            imgW,
            imgH
        );

        
        const smoothed = handsVisible
            ? this.smoother.update(vector)
            : vector;

        
        this._frameCount++;
        const now = performance.now();
        if (now - this._lastFpsTime >= 1000) {
            this.fps = this._frameCount;
            this._frameCount = 0;
            this._lastFpsTime = now;
        }

        
        this.onResult({
            vector: Array.from(smoothed),
            handsVisible,
            fps: this.fps,
        });
    }

    _onResultsHolistic(results) {
        const ctx = this.canvasCtx;
        const imgW = results.image.width || this.videoEl.videoWidth || this.width;
        const imgH = results.image.height || this.videoEl.videoHeight || this.height;
        const cw = this.canvasEl.width = imgW;
        const ch = this.canvasEl.height = imgH;

        ctx.save();
        ctx.clearRect(0, 0, cw, ch);
        ctx.drawImage(results.image, 0, 0, cw, ch);

        
        if (results.poseLandmarks) {
            drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
            drawLandmarks(ctx, results.poseLandmarks, { color: '#FF0000', lineWidth: 1, radius: 2 });
        }
        if (results.faceLandmarks) {
            drawConnectors(ctx, results.faceLandmarks, FACEMESH_TESSELATION, { color: '#C0C0C070', lineWidth: 1 });
        }
        if (results.leftHandLandmarks) {
            drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#CC0000', lineWidth: 2 });
            drawLandmarks(ctx, results.leftHandLandmarks, { color: '#00FF00', lineWidth: 1, radius: 2 });
        }
        if (results.rightHandLandmarks) {
            drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#00CC00', lineWidth: 2 });
            drawLandmarks(ctx, results.rightHandLandmarks, { color: '#FF0000', lineWidth: 1, radius: 2 });
        }
        ctx.restore();

        const { vector, handsVisible } = buildFeatureVectorHolistic(results, imgW, imgH);
        const smoothed = handsVisible ? this.smoother.update(vector) : vector;

        this._frameCount++;
        const now = performance.now();
        if (now - this._lastFpsTime >= 1000) {
            this.fps = this._frameCount;
            this._frameCount = 0;
            this._lastFpsTime = now;
        }

        this.onResult({
            vector: Array.from(smoothed),
            handsVisible,
            fps: this.fps,
        });
    }
}
