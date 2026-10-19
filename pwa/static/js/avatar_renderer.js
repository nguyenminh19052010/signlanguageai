class AvatarRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) return;

        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);
        this.scene.fog = new THREE.Fog(0x1a1a2e, 10, 50);

        this.camera = new THREE.PerspectiveCamera(45, this.width / this.height, 0.25, 100);
        this.camera.position.set(-5, 3, 10);
        this.camera.lookAt(new THREE.Vector3(0, 2, 0));

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(this.width, this.height);
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.container.appendChild(this.renderer.domElement);

        
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444);
        hemiLight.position.set(0, 20, 0);
        this.scene.add(hemiLight);

        const dirLight = new THREE.DirectionalLight(0xffffff);
        dirLight.position.set(-3, 10, -10);
        dirLight.castShadow = true;
        this.scene.add(dirLight);

        
        const mesh = new THREE.Mesh(
            new THREE.PlaneGeometry(200, 200),
            new THREE.MeshPhongMaterial({ color: 0x999999, depthWrite: false })
        );
        mesh.rotation.x = -Math.PI / 2;
        this.scene.add(mesh);

        const grid = new THREE.GridHelper(200, 40, 0x000000, 0x000000);
        grid.material.opacity = 0.2;
        grid.material.transparent = true;
        this.scene.add(grid);

        
        this.mixer = null;
        this.actions = {};
        this.activeAction = null;
        this.queue = [];
        this.isPlayingQueue = false;

        this.clock = new THREE.Clock();
        
        
        window.addEventListener('resize', this.onWindowResize.bind(this));

        
        this.animate();

        
        this.loadModel();
    }

    createFallbackAvatar() {
        
        this.fallbackGroup = new THREE.Group();
        
        const mat = new THREE.MeshPhongMaterial({ color: 0x00ffcc });
        
        const head = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat);
        head.position.y = 3.5;
        this.fallbackGroup.add(head);
        
        
        const body = new THREE.Mesh(new THREE.BoxGeometry(1.5, 2, 1), mat);
        body.position.y = 2;
        this.fallbackGroup.add(body);

        
        this.leftArm = new THREE.Mesh(new THREE.BoxGeometry(0.4, 2, 0.4), mat);
        this.leftArm.position.set(-1, 2, 0);
        this.fallbackGroup.add(this.leftArm);

        this.rightArm = new THREE.Mesh(new THREE.BoxGeometry(0.4, 2, 0.4), mat);
        this.rightArm.position.set(1, 2, 0);
        this.fallbackGroup.add(this.rightArm);

        this.scene.add(this.fallbackGroup);

        const loadingText = document.getElementById('avatar-loading');
        if (loadingText) loadingText.textContent = 'Đang dùng Avatar dự phòng (Do mạng chặn GLTF)';
        
        this.isFallback = true;
    }

    loadModel() {
        if (typeof THREE.GLTFLoader === 'undefined') {
            console.warn('GLTFLoader is undefined. Using fallback avatar.');
            this.createFallbackAvatar();
            return;
        }

        const loader = new THREE.GLTFLoader();
        
        const url = 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r128/examples/models/gltf/RobotExpressive/RobotExpressive.glb';
        
        loader.load(url, (gltf) => {
            const model = gltf.scene;
            this.scene.add(model);
            
            
            const loadingText = document.getElementById('avatar-loading');
            if (loadingText) loadingText.style.display = 'none';

            this.mixer = new THREE.AnimationMixer(model);

            
            const animations = gltf.animations;
            for (let i = 0; i < animations.length; i++) {
                const clip = animations[i];
                const action = this.mixer.clipAction(clip);
                this.actions[clip.name] = action;
            }

            
            this.fadeToAction('Idle', 0.5);

            
            this.mixer.addEventListener('finished', (e) => {
                this.playNextInQueue();
            });

            
            const loadingText = document.getElementById('avatar-loading');
            if (loadingText) loadingText.style.display = 'none';

        }, undefined, (e) => {
            console.error('Failed to load GLTF model:', e);
            this.createFallbackAvatar();
        });
    }

    fadeToAction(name, duration) {
        const nextAction = this.actions[name];
        if (!nextAction) return;

        if (this.activeAction !== nextAction) {
            const previousAction = this.activeAction;
            this.activeAction = nextAction;
            
            nextAction.reset();
            nextAction.setEffectiveTimeScale(1);
            nextAction.setEffectiveWeight(1);
            nextAction.play();

            if (previousAction) {
                previousAction.crossFadeTo(nextAction, duration, true);
            }
        }
    }

    
    mapGlossToAnim(gloss) {
        const g = gloss.toLowerCase();
        
        if (g.includes('chào') || g.includes('hello')) return 'Wave';
        if (g.includes('có') || g.includes('yes') || g.includes('đồng ý')) return 'Yes';
        if (g.includes('không') || g.includes('no') || g.includes('từ chối')) return 'No';
        if (g.includes('nhanh') || g.includes('chạy')) return 'Running';
        if (g.includes('tuyệt') || g.includes('giỏi')) return 'ThumbsUp';
        if (g.includes('buồn') || g.includes('chết')) return 'Death';
        if (g.includes('ngồi')) return 'Sitting';
        if (g.includes('đi')) return 'Walking';
        
        return 'Dance';
    }

    playSequence(glosses) {
        this.queue = [];
        
        if (this.isFallback) {
            
            this.queue = [...glosses];
            if (!this.isPlayingQueue) {
                this.isPlayingQueue = true;
                this.playNextFallback();
            }
            return;
        }

        if (!this.mixer) return;
        
        for (const gloss of glosses) {
            const animName = this.mapGlossToAnim(gloss);
            if (this.actions[animName]) {
                this.queue.push(animName);
            }
        }
        
        if (this.queue.length > 0 && !this.isPlayingQueue) {
            this.isPlayingQueue = true;
            this.playNextInQueue();
        }
    }

    playNextFallback() {
        if (this.queue.length === 0) {
            this.isPlayingQueue = false;
            if (this.leftArm && this.rightArm) {
                this.leftArm.rotation.x = 0;
                this.rightArm.rotation.x = 0;
            }
            return;
        }
        
        const word = this.queue.shift();
        
        let t = 0;
        const interval = setInterval(() => {
            t += 0.1;
            if (this.leftArm && this.rightArm) {
                this.leftArm.rotation.x = Math.sin(t) * 2;
                this.rightArm.rotation.x = Math.sin(t) * 2;
            }
            if (t > Math.PI) {
                clearInterval(interval);
                setTimeout(() => this.playNextFallback(), 200);
            }
        }, 30);
    }

    playNextInQueue() {
        if (this.queue.length === 0) {
            
            this.fadeToAction('Idle', 0.5);
            this.isPlayingQueue = false;
            return;
        }

        const nextAnim = this.queue.shift();
        const action = this.actions[nextAnim];
        
        if (action) {
            
            action.setLoop(THREE.LoopOnce, 1);
            action.clampWhenFinished = true;
            this.fadeToAction(nextAnim, 0.2);
        } else {
            this.playNextInQueue();
        }
    }

    onWindowResize() {
        if (!this.camera || !this.renderer) return;
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        if (this.width === 0 || this.height === 0) return;
        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.width, this.height);
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));
        const dt = this.clock.getDelta();
        if (this.mixer) this.mixer.update(dt);
        this.renderer.render(this.scene, this.camera);
    }
}
