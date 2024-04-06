class Utils {
    constructor(errorOutputId) {
        this.errorOutput = document.getElementById(errorOutputId);
        this.video = null;
        this.stream = null;
        this.onCameraStartedCallback = null;
    }

    loadOpenCv(onloadCallback) {
        const script = this.createScriptTag('opencv3_4.js', onloadCallback);
        this.insertScript(script);
    }

    createFileFromUrl(path, url, callback) {
        const request = new XMLHttpRequest();
        this.setupRequest(request, path, callback);
        request.send();
    }

    loadImageToCanvas(url, canvasId) {
        const img = new Image();
        this.setupImageLoader(img, canvasId);
        img.src = url;
    }

    executeCode(textAreaId) {
        try {
            this.clearError();
            const code = document.getElementById(textAreaId).value;
            eval(code);
        } catch (err) {
            this.printError(err);
        }
    }

    clearError() {
        this.errorOutput.innerHTML = '';
    }

    printError(err) {
        const errorString = this.getErrorString(err);
        this.errorOutput.innerHTML = errorString;
    }

    addFileInputHandler(fileInputId, canvasId) {
        const inputElement = document.getElementById(fileInputId);
        this.setupFileInput(inputElement, canvasId);
    }

    startCamera(resolution, callback, videoId) {
        const constraints = this.getVideoConstraints(resolution);
        this.initCamera(constraints, callback, videoId);
    }

    stopCamera() {
        if (this.video) {
            this.video.pause();
            this.video.srcObject = null;
            this.video.removeEventListener('canplay', this.onVideoCanPlay.bind(this));
        }
        if (this.stream) {
            this.stream.getVideoTracks().forEach(track => track.stop());
        }
    }

    // Helper methods from here on
    createScriptTag(src, onloadCallback) {
        let script = document.createElement('script');
        script.async = true;
        script.type = 'text/javascript';
        script.addEventListener('load', onloadCallback);
        script.addEventListener('error', () => this.printError(`Failed to load ${src}`));
        script.src = src;
        return script;
    }

    insertScript(script) {
        const firstScriptTag = document.getElementsByTagName('script')[0];
        firstScriptTag.parentNode.insertBefore(script, firstScriptTag);
    }

    setupRequest(request, path, callback) {
        request.open('GET', path, true);
        request.responseType = 'arraybuffer';
        request.onload = () => {
            if (request.status === 200) {
                const data = new Uint8Array(request.response);
                cv.FS_createDataFile('/', path, data, true, false, false);
                callback();
            } else {
                this.printError(`Failed to load ${path} status: ${request.status}`);
            }
        };
    }

    setupImageLoader(img, canvasId) {
        img.onload = () => {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0, img.width, img.height);
        };
    }

    getErrorString(err) {
        if (typeof err === 'undefined') {
            return '';
        } else if (typeof err === 'number' && !isNaN(err) && typeof cv !== 'undefined') {
            return 'Exception: ' + cv.exceptionFromPtr(err).msg;
        } else if (typeof err === 'string') {
            const ptr = Number(err.split(' ')[0]);
            return !isNaN(ptr) && typeof cv !== 'undefined' ? 'Exception: ' + cv.exceptionFromPtr(ptr).msg : err;
        } else if (err instanceof Error) {
            return err.stack.replace(/\n/g, '<br>');
        }
        return err.toString();
    }

    setupFileInput(inputElement, canvasId) {
        inputElement.addEventListener('change', (e) => {
            const files = e.target.files;
            if (files.length > 0) {
                const imgUrl = URL.createObjectURL(files[0]);
                this.loadImageToCanvas(imgUrl, canvasId);
            }
        }, false);
    }

    getVideoConstraints(resolution) {
        const constraints = {
            'qvga': { width: { exact: 320 }, height: { exact: 240 } },
            'vga': { width: { exact: 640 }, height: { exact: 480 } }
        };
        return constraints[resolution] || true;
    }

    initCamera(constraints, callback, videoId) {
        const video = document.getElementById(videoId) || document.createElement('video');
        navigator.mediaDevices.getUserMedia({ video: constraints, audio: false })
            .then(stream => {
                video.srcObject = stream;
                video.play();
                this.video = video;
                this.stream = stream;
                this.onCameraStartedCallback = callback;
                video.addEventListener('canplay', this.onVideoCanPlay.bind(this), false);
            })
            .catch(err => {
                this.printError(`Camera Error: ${err.name} ${err.message}`);
            });
    }

    onVideoCanPlay() {
        if (this.onCameraStartedCallback) {
            this.onCameraStartedCallback(this.stream, this.video);
        }
    }
}
