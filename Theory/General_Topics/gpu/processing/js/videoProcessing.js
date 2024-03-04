// **************************************************************** //
// *********** GPU Programming with WebGL and GLSL  *************** //
// **************************************************************** //

// Define global variables
var gl, program, canvas, aspect, imageAspect, textureId, videoId;
var copyVideo = false;
var webcam = false;

var dimAndKernelWeight = vec3(1241.0, 639.0, 16.0);
dimAndKernelWeight[2] = 1.0; // Default normalization factor
var kernal = mat3();
var kernelSize = 3;
var filter = "normal";
var brightness = 1.0;
var color = false;
var blackWhite = false;
var left = -2, right = 2, bottom = -2, topBound = 2, near = -10, far = 10;

var videofile = "files/video.mp4";

// Initialize the WebGL context and shaders
window.onload = function() {
    canvas = document.getElementById("gl-canvas");
    gl = canvas.getContext("webgl2");
    if (!gl) {
        alert("WebGL isn't available");
        return;
    }

    aspect = canvas.width / canvas.height;
    left *= aspect;
    right *= aspect;

    program = initShaders(gl, "vertex-shader", "fragment-shader");
    gl.useProgram(program);

    var image = document.getElementById("texImage");
    configureTexture2(image);

    imageAspect = image.width / image.height;
    dimAndKernelWeight[0] = image.width;
    dimAndKernelWeight[1] = image.height;

    var vertices = [
        vec2(-2.0 * imageAspect,  2.0),
        vec2(-2.0 * imageAspect, -2.0),
        vec2( 2.0 * imageAspect, -2.0),
        vec2(-2.0 * imageAspect,  2.0),
        vec2( 2.0 * imageAspect,  2.0),
        vec2( 2.0 * imageAspect, -2.0)
    ];

    var texCoordsArray = [
        vec2(0.0, 1.0),
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(1.0, 1.0),
        vec2(1.0, 0.0)
    ];

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    var tBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, tBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, flatten(texCoordsArray), gl.STATIC_DRAW);

    var vTexCoord = gl.getAttribLocation(program, "vTexCoord");
    gl.vertexAttribPointer(vTexCoord, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vTexCoord);

    var bufferId = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, bufferId);
    gl.bufferData(gl.ARRAY_BUFFER, flatten(vertices), gl.STATIC_DRAW);

    var vPosition = gl.getAttribLocation(program, "vPosition");
    gl.vertexAttribPointer(vPosition, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vPosition);

    textureId = initTexture(gl);
    videoId = setupVideo(videofile);
    render();
};

var colorButton = document.getElementById("color");
colorButton.addEventListener("click", function () {
    dimAndKernelWeight[2] = 1.0;
    filter = "normal";
    color = !color;
    if (color) {
        blackWhite = false;
    }
});

var blackWhiteButton = document.getElementById("blackWhite");
blackWhiteButton.addEventListener("click", function () {
    dimAndKernelWeight[2] = 1.0;
    filter = "normal";
    blackWhite = !blackWhite;
    if (blackWhite) {
        color = false;
    }
});

var myVideo = document.getElementById("myvideo");
myVideo.addEventListener("change", function() {
    const selectedFile = myVideo.files[0];
    if (selectedFile) {
        const videoURL = URL.createObjectURL(selectedFile);
        videoId = setupVideo(videoURL);
    }
});

var webcamButton = document.getElementById("webcam");
webcamButton.addEventListener("click", function () {
    webcam = !webcam;
    if(webcam) {
        onCamera();
    }
    else {
        if(videoStream) {
            const tracks = videoStream.getTracks();
            tracks.forEach(track => track.stop());
            videoStream = null;
        }
        videoId = setupVideo(video_path);
    }
});

// Handle filter selection from the menu
var m = document.getElementById("mymenu");
m.addEventListener("click", function () {
    switch (m.selectedIndex) {
        case 0:
            dimAndKernelWeight[2] = 1.0; filter = "normal"; kernelSize = 3;
            break;
        case 1:
            // 3x3 Gaussian blur kernel ~ sigma = 16
            dimAndKernelWeight[2] = 1.0; filter = "gaussianBlur"; kernelSize = 3;
            break;
        case 2:
            // 5x5 Gaussian blur kernel ~ sigma = 273
            dimAndKernelWeight[2] = 1.0; filter = "gaussianBlur2"; kernelSize = 5;
            break;
        case 3:
            // 7x7 Gaussian blur kernel ~ sigma = 1003
            dimAndKernelWeight[2] = 1.0; filter = "gaussianBlur3"; kernelSize = 7;
            break;
        case 4:
            dimAndKernelWeight[2] = 1.0; filter = "unsharpen"; kernelSize = 3;
            break;
        case 5:
            dimAndKernelWeight[2] = 1.0; filter = "sharpness"; kernelSize = 3;
            break;
        case 6:
            dimAndKernelWeight[2] = 8.0; filter = "sharpen"; kernelSize = 3;
            break;
        case 7:
            dimAndKernelWeight[2] = 1.0; filter = "edgeDetect2"; kernelSize = 3;
            break;
        case 8:
            dimAndKernelWeight[2] = 1.0; filter = "edgeDetect3"; kernelSize = 3;
            break;
        case 9:
            dimAndKernelWeight[2] = 1.0; filter = "edgeDetect4"; kernelSize = 3;
            break;
        case 10:
            dimAndKernelWeight[2] = 1.0; filter = "edgeDetect5"; kernelSize = 3;
            break;
        case 11:
            dimAndKernelWeight[2] = 1.0; filter = "edgeDetect6"; kernelSize = 3;
            break;
        case 12:
            dimAndKernelWeight[2] = 1.0; filter = "sobelHorizontal"; kernelSize = 3;
            break;
        case 13:
            dimAndKernelWeight[2] = 1.0; filter = "sobelVertical"; kernelSize = 3;
            break;
        case 14:
            dimAndKernelWeight[2] = 1.0; filter = "previtHorizontal"; kernelSize = 3;
            break;
        case 15:
            dimAndKernelWeight[2] = 1.0; filter = "previtVertical"; kernelSize = 3;
            break;
        case 16:
            dimAndKernelWeight[2] = 1.0; filter = "boxBlur"; kernelSize = 3;
            break;
        case 17:
            dimAndKernelWeight[2] = 1.0; filter = "triangleBlur"; kernelSize = 3;
            break;
        case 18:
            dimAndKernelWeight[2] = 1.0; filter = "emboss"; kernelSize = 3;
            break;
    }
});

// Handle keyboard events for panning and zooming
window.addEventListener("keydown", dealWithKeyboard, false);

function dealWithKeyboard(e) {
    switch (e.keyCode) {
        case 33: // PageUp key, zoom in
            {
                var range = (right - left);
                var delta = (range - range * 0.9) * 0.5;
                left += delta;
                right -= delta;
                range = topBound - bottom;
                delta = (range - range * 0.9) * 0.5;
                bottom += delta;
                topBound -= delta;
            }
            break;
        case 34: // PageDown key, zoom out
            {
                var range = (right - left);
                var delta = (range * 1.1 - range) * 0.5;
                left -= delta;
                right += delta;
                range = topBound - bottom;
                delta = (range * 1.1 - range) * 0.5;
                bottom -= delta;
                topBound += delta;
            }
            break;
        case 37: // Left arrow key, pan left
            {
                left += -0.1;
                right += -0.1;
            }
            break;
        case 38: // Up arrow key, pan up
            {
                bottom += 0.1;
                topBound += 0.1;
            }
            break;
        case 39: // Right arrow key, pan right
            {
                left += 0.1;
                right += 0.1;
            }
            break;
        case 40: // Down arrow key, pan down
            {
                bottom += -0.1;
                topBound += -0.1;
            }
            break;
    }
}

// Render function to draw the scene
function render() {
    gl.clear(gl.COLOR_BUFFER_BIT);

    if(copyVideo) {
        updateTexture(gl, textureId, videoId);
    }

    var PMat = ortho(left, right, bottom, topBound, near, far);
    var P_loc = gl.getUniformLocation(program, "P");
    gl.uniformMatrix4fv(P_loc, false, flatten(PMat));

    var kernels = {
        normal: [
            0, 0, 0,
            0, 1, 0,
            0, 0, 0
        ],
        // 3x3 Gaussian blur kernel ~ sigma = 16
        gaussianBlur: [
            0.062, 0.125, 0.062, 
            0.125, 0.25, 0.125, 
            0.062, 0.125, 0.062
        ],
        // 5x5 Gaussian blur kernel (rounded to 5 decimal places) ~ sigma = 273
        gaussianBlur2: [
            0.004, 0.015, 0.026, 0.015, 0.004, 
            0.015, 0.059, 0.095, 0.059, 0.015, 
            0.026, 0.095, 0.15, 0.095, 0.026, 
            0.015, 0.059, 0.095, 0.059, 0.015, 
            0.004, 0.015, 0.026, 0.015, 0.004
        ],
        // 7x7 Gaussian blur kernel (rounded to 5 decimal places) ~ sigma = 1003
        gaussianBlur3: [
            0.0, 0.0, 0.001, 0.002, 0.001, 0.0, 0.0, 
            0.0, 0.003, 0.013, 0.022, 0.013, 0.003, 0.0, 
            0.001, 0.013, 0.059, 0.097, 0.059, 0.013, 0.001, 
            0.002, 0.022, 0.097, 0.159, 0.097, 0.022, 0.002, 
            0.001, 0.013, 0.059, 0.097, 0.059, 0.013, 0.001, 
            0.0, 0.003, 0.013, 0.022, 0.013, 0.003, 0.0, 
            0.0, 0.0, 0.001, 0.002, 0.001, 0.0, 0.0
        ],
        unsharpen: [
            -1, -1, -1,
            -1,  9, -1,
            -1, -1, -1
        ],
        sharpness: [
            0, -1,  0,
            -1,  5, -1,
            0, -1,  0
        ],
        sharpen: [
            -1, -1, -1,
            -1, 16, -1,
            -1, -1, -1
        ],
        edgeDetect: [
            -0.125, -0.125, -0.125,
            -0.125,  1, -0.125,
            -0.125, -0.125, -0.125
        ],
        edgeDetect2: [
            -1, -1, -1,
            -1,  8, -1,
            -1, -1, -1
        ],
        edgeDetect3: [
            -5, 0, 0,
            0, 0, 0,
            0, 0, 5
        ],
        edgeDetect4: [
            -1, -1, -1,
            0,  0,  0,
            1,  1,  1
        ],
        edgeDetect5: [
            -1, -1, -1,
            2,  2,  2,
            -1, -1, -1
        ],
        edgeDetect6: [
            -5, -5, -5,
            -5, 39, -5,
            -5, -5, -5
        ],
        sobelHorizontal: [
            1,  2,  1,
            0,  0,  0,
           -1, -2, -1
        ],
        sobelVertical: [
            1,  0, -1,
            2,  0, -2,
            1,  0, -1
        ],
        previtHorizontal: [
            1,  1,  1,
            0,  0,  0,
           -1, -1, -1
        ],
        previtVertical: [
            1,  0, -1,
            1,  0, -1,
            1,  0, -1
        ],
        boxBlur: [
            0.111, 0.111, 0.111,
            0.111, 0.111, 0.111,
            0.111, 0.111, 0.111
        ],
        triangleBlur: [
            0.0625, 0.125, 0.0625,
            0.125,  0.25,  0.125,
            0.0625, 0.125, 0.0625
        ],
        emboss: [
           -2, -1,  0,
           -1,  1,  1,
            0,  1,  2
        ]
    };
    
    // set black and white state
    var blackWhiteLoc = gl.getUniformLocation(program, "blackWhite");
    gl.uniform1i(blackWhiteLoc, blackWhite);

    // set color state
    var colorLoc = gl.getUniformLocation(program, "color");
    gl.uniform1i(colorLoc, color);

    // set the kernel
    var kernelArray = kernels[filter];
    if (!kernelArray) {
        console.error('No kernel found for filter:', filter);
    } else {        
        var kernelFloatArray = new Float32Array(kernelArray);
        var kernelLocation = gl.getUniformLocation(program, "kernel[0]");
        gl.uniform1fv(kernelLocation, kernelFloatArray);
    }
   
    // set the dimAndKernelWeight
    var dimAndKernelWeightLoc = gl.getUniformLocation(program, "dimAndKernelWeight");
    gl.uniform3fv(dimAndKernelWeightLoc, dimAndKernelWeight);

    // set the kernelSize uniform
    var kernelSizeLoc = gl.getUniformLocation(program, "kernelSize");
    gl.uniform1i(kernelSizeLoc, kernelSize);

    if (kernelSize == 3) {
        var kernelLocation = gl.getUniformLocation(program, "kernel[0]");
        gl.uniform1fv(kernelLocation, kernelFloatArray);
    } else if (kernelSize == 5) {
        var kernelFiveLocation = gl.getUniformLocation(program, "kernelFive[0]");
        gl.uniform1fv(kernelFiveLocation, kernelFloatArray);
    } else if (kernelSize == 7) {
        var kernelSevenLocation = gl.getUniformLocation(program, "kernelSeven[0]");
        gl.uniform1fv(kernelSevenLocation, kernelFloatArray);
    }    

    // set the brightness uniform
    var brightnessLoc = gl.getUniformLocation(program, "brightness");
    gl.uniform1f(brightnessLoc, brightness);
    
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(render);
}

// Configure the texture for rendering
function configureTexture2(image) {
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    gl.uniform1i(gl.getUniformLocation(program, "texture"), 0);
}

// Initialize the texture for rendering
function initTexture(gl) {
    const texture = gl.createTexture();

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

    const level = 0;
    const internalFormat = gl.RGBA;
    const width = 1;
    const height = 1;
    const border = 0;
    const srcFormat = gl.RGBA;
    const srcType = gl.UNSIGNED_BYTE;
    const pixel = new Uint8Array([0, 0, 255, 255]); // opaque blue
    
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, width, height, border, srcFormat, srcType, pixel);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    
    return texture;
}

function updateTexture(gl, texture, video) {
    const level = 0;
    const internalFormat = gl.RGBA;
    const sourceFormat = gl.RGBA;
    const sourceType = gl.UNSIGNED_BYTE;

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, sourceFormat, sourceType, video);
}

// Setup the video for rendering
function setupVideo(url) {
    const video = document.createElement("video");

    var isPlaying = false;
    var canTimeUpdate = false;

    video.autoplay = true;
    video.muted = true;
    video.loop = true;

    video.addEventListener("playing", function() {
        isPlaying = true;
        checkReady();
    }, true);

    video.addEventListener("timeupdate", function() {
        canTimeUpdate = true;
        checkReady();
    }, true);

    video.src = url;
    video.play();

    function checkReady() {
        if(isPlaying && canTimeUpdate) {
            copyVideo = true;
        }
    }

    return video;
}


function onCamera() {
    const video = videoId; 
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const constraints = {
        video: true,
        audio: false
    };
    navigator.mediaDevices.getUserMedia(constraints)
    .then(function(stream) {
        videoStream = stream;
        video.srcObject = stream;
        video.play();
    })
    .catch(function(error) {
        console.log('Error accessing webcam:', error);
    });
} else {
    console.log('Media is not supported'); 
}}