// **************************************************************** //
// *********** GPU Programming with WebGL and GLSL  *************** //
// **************************************************************** //

// This file contains the JavaScript code for the image processing

// Global Variables
var gl, program, canvas, aspect;
var warmth = 0.0;
var brightness = 0.0;
var contrast = 0.0;
var saturation = 0.0;
var invert = false;
var blackWhite = false;
var viewLimits = { left: -2, right: 2, bottom: -2, top: 2, near: -10, far: 10 };

// Initialization on Window Load
window.onload = function init() {
    setupCanvas();
    setupWebGL();
    setupShaders();
    setupBuffers();
    setupTexture();
    setupEventListeners();
    render();
};

function setupCanvas() {
    canvas = document.getElementById("gl-canvas");
    aspect = canvas.width / canvas.height;
    viewLimits.left *= aspect;
    viewLimits.right *= aspect;
}

function setupWebGL() {
    gl = canvas.getContext('webgl2');
    if (!gl) { alert("WebGL isn't available"); }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
}

function setupShaders() {
    program = initShaders(gl, "vertex-shader", "fragment-shader");
    gl.useProgram(program);
}

function setupBuffers() {
    // Vertices and Texture Coordinates
    const vertices = [
        vec2(-3.884, 2.0), vec2(-3.884, -2.0), vec2(3.884, -2.0),
        vec2(-3.884, 2.0), vec2(3.884, 2.0), vec2(3.884, -2.0)
    ];
    const texCoords = [
        vec2(0.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 0.0),
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0)
    ];

    // Buffer for texture coordinates
    createBuffer(texCoords, "vTexCoord", 2);

    // Buffer for vertices
    createBuffer(vertices, "vPosition", 2);
}

function createBuffer(data, attribLocation, size) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, flatten(data), gl.STATIC_DRAW);

    const location = gl.getAttribLocation(program, attribLocation);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(location);
}

function setupTexture() {
    const image = document.getElementById("texImage");
    configureTexture(image);
}

function setupEventListeners() {
    document.getElementById("warmth").oninput = function(event) {
        warmth = event.target.value / 128 * 0.5;
    };
    document.getElementById("brightness").oninput = function(event) {
        brightness = event.target.value / 128 * 0.5;
    };
    document.getElementById("contrast").oninput = function(event) {
        contrast = event.target.value / 128 * 0.5;
    };
    document.getElementById("saturation").oninput = function(event) {
        saturation = event.target.value / 128 * 0.5;
    };
    document.getElementById("blackWhite").onclick = function() {
        blackWhite = !blackWhite;
    };    
    document.getElementById("invert").onclick = function() {
        invert = !invert;
    };
    window.addEventListener("keydown", dealWithKeyboard, false);
}

function configureTexture(image) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.uniform1i(gl.getUniformLocation(program, "texture"), 0);
}

function render() {
    gl.clear(gl.COLOR_BUFFER_BIT);
    updateProjectionMatrix();
    
    // Set uniform variables for the shader
    gl.uniform1f(gl.getUniformLocation(program, 'brightness'), brightness);
    gl.uniform1f(gl.getUniformLocation(program, 'warmth'), warmth);
    gl.uniform1f(gl.getUniformLocation(program, 'contrast'), contrast);
    gl.uniform1f(gl.getUniformLocation(program, 'saturation'), saturation);
    gl.uniform1i(gl.getUniformLocation(program, 'invert'), invert ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(program, 'blackWhite'), blackWhite ? 1 : 0);
    
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(render);
}

function updateProjectionMatrix() {
    const PMat = ortho(viewLimits.left, viewLimits.right, viewLimits.bottom, viewLimits.top, viewLimits.near, viewLimits.far);
    gl.uniformMatrix4fv(gl.getUniformLocation(program, "P"), false, flatten(PMat));
}

function dealWithKeyboard(e) {
    // Zoom and Pan Controls
    const zoomFactor = 0.1;
    const panFactor = 0.1;
    switch (e.keyCode) {
        case 33: // PageUp
            zoom(-zoomFactor);
            break;
        case 34: // PageDown
            zoom(zoomFactor);
            break;
        case 37: // Left Arrow
            pan(-panFactor, 0);
            break;
        case 38: // Up Arrow
            pan(0, panFactor);
            break;
        case 39: // Right Arrow
            pan(panFactor, 0);
            break;
        case 40: // Down Arrow
            pan(0, -panFactor);
            break;
    }
}

function zoom(factor) {
    const width = (viewLimits.right - viewLimits.left) * factor;
    const height = (viewLimits.top - viewLimits.bottom) * factor;
    viewLimits.left += width;
    viewLimits.right -= width;
    viewLimits.bottom += height;
    viewLimits.top -= height;
}

function pan(x, y) {
    viewLimits.left += x;
    viewLimits.right += x;
    viewLimits.bottom += y;
    viewLimits.top += y;
}
