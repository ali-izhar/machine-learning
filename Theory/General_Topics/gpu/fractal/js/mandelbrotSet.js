// --------------------------------------------------------------------------------------- //
// GPU Programming
// This program draws the Mandelbrot set using WebGL 2.0
// --------------------------------------------------------------------------------------- //

var gl;
var program;
var canvas;
var aspect;

var left = -2;              // left limit of world coords
var right = 2;              // right limit of world coords
var bottom = -2;            // bottom limit of world coords
var topBound = 2;           // top limit of worlds coord
var near = -10;             // near clip plane
var far = 10;               // far clip plane

window.onload = function init() {
    canvas = document.getElementById( "gl-canvas" );        // Get HTML canvas
    
    gl = canvas.getContext('webgl2');                       // Get a WebGL 2.0 context
    if ( !gl ) { alert( "WebGL isn't available" ); }

    aspect = canvas.width / canvas.height;                  // get the aspect ratio of the canvas
    left *= aspect;                                         // left limit of world coords
    right *= aspect;                                        // right limit of world coords

    // Vertices of two triangles in complex plane
    var vertices = [
        vec2(-2.0, 2.0),
        vec2(-2.0,-2.0),
        vec2( 2.0,-2.0),
        vec2(-2.0, 2.0),
        vec2( 2.0, 2.0),
        vec2( 2.0,-2.0)
    ];

    // Configure WebGL
    gl.viewport( 0, 0, canvas.width, canvas.height );  // What part of html are we looking at?
    gl.clearColor( 0.0, 0.0, 0.0, 1.0 );               // Set background color of the viewport to black
    
    // Load shaders and initialize attribute buffers
    program = initShaders( gl, "vertex-shader", "fragment-shader" );    // Compile and link shaders to form a program
    gl.useProgram(program);                                             // Make this the active shaer program
    
    // Load the data into the GPU
    var bufferId = gl.createBuffer();                                    // Generate a VBO id
    gl.bindBuffer( gl.ARRAY_BUFFER, bufferId );                          // Bind this VBO to be the active one
    gl.bufferData( gl.ARRAY_BUFFER, flatten(vertices), gl.STATIC_DRAW ); // Load the VBO with vertex data

    // Associate our shader variables with our data buffer
    var vPosition = gl.getAttribLocation( program, "vPosition" );        // Link js vPosition with "vertex shader attribute variable" - vPosition
    gl.vertexAttribPointer(vPosition, 2, gl.FLOAT, false, 0, 0 );        // Specify layout of VBO memory
    gl.enableVertexAttribArray( vPosition );                             // Enable this attribute

    render();
};

let mouseDown = false;
window.onmousedown = function () {
    mouseDown = !mouseDown
}

window.onmouseup = function () {
    mouseDown = !mouseDown
}

window.addEventListener('wheel', function(event) {
    if (event.deltaY < 0) {
        var range = (right - left);
        var delta = (range - range * 0.9) * 0.5;
        left += delta;
        right -= delta;
        range = topBound - bottom;
        delta = (range - range * 0.9) * 0.5;
        bottom += delta;
        topBound -= delta;
    } else if (event.deltaY > 0) {
        var range = (right - left);
        var delta = (range * 1.1 - range) * 0.5;
        left -= delta;
        right += delta;
        range = topBound - bottom;
        delta = (range * 1.1 - range) * 0.5;
        bottom -= delta;
        topBound += delta;
    }
});

// Callback function for keydown events, rgeisters function dealWithKeyboard
window.addEventListener("keydown", dealWithKeyboard, false);

console.log("bot is " + bot);

// Functions that gets called to parse keydown events
function dealWithKeyboard(e) {
    switch (e.keyCode) {
       case 33: // PageUp key , Zoom in
           {
               var range = (right - left);
               var delta = (range - range * 0.9) * 0.5;
               left += delta; right -= delta;
               range = topBound - bottom;
               delta = (range - range * 0.9) * 0.5;
               bottom += delta; topBound -= delta;
           }
       break;
       case 34: // PageDown key, zoom out
           {
               var range = (right - left);
               var delta = (range * 1.1 - range) * 0.5;
               left -= delta; right += delta;
               range = topBound - bottom;
               delta = (range * 1.1 - range) * 0.5;
               bottom -= delta; topBound += delta;
           }
       break;
       case 37: // left arrow pan left
           { left += -0.001; right += -0.001; }
       break;
       case 38: // up arrow pan up
           { bottom += 0.001; topBound += 0.001; }
       break;
       case 39: // right arrow pan right
           { left += 0.001; right += 0.001; }
       break;
       case 40: // down arrow pan down
           { bottom += -0.001; topBound += -0.001;}
       break;
    }
}

function render() {
    gl.clear(gl.COLOR_BUFFER_BIT);            // Clear viewport with gl.clearColor defined above

    var zoomScalingFactor = Math.abs(topBound-bottom) / 2.0;
    window.addEventListener('mousemove', (event) => {
        if (mouseDown) {            
            left = originalLeft + zoomScalingFactor*0.003*(originalMousePos.x - event.clientX);
            right = originalRight + zoomScalingFactor*0.003*(originalMousePos.x - event.clientX);
            bottom = originalBottom - zoomScalingFactor*0.003*(originalMousePos.y - event.clientY);
            topBound = originalTopBound - zoomScalingFactor*0.003*(originalMousePos.y - event.clientY);
            
        } else if (!mouseDown) {
            originalMousePos = { x: event.clientX, y: event.clientY };
            originalLeft = left
            originalRight = right
            originalBottom = bottom
            originalTopBound = topBound
            
        }
      });

    var PMat;                                                  // js variable to hold projection matrix
    console.log(left + " " + right);
    PMat = ortho(left, right, bottom, topBound, near, far);    // Call function to compute orthographic projection matrix

    var P_loc = gl.getUniformLocation(program, "P");           // Get Vertex shader memory location for P
    gl.uniformMatrix4fv(P_loc, false, flatten(PMat));          // Set uniform variable P on GPU 

    // Get uniform locations
    // Set CPU-side variables for all of our shader variables
    var viewportDimensions = vec2(canvas.width, canvas.height);

    var viewportDimensionsLoc = gl.getUniformLocation(program, 'viewportDimensions');
    var leftLoc = gl.getUniformLocation(program, 'left');
    var rightLoc = gl.getUniformLocation(program, 'right');
    var bottomLoc = gl.getUniformLocation(program, 'bottom');
    var topBoundLoc = gl.getUniformLocation(program, 'topBound');

    gl.uniform2fv(viewportDimensionsLoc, viewportDimensions);
    gl.uniform1f(leftLoc,  left);
    gl.uniform1f(rightLoc, right);
    gl.uniform1f(bottomLoc, bottom);
    gl.uniform1f(topBoundLoc, topBound);

    gl.drawArrays(gl.TRIANGLES, 0, 6);         // Draw two triangles using the TRIANGLES primitive using 6 vertices
    requestAnimationFrame(render);             // swap buffers, continue render loop
}
