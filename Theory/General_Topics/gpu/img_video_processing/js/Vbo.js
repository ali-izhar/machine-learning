// Specify layout of VBO memory
function Vbo(gl, verts, dim) {
    var data = {};

    data.nVerts = verts.length;
    var id = gl.createBuffer();                                              // Creates buffer object id
    gl.bindBuffer(gl.ARRAY_BUFFER, id);                                      // Bind this VBO as current
    // push data to GPU, gl.STATIC_DRAW is set to optimize usuage
    gl.bufferData(gl.ARRAY_BUFFER, flatten(verts), gl.STATIC_DRAW);


    data.id = id;
    data.gl = gl;
    data.dim = dim;

    function BindToAttribute(attribute) {
        // Tell which buffer object we want to operate on as a VBO
        gl.bindBuffer(gl.ARRAY_BUFFER, data.id);                   
        gl.enableVertexAttribArray(attribute);                               // Enable this attribute in the shader
        // Define format of the attribute array. Must match parameters in shader
        gl.vertexAttribPointer(attribute, data.dim, gl.FLOAT, false, 0, 0); }

    function Delete() {
        gl.deleteBuffer(data.id);
    }

    data.BindToAttribute = BindToAttribute;
    data.Delete = Delete;

    return data;
}
