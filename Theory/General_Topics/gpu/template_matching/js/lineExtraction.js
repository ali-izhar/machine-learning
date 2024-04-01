// Notifies when OpenCV.js is ready
function onOpenCvReady() {
    document.getElementById('status').textContent = 'OpenCV.js is ready.';
  }
  
  // Sets up the image input for processing
  function setupImageInput() {
    const imgElement = document.getElementById('imageSrc');
    const inputElement = document.getElementById('fileInput');
  
    inputElement.addEventListener('change', (e) => {
      const file = e.target.files[0];
      imgElement.src = URL.createObjectURL(file);
    }, false);
  
    imgElement.onload = processImageForLines;
  }
  
  // Processes the loaded image to detect and draw lines
  function processImageForLines() {
    const src = cv.imread('imageSrc');
    const dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);
    performLineDetection(src, dst);
    cv.imshow('canvasOutput', dst);
    
    // Clean up resources
    src.delete();
    dst.delete();
  }
  
  // Performs line detection and drawing on the destination image
  function performLineDetection(src, dst) {
    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);

    // Apply Gaussian Blur to reduce noise
    const ksize = new cv.Size(3, 3);
    cv.GaussianBlur(src, src, ksize, 50, 50, cv.BORDER_DEFAULT);
    
    cv.Canny(src, src, 50, 150, 3);
    const lines = new cv.Mat();
    cv.HoughLinesP(src, lines, 0.4, Math.PI / 90, 2, 0, 1);
    // cv.HoughLinesP(src, lines, 1, Math.PI / 90, 10, 5, 5);
    drawDetectedLines(dst, lines);
  
    // Clean up the lines matrix
    lines.delete();
  }
  
  // Draws detected lines on the destination image
  function drawDetectedLines(dst, lines) {
    const color = new cv.Scalar(113, 170, 223);
  
    for (let i = 0; i < lines.rows; ++i) {
      const startPoint = new cv.Point(lines.data32S[i * 4], lines.data32S[i * 4 + 1]);
      const endPoint = new cv.Point(lines.data32S[i * 4 + 2], lines.data32S[i * 4 + 3]);
      cv.line(dst, startPoint, endPoint, color);
    }
  }
  
  // Initialize the app once OpenCV is ready
  onOpenCvReady();
  setupImageInput();
  