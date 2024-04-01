// Notifies when OpenCV.js is ready and initializes event listeners
function onOpenCvReady() {
    document.getElementById('status').textContent = 'OpenCV.js is ready.';
    setupImageInputListener();
  }
  
  // Sets up the listener for image file input
  function setupImageInputListener() {
    const imgElement = document.getElementById('imageSrc');
    const inputElement = document.getElementById('fileInput');
  
    inputElement.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        imgElement.src = URL.createObjectURL(e.target.files[0]);
      }
    }, false);
  
    imgElement.onload = processImageForCircleDetection;
  }
  
  // Processes the loaded image to detect circles
  function processImageForCircleDetection() {
    const src = cv.imread('imageSrc');
    const dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8U);
    detectAndDrawCircles(src, dst);
    cv.imshow('canvasOutput', dst);
  
    // Cleanup resources
    src.delete(); dst.delete();
  }
  
  // Detects circles in the source image and draws them on the destination image
  function detectAndDrawCircles(src, dst) {

    // Convert the image to grayscale and use the Hough gradient which requires edge detection
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.Canny(gray, gray, 50, 150, 3);

    // Use the Hough gradient method to detect circles
    const circles = new cv.Mat();
    cv.HoughCircles(gray, circles, cv.HOUGH_GRADIENT, 1, 45, 75, 40, 0, 0);

    // Draw detected circles
    for (let i = 0; i < circles.cols; i++) {
      const x = circles.data32F[i * 3];
      const y = circles.data32F[i * 3 + 1];
      const radius = circles.data32F[i * 3 + 2];
      const center = new cv.Point(x, y);
      cv.circle(dst, center, radius, new cv.Scalar(255, 0, 0), 2);
    }
  
    circles.delete();
  }
  
  // Initialize when OpenCV.js is ready
  onOpenCvReady();