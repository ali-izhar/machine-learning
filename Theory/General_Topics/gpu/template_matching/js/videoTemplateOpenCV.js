// Indicates when OpenCV is ready by updating the page status
function onOpenCvReady() {
  document.getElementById('status').textContent = 'OpenCV.js is ready.';
}

// Initialize video processing with OpenCV.js for Canny Edge Detection
function initializeVideoProcessing() {
  const videoElement = document.getElementById("videoInput");
  configureVideoElement(videoElement, 640, 480);
  startVideoCapture(videoElement);
}

// Configure the video element properties
function configureVideoElement(videoElement, width, height) {
  videoElement.width = width;
  videoElement.height = height;
}

// Start capturing video from the user's webcam
function startVideoCapture(videoElement) {
  navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(stream => {
      videoElement.srcObject = stream;
      videoElement.play();
      processVideoFrames(videoElement);
    })
    .catch(err => {
      console.error("An error occurred during video capture: ", err);
    });
}

// Process video frames for edge detection
function processVideoFrames(videoElement) {
  const srcMat = new cv.Mat(videoElement.height, videoElement.width, cv.CV_8UC4);
  const dstMat = new cv.Mat(videoElement.height, videoElement.width, cv.CV_8UC1);
  const capture = new cv.VideoCapture(videoElement);
  const FPS = 30;

  // Function to process each frame
  function processFrame() {
    try {
      let beginTime = Date.now();
      capture.read(srcMat);
      preprocessAndDisplayFrame(srcMat, dstMat);
      // Schedule the next frame processing
      let delay = 1000 / FPS - (Date.now() - beginTime);
      setTimeout(processFrame, delay);
    } catch (err) {
      console.error("Error processing video frame: ", err);
    }
  }

  // Start processing frames immediately
  setTimeout(processFrame, 0);
}

// Preprocess the frame (grayscale conversion, Gaussian blur, etc.) and display
function preprocessAndDisplayFrame(srcMat, dstMat) {
  cv.cvtColor(srcMat, dstMat, cv.COLOR_RGBA2GRAY);
  const kernelSize = new cv.Size(9, 9);
  cv.GaussianBlur(dstMat, dstMat, kernelSize, 50, 50, cv.BORDER_DEFAULT);
  // Uncomment to apply Canny edge detection
  // cv.Canny(dstMat, dstMat, 50, 100, 3, false);
  
  cv.imshow("canvasOutput", dstMat);
}

// Call to indicate OpenCV is ready
onOpenCvReady();
// Call to initialize video processing
initializeVideoProcessing();