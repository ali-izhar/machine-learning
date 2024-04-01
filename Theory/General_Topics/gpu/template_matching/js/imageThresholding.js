// Select the video element on the page
let video = document.getElementById("videoInput");
video.width = 640;
video.height = 480;

// Access the user's webcam
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
  .then(function(stream) {
    video.srcObject = stream;
    video.play();

    // Create OpenCV objects for processing
    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let cap = new cv.VideoCapture(video);

    const FPS = 30; // Frames per second
    function processVideo() {
      try {
        let begin = Date.now();
        
        // Capture and process the video frame
        cap.read(src);
        cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);

        let thresholdValue = 0; // Initial value for Otsu's method
        let maxValue = 255; // Maximum value for binary thresholding

        // Apply Otsu's thresholding method
        cv.threshold(dst, dst, thresholdValue, maxValue, cv.THRESH_BINARY + cv.THRESH_OTSU);
        
        // Apply binary thresholding
        // cv.threshold(dst, dst, thresholdValue, maxValue, cv.THRESH_BINARY);
             
        cv.imshow("canvasOutput", dst);

        // Schedule the next frame processing
        let delay = 1000 / FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
      } catch (err) {
        // Log errors to the console
        console.error(err);
      }
    }

    // Start the video processing
    setTimeout(processVideo, 0);
  })
  .catch(function(err) {
    // Handle errors with webcam access
    console.log("An error occurred! " + err);
  });
