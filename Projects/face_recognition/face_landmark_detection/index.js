import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const imageDisplay = document.getElementById("upload-image");
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");
const webcamVideo = document.getElementById("webcam");
const webcamButton = document.getElementById("webcamButton");
const outputCanvas = document.getElementById("output_canvas");

let faceLandmarker;
let webcamRunning = false;

async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    },
    outputFaceBlendshapes: true,
    runningMode: "IMAGE",
  });
}

function clearCanvas() {
  const canvas = document.getElementById('output_canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const imageCanvas = document.getElementById('image-canvas');
  const imageCtx = imageCanvas.getContext('2d');
  imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
}

function displayLandmarks(result, mediaElement, isVideo = false) {
  const canvas = document.getElementById(isVideo ? 'output_canvas' : 'image-canvas');
  
  // Determine the size of the mediaElement
  let mediaWidth, mediaHeight;
  if (isVideo) {
    // For video, use the intrinsic video dimensions
    mediaWidth = mediaElement.videoWidth;
    mediaHeight = mediaElement.videoHeight;
  } else {
    // For images, use the natural dimensions
    mediaWidth = mediaElement.naturalWidth;
    mediaHeight = mediaElement.naturalHeight;
  }

  // Get the size as displayed on screen
  const { width, height } = mediaElement.getBoundingClientRect();
  
  // Adjust the canvas to match the displayed size
  canvas.width = width;
  canvas.height = height;
  canvas.style.display = 'block';
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.style.top = `${mediaElement.offsetTop}px`;
  canvas.style.left = `${mediaElement.offsetLeft}px`;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Compute the scale factors
  const scaleX = width / mediaWidth;
  const scaleY = height / mediaHeight;

  // Create a DrawingUtils instance with scaling
  const drawingUtils = new DrawingUtils(ctx, scaleX, scaleY);

  // Draw the landmarks scaled to the canvas size
  result.faceLandmarks.forEach((faceLandmarks) => { 
    drawingUtils.drawConnectors(faceLandmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
    drawingUtils.drawConnectors(faceLandmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
    drawingUtils.drawConnectors(faceLandmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
    drawingUtils.drawConnectors(faceLandmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
    drawingUtils.drawConnectors(faceLandmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
    drawingUtils.drawConnectors(faceLandmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
    drawingUtils.drawConnectors(faceLandmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
    drawingUtils.drawConnectors(faceLandmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
    drawingUtils.drawConnectors(faceLandmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });
  });
  if (isVideo) {
    drawBlendShapes(videoBlendShapes, result.faceBlendshapes);
  } else {
    canvas.style.display = 'block';
    drawBlendShapes(imageBlendShapes, result.faceBlendshapes);
  }
}

function startWebcam() {
  const constraints = { video: true };
  navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      webcamVideo.srcObject = stream;
      outputCanvas.style.display = 'block'; // Make the canvas visible
      webcamRunning = true;
      webcamButton.innerText = "Stop Webcam";
      predictWebcam();
  }).catch(function(error) {
      console.log("Error accessing the webcam", error);
  });
}

function stopWebcam() {
  webcamVideo.srcObject.getTracks().forEach(track => track.stop());
  webcamVideo.srcObject = null;
  clearCanvas();
}

async function predictWebcam() {
  if (!webcamRunning) {
    console.log("Webcam has been stopped."); // Log when the webcam is not running
    return; // Exit if the webcam has been stopped
  }

  if (webcamVideo.readyState === webcamVideo.HAVE_ENOUGH_DATA) {
    try {
      const result = await faceLandmarker.detect(webcamVideo);
      if (result && result.faceLandmarks.length > 0) {
        displayLandmarks(result, webcamVideo, true); // Pass 'true' for isVideo
      } else {
        console.log("No landmarks detected."); // Log if no landmarks are detected
      }
    } catch (error) {
      console.error("Error during face landmark detection:", error); // Log any errors during detection
    }
  } else {
    console.log("The webcam video data is not ready yet."); // Log if video data is not ready
  }
  requestAnimationFrame(predictWebcam); // Ensure the loop continues
}

function drawBlendShapes(element, blendShapes) {
  if (!blendShapes || blendShapes.length === 0) {
      console.log("No blend shapes data to draw.");
      return;
  }
  let html = blendShapes.map(shape => {
      if (shape && shape.score !== undefined) {
          return `<li>${shape.categoryName}: ${shape.score.toFixed(4)}</li>`;
      }
      return ''; // Return an empty string if the shape data is not valid
  }).join('');
  element.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', function() {
  const imageInput = document.getElementById('image-input');
  imageInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function(e) {
        const uploadedImage = document.getElementById('upload-image');
        uploadedImage.src = e.target.result;
        clearCanvas(); // Ensure the previous drawings are cleared
      };
      reader.readAsDataURL(file);
    }
  });

  createFaceLandmarker();

  const detectButton = document.getElementById('detect-image');
  detectButton.addEventListener('click', async function() {
    if (!faceLandmarker) {
      console.log("Face Landmarker model not loaded yet!");
      return;
    }
    // Use the uploaded image for detection
    const result = await faceLandmarker.detect(imageDisplay);
    if (result && result.faceLandmarks.length > 0) {
      // Call displayLandmarks with the imageDisplay and false for isVideo
      displayLandmarks(result, imageDisplay, false);
      console.log(`Detected ${result.faceLandmarks.length} faces`);
    } else {
      console.log("No landmarks detected on the image.");
    }
  });

  const webcamButton = document.getElementById('webcamButton');
  webcamButton.addEventListener('click', function() {
      if (!webcamRunning) {
          startWebcam();
          webcamRunning = true;
          webcamButton.innerText = "Stop Webcam";
      } else {
          stopWebcam();
          webcamRunning = false;
          webcamButton.innerText = "Start Webcam";
      }
  });

});
