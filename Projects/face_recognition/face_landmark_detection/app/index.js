import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const imageDisplay = document.getElementById("upload-image");
const imageInput = document.getElementById("image-input");
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");
const webcamVideo = document.getElementById("webcam");
const webcamButton = document.getElementById("webcamButton");
const outputCanvas = document.getElementById("output_canvas");
const canvasCtx = outputCanvas.getContext("2d");

let faceLandmarker;
let webcamRunning = false;
const videoWidth = 480;

async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU"
    },
    outputFaceBlendshapes: true,
    runningMode: "IMAGE",
    numFaces: 1
  });
}

function displayLandmarks(result, mediaElement, isVideo = false) {
  const canvas = document.getElementById('output_canvas');
  
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
  for (const landmarks of result.faceLandmarks) {
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });
  }
  if (isVideo) {
    drawBlendShapes(videoBlendShapes, result.faceBlendshapes);
  } else {
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
}

async function predictWebcam() {
  if (!webcamRunning) {
    return; // Exit if the webcam has been stopped
  }
  
  if (webcamVideo.readyState === webcamVideo.HAVE_ENOUGH_DATA) {
    const result = await faceLandmarker.detect(webcamVideo);
    if (result && result.faceLandmarks.length > 0) {
      displayLandmarks(result, webcamVideo, true); // Pass 'true' for isVideo
    }
  }
  requestAnimationFrame(predictWebcam);
}

function drawBlendShapes(element, blendShapes) {
  if (!blendShapes.length) {
      return;
  }
  let html = blendShapes.map(shape => `<li>${shape.categoryName}: ${shape.score.toFixed(4)}</li>`).join('');
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
      const result = await faceLandmarker.detect(imageDisplay);
      displayLandmarks(result, imageDisplay);
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
