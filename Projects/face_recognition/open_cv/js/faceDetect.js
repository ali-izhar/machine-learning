function setupVideoStream() {
    let video = document.getElementById("cam_input");
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then(function (stream) {
        video.srcObject = stream;
        video.play();
      })
      .catch(function (err) {
        console.log("An error occurred! " + err);
      });
  }
  
  function loadCascadeClassifier(utils, classifier, cascadeFile, callback) {
    utils.createFileFromUrl(cascadeFile, cascadeFile, () => {
      classifier.load(cascadeFile);
      if (typeof callback === 'function') callback();
    });
  }
  
  function detectAndDrawFaces(src, dst, gray, cap, faceClassifier, eyeClassifier, faces, eyes) {
    const FPS = 24;
    function processVideo() {
      let begin = Date.now();
      cap.read(src);
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      try {
        faceClassifier.detectMultiScale(gray, faces, 1.1, 3, 0);
        drawFaces(faces, gray, src, eyeClassifier, eyes);
      } catch (err) {
        console.log(err);
      }
      cv.imshow("canvas_output", src);
      scheduleNextFrame(processVideo, begin, FPS);
    }
    processVideo(); // Start the processing loop
  }
  
  function drawFaces(faces, gray, src, eyeClassifier, eyes) {
    for (let i = 0; i < faces.size(); ++i) {
      let face = faces.get(i);
      let point1 = new cv.Point(face.x, face.y);
      let point2 = new cv.Point(face.x + face.width, face.y + face.height);
      cv.rectangle(src, point1, point2, [255, 0, 0, 255]);
      let roiGray = gray.roi(face);
      eyeClassifier.detectMultiScale(roiGray, eyes);
      drawEyes(eyes, face, src);
      roiGray.delete();
    }
  }
  
  function drawEyes(eyes, face, src) {
    for (let j = 0; j < eyes.size(); ++j) {
      let eye = eyes.get(j);
      let point1 = new cv.Point(face.x + eye.x, face.y + eye.y);
      let point2 = new cv.Point(face.x + eye.x + eye.width, face.y + eye.y + eye.height);
      cv.rectangle(src, point1, point2, [0, 0, 255, 255]);
    }
  }
  
  function scheduleNextFrame(processVideoCallback, beginTime, FPS) {
    let delay = 1000 / FPS - (Date.now() - beginTime);
    setTimeout(processVideoCallback, delay);
  }
  
  function openCvReady() {
    cv['onRuntimeInitialized'] = () => {
      document.getElementById("status").innerHTML = "OpenCV.js is ready.";
      setupVideoStream();
      let video = document.getElementById("cam_input");
      let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
      let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
      let gray = new cv.Mat();
      let cap = new cv.VideoCapture(cam_input);
      let faces = new cv.RectVector();
      let eyes = new cv.RectVector();
      let faceClassifier = new cv.CascadeClassifier();
      let eyeClassifier = new cv.CascadeClassifier();
      let utils = new Utils('errorMessage');
      let faceCascadeFile = 'haarcascade_frontalface_default.xml';
      let eyeCascadeFile = 'haarcascade_eye.xml';
  
      loadCascadeClassifier(utils, faceClassifier, faceCascadeFile, () => {
        loadCascadeClassifier(utils, eyeClassifier, eyeCascadeFile, () => {
          detectAndDrawFaces(src, dst, gray, cap, faceClassifier, eyeClassifier, faces, eyes);
        });
      });
    };
  }