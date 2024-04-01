// Indicates when OpenCV is ready by updating the page status
function onOpenCvReady() {
  document.getElementById('status').textContent = 'OpenCV.js is ready.';
}

// Initializes file input elements and sets up event listeners for image loading
function initializeImageInputHandlers() {
  setupImageLoadListener('imageSrc', 'fileInput');
  setupImageLoadListener('imageSrc2', 'fileInput2');
}

// Sets up an event listener for when a new image is loaded via the input
function setupImageLoadListener(imageElementId, inputElementId) {
  const imgElement = document.getElementById(imageElementId);
  const inputElement = document.getElementById(inputElementId);

  inputElement.addEventListener('change', (e) => {
    loadImageIntoElement(e.target.files[0], imgElement);
  }, false);

  // Special handling for the first image element to perform template matching when loaded
  if (imageElementId === 'imageSrc') {
    imgElement.onload = processTemplateMatching;
  }
}

// Loads the selected image into the specified element
function loadImageIntoElement(file, element) {
  element.src = URL.createObjectURL(file);
}

// Processes template matching between two images
function processTemplateMatching() {
  const src = cv.imread('imageSrc');
  const template = cv.imread('imageSrc2'); // Assuming the second image is already loaded
  const dst = new cv.Mat();
  const mask = new cv.Mat();

  cv.matchTemplate(src, template, dst, cv.TM_CCOEFF, mask);
  const result = cv.minMaxLoc(dst, mask);
  const maxPoint = result.maxLoc;
  const color = new cv.Scalar(0, 255, 0, 255);
  const point = new cv.Point(maxPoint.x + template.cols, maxPoint.y + template.rows);

  cv.rectangle(src, maxPoint, point, color, 2, cv.LINE_8, 0);

  cv.imshow('canvasOutput', src);
  src.delete(); dst.delete(); template.delete(); mask.delete();
}

// Call the function to set everything up
onOpenCvReady();
initializeImageInputHandlers();
