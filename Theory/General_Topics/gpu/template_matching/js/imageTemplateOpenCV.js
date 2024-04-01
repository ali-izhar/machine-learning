// Indicates when OpenCV is ready by updating the page status
function onOpenCvReady() {
  document.getElementById('status').textContent = 'OpenCV.js is ready.';
}

// Initializes OpenCV objects and listeners for the Canny Edge Detection
function initializeCannyEdgeDetection() {
   const imgElement = document.getElementById('imageSrc');
   const inputElement = document.getElementById('fileInput');
   setupImageLoadListener(imgElement, inputElement);
 }
 
 // Sets up an event listener for when a new image is loaded via the input
 function setupImageLoadListener(imgElement, inputElement) {
   inputElement.addEventListener('change', (e) => {
     loadImageIntoElement(e.target.files[0], imgElement);
   }, false);
 
   imgElement.onload = processImageWithOpenCV;
 }
 
 // Loads the selected image into the specified element
 function loadImageIntoElement(file, element) {
   element.src = URL.createObjectURL(file);
 }
 
 // Processes the loaded image using OpenCV
 function processImageWithOpenCV() {
   const srcMat = cv.imread('imageSrc');
   const dstMat = new cv.Mat();
 
   preprocessImage(srcMat, dstMat);
   cv.imshow('canvasOutput', dstMat);
 
   // Free memory
   srcMat.delete();
   dstMat.delete();
 }
 
 // Preprocesses the image with OpenCV (grayscale conversion, Gaussian blur, etc.)
 function preprocessImage(srcMat, dstMat) {
   cv.cvtColor(srcMat, dstMat, cv.COLOR_RGBA2GRAY);
   // original kernelSize = new cv.Size(5, 5);
   const kernelSize = new cv.Size(25, 25);
   // original cv.GaussianBlur(dstMat, dstMat, kernelSize, 100, 100, cv.BORDER_DEFAULT);
   cv.GaussianBlur(dstMat, dstMat, kernelSize, 100, 100, cv.BORDER_DEFAULT);
   // Uncomment to apply Canny edge detection
   // cv.Canny(dstMat, dstMat, 50, 100, 3, false);
 }
 
// Call the function to set everything up
onOpenCvReady();
// Call the function to initialize everything
initializeCannyEdgeDetection();
 