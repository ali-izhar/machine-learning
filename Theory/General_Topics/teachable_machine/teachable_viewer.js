let model, webcam, labelContainer, maxPredictions;
let modelReady = false; // Flag to indicate model readiness

async function init() {
    const modelURL = "model.json";
    const metadataURL = "metadata.json";

    model = await tmImage.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();
    modelReady = true;

    labelContainer = document.getElementById("label-container");
    for (let i = 0; i < maxPredictions; i++) {
        const div = document.createElement("div");
        div.classList.add("prediction-label"); // Optionally add a class for styling
        labelContainer.appendChild(div);
    }

    // Buttons are initially disabled and will be enabled after model is ready
    document.getElementById('imageUpload').disabled = !modelReady;
    document.getElementById('predictButton').disabled = !modelReady;
}

async function startWebcam() {
    if (!modelReady) {
        console.log('Model not ready for webcam.');
        return;
    }

    if (!webcam) {
        const flip = true; // Flip the webcam
        webcam = new tmImage.Webcam(640, 480, flip); // Adjusted size for better visibility
        await webcam.setup({ facingMode: "user" }); // Setup with desired facingMode
        await webcam.play();
        document.getElementById("webcam-container").innerHTML = ''; // Clear "Webcam Preview" text
        document.getElementById("webcam-container").appendChild(webcam.canvas);
        webcam.canvas.classList.add("webcam-feed"); // Optionally add a class for styling
    }
    window.requestAnimationFrame(loop);
}

async function loop() {
    webcam.update(); // update the webcam frame
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    // predict can take in an image, video or canvas html element
    const prediction = await model.predict(webcam.canvas);
    for (let i = 0; i < maxPredictions; i++) {
        const classPrediction =
            prediction[i].className + ": " + prediction[i].probability.toFixed(2);
        labelContainer.childNodes[i].innerHTML = classPrediction;
    }
}

function previewImage() {
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');

    if (imageUpload.files.length > 0) {
        const file = imageUpload.files[0];
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.onload = () => {
            // Adjust image styles as needed
            img.style.maxWidth = '100%';
            img.style.height = 'auto';
        };
        // Clear the previous image
        imagePreview.innerHTML = '';
        imagePreview.appendChild(img);

        // Enable the prediction button once an image is selected
        document.getElementById('predictButton').disabled = false;
    }
}

async function predictUpload() {
    const imagePreview = document.getElementById('imagePreview').getElementsByTagName('img')[0];
    if (!modelReady) {
        console.log('Model not ready for predictions');
        return;
    }
    if (imagePreview) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 200; // Adjust according to your model's input size
        canvas.height = 200;
        ctx.drawImage(imagePreview, 0, 0, canvas.width, canvas.height);
        const prediction = await model.predict(canvas);
        
        // Initialize the list
        const uploadPredictionContainer = document.getElementById('upload-prediction');
        uploadPredictionContainer.innerHTML = ''; // Clear previous predictions
        const list = document.createElement('ul');
        
        for (let i = 0; i < maxPredictions; i++) {
            const classPrediction = `${prediction[i].className}: ${prediction[i].probability.toFixed(2)}`;
            const item = document.createElement('li');
            item.innerHTML = classPrediction;
            list.appendChild(item); // Add each prediction as a list item
        }

        uploadPredictionContainer.appendChild(list); // Append the list to the container
    }
}