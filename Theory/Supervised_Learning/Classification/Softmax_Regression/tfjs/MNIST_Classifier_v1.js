/*
 * Hand Written Digit Classifier trained using the Modified-National Institute of 
 * Standards and Technology (M-NIST) data set
 * 
 * Original code base derived from various material from Jason Mayes, tfjs 
 * development team https://github.com/jasonmayes
 *
 * Modified v1.0, 6/12/2023 Prof R CV class
 */

/*
 * To save disk space import the MNIST training data from this URL 
 *   - The training data is in normalized form (0.0 to 1.0 grayscale)
 *   - The import contains 10,000 samples all in order (0, first, and 9 last)
 */
import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js';

const INPUTS = TRAINING_DATA.inputs;   // Reference to the MNIST input values (pixel data).
const OUTPUTS = TRAINING_DATA.outputs; // Reference to the MNIST output values.

// Shuffle to make odering of digita smples random, but still keep the input-output pairing
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Create two tensors
const INPUTS_TENSOR = tf.tensor2d(INPUTS); // Input feature Array is 2 dimensional.
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10); // Output feature Array is 1 dimensional.

/////////////////////////////
//
// Create model (simple NN)
// - 784 inputs via 28x28 MNIST image
// - first input layer contains 50 neurons with a ReLu activation function
// - One hidden layer with 16 neurons and a ReLu activation function
// - Output layer must have 10 neurons, one for each digit the model can predict
//     - The activation function is softmax so that get percentage confidences
//
/////////////////////////////
const model = tf.sequential();

model.add(tf.layers.dense({inputShape: [784], units: 50, activation: 'relu'}));
model.add(tf.layers.dense({units: 16, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
model.summary();  // print model summary to console

train();  // Call the function to train the model

/////////////////////////////
//
// Train the model
// - Choose an optimizer
//      - stochastic gradient descent
//      - adam, automatically changes the learning rate over time
// - Choose a loss function
//      - categoricalCrossentropy
// - Choose metric
//      - accuracy, be sure that validation accuracy is going up over time
/////////////////////////////

async function train() { 
  // Compile the model with the defined optimizer and loss function
  model.compile({
    optimizer: 'adam', 
    loss: 'categoricalCrossentropy', 
    metrics: ['accuracy']  
  });

  // Do the actual training
  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true,        // Ensure data is shuffled again before using each time.
    validationSplit: 0.2, // set aside 20% of the data for validation testing
    batchSize: 512,       // Update weights after every 512 examples, this can be tweaked      
    epochs: 50,           // This is also a tweakable variable, if the convergence is fast it can be low
    callbacks: {onEpochEnd: logProgress} // callback after each epoch for loss and accuracy score
  });
  
  // Training complete, delete input and output tensors
  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();
    
  evaluate();  // After the mode is trained it can be evaluated
}


function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
}

//////////////////////
//
// Test the model with some MNIST sample images.
// - Display the test MNIST image
// - Display predicted digit 
//   - Green for a correct prediction
//   - Red for an incorrect prediction
////////////////////


// reference to the element('prediction') in HTML where predictions will be rendered
const PREDICTION_ELEMENT = document.getElementById('prediction');

function evaluate() {
  const OFFSET = Math.floor((Math.random() * INPUTS.length)); // a random index into the example images
  
  // Clean up created tensors automatically.
  let answer = tf.tidy(function() {
    let randomImg = tf.tensor1d(INPUTS[OFFSET]); // Create a 1D tensor of the gryscale image
    
    // expandDims turns randomImg into a 2D tensor since predict expexts a batch of images
    let output = model.predict(randomImg.expandDims());
    output.print(); // print the output to the console
    
    // squeeze coverts output to a 1D tensor & argMax returns index of the highwar number in the tensor
    return output.squeeze().argMax();    
  });
  
  answer.array().then(function(index) {
    PREDICTION_ELEMENT.innerText = index;
    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  });
}

/////////////
//
// Code to draw the MNIST digit
//
////////////
const CANVAS = document.getElementById('canvas'); // Reference to HTML element 'canvas'
const CTX = CANVAS.getContext('2d');              // 2D context for this canvas


function drawImage(digit) {
  var imageData = CTX.getImageData(0, 0, 28, 28); // Get current canvas data
  
  // Create and RBGA image by mutiplying the intensity in digit by RGB color component, alpha set to 255
  for (let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255;      // Red Channel.
    imageData.data[i * 4 + 1] = digit[i] * 255;  // Green Channel.
    imageData.data[i * 4 + 2] = digit[i] * 255;  // Blue Channel.
    imageData.data[i * 4 + 3] = 255;             // Alpha Channel.
  }

  // Render the updated array of data to the canvas itself.
  CTX.putImageData(imageData, 0, 0);

  // Perform a new classification after a certain interval.
  setTimeout(evaluate, interval); // classify a random handwritten digit every (interval) ms
}


var interval = 2000;
const RANGER = document.getElementById('ranger');
const DOM_SPEED = document.getElementById('domSpeed');

// When user drags slider update interval.
RANGER.addEventListener('input', function(e) {
  interval = this.value;
  DOM_SPEED.innerText = 'Change speed of classification! Currently: ' + interval + 'ms';
});