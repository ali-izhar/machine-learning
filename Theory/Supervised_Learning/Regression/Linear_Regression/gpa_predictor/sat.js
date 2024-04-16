let univGPAModel, compGPAModel;
let isUnivModelTrained = false;
let isCompModelTrained = false;
let minMathSAT, maxMathSAT, minCompGPA, maxCompGPA;

async function loadAndPredict() {
    // Load and process data
    const response = await fetch('sat.csv');
    const data = await response.arrayBuffer();
    const workbook = XLSX.read(data, {type: 'buffer'});
    const sheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[sheetName];
    const csvData = XLSX.utils.sheet_to_csv(worksheet);

    // Parse CSV data
    Papa.parse(csvData, {
        header: true,
        dynamicTyping: true,
        complete: async function(results) {
            const data = results.data;
            
            // Split data for different models
            const highGPAData = data.map(d => d.high_GPA);
            const univGPAData = data.map(d => d.univ_GPA);
            const mathSATData = data.map(d => d.math_SAT);
            const compGPAData = data.map(d => d.comp_GPA);

            // Normalize SAT scores and GPAs
            const normalizedMathSATData = normalize(mathSATData);
            const normalizedCompGPAData = normalize(compGPAData);

            // Find min and max values for SAT scores and GPAs
            minMathSAT = Math.min(...mathSATData);
            maxMathSAT = Math.max(...mathSATData);
            minCompGPA = Math.min(...compGPAData);
            maxCompGPA = Math.max(...compGPAData);

            // University GPA Model
            univGPAModel = createModel();
            await trainModel(univGPAModel, highGPAData, univGPAData, 'univ');

            // Computer Science GPA Model
            compGPAModel = createModel();
            await trainModel(compGPAModel, normalizedMathSATData, normalizedCompGPAData, 'comp');

            // Generate predictions for the scatter plot using univGPAModel
            const predictedUnivGPAs = highGPAData.map(highGPA => 
                univGPAModel.predict(tf.tensor2d([highGPA], [1, 1])).dataSync()[0]
            );

            // Create the scatter plot chart
            createChart(univGPAData, highGPAData, predictedUnivGPAs);
        }
    });
}

function normalize(data) {
    const maxVal = Math.max(...data);
    const minVal = Math.min(...data);
    return data.map(val => (val - minVal) / (maxVal - minVal));
}

function normalizeValue(value, min, max) {
    return (value - min) / (max - min);
}

function denormalizeValue(normalized, min, max) {
    return normalized * (max - min) + min;
}

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    return model;
}

async function trainModel(model, inputs, labels, type) {
    if ((type === 'univ' && isUnivModelTrained) || (type === 'comp' && isCompModelTrained)) {
        console.log(`Model for ${type} is already trained`);

        return;
    }

    document.getElementById('trainingAlert').style.display = 'block';

    try {
        model.compile({
            optimizer: tf.train.sgd(0.01),
            loss: 'meanSquaredError'
        });

        const xs = tf.tensor2d(inputs, [inputs.length, 1]);
        const ys = tf.tensor2d(labels, [labels.length, 1]);

        await model.fit(xs, ys, {
            epochs: 100,
            callbacks: {
                onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
            }
        });
        console.log(`Model for ${type} trained successfully`);
        document.getElementById('trainingAlert').textContent = 'Training complete!';
        document.getElementById('trainingAlert').style.display = 'none';
        if (type === 'univ') {
            isUnivModelTrained = true;
        } else {
            isCompModelTrained = true;
        }
    } catch (error) {
        console.error(`Error training model for ${type}:`, error);
    }
}

// Function to predict university GPA based on high school GPA
async function predictUnivGPA() {
    const highGPAInput = document.getElementById('highGPAInput');
    if (highGPAInput && univGPAModel) {
        const highGPA = parseFloat(highGPAInput.value);
        if (!isNaN(highGPA)) {
            try {
                const predictionTensor = univGPAModel.predict(tf.tensor2d([highGPA], [1, 1]));
                const prediction = predictionTensor.dataSync()[0];
                predictionTensor.dispose();
                document.getElementById('predictedUnivGPA').textContent = prediction.toFixed(2);
            } catch (error) {
                console.error('Error during prediction:', error);
                document.getElementById('predictedUnivGPA').textContent = 'Error in prediction';
            }
        } else {
            document.getElementById('predictedUnivGPA').textContent = 'Invalid input';
        }
    } else {
        document.getElementById('predictedUnivGPA').textContent = 'Model not ready or input not found';
    }
}

// Function to predict computer science GPA based on normalized Math SAT score
async function predictCompGPA() {
    const satScoreInput = document.getElementById('mathSATInput');
    if (satScoreInput && compGPAModel) {
        const satScore = parseFloat(satScoreInput.value);
        if (!isNaN(satScore)) {
            try {
                // Normalize SAT score for prediction
                const normalizedSAT = normalizeValue(satScore, minMathSAT, maxMathSAT);
                const predictionTensor = compGPAModel.predict(tf.tensor2d([normalizedSAT], [1, 1]));
                let prediction = predictionTensor.dataSync()[0];
                predictionTensor.dispose();

                // Denormalize the predicted GPA if necessary
                prediction = denormalizeValue(prediction, minCompGPA, maxCompGPA);

                document.getElementById('predictedCompGPA').textContent = prediction.toFixed(2);
            } catch (error) {
                console.error('Error during prediction:', error);
                document.getElementById('predictedCompGPA').textContent = 'Error in prediction';
            }
        } else {
            document.getElementById('predictedCompGPA').textContent = 'Invalid input';
        }
    } else {
        document.getElementById('predictedCompGPA').textContent = 'Model not ready or input not found';
    }
}

function createChart(actualGPAs, highGPAValues) {
    const ctx = document.getElementById('gpaChart').getContext('2d');
    
    // Generate predictions for the range of high school GPAs
    const predictionLine = highGPAValues.map(highGPA => {
        const prediction = univGPAModel.predict(tf.tensor2d([highGPA], [1, 1])).dataSync()[0];
        return { x: highGPA, y: prediction };
    });

    // Sort the prediction line by high school GPA to ensure the line is plotted correctly
    predictionLine.sort((a, b) => a.x - b.x);

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Actual University GPA',
                data: actualGPAs.map((gpa, index) => ({ x: highGPAValues[index], y: gpa })),
                backgroundColor: 'rgba(255, 99, 132, 0.6)',
                pointRadius: 5
            }, {
                label: 'Predicted University GPA',
                data: predictionLine,
                type: 'line', // Define this dataset as a line chart
                fill: false,
                borderColor: 'rgba(54, 162, 235, 0.6)',
                borderWidth: 2,
                pointRadius: 0 // No points on the prediction line
            }]
        },
        options: {
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'High School GPA'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'University GPA'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                }
            }
        }
    });
}


window.onload = loadAndPredict;