/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
// import * as fs from "fs"

async function getData() {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    })).filter(car => (car.mpg != null && car.horsepower != null));
    // cleaned.length = 200
    return cleaned;
}

async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
      x: d.horsepower,
      y: d.mpg,
    }));
  
    tfvis.render.scatterplot(
      {name: 'Horsepower v MPG'},
      {values},
      {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300
      }
    );
    // More code will be added below
    // Create the model
    let modelResponse = await UrlExists("https://neu-prd-devops-01.northeurope.cloudapp.azure.com/my-model.json")
    
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;
    //check if trained json model exists, only load
    console.log(modelResponse)
    if(modelResponse == '404' || modelResponse == 'CORS Error'){
      console.log('hello i am here');
      var model = createModel();
      await trainModel(model, inputs, labels);
      // await model.save('https://neu-prd-devops-01.northeurope.cloudapp.azure.com/my-model.json')
      // await model.save('https://neu-prd-devops-01.northeurope.cloudapp.azure.com/my-model.json');
      await model.save('downloads://my-model');
      console.log(model)
    } else if (modelResponse === '200'){
      var model = await tf.loadLayersModel("https://neu-prd-devops-01.northeurope.cloudapp.azure.com/my-model.json")
    }



    // let modelResponse = await fetch("https://neu-prd-devops-01.northeurope.cloudapp.azure.com/my-model.json")
    // const model = await modelResponse.json()
    // $.getJSON("https://neu-prd-devops-01.northeurope.cloudapp.azure.com/my-model.json", function(result){
    //   console.log(result)
    // })
    // Convert the data to a form we can use for training.
    tfvis.show.modelSummary({name: 'Model Summary'}, model);
    // if(!model){
      // Train the model
    console.log('Done Training');
    testModel(model, data, tensorData);
    // }
    // for dev
}

function UrlExists(url) {
  var http = new XMLHttpRequest();
  try{
    http.open('HEAD', url, false);
    // if json model file does not exist, then the No CORS policy would not apply on it, therefore we need to check for this instead of 404 error
    http.send();
  } catch(err){
    return 'CORS Error'
    // console.log(err.indexOf("DOMException: Failed to execute 'send' on 'XMLHttpRequest': Failed to load") >= 0)
    
  }
  // console.log(http)
  // if(http.status == 0){
  //   return 'CORS Error'
  // }
  if (http.status != 404) {
    //  do something
    return "200"
  }
  else {
    return "404"
    
  }
}
  
document.addEventListener('DOMContentLoaded', run);

function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    // inputShape is 1 because we have 1 number as the input (the horsepower of a car)
    // unit is the weight
    // model.add(tf.layers.dense({inputShape: [1], units: 100, useBias: true}));
    // 500 is arbitrary, can be anything
    model.add(tf.layers.dense({inputShape: [1], units: 500, activation: 'sigmoid'}));
    model.add(tf.layers.dense({inputShape:[500], units: 1, activation: 'sigmoid'}));

    // Add an output layer
    // set units to 1 because we want to output 1 number (mpg)
    model.add(tf.layers.dense({units: 1, useBias: true}));
    // model.add(tf.layers.dense({units: [392], activation: 'sigmoid'}));

    return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.
  
    return tf.tidy(() => {
        // Step 1. Shuffle the data
        // important so that the model does not lean things that depend on the order the data was fed in
        // not be sensitive to the structure in subgroups (such as when only high horsepower cars are found in the first half of training)
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        // machine learning models are designed to work with numbers that are not too big
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
        }
    });
}

async function trainModel(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });
  
    const batchSize = 32;
    const epochs = 300;
  
    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    });
}

function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {

        const xsNorm = tf.linspace(0, 1, 100);
        const predictions = model.predict(xsNorm.reshape([100, 1]));

        const unNormXs = xsNorm
        .mul(inputMax.sub(inputMin))
        .add(inputMin);

        const unNormPreds = predictions
        .mul(labelMax.sub(labelMin))
        .add(labelMin);

        // Un-normalize the data
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });


    const predictedPoints = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]}
    });

    const originalPoints = inputData.map(d => ({
        x: d.horsepower, y: d.mpg,
    }));


    tfvis.render.scatterplot(
        {name: 'Model Predictions vs Original Data'},
        {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
        {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300
        }
    );

    
}
