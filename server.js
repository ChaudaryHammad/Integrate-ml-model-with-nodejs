const cors = require('cors');
const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));

app.post('/predict', async (req, res) => {
  try {
    const imageData = req.body.imageData;

    // Check if the image data is base64-encoded
    if (!imageData.startsWith('data:image')) {
      throw new Error('Invalid image data. Expected base64-encoded image.');
    }

    // Extract image type (e.g., 'image/jpeg') from base64 data
    const imageType = imageData.split(';')[0].split(':')[1];

    // Check if the image type is supported
    if (!['image/jpeg', 'image/png', 'image/gif', 'image/bmp'].includes(imageType)) {
      throw new Error('Unsupported image type. Expected BMP, JPEG, PNG, or GIF.');
    }

    // Remove the data:image/jpeg;base64, prefix before decoding
    const base64Data = imageData.replace(/^data:image\/\w+;base64,/, '');

    // Decode the base64 data and create a TensorFlow tensor
    let tensor = tf.node.decodeImage(Buffer.from(base64Data, 'base64'));

    // Resize the image to match the model input size
    tensor = tf.image.resizeBilinear(tensor, [256, 256]);

    // Add an extra dimension to represent the batch size
    tensor = tf.expandDims(tensor, 0);

    // Load the TensorFlow.js model
    const modelPath = path.join(__dirname, 'model.json');
    const model = await tf.loadLayersModel('file://' + modelPath);

    // Make a prediction
    const prediction = model.predict(tensor);

    // Send the prediction back to the client
    res.json({ prediction: prediction.arraySync() });
  } catch (error) {
    console.error(error.message);
    res.status(400).json({ error: error.message });
  }
});

app.listen(8000, () => {
  console.log('Server started on port 8000');
});
