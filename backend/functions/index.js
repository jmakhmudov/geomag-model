const functions = require("firebase-functions");
const tf = require("@tensorflow/tfjs");

exports.predictWithTF = functions.https.onRequest(async (req, res) => {
  try {
    const jsonData = await fetch("https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json").then((response) => response.json());
    const currentHour = new Date().getHours();
    const hourlyData = [];

    for (let i = currentHour - 23; i <= currentHour; i++) {
      let hourData;
      if (i >= 0) {
        hourData = jsonData.filter((data) => {
          const hour = new Date(data[0]).getHours();
          return hour === i;
        });
      } else {
        hourData = jsonData.filter((data) => {
          const hour = new Date(data[0]).getHours();
          return hour === i + 24;
        });
      }

      const filteredData = hourData.filter((row) => row[3] !== null);

      const averageSpeed = filteredData.reduce((sum, data) =>
        sum + parseFloat(data[2]), 0) / filteredData.length;
      const averageDensity = filteredData.reduce((sum, data) =>
        sum + parseFloat(data[1]), 0) / filteredData.length;
      const averageTemperature = filteredData.reduce((sum, data) =>
        sum + parseFloat(data[3]), 0) / filteredData.length;

      hourlyData.push(averageSpeed / 100,
          averageDensity, averageTemperature / 100000);
    }
    const pred = await predict(hourlyData.reverse());
    const jsonResponse = {
      prediction:
        pred < 0 ? 0.00 : Math.abs(roundToNearestTemplate(pred)).toFixed(2)};
    res.status(200).json(jsonResponse);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({error: "An error occurred"});
  }
});

/**
 * Rounds a float number to the nearest template value of ".00", ".33", or ".67"
 *
 * @param {number} number - The input float number to be rounded.
 * @return {number} The rounded number using the nearest template.
 *
 * @example
 * const originalNumber = 4.56;
 * const roundedNumber = roundToNearestTemplate(originalNumber);
 * // Output: 4.67
 */
function roundToNearestTemplate(number) {
  const fractionalPart = number - Math.floor(number);
  if (fractionalPart < 0.165) {
    return Math.floor(number) + 0.00;
  } else if (fractionalPart < 0.5) {
    return Math.floor(number) + 0.33;
  } else if (fractionalPart < 0.835) {
    return Math.floor(number) + 0.67;
  } else {
    return Math.floor(number + 1) + 0.00;
  }
}

/**
 * This is a simple function that adds two numbers.
 * @param {Array} data - The first number.
 * @return {number} - The sum of a and b.
 */
async function predict(data) {
  const tensor = tf.tensor(data, [1, 72]);
  const model = await tf.loadLayersModel("https://firebasestorage.googleapis.com/v0/b/geomag-ml-api.appspot.com/o/model.json?alt=media&token=db016038-4c56-4c85-97ae-3ec764638d5d");
  const pred = await model.predict(tensor).dataSync();
  console.log(pred)
  return pred[0];
}
