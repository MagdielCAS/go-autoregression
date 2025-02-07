go-autoregression
=======
[![GoDoc](https://godoc.org/github.com/MagdielCAS/go-autoregression?status.svg)](https://godoc.org/github.com/MagdielCAS/go-autoregression)
[![Go Report Card](https://goreportcard.com/badge/MagdielCAS/go-autoregression)](https://goreportcard.com/report/MagdielCAS/go-autoregression)
[![License][license-image]][license-url]

[license-image]: https://img.shields.io/badge/license-BS2-green.svg?style=flat-square
[license-url]: LICENSE


# ar: Autoregressive Time Series Prediction Package

Package `ar` provides a simple implementation of an Autoregressive (AR) model for time series prediction in Go.  It allows you to predict future values of a time series based on its past values and associated parameters.  This package is designed to be easy to use and understand, making it suitable for learning about AR models or for quick prototyping.

## Overview

The AR model predicts future values based on a weighted sum of past values and associated parameters. The key concept is that the future depends on the past. This implementation uses the "least squares" method to determine the optimal weights for the model.

## Features

* **Configurable Lags:** Control the number of past data points (`na`) and past parameter values (`nb`) used for prediction.
* **Adjustable Step Size:** Specify the time interval (`StepSize`) between data points for accurate future time projections.
* **Clear Error Handling:** Returns errors for invalid parameters or insufficient data.
* **Simple API:** Easy-to-use `NewPredictor` and `Predict` functions.
* Uses an additional array of parameters `P`, which are used alongside the main data to improve the predictive capabilities making it a more versatile forecasting system.

## Installation

```bash
go get github.com/MagdielCAS/go-autoregression
```

## Usage

```go
package main

import (
 "encoding/json"
 "fmt"
 "log"
 "os"
 "github.com/MagdielCAS/go-autoregression" // Replace with the actual path to your 'ar' package.
)

func main() {
 // 1. Sample Data (replace with your actual data).
 //    Each inner slice is [data_value, time_value/external_input].
 data := [][]float64{
  {1578.0077, 0}, {1581.1876, 5}, {1452.4627, 33},
  {1449.7326, 58}, {1501.0392, 80}, {1460.4557, 110},
      //... More Data
  {2261.5, 2070},
 }

 // 2. Define AR Model Parameters.
 params := ar.ModelParameters{
  AutoregressiveLags: 3,   // 'na' -  How many past 'data_value' to consider.
  ExternalInputLags:  3,   // 'nb' -  How many past 'time_value' to consider.
  StepSize:           25.0, // 'stepSize' - The interval between 'time_value' samples.
 }

 // 3. Create a new AR model predictor.
 predictor, err := ar.NewPredictor(data, params)
 if err != nil {
  log.Fatalf("Failed to create predictor: %v", err)
 }

 // 4. Set the number of prediction steps.
 numToPredict := 25

 // 5. Perform prediction.
 predictedData, err := predictor.Predict(numToPredict)
 if err != nil {
  log.Fatalf("Prediction failed: %v", err)
 }

 // 6. Print and/or Marshal Results.
 fmt.Println("Predicted Data:")
    fmt.Println(predictedData)
 // To get a json:
    // jsonParsed, _ := json.MarshalIndent(predictedData, "", "  ")
 // fmt.Println(jsonParsed)
}
```

------------

## API Reference

### `type ModelParameters`

```go
type ModelParameters struct {
 AutoregressiveLags int     // na: Number of past data points to consider.
 ExternalInputLags  int     // nb: Number of past external input values to consider.
 StepSize           float64 // StepSize:  The time interval ('delta Time') to use.
}
```

Defines the parameters for the AR model.

* `AutoregressiveLags (na)`:  The number of past data values to use in the autoregressive component.  Must be greater than 0.
* `ExternalInputLags (nb)`: The number of past external input/time values to use. Must be greater than or equal to 0.
* `StepSize`: The time step to use for projecting future time values. Must be greater than 0. Represents the regular interval at which the original data was sampled.

### `func NewPredictor(data [][]float64, params ModelParameters) (*Predictor, error)`

Creates a new `Predictor` instance.

* `data`:  A slice of slices, where each inner slice contains two floats: `[data_value, time_value/external_input]`.  This is your historical time series data. The `time_value` should ideally be monotonically increasing.
* `params`:  The `ModelParameters` to configure the AR model.

Returns a pointer to the `Predictor` or an error if the parameters are invalid (e.g., non-positive lags or step size) or if there's not enough data.

### `func (p *Predictor) Predict(numToPredict int) ([][]float64, error)`

Performs the prediction.

* `numToPredict`:  The number of steps into the future to predict.

Returns a slice of slices, where each inner slice contains two floats: `[predicted_time, predicted_value]`.  Returns an error if the prediction fails (e.g., not enough data to perform the prediction given the specified lags).

------------

## What the Code Does: Prediction Based on Past Behavior

Imagine you have a series of data points collected over time.  This could be anything:

* The daily price of a stock.
* The hourly temperature in a city.
* The weekly sales of a product.
* The number of website visitors each month.

This code tries to *predict* future values in that series based on the patterns it sees in the *past* data. It's like saying, "If the temperature has been steadily increasing over the last few days, it's likely to continue increasing tomorrow."  Of course, it's more sophisticated than just simple extrapolation, but that's the general idea.

## The Algorithm: Autoregressive (AR) Model

The specific method the code uses is called an **Autoregressive (AR) model**.  Let's break down that term:

* **Auto:**  This means "self."
* **Regressive:**  This refers to "regression," a statistical technique that looks at the relationship between variables.

So, an "Autoregressive" model is one that uses the *past values of a variable to predict its future values*.  It's like the variable is "regressing" on itself.

## How the AR Model Works (Simplified Analogy)

Think of it like a simple recipe:

1. **Ingredients (Past Data):** The AR model looks at the *recent* past values of your data.  The "recent" part is important.  The code uses a parameter called `na` (which is set to 3 in this case) that determines how many past values it considers.  So, it might look at the values from the last 3 days, 3 weeks, 3 months, etc., depending on your data.
    Also it use other parameter `nb` related with a secondary set of values (`P` in the code) `P` values are parameters that are related with input data (can be time or other data)

2. **Recipe (The Equation):** The AR model uses a simple equation to make the prediction.  It's basically a weighted average of the past values, plus some influence from the associated data (P):

    ```text
    Predicted Value = (Weight1 * Past Value 1) + (Weight2 * Past Value 2) + (Weight3 * Past Value 3) + ... + (Weight7 x parameter value)
    ```

    * `Past Value 1`, `Past Value 2`, `Past Value 3` are the values from the previous time steps (up to `na` steps back).
    * `Weight1`, `Weight2`, `Weight3`... ...`Weight7` are numbers that the model *learns* from the data. These weights determine how much influence each past value (and parameter) has on the prediction.  A larger weight means that past value is more important.

3. **Cooking (Finding the Weights):**  The tricky part is figuring out the best values for those weights.  This is where the code does some mathematical magic (using matrix operations). It's essentially trying to find the weights that would have *best predicted the past data*.  If a particular set of weights did a good job of predicting the past, it's likely to do a reasonably good job of predicting the future (at least in the short term).

4. **Baking the Results. (Prediction):** Predict the requested values by calculating the equation with the past values, secondary values `P`, and the calculated weigths.

### The Theorem:  Least Squares (in a nutshell)

The mathematical technique used to find the best weights is based on a fundamental concept called **least squares**. Here's the gist:

1. **Error:** For any given set of weights, the model can make predictions for the *past* data (since we already know the actual values). We can then calculate the *error* between the model's predictions and the actual values.

2. **Squaring the Error:** We square these errors (multiply each error by itself).  This has a couple of benefits:
    * It makes all errors positive (so positive and negative errors don't cancel each other out).
    * It gives more weight to larger errors (which we want to minimize more).

3. **Minimizing the Total Squared Error:**  The "least squares" method finds the set of weights that *minimizes the sum of all the squared errors* across the entire past dataset.  It's trying to find the weights that make the model fit the past data as closely as possible.

4. **Matrix Algebra:** The code uses matrix algebra (from the `gonum.org/v1/gonum/mat` library) to perform this minimization efficiently.  The key steps involve:
    * Creating a matrix called `phi` that organizes the past data.
    * Using matrix operations (transpose, multiplication, and solving a system of linear equations) to find the weights (stored in the `th` variable).

### In Summary: What the Code Does Step-by-Step

1. **Input:** Takes a series of data points (time, value pairs) and the number of future values to predict as input.
2. **Data Preparation:**  Organizes the data into a format suitable for the AR model (the `phi` matrix).
3. **Weight Calculation:**  Uses the "least squares" method (with matrix algebra) to find the best weights for the AR model's equation.
4. **Prediction:** Uses the learned weights and the AR equation to predict future values.
5. **Output:**  Returns the predicted values.

### Limitations and Important Considerations

* **Stationarity:** AR models work best when the data is *stationary*. This means that the statistical properties of the data (like the average and variance) don't change over time.  If your data has a strong trend (consistently going up or down) or seasonality (repeating patterns), the AR model might not perform well without some preprocessing (like differencing the data).
* **Short-Term Predictions:** AR models are generally better for short-term predictions.  The further out you try to predict, the less reliable the predictions become.
* **Order (na, nb):** The choice of `na` (and `nb`)—the number of past values to consider—is important.  Too small, and you might miss important patterns.  Too large, and the model might overfit the data (learn the noise instead of the signal).  There are techniques (like AIC and BIC) to help choose the optimal order, but they aren't implemented in this simplified code.
* **External Factors. (P Values)** Uses and additional array `P` of parameters that has influence in the prediction, this improves the approach, and provides versatility.

The code you provided is a basic implementation of an AR model.  Real-world time series forecasting often involves more complex models and techniques, but this gives you a fundamental understanding of the core concepts.

------------

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the BSD 2 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

The package leverages the powerful `gonum/mat` library for efficient matrix operations.
