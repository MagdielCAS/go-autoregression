# AutoRegression Package

The `ar` package is a simple yet powerful Go library for implementing Autoregressive (AR) modeling for time series prediction. It utilizes past data to predict future values, and is highly configurable with parameters such as the number of past data points to consider and the step size for future predictions.

## Features

- **Autoregressive Modeling**: Utilize past values of a series to predict its future values.
- **Configurable Parameters**: Define the number of past values to consider (`na`), the number of past external input values (`nb`), and the step size (`stepSize`).
- **Matrix Operations**: Efficient computation using the `gonum/mat` package.

## Installation

To use the `ar` package in your project, first install it using `go get`:

```bash
go get github.com/MagdielCAS/go-autoregression
```

## Usage

Here's a quick guide on how to get started with using the AR package:

### Example Code

```go
package main

import (
	"fmt"
	"log"
	"github.com/MagdielCAS/go-autoregression" // Replace with the actual import path for your project
)

func main() {
	// Sample Data: Replace with your actual data
	data := [][]float64{
		{1578.0077, 0}, {1581.1876, 5}, {1452.4627, 33}, // ... Add more data points
	}

	// Define the AR Model Parameters
	params := ar.ModelParameters{
		AutoregressiveLags: 3,    // Consider 3 past data values
		ExternalInputLags:  3,    // Consider 3 past external input values
		StepSize:           25.0, // Interval between 'time_value' samples
	}

	// Create a new AR Predictor
	predictor, err := ar.NewPredictor(data, params)
	if err != nil {
		log.Fatalf("Failed to create predictor: %v", err)
	}

	// Define how many steps to predict into the future
	numToPredict := 25

	// Perform Prediction
	predictedData, err := predictor.Predict(numToPredict)
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}

	// Print Predicted Data
	fmt.Println("Predicted Data:")
	fmt.Println(predictedData)
}
```

### Parameters

- **AutoregressiveLags (`na`)**: Number of past data points to consider for the autoregressive component.
- **ExternalInputLags (`nb`)**: Number of past external input values (e.g., time or other influential data).
- **StepSize**: Time interval or 'delta Time' between data points.

## How It Works

1. **Input**: Provide historical data as a series of [data_value, time_value] pairs.
2. **Model**: The AR method uses past values to predict future ones based on calculated weights.
3. **Prediction**: The model outputs the future values based on the specified number of prediction steps.

## Limitations

- **Stationarity**: Best suited for stationary data without strong trends or seasonality.
- **Short-Term Predictions**: Generally more accurate for short-term predictions.
- **Parameter Sensitivity**: Proper choice of `na`, `nb`, and `stepSize` is necessary for optimal performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

The package leverages the powerful `gonum/mat` library for efficient matrix operations.

Feel free to contribute and improve the package by submitting issues or pull requests!
