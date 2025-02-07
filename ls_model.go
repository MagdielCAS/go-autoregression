// Package ar implements Autoregressive (AR) modeling for time series prediction.
// It allows configuring model parameters like the number of past values to consider (lags)
// for both autoregressive (na) and external input (nb) components, as well as the step size
// for future input extrapolation.
package ar

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// LSModelParameters holds the configuration for the Autoregressive model.
type LSModelParameters struct {
	StepSize float64 // StepSize: the historic 'delta Time' in the original data to use.
}

// Predictor struct encapsulates the AR model, it will store the data and params to be used for the prediction.
type LSPredictor struct {
	Data   [][]float64       // Historical data: each row is [data_value, time_value].
	Params LSModelParameters // Model parameters.
}

// NewPredictor creates a new AR model predictor with the given data and parameters.
// It performs basic validation of the parameters.
func NewLSPredictor(data [][]float64, params LSModelParameters) (*LSPredictor, error) {
	if params.StepSize <= 0 {
		return nil, fmt.Errorf("step size must be a positive number, step size: %f", params.StepSize)
	}

	return &LSPredictor{Data: data, Params: params}, nil
}

// Predict performs AR model prediction for the given number of steps in the future.
// It returns the predicted data as a slice of [time, value] pairs or an error if prediction fails.
func (p *LSPredictor) Predict(numToPredict int) ([][]float64, error) {
	timeValues := make([]float64, len(p.Data))
	dataValues := make([]float64, len(p.Data))
	for i, row := range p.Data {
		dataValues[i] = row[0] // 'Y' values (historical data values).
		timeValues[i] = row[1] // 'P' values (historical time values or external input).
	}

	P := timeValues
	Y := dataValues
	Pl := extendTimeValues(timeValues, numToPredict, p.Params.StepSize)

	// Create A matrix
	A := mat.NewDense(len(P), 4, nil)
	for i := 0; i < len(P); i++ {
		A.Set(i, 0, math.Pow(P[i], 2))
		A.Set(i, 1, P[i])
		A.Set(i, 2, 1)
		A.Set(i, 3, math.Cos(P[i]))
	}

	// Create Atest matrix
	Atest := mat.NewDense(len(Pl), 4, nil)
	for i := 0; i < len(Pl); i++ {
		Atest.Set(i, 0, math.Pow(Pl[i], 2))
		Atest.Set(i, 1, Pl[i])
		Atest.Set(i, 2, 1)
		Atest.Set(i, 3, math.Cos(Pl[i]))
	}

	// Calculate theta (th) using pseudo-inverse  (equivalent of np.linalg.pinv)
	At := A.T()
	var ATA mat.Dense
	ATA.Mul(At, A) // A' * A

	var ATAInv mat.Dense
	err := ATAInv.Inverse(&ATA) // (A' * A)^-1
	if err != nil {
		return [][]float64{}, fmt.Errorf("error inverting ATA matrix: %w", err)
	}

	var AtAInvAt mat.Dense
	AtAInvAt.Mul(&ATAInv, At) // (A' * A)^-1 * A'

	// Create a matrix from the vector Y
	YMatrix := mat.NewDense(len(Y), 1, Y)

	var th mat.Dense
	th.Mul(&AtAInvAt, YMatrix) //(A' * A)^-1 * A' * y

	// Calculate y_ap (predicted Y values)

	var yAp mat.Dense
	yAp.Mul(Atest, &th)

	// Create the result matrix
	result := make([][]float64, len(Pl))
	for i := 0; i < len(Pl); i++ {
		result[i] = []float64{Pl[i], yAp.At(i, 0)} //fixed row/col indexing
	}

	return result, nil
}

// --------------------------------------------------
// Example Usage (in a separate `main` package):
// --------------------------------------------------
//package main
//
//import (
//	"encoding/json"
//	"fmt"
//	"log"
//	"os"
//	"path/to/ar" // Replace with the actual path to your 'ar' package.
//)
//
//func main() {
//	// 1. Sample Data (replace with your actual data).
//	//    Each inner slice is [data_value, time_value].
//	data := [][]float64{
//		{1578.0077, 0}, {1581.1876, 5}, {1452.4627, 33},
//		{1449.7326, 58}, {1501.0392, 80}, {1460.4557, 110},
//		{1492.824, 130}, {1422.3826, 155}, {1404.3431, 180},
//		{1480.74, 210}, {1410.3936, 230}, {1612.336, 255},
//		{1729.343, 280}, {1735.5231, 305}, {1632.595, 330},
//		{1648.3143, 355}, {1640.1972, 380}, {1658.7949, 405},
//		{1675.4953, 430}, {1712.2672, 455}, {1623.8666, 480},
//		{1622.154, 505}, {1630.9466, 530}, {1595.8407, 555},
//		{1548.5976, 580}, {1598.6558, 605}, {1624.0902, 630},
//		{1616.8663, 655}, {1661.251, 680}, {2012.605, 705},
//		{1904.3356, 730}, {1760.5438, 755}, {2449.3183, 780},
//		{2417.4744, 805}, {2431.7134, 830}, {2391.2651, 855},
//		{2402.8298, 885}, {2417.0901, 905}, {2403.8137, 930},
//		{2407.1756, 955}, {2363.049, 980}, {2364.4589, 1010},
//		{2368.4206, 1030}, {2338.8434, 1055}, {2369.9809, 1080},
//		{2353.5891, 1105}, {2380.8422, 1130}, {2519.2731, 1155},
//		{2557.5253, 1180}, {2536.3437, 1205}, {2517.6042, 1235},
//		{2543.7378, 1255}, {2355.5603, 1280}, {2347.445, 1305},
//		{2269.8631, 1335}, {2307.6435, 1355}, {2274.5249, 1380},
//		{2319.0633, 1405}, {2251.9456, 1430}, {2273.7241, 1455},
//		{2250.0617, 1480}, {2272.8212, 1505}, {2367.9611, 1530},
//		{2351.8406, 1555}, {2348.4958, 1580}, {2308.7974, 1605},
//		{2290.4632, 1630}, {2303.6924, 1655}, {2218.8104, 1680},
//		{2260.9153, 1705}, {2236.759, 1730}, {2238.0003, 1755},
//		{2222.3537, 1780}, {2288.0802, 1805}, {2240.4641, 1830},
//		{2258.3908, 1855}, {2175.4428, 1880}, {2247.978, 1905},
//		{2234.6417, 1930}, {2232.0709, 1955}, {2216.933, 1980},
//		{2219.6263, 2005}, {2304.114, 2030}, {2230.2487, 2055},
//		{2261.5, 2070},
//	}
//
//	// 2. Define AR Model Parameters.
//	params := ar.LSModelParameters{
//		AutoregressiveLags: 3,   // 'na' -  How many past 'data_value' to consider.
//		ExternalInputLags:  3,   // 'nb' -  How many past 'time_value' to consider.
//		StepSize:           25.0, // 'stepSize' - The interval between 'time_value' samples.
//	}
//
//	// 3. Create a new AR model predictor.
//	predictor, err := ar.NewPredictor(data, params)
//	if err != nil {
//		log.Fatalf("Failed to create predictor: %v", err)
//	}
//
//	// 4. Set the number of prediction steps.
//	numToPredict := 25
//
//	// 5. Perform prediction.
//	predictedData, err := predictor.Predict(numToPredict)
//	if err != nil {
//		log.Fatalf("Prediction failed: %v", err)
//	}
//
//	// 6. Print Results.
//	fmt.Println("Predicted Data:")
//	//jsonParsed, _ := json.MarshalIndent(predictedData, "", "  ")
//	fmt.Println(predictedData)
//
//}
