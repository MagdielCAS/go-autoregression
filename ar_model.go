// Package ar implements Autoregressive (AR) modeling for time series prediction.
// It allows configuring model parameters like the number of past values to consider (lags)
// for both autoregressive (na) and external input (nb) components, as well as the step size
// for future input extrapolation.
package ar

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// ModelParameters holds the configuration for the Autoregressive model.
type ModelParameters struct {
	AutoregressiveLags int     // na: Number of past data points to consider for the autoregressive component.
	ExternalInputLags  int     // nb: Number of past external input values to consider.
	StepSize           float64 // StepSize: the historic 'delta Time' in the original data to use.
}

// Predictor struct encapsulates the AR model, it will store the data and params to be used for the prediction.
type Predictor struct {
	Data   [][]float64     // Historical data: each row is [data_value, time_value].
	Params ModelParameters // Model parameters.
}

// NewPredictor creates a new AR model predictor with the given data and parameters.
// It performs basic validation of the parameters.
func NewPredictor(data [][]float64, params ModelParameters) (*Predictor, error) {
	if params.AutoregressiveLags <= 0 || params.ExternalInputLags < 0 {
		return nil, fmt.Errorf("lags must be positive integers, autoregressive lags: %d, external input lags: %d", params.AutoregressiveLags, params.ExternalInputLags)
	}

	if params.StepSize <= 0 {
		return nil, fmt.Errorf("step size must be a positive number, step size: %f", params.StepSize)
	}

	return &Predictor{Data: data, Params: params}, nil
}

// Predict performs AR model prediction for the given number of steps in the future.
// It returns the predicted data as a slice of [time, value] pairs or an error if prediction fails.
func (p *Predictor) Predict(numToPredict int) ([][]float64, error) {
	na := p.Params.AutoregressiveLags
	nb := p.Params.ExternalInputLags
	stepSize := p.Params.StepSize

	// Ensure m is at least na to have enough history
	m := max(na, nb)

	// Check if we have enough data
	if len(p.Data) <= m {
		return nil, fmt.Errorf("not enough data points for prediction, need at least %d points", m+1)
	}

	// 1. Separate the input and output data from the historical dataset.
	timeValues := make([]float64, len(p.Data))
	dataValues := make([]float64, len(p.Data))
	for i, row := range p.Data {
		dataValues[i] = row[0] // 'Y' values (historical data values).
		timeValues[i] = row[1] // 'P' values (historical time values or external input).
	}

	// 2. Extend historical data with projected future time values, using a linear projection.
	//    These future time values serve as inputs for the prediction.
	pl := extendTimeValues(timeValues, numToPredict, stepSize)

	// 3. Construct the 'phi' matrix, which contains lagged values of both data and time.
	phi := constructPhiMatrix(dataValues, timeValues, na, nb, m)
	if phi == nil {
		return nil, fmt.Errorf("failed to construct phi matrix")
	}

	// 4. Calculate 'theta' (th), coefficients of AR model, use Least Squares to estimate the vector th.
	th, err := calculateTheta(phi, dataValues)
	if err != nil && err != mat.ErrSingular {
		return nil, fmt.Errorf("error calculating theta: %w", err)
	}

	// 5. Perform prediction using the computed 'theta' and the extended time values.
	yAp := performPrediction(dataValues, pl, th, m, na, nb) // yAp stands for "Y Approximate"

	// 6. Combine Pl and yAp into the result
	// Combine the extended time values (pl) and predicted data values (yAp) into the final result.
	result := make([][]float64, len(pl))
	for i := range pl {
		result[i] = []float64{pl[i], yAp[i]}
	}

	return result, err
}

// extendTimeValues extends the time values array with projected future time values, using a linear projection.
func extendTimeValues(timeValues []float64, numToPredict int, stepSize float64) []float64 {
	pl := make([]float64, len(timeValues)+numToPredict)
	copy(pl, timeValues)
	lastTimeValue := timeValues[len(timeValues)-1]

	for i := 0; i < numToPredict; i++ {
		pl[len(timeValues)+i] = lastTimeValue + float64(i+1)*stepSize
	}

	return pl
}

// constructPhiMatrix constructs phi matrix, which contains lagged values of both data and time.
// dataValues: Y
// timeValues: P
func constructPhiMatrix(dataValues []float64, timeValues []float64, na int, nb int, m int) *mat.Dense {
	dim := na + nb + 1
	numRows := len(dataValues) - m // Adjust the number of rows to account for the lag
	if numRows <= 0 {
		return nil
	}
	phi := mat.NewDense(numRows, dim, nil)

	// Fill the phi matrix starting from index m
	for i := 0; i < numRows; i++ {
		row := make([]float64, dim)
		actualIndex := i + m // Actual index in the original data

		// Add -Y values (negative past data values)
		for j := 1; j <= na; j++ {
			if actualIndex-j >= 0 {
				row[j-1] = -dataValues[actualIndex-j]
			}
		}

		// Add P values (past time/external input values)
		for j := 0; j <= nb; j++ {
			if actualIndex-j >= 0 {
				row[na+j] = timeValues[actualIndex-j]
			}
		}

		phi.SetRow(i, row)
	}

	return phi
}

// performPrediction performs the prediction based on theta and the dataValues
func performPrediction(dataValues []float64, pl []float64, th *mat.Dense, m int, na int, nb int) []float64 {
	yAp := make([]float64, len(pl)) // yAp stands for "Y Approximate"

	// Initialize predicted output with historical data for first 'm+1' values
	copy(yAp, dataValues) // Copy initial values from dataValues

	// Start prediction from m+1 to ensure we have enough history
	for i := m + 1; i < len(pl); i++ {
		sum := 0.0

		// Autoregressive part
		for j := 1; j <= na; j++ {
			if i-j >= 0 {
				sum -= yAp[i-j] * th.At(j-1, 0)
			}
		}

		// External input part
		for j := 0; j <= nb; j++ {
			if i-j >= 0 {
				sum += pl[i-j] * th.At(na+j, 0)
			}
		}

		yAp[i] = sum
	}

	return yAp
}

// calculateTheta calculates the 'theta' (th)  coefficients of AR mode.
func calculateTheta(phi *mat.Dense, dataValues []float64) (*mat.Dense, error) {
	rows, cols := phi.Dims()

	// Create Y vector with the correct dimensions (excluding the first m points)
	y := make([]float64, rows)
	copy(y, dataValues[len(dataValues)-rows:])

	// Calculate theta using the normal equation: (phi' * phi) * th = phi' * Y
	phiT := phi.T()
	phiTphi := mat.NewDense(cols, cols, nil)
	phiTphi.Mul(phiT, phi)

	phiTP := mat.NewDense(cols, 1, nil)
	yVec := mat.NewDense(rows, 1, y)
	phiTP.Mul(phiT, yVec)

	phiTphiInv := mat.NewDense(cols, cols, nil)
	err := phiTphiInv.Inverse(phiTphi)
	// if resulting is close to singular it might be imprecise but still computable
	if err != nil && err != mat.ErrSingular {
		return nil, fmt.Errorf("matrix inversion failed: %w", err)
	}

	th := mat.NewDense(cols, 1, nil)
	th.Mul(phiTphiInv, phiTP)

	return th, err
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
//	params := ar.ModelParameters{
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
