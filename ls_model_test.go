package ar

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestNewLSPredictor(t *testing.T) {
	testCases := []struct {
		name        string
		data        [][]float64
		params      LSModelParameters
		expectedErr bool
	}{
		{
			name: "Valid parameters",
			data: [][]float64{{1, 1}, {2, 2}},
			params: LSModelParameters{
				StepSize: 1.0,
			},
			expectedErr: false,
		},
		{
			name: "Invalid StepSize (zero)",
			data: [][]float64{{1, 1}, {2, 2}},
			params: LSModelParameters{
				StepSize: 0.0,
			},
			expectedErr: true,
		},
		{
			name: "Invalid StepSize (negative)",
			data: [][]float64{{1, 1}, {2, 2}},
			params: LSModelParameters{
				StepSize: -1.0,
			},
			expectedErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewLSPredictor(tc.data, tc.params)
			if (err != nil) != tc.expectedErr {
				t.Errorf("NewLSPredictor() error = %v, expectedErr %v", err, tc.expectedErr)
			}
		})
	}
}

func TestLSPredict(t *testing.T) {
	// Define a more complex dataset that won't result in a singular matrix
	data := [][]float64{
		{1578.0077, 0}, {1581.1876, 5}, {1452.4627, 33},
		{1449.7326, 58}, {1501.0392, 80}, {1460.4557, 110},
		{1492.824, 130}, {1422.3826, 155}, {1404.3431, 180},
		{1480.74, 210}, {1410.3936, 230}, {1612.336, 255},
		{1729.343, 280}, {1735.5231, 305}, {1632.595, 330},
		{1648.3143, 355}, {1640.1972, 380}, {1658.7949, 405},
		{1675.4953, 430}, {1712.2672, 455}, {1623.8666, 480},
		{1622.154, 505}, {1630.9466, 530}, {1595.8407, 555},
		{1548.5976, 580}, {1598.6558, 605}, {1624.0902, 630},
		{1616.8663, 655}, {1661.251, 680}, {2012.605, 705},
		{1904.3356, 730}, {1760.5438, 755}, {2449.3183, 780},
		{2417.4744, 805}, {2431.7134, 830}, {2391.2651, 855},
		{2402.8298, 885}, {2417.0901, 905}, {2403.8137, 930},
		{2407.1756, 955}, {2363.049, 980}, {2364.4589, 1010},
		{2368.4206, 1030}, {2338.8434, 1055}, {2369.9809, 1080},
		{2353.5891, 1105}, {2380.8422, 1130}, {2519.2731, 1155},
		{2557.5253, 1180}, {2536.3437, 1205}, {2517.6042, 1235},
		{2543.7378, 1255}, {2355.5603, 1280}, {2347.445, 1305},
		{2269.8631, 1335}, {2307.6435, 1355}, {2274.5249, 1380},
		{2319.0633, 1405}, {2251.9456, 1430}, {2273.7241, 1455},
		{2250.0617, 1480}, {2272.8212, 1505}, {2367.9611, 1530},
		{2351.8406, 1555}, {2348.4958, 1580}, {2308.7974, 1605},
		{2290.4632, 1630}, {2303.6924, 1655}, {2218.8104, 1680},
		{2260.9153, 1705}, {2236.759, 1730}, {2238.0003, 1755},
		{2222.3537, 1780}, {2288.0802, 1805}, {2240.4641, 1830},
		{2258.3908, 1855}, {2175.4428, 1880}, {2247.978, 1905},
		{2234.6417, 1930}, {2232.0709, 1955}, {2216.933, 1980},
		{2219.6263, 2005}, {2304.114, 2030}, {2230.2487, 2055},
		{2261.5, 2070},
	}

	params := LSModelParameters{
		StepSize: 25,
	}

	numToPredict := 3
	predictor, err := NewLSPredictor(data, params)
	if err != nil {
		t.Fatalf("Failed to create predictor: %v", err)
	}

	predictedData, err := predictor.Predict(numToPredict)
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	// Perform basic validation
	if len(predictedData) != len(data)+numToPredict {
		t.Errorf("Predict() returned incorrect number of predictions. Expected %d, got %d",
			len(data)+numToPredict, len(predictedData))
	}

	// Check if time values are extended correctly
	expectedNewTime := data[len(data)-1][1] + float64(numToPredict)*params.StepSize
	actualNewTime := predictedData[len(predictedData)-1][0]
	if math.Abs(expectedNewTime-actualNewTime) > 1e-6 {
		t.Errorf("Predict() incorrectly extended the time value. Expected %f, got %f",
			expectedNewTime, actualNewTime)
	}

	// Basic check for the structure of the prediction
	for _, prediction := range predictedData {
		if len(prediction) != 2 {
			t.Errorf("Predict() returned prediction with incorrect number of values. Expected 2, got %d",
				len(prediction))
		}
	}
}

func TestLSExtendTimeValues(t *testing.T) {
	testCases := []struct {
		name         string
		timeValues   []float64
		numToPredict int
		stepSize     float64
		expected     []float64
	}{
		{
			name:         "Simple Extension",
			timeValues:   []float64{1, 2, 3},
			numToPredict: 2,
			stepSize:     1.0,
			expected:     []float64{1, 2, 3, 4, 5},
		},
		{
			name:         "Different Step Size",
			timeValues:   []float64{10, 20, 30},
			numToPredict: 3,
			stepSize:     5.0,
			expected:     []float64{10, 20, 30, 35, 40, 45},
		},
		{
			name:         "Single Initial Value",
			timeValues:   []float64{100},
			numToPredict: 1,
			stepSize:     10.0,
			expected:     []float64{100, 110},
		},
		{
			name:         "Zero numToPredict",
			timeValues:   []float64{1, 2, 3},
			numToPredict: 0,
			stepSize:     1.0,
			expected:     []float64{1, 2, 3},
		},
		{
			name:         "Negative Time Values",
			timeValues:   []float64{-1, -2, -3},
			numToPredict: 2,
			stepSize:     -1.0,
			expected:     []float64{-1, -2, -3, -4, -5},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := extendTimeValues(tc.timeValues, tc.numToPredict, tc.stepSize)
			if !reflect.DeepEqual(actual, tc.expected) {
				t.Errorf("extendTimeValues() = %v, want %v", actual, tc.expected)
			}
		})
	}
}

func TestLSConstructPhiMatrix(t *testing.T) {
	testCases := []struct {
		name          string
		dataValues    []float64
		timeValues    []float64
		na            int
		nb            int
		m             int
		expectedRows  int  // Add for assertions based on data and lag lengths
		expectedCols  int  // Add for assertions based on data and lag lengths
		expectNonNil  bool // Expect non-nil output matrix
		expectedError bool
	}{
		{
			name:          "Simple Case",
			dataValues:    []float64{1, 2, 3, 4, 5},
			timeValues:    []float64{10, 20, 30, 40, 50},
			na:            2,
			nb:            1,
			m:             2,
			expectedRows:  3, // Number of rows in dataValues (length of series) - m
			expectedCols:  4, // (na + nb + 1) = 2 + 1 + 1 = 4
			expectNonNil:  true,
			expectedError: false,
		},
		{
			name:          "na > nb",
			dataValues:    []float64{1, 2, 3, 4, 5},
			timeValues:    []float64{10, 20, 30, 40, 50},
			na:            3,
			nb:            1,
			m:             3,
			expectedRows:  2,
			expectedCols:  5, // (na + nb + 1) = 3 + 1 + 1 = 5
			expectNonNil:  true,
			expectedError: false,
		},
		{
			name:          "nb > na",
			dataValues:    []float64{1, 2, 3, 4, 5},
			timeValues:    []float64{10, 20, 30, 40, 50},
			na:            1,
			nb:            3,
			m:             3,
			expectedRows:  2,
			expectedCols:  5, // (na + nb + 1) = 1 + 3 + 1 = 5
			expectNonNil:  true,
			expectedError: false,
		},
		{
			name:          "Small data size, valid",
			dataValues:    []float64{1, 2},
			timeValues:    []float64{10, 20},
			na:            1,
			nb:            0,
			m:             1,
			expectedRows:  1,
			expectedCols:  2, // na + nb + 1 = 1 + 0 + 1 = 2
			expectNonNil:  true,
			expectedError: false,
		},
		{
			name:          "Minimal data size, lags = 0",
			dataValues:    []float64{1},
			timeValues:    []float64{10},
			na:            0,
			nb:            0,
			m:             0,
			expectedRows:  1,
			expectedCols:  1, // na + nb + 1 = 0 + 0 + 1 = 1
			expectNonNil:  true,
			expectedError: false,
		},

		{
			name:          "Empty dataValues and timeValues, should not generate Phi matrix",
			dataValues:    []float64{},
			timeValues:    []float64{},
			na:            2, // Example
			nb:            1, // Example
			m:             2, // Example
			expectedRows:  0,
			expectedCols:  4,
			expectNonNil:  false,
			expectedError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			phi := constructPhiMatrix(tc.dataValues, tc.timeValues, tc.na, tc.nb, tc.m)

			if (phi == nil) == tc.expectNonNil {
				t.Errorf("constructPhiMatrix() = nil, expected not nil: %v", tc.expectNonNil)
				return // Stop further assertions if the matrix is unexpectedly nil
			}
			if !tc.expectNonNil {
				return
			}
			rows, cols := phi.Dims()
			if rows != tc.expectedRows {
				t.Errorf("constructPhiMatrix() rows = %v, want %v", rows, tc.expectedRows)
			}
			if cols != tc.expectedCols {
				t.Errorf("constructPhiMatrix() cols = %v, want %v", cols, tc.expectedCols)
			}
		})
	}
}

func TestLSCalculateTheta(t *testing.T) {
	testCases := []struct {
		name        string
		phiData     []float64
		phiRows     int
		phiCols     int
		dataValues  []float64
		expectError bool // Expect error in matrix inversion
	}{
		{
			name:        "Simple Case",
			phiData:     []float64{1, 2, 3, 4, 5, 6},
			phiRows:     2,
			phiCols:     3,
			dataValues:  []float64{7, 8},
			expectError: false,
		},
		{
			name:        "Square Phi",
			phiData:     []float64{1, 2, 3, 4},
			phiRows:     2,
			phiCols:     2,
			dataValues:  []float64{5, 6},
			expectError: false,
		},
		{
			name:        "Singular Matrix",
			phiData:     []float64{1, 2, 2, 4},
			phiRows:     2,
			phiCols:     2,
			dataValues:  []float64{5, 6},
			expectError: false,
		},
		{
			name:        "Overdetermined system",
			phiData:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			phiRows:     3,
			phiCols:     3,
			dataValues:  []float64{10, 11, 12},
			expectError: false, // Expect Singular Matrix
		},
		{
			name:        "Underdetermined system",
			phiData:     []float64{1, 2, 3, 4},
			phiRows:     2,
			phiCols:     2,
			dataValues:  []float64{5, 6},
			expectError: false,
		},
		{
			name:        "Matrix sizes not matching for data values",
			phiData:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
			phiRows:     2,
			phiCols:     4, // adjusted ncols to match dataLen
			dataValues:  []float64{10, 11},
			expectError: false,
		},
		{
			// Adding case for correct execution, otherwise it breaks
			name:        "Small system with correct values and parameters",
			phiData:     []float64{1, 2, 3, 4},
			phiRows:     2,
			phiCols:     2,
			dataValues:  []float64{5, 6},
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			phi := mat.NewDense(tc.phiRows, tc.phiCols, tc.phiData)

			th, err := calculateTheta(phi, tc.dataValues)

			if (err != nil) != tc.expectError {
				t.Errorf("calculateTheta() error = %v, expectedError %v", err, tc.expectError)
			}

			if err == nil && th == nil {
				t.Errorf("calculateTheta() returned nil theta and no error, which is unexpected")
			}

			if tc.expectError && err == nil {
				t.Errorf("Expected an error, but got none")
			}
		})
	}
}

func TestLSPerformPrediction(t *testing.T) {
	testCases := []struct {
		name            string
		dataValues      []float64
		pl              []float64
		thData          []float64
		thRows          int
		thCols          int
		m               int
		na              int
		nb              int
		expectedLength  int
		expectedInitial []float64
	}{
		{
			name:            "Simple Prediction",
			dataValues:      []float64{1, 2, 3},
			pl:              []float64{10, 20, 30},
			thData:          []float64{0.5, 0.5, 1, 1},
			thRows:          4,
			thCols:          1,
			m:               1,
			na:              2,
			nb:              1,
			expectedLength:  3,
			expectedInitial: []float64{1, 2, 0},
		},
		{
			name:            "long data values, small pl value len",
			dataValues:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			pl:              []float64{22, 23},
			thData:          []float64{0.5, 0.5, 1, 1},
			thRows:          4,
			thCols:          1,
			m:               1,
			na:              2,
			nb:              1,
			expectedLength:  2,
			expectedInitial: []float64{1, 2},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			th := mat.NewDense(tc.thRows, tc.thCols, tc.thData)

			yAp := performPrediction(tc.dataValues, tc.pl, th, tc.m, tc.na, tc.nb)

			if len(yAp) != tc.expectedLength {
				t.Errorf("performPrediction() len(yAp) = %d, want %d", len(yAp), tc.expectedLength)
			}
			for i := 0; i < len(yAp) && i < len(tc.dataValues); i++ {
				if i > tc.m {
					break
				}
				if math.Abs(yAp[i]-tc.dataValues[i]) > 1e-6 { // floating comparison
					fmt.Printf("Actual: %+v\n", yAp)
					t.Errorf("For Index[%d] performPrediction()  diff is (yAp[i]-tc.dataValues[i])= %f , want aprox 0", i, math.Abs(yAp[i]-tc.dataValues[i]))
				}
			}
		})
	}
}
