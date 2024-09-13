package sequential

import (
	"ann/preprocess"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Estructura de la red neuronal
type NeuralNetwork struct {
	InputNeurons  int
	HiddenNeurons int
	OutputNeurons int
	LearningRate  float64
	WeightsIH     [][]float64 // Pesos entre la capa de entrada y la oculta
	WeightsHO     []float64   // Pesos entre la capa oculta y la de salida
	BiasH         []float64   // Sesgo para las neuronas ocultas
	BiasO         float64     // Sesgo para la neurona de salida
}

// Función para crear y inicializar una red neuronal
func NewNeuralNetwork(inputNeurons, hiddenNeurons, outputNeurons int, learningRate float64) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())
	nn := &NeuralNetwork{
		InputNeurons:  inputNeurons,
		HiddenNeurons: hiddenNeurons,
		OutputNeurons: outputNeurons,
		LearningRate:  learningRate,
		WeightsIH:     make([][]float64, hiddenNeurons),
		WeightsHO:     make([]float64, hiddenNeurons),
		BiasH:         make([]float64, hiddenNeurons),
		BiasO:         rand.Float64(),
	}

	// Inicializar los pesos aleatoriamente
	for i := range nn.WeightsIH {
		nn.WeightsIH[i] = make([]float64, inputNeurons)
		for j := range nn.WeightsIH[i] {
			nn.WeightsIH[i][j] = rand.Float64() * 0.1
		}
		nn.WeightsHO[i] = rand.Float64() * 0.1
		nn.BiasH[i] = rand.Float64() * 0.1
	}

	return nn
}

// Función de activación (sigmoide)
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivada de la función sigmoide
func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

// Función para entrenar la red neuronal (gradiente descendente)
func (nn *NeuralNetwork) Train(record preprocess.Record, target float64) {
	// Fase de forward pass
	features := extractFeatures(record)
	hiddenInputs := make([]float64, nn.HiddenNeurons)
	hiddenOutputs := make([]float64, nn.HiddenNeurons)

	// Cálculo de la capa oculta
	for i := 0; i < nn.HiddenNeurons; i++ {
		sum := nn.BiasH[i]
		for j := 0; j < nn.InputNeurons; j++ {
			sum += nn.WeightsIH[i][j] * features[j]
		}
		hiddenInputs[i] = sum
		hiddenOutputs[i] = sigmoid(sum)
	}

	// Cálculo de la salida final
	finalInput := nn.BiasO
	for i := 0; i < nn.HiddenNeurons; i++ {
		finalInput += nn.WeightsHO[i] * hiddenOutputs[i]
	}
	finalOutput := sigmoid(finalInput)

	// Fase de retropropagación del error (backpropagation)
	outputError := target - finalOutput
	gradient := outputError * sigmoidDerivative(finalOutput)

	// Actualizar los pesos de la capa de salida
	for i := 0; i < nn.HiddenNeurons; i++ {
		delta := gradient * hiddenOutputs[i]
		nn.WeightsHO[i] += nn.LearningRate * delta
	}
	nn.BiasO += nn.LearningRate * gradient

	// Calcular el error de la capa oculta y actualizar los pesos de la capa oculta
	hiddenErrors := make([]float64, nn.HiddenNeurons)
	for i := 0; i < nn.HiddenNeurons; i++ {
		hiddenErrors[i] = gradient * nn.WeightsHO[i]
		hiddenGradient := hiddenErrors[i] * sigmoidDerivative(hiddenOutputs[i])

		for j := 0; j < nn.InputNeurons; j++ {
			nn.WeightsIH[i][j] += nn.LearningRate * hiddenGradient * features[j]
		}
		nn.BiasH[i] += nn.LearningRate * hiddenGradient
	}
}

// Función para predecir usando la red neuronal
func (nn *NeuralNetwork) Predict(record preprocess.Record) float64 {
	features := extractFeatures(record)
	hiddenOutputs := make([]float64, nn.HiddenNeurons)

	// Cálculo de la capa oculta
	for i := 0; i < nn.HiddenNeurons; i++ {
		sum := nn.BiasH[i]
		for j := 0; j < nn.InputNeurons; j++ {
			sum += nn.WeightsIH[i][j] * features[j]
		}
		hiddenOutputs[i] = sigmoid(sum)
	}

	// Cálculo de la salida final
	finalInput := nn.BiasO
	for i := 0; i < nn.HiddenNeurons; i++ {
		finalInput += nn.WeightsHO[i] * hiddenOutputs[i]
	}
	finalOutput := sigmoid(finalInput)

	return finalOutput
}

// Función para extraer características numéricas del registro
func extractFeatures(record preprocess.Record) []float64 {
	return []float64{
		float64(record.Age),
		float64(record.Fnlwgt),
		float64(record.EducationNum),
		float64(record.CapitalGain),
		float64(record.CapitalLoss),
		float64(record.HoursPerWeek),
	}
}

// Función para convertir la etiqueta de ingreso a 0 o 1
func convertLabel(income string) float64 {
	if income == ">50K" {
		return 1.0
	}
	return 0.0
}

// Función para probar la red neuronal secuencial
func TestSequentialNN(records []preprocess.Record) {
	// Dividir datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
	numTrain := int(0.8 * float64(len(records)))
	trainData := records[:numTrain]
	testData := records[numTrain:]

	// Crear y entrenar la red neuronal
	fmt.Println("Entrenando Red Neuronal Secuencial...")
	nn := NewNeuralNetwork(6, 10, 1, 0.01) // 6 neuronas de entrada, 10 ocultas, 1 de salida, tasa de aprendizaje 0.01

	start := time.Now()
	epochs := 50 // Número de épocas
	for epoch := 0; epoch < epochs; epoch++ {
		for _, record := range trainData {
			label := convertLabel(record.Income)
			nn.Train(record, label)
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("Tiempo de entrenamiento: %s\n", elapsed)

	// Probar el modelo
	fmt.Println("Probando Red Neuronal Secuencial...")
	correct := 0
	for _, record := range testData {
		output := nn.Predict(record)
		prediction := 0.0
		if output > 0.5 {
			prediction = 1.0
		}
		label := convertLabel(record.Income)
		if prediction == label {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(testData))
	fmt.Printf("Precisión: %.2f%%\n", accuracy*100)
}
