package concurrent

import (
	"ann/preprocess"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// Estructura de la red neuronal concurrente
type NeuralNetwork struct {
	InputNeurons  int
	HiddenNeurons int
	OutputNeurons int
	LearningRate  float64
	WeightsIH     [][]float64 // Pesos entre la capa de entrada y la oculta
	WeightsHO     []float64   // Pesos entre la capa oculta y la de salida
	BiasH         []float64   // Sesgo para las neuronas ocultas
	BiasO         float64     // Sesgo para la neurona de salida
	mu            sync.Mutex  // Mutex para evitar condición de carrera
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

// Función para entrenar la red neuronal concurrentemente con optimizaciones
func (nn *NeuralNetwork) Train(records []preprocess.Record, epochs int, workers int) {
	var wg sync.WaitGroup
	recordChan := make(chan preprocess.Record, len(records))
	workerChan := make(chan int, workers) // Limitar el número de workers concurrentes

	for epoch := 0; epoch < epochs; epoch++ {
		gradientsIH := make([][]float64, nn.HiddenNeurons) // Gradientes acumulados para IH
		gradientsHO := make([]float64, nn.HiddenNeurons)   // Gradientes acumulados para HO
		biasGradientsH := make([]float64, nn.HiddenNeurons)
		biasGradientO := 0.0

		for i := range gradientsIH {
			gradientsIH[i] = make([]float64, nn.InputNeurons)
		}

		// Crear los workers para procesar los registros concurrentemente
		for w := 0; w < workers; w++ {
			workerChan <- w
			wg.Add(1)
			go func() {
				defer wg.Done()

				// Gradientes locales en cada goroutine
				localGradientsIH := make([][]float64, nn.HiddenNeurons)
				localGradientsHO := make([]float64, nn.HiddenNeurons)
				localBiasGradientsH := make([]float64, nn.HiddenNeurons)
				localBiasGradientO := 0.0

				for i := range localGradientsIH {
					localGradientsIH[i] = make([]float64, nn.InputNeurons)
				}

				for record := range recordChan {
					features := extractFeatures(record)
					label := convertLabel(record.Income)

					// Forward pass
					hiddenOutputs := make([]float64, nn.HiddenNeurons)
					for i := 0; i < nn.HiddenNeurons; i++ {
						sum := nn.BiasH[i]
						for j := 0; j < nn.InputNeurons; j++ {
							sum += nn.WeightsIH[i][j] * features[j]
						}
						hiddenOutputs[i] = sigmoid(sum)
					}

					finalInput := nn.BiasO
					for i := 0; i < nn.HiddenNeurons; i++ {
						finalInput += nn.WeightsHO[i] * hiddenOutputs[i]
					}
					finalOutput := sigmoid(finalInput)

					// Backpropagation
					outputError := label - finalOutput
					gradient := outputError * sigmoidDerivative(finalOutput)

					// Actualizar gradientes locales
					for i := 0; i < nn.HiddenNeurons; i++ {
						deltaHO := gradient * hiddenOutputs[i]
						localGradientsHO[i] += deltaHO
					}
					localBiasGradientO += gradient

					// Backpropagation para la capa oculta
					for i := 0; i < nn.HiddenNeurons; i++ {
						hiddenError := gradient * nn.WeightsHO[i]
						hiddenGradient := hiddenError * sigmoidDerivative(hiddenOutputs[i])

						for j := 0; j < nn.InputNeurons; j++ {
							localGradientsIH[i][j] += hiddenGradient * features[j]
						}
						localBiasGradientsH[i] += hiddenGradient
					}
				}

				// Aplicar los gradientes locales acumulados a los gradientes globales
				nn.mu.Lock()
				for i := 0; i < nn.HiddenNeurons; i++ {
					gradientsHO[i] += localGradientsHO[i]
					biasGradientsH[i] += localBiasGradientsH[i]
					for j := 0; j < nn.InputNeurons; j++ {
						gradientsIH[i][j] += localGradientsIH[i][j]
					}
				}
				biasGradientO += localBiasGradientO
				nn.mu.Unlock()

				<-workerChan // Liberar el worker para la próxima tarea
			}()
		}

		// Enviar registros a las goroutines
		for _, record := range records {
			recordChan <- record
		}

		// Cerrar el canal y esperar a que los workers terminen
		close(recordChan)
		wg.Wait()

		// Actualizar los pesos y sesgos después de procesar todos los registros
		nn.mu.Lock()
		for i := 0; i < nn.HiddenNeurons; i++ {
			nn.WeightsHO[i] += nn.LearningRate * gradientsHO[i]
			nn.BiasH[i] += nn.LearningRate * biasGradientsH[i]
			for j := 0; j < nn.InputNeurons; j++ {
				nn.WeightsIH[i][j] += nn.LearningRate * gradientsIH[i][j]
			}
		}
		nn.BiasO += nn.LearningRate * biasGradientO
		nn.mu.Unlock()

		// Resetear los canales para la próxima época
		recordChan = make(chan preprocess.Record, len(records))
	}
}

// Función para predecir usando la red neuronal (sin concurrencia)
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

// Función para probar la red neuronal concurrente
func TestConcurrentNN(records []preprocess.Record) {
	// Dividir datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
	numTrain := int(0.8 * float64(len(records)))
	trainData := records[:numTrain]
	testData := records[numTrain:]

	// Crear y entrenar la red neuronal concurrente
	fmt.Println("Entrenando Red Neuronal Concurrente...")
	nn := NewNeuralNetwork(6, 10, 1, 0.01) // 6 neuronas de entrada, 10 ocultas, 1 de salida, tasa de aprendizaje 0.01

	start := time.Now()
	nn.Train(trainData, 50, 4) // Entrenamiento con 50 épocas y 4 workers
	elapsed := time.Since(start)
	fmt.Printf("Tiempo de entrenamiento: %s\n", elapsed)

	// Probar el modelo
	fmt.Println("Probando Red Neuronal Concurrente...")
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
