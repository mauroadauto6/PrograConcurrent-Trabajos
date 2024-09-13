package sequential

import (
	"fmt"
	"svm/preprocess"
	"time"
)

// Estructura para representar un modelo SVM
type SVM struct {
	Weights []float64
	Bias    float64
	Lambda  float64 // Parámetro de regularización
	LR      float64 // Tasa de aprendizaje
}

// Función para entrenar el modelo SVM secuencial
func TrainSVM(records []preprocess.Record, epochs int, lambda float64, lr float64) *SVM {
	svm := &SVM{
		Weights: make([]float64, 6), // Asumimos 6 características numéricas (edad, fnlwgt, etc.)
		Bias:    0,
		Lambda:  lambda,
		LR:      lr,
	}

	for epoch := 0; epoch < epochs; epoch++ {
		for _, record := range records {
			features := extractFeatures(record)
			label := convertLabel(record.Income)

			// Verificar si el ejemplo actual está mal clasificado
			if label*(dotProduct(svm.Weights, features)+svm.Bias) < 1 {
				// Actualizar los pesos y el sesgo (bias)
				for i := range svm.Weights {
					svm.Weights[i] = (1-lr*lambda)*svm.Weights[i] + lr*label*features[i]
				}
				svm.Bias += lr * label
			} else {
				// Solo aplicar la penalización de regularización
				for i := range svm.Weights {
					svm.Weights[i] *= (1 - lr*lambda)
				}
			}
		}
	}
	return svm
}

// Función para predecir con SVM
func (svm *SVM) Predict(record preprocess.Record) string {
	features := extractFeatures(record)
	score := dotProduct(svm.Weights, features) + svm.Bias

	if score >= 0 {
		return ">50K"
	}
	return "<=50K"
}

// Función para extraer características numéricas de un registro
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

// Función para convertir la etiqueta de ingreso a -1 o 1
func convertLabel(income string) float64 {
	if income == ">50K" {
		return 1.0
	}
	return -1.0
}

// Producto punto entre dos vectores
func dotProduct(vec1, vec2 []float64) float64 {
	result := 0.0
	for i := range vec1 {
		result += vec1[i] * vec2[i]
	}
	return result
}

// Función para probar el SVM secuencial
func TestSequentialSVM(records []preprocess.Record) {
	// Dividir datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
	numTrain := int(0.8 * float64(len(records)))
	trainData := records[:numTrain]
	testData := records[numTrain:]

	// Entrenar SVM
	fmt.Println("Entrenando SVM Secuencial...")
	start := time.Now()
	svm := TrainSVM(trainData, 100, 0.01, 0.001) // 100 épocas, lambda = 0.01, tasa de aprendizaje = 0.001
	elapsed := time.Since(start)
	fmt.Printf("Tiempo de entrenamiento: %s\n", elapsed)

	// Probar el modelo
	fmt.Println("Probando SVM Secuencial...")
	correct := 0
	for _, record := range testData {
		prediction := svm.Predict(record)
		if prediction == record.Income {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(testData))
	fmt.Printf("Precisión: %.2f%%\n", accuracy*100)
}
