package sequential

import (
	"fmt"
	"math/rand"
	"rf/preprocess"
	"time"
)

// Estructura para representar un árbol de decisión (nodo)
type DecisionTree struct {
	SplitFeature int
	Threshold    float64
	Left         *DecisionTree
	Right        *DecisionTree
	Prediction   string
}

// Estructura del modelo Random Forest
type RandomForest struct {
	Trees []*DecisionTree
}

// Función para entrenar el modelo Random Forest
func TrainRandomForest(records []preprocess.Record, numTrees int, maxDepth int) *RandomForest {
	var forest RandomForest
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numTrees; i++ {
		sample := bootstrapSample(records)
		tree := buildTree(sample, maxDepth)
		forest.Trees = append(forest.Trees, tree)
	}

	return &forest
}

// Función para realizar predicciones
func (forest *RandomForest) Predict(record preprocess.Record) string {
	votes := make(map[string]int)

	for _, tree := range forest.Trees {
		prediction := tree.predict(record)
		votes[prediction]++
	}

	// Devolver el voto mayoritario
	var maxVotes int
	var majorityLabel string
	for label, count := range votes {
		if count > maxVotes {
			maxVotes = count
			majorityLabel = label
		}
	}

	return majorityLabel
}

// Función recursiva para construir un árbol de decisión
func buildTree(records []preprocess.Record, depth int) *DecisionTree {
	if depth == 0 || len(records) == 0 {
		return &DecisionTree{Prediction: majorityLabel(records)}
	}

	feature, threshold := chooseBestSplit(records)
	leftRecords, rightRecords := splitRecords(records, feature, threshold)

	leftChild := buildTree(leftRecords, depth-1)
	rightChild := buildTree(rightRecords, depth-1)

	return &DecisionTree{
		SplitFeature: feature,
		Threshold:    threshold,
		Left:         leftChild,
		Right:        rightChild,
	}
}

// Función para hacer predicciones con un árbol de decisión
func (tree *DecisionTree) predict(record preprocess.Record) string {
	if tree.Prediction != "" {
		return tree.Prediction
	}

	// Usar el valor de la característica para determinar el camino en el árbol
	var featureValue float64
	switch tree.SplitFeature {
	case 0:
		featureValue = float64(record.Age)
	case 2:
		featureValue = float64(record.Fnlwgt)
	case 4:
		featureValue = float64(record.EducationNum)
	case 10:
		featureValue = float64(record.CapitalGain)
	case 11:
		featureValue = float64(record.CapitalLoss)
	case 12:
		featureValue = float64(record.HoursPerWeek)
	default:
		featureValue = 0 // Otras características no numéricas
	}

	if featureValue < tree.Threshold {
		return tree.Left.predict(record)
	}
	return tree.Right.predict(record)
}

// Elegir el mejor punto de división basado en ganancia de información
func chooseBestSplit(records []preprocess.Record) (int, float64) {
	// De forma simplificada, elegimos características aleatorias para simular la elección de división
	feature := rand.Intn(6)           // Elegimos entre las características numéricas (0: age, 2: fnlwgt, etc.)
	threshold := rand.Float64() * 100 // Umbral aleatorio

	return feature, threshold
}

// Dividir los registros en dos subconjuntos según la característica y el umbral
func splitRecords(records []preprocess.Record, feature int, threshold float64) ([]preprocess.Record, []preprocess.Record) {
	var left, right []preprocess.Record

	for _, record := range records {
		var featureValue float64
		switch feature {
		case 0:
			featureValue = float64(record.Age)
		case 2:
			featureValue = float64(record.Fnlwgt)
		case 4:
			featureValue = float64(record.EducationNum)
		case 10:
			featureValue = float64(record.CapitalGain)
		case 11:
			featureValue = float64(record.CapitalLoss)
		case 12:
			featureValue = float64(record.HoursPerWeek)
		}

		if featureValue < threshold {
			left = append(left, record)
		} else {
			right = append(right, record)
		}
	}

	return left, right
}

// Función de votación para determinar la etiqueta mayoritaria
func majorityLabel(records []preprocess.Record) string {
	labelCount := make(map[string]int)

	for _, record := range records {
		labelCount[record.Income]++
	}

	var maxCount int
	var majorityLabel string
	for label, count := range labelCount {
		if count > maxCount {
			maxCount = count
			majorityLabel = label
		}
	}

	return majorityLabel
}

// Bootstrap sample: obtener una muestra aleatoria con reemplazo
func bootstrapSample(records []preprocess.Record) []preprocess.Record {
	var sample []preprocess.Record
	for i := 0; i < len(records); i++ {
		sample = append(sample, records[rand.Intn(len(records))])
	}
	return sample
}

// Función para probar el Random Forest secuencial
func TestSequentialRandomForest(records []preprocess.Record) {
	// Dividir datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
	numTrain := int(0.8 * float64(len(records)))
	trainData := records[:numTrain]
	testData := records[numTrain:]

	// Entrenar Random Forest
	fmt.Println("Entrenando Random Forest Secuencial...")
	start := time.Now()
	rf := TrainRandomForest(trainData, 10, 5) // 10 árboles, profundidad máxima 5
	elapsed := time.Since(start)
	fmt.Printf("Tiempo de entrenamiento: %s\n", elapsed)

	// Probar el modelo
	fmt.Println("Probando Random Forest Secuencial...")
	correct := 0
	for _, record := range testData {
		prediction := rf.Predict(record)
		if prediction == record.Income {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(testData))
	fmt.Printf("Precisión: %.2f%%\n", accuracy*100)
}
