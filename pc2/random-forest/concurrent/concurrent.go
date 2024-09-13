package concurrent

import (
	"fmt"
	"math/rand"
	"rf/preprocess"
	"sync"
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

// Estructura del modelo Random Forest concurrente
type RandomForest struct {
	Trees []*DecisionTree
	mu    sync.Mutex // Mutex para evitar condición de carrera
}

// Función para entrenar el modelo Random Forest concurrentemente con pool de workers
func TrainRandomForest(records []preprocess.Record, numTrees int, maxDepth int) *RandomForest {
	var forest RandomForest
	rand.Seed(time.Now().UnixNano())

	// Canal para recibir los árboles construidos en paralelo
	treeChan := make(chan *DecisionTree, numTrees)
	workerChan := make(chan int, 4) // Limitar el número de workers concurrentes (pool de 4)
	var wg sync.WaitGroup

	// Iniciar el entrenamiento de los árboles concurrentemente
	for i := 0; i < numTrees; i++ {
		workerChan <- i // Limitar a 4 goroutines concurrentes
		wg.Add(1)
		go func() {
			defer wg.Done()
			sample := bootstrapSample(records)
			tree := buildTree(sample, maxDepth)
			treeChan <- tree
			<-workerChan // Liberar espacio en el worker pool
		}()
	}

	// Goroutine para cerrar el canal cuando todos los árboles estén listos
	go func() {
		wg.Wait()
		close(treeChan)
	}()

	// Recibir los árboles construidos
	for tree := range treeChan {
		forest.mu.Lock()
		forest.Trees = append(forest.Trees, tree)
		forest.mu.Unlock() // Evitar condición de carrera
	}

	return &forest
}

// Función para realizar predicciones de manera concurrente
func (forest *RandomForest) Predict(record preprocess.Record) string {
	votes := make(map[string]int)
	var mu sync.Mutex

	var wg sync.WaitGroup
	for _, tree := range forest.Trees {
		wg.Add(1)
		go func(tree *DecisionTree) {
			defer wg.Done()
			prediction := tree.predict(record)
			mu.Lock()
			votes[prediction]++
			mu.Unlock() // Proteger el acceso concurrente al mapa de votos
		}(tree)
	}

	wg.Wait()

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

// Función recursiva para construir un árbol de decisión (igual que en la versión secuencial)
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
		featureValue = 0
	}

	if featureValue < tree.Threshold {
		return tree.Left.predict(record)
	}
	return tree.Right.predict(record)
}

// Elegir el mejor punto de división basado en los datos reales
func chooseBestSplit(records []preprocess.Record) (int, float64) {
	bestFeature := rand.Intn(6) // Elegimos características numéricas
	bestThreshold := 0.0

	// Para simplificar, elegimos el valor promedio como umbral para la característica seleccionada
	sum := 0.0
	for _, record := range records {
		var featureValue float64
		switch bestFeature {
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
		sum += featureValue
	}
	bestThreshold = sum / float64(len(records)) // Promedio

	return bestFeature, bestThreshold
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

// Función para probar el Random Forest concurrente
func TestConcurrentRandomForest(records []preprocess.Record) {
	// Dividir datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
	numTrain := int(0.8 * float64(len(records)))
	trainData := records[:numTrain]
	testData := records[numTrain:]

	// Entrenar Random Forest concurrente
	fmt.Println("Entrenando Random Forest Concurrente...")
	start := time.Now()
	rf := TrainRandomForest(trainData, 10, 5) // 10 árboles, profundidad máxima 5
	elapsed := time.Since(start)
	fmt.Printf("Tiempo de entrenamiento: %s\n", elapsed)

	// Probar el modelo
	fmt.Println("Probando Random Forest Concurrente...")
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
