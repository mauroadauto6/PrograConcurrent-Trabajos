package concurrent

import (
	"filtrado/preprocess"
	"math"
	"math/rand"
	"sync"
	"time"
)

// Factores Latentes
const K = 10 // Número de factores latentes
const epochs = 50
const alpha = 0.01  // Tasa de aprendizaje
const lambda = 0.02 // Regularización

// Matrices para los factores
var P [][]float64
var Q [][]float64

// Inicializa las matrices P y Q
func InitializeMatrices(numUsers, numMovies int) {
	rand.Seed(time.Now().UnixNano())
	P = make([][]float64, numUsers+1)
	Q = make([][]float64, numMovies+1)

	for i := range P {
		P[i] = make([]float64, K)
		for k := range P[i] {
			P[i][k] = rand.Float64()
		}
	}

	for i := range Q {
		Q[i] = make([]float64, K)
		for k := range Q[i] {
			Q[i][k] = rand.Float64()
		}
	}
}

// Función concurrente para actualizar P y Q
func update(user, movie int, rating float64, wg *sync.WaitGroup, mu *sync.Mutex) {
	defer wg.Done()

	pred := PredictRating(user, movie)
	err := rating - pred

	mu.Lock() // Adquiere el mutex para asegurar que solo una goroutine modifique las matrices a la vez
	// Actualización de P y Q
	for k := 0; k < K; k++ {
		P[user][k] += alpha * (err*Q[movie][k] - lambda*P[user][k])
		Q[movie][k] += alpha * (err*P[user][k] - lambda*Q[movie][k])
	}
	mu.Unlock() // Libera el mutex después de la actualización
}

// Predice la calificación de un usuario a una película
func PredictRating(user, movie int) float64 {
	var pred float64
	for k := 0; k < K; k++ {
		pred += P[user][k] * Q[movie][k]
	}
	return pred
}

// Entrenamiento concurrente
func TrainConcurrent(ratings []preprocess.Rating, numUsers, numMovies int) {
	InitializeMatrices(numUsers, numMovies)
	var wg sync.WaitGroup
	var mu sync.Mutex

	for epoch := 0; epoch < epochs; epoch++ {
		for _, r := range ratings {
			wg.Add(1)
			go update(r.UserID, r.MovieID, r.Rating, &wg, &mu)
		}
		wg.Wait() // Esperar a que todas las goroutines terminen
	}
}

// Evalúa el modelo concurrente usando el conjunto de prueba
func EvaluateConcurrent(testSet []preprocess.Rating) float64 {
	var mse float64
	for _, r := range testSet {
		user, movie, rating := r.UserID, r.MovieID, r.Rating
		pred := PredictRating(user, movie)
		mse += math.Pow(rating-pred, 2)
	}
	return mse / float64(len(testSet))
}
