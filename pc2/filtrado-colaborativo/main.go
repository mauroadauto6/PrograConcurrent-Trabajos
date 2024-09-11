package main

import (
	"filtrado/concurrent"
	"filtrado/preprocess"
	"filtrado/sequential"
	"fmt"
	"time"
)

func main() {
	// Cargar los datos
	ratings, err := preprocess.LoadData("ratings.dat")
	if err != nil {
		fmt.Println("Error al cargar los datos:", err)
		return
	}

	// Mostrar cuántas líneas se han cargado
	fmt.Printf("Cantidad de líneas cargadas: %d\n", len(ratings))

	// Dividir los datos en entrenamiento y prueba
	trainSet, testSet := preprocess.SplitData(ratings, 0.8)

	numUsers := 6040
	numMovies := 3952

	// Entrenamiento secuencial
	start := time.Now()
	sequential.TrainSequential(trainSet, numUsers, numMovies)
	duration := time.Since(start)
	fmt.Println("Tiempo de entrenamiento secuencial:", duration)

	// Evaluación del modelo secuencial
	mseSequential := sequential.EvaluateSequential(testSet)
	fmt.Printf("Error cuadrático medio secuencial: %.4f\n", mseSequential)

	// Entrenamiento concurrente
	start = time.Now()
	concurrent.TrainConcurrent(trainSet, numUsers, numMovies)
	duration = time.Since(start)
	fmt.Println("Tiempo de entrenamiento concurrente:", duration)

	// Evaluación del modelo concurrente
	mseConcurrent := concurrent.EvaluateConcurrent(testSet)
	fmt.Printf("Error cuadrático medio concurrente: %.4f\n", mseConcurrent)
}
