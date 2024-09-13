package main

import (
	"ann/concurrent"
	"ann/preprocess"
	"ann/sequential"
	"fmt"
)

func main() {
	// Cargar y preprocesar los datos
	fmt.Println("Cargando y preprocesando datos...")
	records, err := preprocess.LoadAndPreprocess("adult.data", 1000000) // Cargar 1 millón de registros
	if err != nil {
		fmt.Printf("Error al cargar los datos: %v\n", err)
		return
	}

	fmt.Println("Datos cargados correctamente.")

	// **Versión secuencial de Redes Neuronales Artificiales**
	fmt.Println("\n--- Red Neuronal Artificial Secuencial ---")
	sequential.TestSequentialNN(records)

	// **Versión concurrente de Redes Neuronales Artificiales**
	fmt.Println("\n--- Red Neuronal Artificial Concurrente ---")
	concurrent.TestConcurrentNN(records)
}
