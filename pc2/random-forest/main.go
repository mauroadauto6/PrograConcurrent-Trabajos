package main

import (
	"fmt"
	"rf/concurrent"
	"rf/preprocess"
	"rf/sequential"
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

	// **Versión secuencial**
	fmt.Println("\n--- Random Forest Secuencial ---")
	sequential.TestSequentialRandomForest(records)

	// **Versión concurrente**
	fmt.Println("\n--- Random Forest Concurrente ---")
	concurrent.TestConcurrentRandomForest(records)
}
