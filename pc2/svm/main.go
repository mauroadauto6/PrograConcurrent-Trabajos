package main

import (
	"fmt"
	"svm/concurrent"
	"svm/preprocess"
	"svm/sequential"
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

	// **Versión secuencial de SVM**
	fmt.Println("\n--- SVM Secuencial ---")
	sequential.TestSequentialSVM(records)

	// **Versión concurrente de SVM**
	fmt.Println("\n--- SVM Concurrente ---")
	concurrent.TestConcurrentSVM(records)
}
