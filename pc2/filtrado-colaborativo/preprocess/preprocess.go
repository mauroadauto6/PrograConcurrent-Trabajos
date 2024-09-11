package preprocess

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

// Estructura para almacenar una calificación
type Rating struct {
	UserID    int
	MovieID   int
	Rating    float64
	Timestamp int64
}

// Función para cargar y preprocesar el dataset
func LoadData(filePath string) ([]Rating, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var ratings []Rating
	scanner := bufio.NewScanner(file)
	lineCount := 0 // Contador de líneas

	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Split(line, "::")
		if len(fields) != 4 {
			continue
		}

		userID, _ := strconv.Atoi(fields[0])
		movieID, _ := strconv.Atoi(fields[1])
		rating, _ := strconv.ParseFloat(fields[2], 64)
		timestamp, _ := strconv.ParseInt(fields[3], 10, 64)

		ratings = append(ratings, Rating{UserID: userID, MovieID: movieID, Rating: rating, Timestamp: timestamp})
	}
	lineCount++ // Incrementar el contador en cada línea procesada

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return ratings, nil
}

// Función para dividir los datos en entrenamiento y prueba
func SplitData(ratings []Rating, trainRatio float64) ([]Rating, []Rating) {
	trainSize := int(trainRatio * float64(len(ratings)))
	trainSet := ratings[:trainSize]
	testSet := ratings[trainSize:]
	return trainSet, testSet
}
