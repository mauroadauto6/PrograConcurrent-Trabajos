package preprocess

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

// Estructura para almacenar una instancia del dataset
type Record struct {
	Age           int
	WorkClass     string
	Fnlwgt        int
	Education     string
	EducationNum  int
	MaritalStatus string
	Occupation    string
	Relationship  string
	Race          string
	Sex           string
	CapitalGain   int
	CapitalLoss   int
	HoursPerWeek  int
	NativeCountry string
	Income        string
}

// Cargar los datos del archivo, reemplazar valores faltantes y aumentar el dataset hasta el millón
func LoadAndPreprocess(filePath string, numRecords int) ([]Record, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error al abrir el archivo: %v", err)
	}
	defer file.Close()

	var records []Record
	scanner := bufio.NewScanner(file)

	// Leer línea por línea
	for scanner.Scan() {
		line := scanner.Text()
		record := parseRecord(line)
		if record != nil {
			records = append(records, *record)
		}
	}

	// Generar datos adicionales si es necesario
	if len(records) < numRecords {
		missing := numRecords - len(records)
		records = append(records, generateAdditionalRecords(records, missing)...)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error al leer el archivo: %v", err)
	}

	return records, nil
}

// Parsear una línea de datos en una estructura Record
func parseRecord(line string) *Record {
	fields := strings.Split(line, ", ")
	if len(fields) != 15 {
		return nil
	}

	age, _ := strconv.Atoi(fields[0])
	fnlwgt, _ := strconv.Atoi(fields[2])
	educationNum, _ := strconv.Atoi(fields[4])
	capitalGain, _ := strconv.Atoi(fields[10])
	capitalLoss, _ := strconv.Atoi(fields[11])
	hoursPerWeek, _ := strconv.Atoi(fields[12])

	// Reemplazar valores faltantes (?)
	workClass := replaceMissing(fields[1])
	occupation := replaceMissing(fields[6])
	nativeCountry := replaceMissing(fields[13])

	return &Record{
		Age:           age,
		WorkClass:     workClass,
		Fnlwgt:        fnlwgt,
		Education:     fields[3],
		EducationNum:  educationNum,
		MaritalStatus: fields[5],
		Occupation:    occupation,
		Relationship:  fields[7],
		Race:          fields[8],
		Sex:           fields[9],
		CapitalGain:   capitalGain,
		CapitalLoss:   capitalLoss,
		HoursPerWeek:  hoursPerWeek,
		NativeCountry: nativeCountry,
		Income:        fields[14],
	}
}

// Reemplazar valores faltantes con una categoría común
func replaceMissing(value string) string {
	if value == "?" {
		return "Unknown"
	}
	return value
}

// Generar registros adicionales basados en el dataset existente
func generateAdditionalRecords(existingRecords []Record, num int) []Record {
	var newRecords []Record
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < num; i++ {
		original := existingRecords[rand.Intn(len(existingRecords))]
		newRecord := Record{
			Age:           original.Age + rand.Intn(10) - 5, // Variar la edad un poco
			WorkClass:     original.WorkClass,
			Fnlwgt:        original.Fnlwgt + rand.Intn(5000) - 2500,
			Education:     original.Education,
			EducationNum:  original.EducationNum,
			MaritalStatus: original.MaritalStatus,
			Occupation:    original.Occupation,
			Relationship:  original.Relationship,
			Race:          original.Race,
			Sex:           original.Sex,
			CapitalGain:   original.CapitalGain + rand.Intn(1000) - 500,
			CapitalLoss:   original.CapitalLoss + rand.Intn(500) - 250,
			HoursPerWeek:  original.HoursPerWeek + rand.Intn(10) - 5,
			NativeCountry: original.NativeCountry,
			Income:        original.Income,
		}
		newRecords = append(newRecords, newRecord)
	}
	return newRecords
}

// Función para imprimir los registros (opcional para debug)
func PrintRecords(records []Record) {
	for _, record := range records {
		fmt.Printf("%+v\n", record)
	}
}
