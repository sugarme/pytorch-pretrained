package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

var task string

func init() {
	flag.StringVar(&task, "task", "", "specify a task, ie., 'convert', 'check'")
}

func main() {
	flag.Parse()
	switch task {
	case "convert":
		convert()
	case "check":
		checkModel("./biobert-v1.1.gt")
	default:
		panic("Unspecified or invalid task. ")
	}
}

// convert converts numpy model weights to `gotch` model weights.
func convert() {
	filepath := "./model.npz"

	namedTensors, err := ts.ReadNpz(filepath)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Num of named tensor: %v\n", len(namedTensors))
	outputFile := "biobert-v1.1.gt"
	err = ts.SaveMultiNew(namedTensors, outputFile)
	if err != nil {
		log.Fatal(err)
	}
}

// checkModel loads model weights from file and prints out tensor names.
func checkModel(file string) {
	vs := nn.NewVarStore(gotch.CPU)
	err := vs.Load(file)

	namedTensors, err := ts.LoadMultiWithDevice(file, vs.Device())
	if err != nil {
		log.Fatal(err)
	}

	// var namedTensorsMap map[string]*ts.Tensor = make(map[string]*ts.Tensor, 0)
	for _, namedTensor := range namedTensors {
		// namedTensorsMap[namedTensor.Name] = namedTensor.Tensor
		fmt.Println(namedTensor.Name)
	}
}
