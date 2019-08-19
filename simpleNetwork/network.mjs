import Perceptron from './perceptron';

export default class {
    constructor(inputCount, layerShape, activationFunctions) {
        this.inputCount = inputCount;
        this.layers = Array(layerShape.length);
        this.lastInputs = Array.from(Array(inputCount).keys()).map(()=>0);
        let previousOutput = inputCount;
        for(let i = 0; i < layerShape.length; i++) {
            // console.log('Creating layer of %d nodes with input shape %d', layerShape[i], previousOutput);
            this.layers[i] = Array.from(Array(layerShape[i]).keys()).map(
                () => new Perceptron(previousOutput, activationFunctions[i])
            );
            previousOutput = this.layers[i].length;
        }
    }

    process(inputs) {
        const self = this;
        let currentInputs = inputs;
        this.layers.forEach((layer, i) => {
            // console.log('Layer %d size %d', i, layer.length);
            self.lastInputs[i]=currentInputs.slice();
            let outputs = Array.from(Array(layer.length).keys()).map(()=>0);
            layer.forEach((node, j) => {
                outputs[j] = node.process(currentInputs);
                // console.log('Output: %f', outputs[j])
            });
            currentInputs = outputs.slice();
        });
        return currentInputs;
    }
    
    adjust(delta, learningRate) {
        const self = this;
        this.layers.forEach((layer, i) => {
            layer.forEach((node, j) => {
               node.adjust(self.lastInputs[i], delta, learningRate);
            });
        });
    }
}