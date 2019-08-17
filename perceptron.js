/**
 * Yes, this isn't a perceptron. It doesn't just do binary classification.
 */
export default class {
  /**
     *
     * @param {(x:Number)->Number} activation the activation function
     */
  constructor(inputCount, activation) {
    this.weights = Array.from(Array(inputCount).keys());
    this.weights = this.weights.map(() => Math.random() * 2.0 - 1.0);
    this.bias = Math.random() * 2.0 - 1.0;
    this.activation = activation;
  }

  activate(value) {
    return this.activation(value);
  }

  process(inputs) {
    const self = this;
    let sum = this.bias;
    inputs.forEach((value, i) => {
      // console.log('Input: %d Value: %s', i, value);
      sum += value * self.weights[i];
    });
    return this.activate(sum);
  }

  adjust(inputs, delta, learningRate) {
    const self = this;
    inputs.forEach((value, i) => {
      // console.log('Input: %d Value: %s', i, value);
      self.weights[i] += value * delta * learningRate;
      // console.log('new weight: %d', self.weights[i]);
    });
    this.bias += delta * learningRate;
  }
}
