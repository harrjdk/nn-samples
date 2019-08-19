import Network from './simpleNetwork/network';

const relu = (x) => {
    if (x > 0) {
        return x;
    }
    return 0;
}

const tanh = (x) => {
    return Math.tanh(x);
}

function nonLin (x) {
    return 3 * x**3 + 4 * x**2 - 5 * x - 6;
}

const TRAINING_ROUNDS = 700;
const TEST_ROUNDS = 300;

const network = new Network(1, [200, 100, 1], [relu, tanh, relu]);
const actual = Array.from(Array(TRAINING_ROUNDS + TEST_ROUNDS).keys()).map(i => nonLin(i));
let predicted = Array.from(Array(TRAINING_ROUNDS + TEST_ROUNDS).keys()).map(() => 0);
let delta = 0;
let percentError = 0;
for (let i = 0; i < TRAINING_ROUNDS; i++) {
    predicted[i] = network.process([i]);
    delta = actual[i] - predicted[i];
    network.adjust(delta, 0.01);
    percentError = Math.abs(actual[i]-predicted[i])/actual[i];
}
console.log('FINAL TRAINING Actual: %f, Predicted: %f, Delta: %f, Percent Error: %f', actual[TRAINING_ROUNDS-1], predicted[TRAINING_ROUNDS-1], delta, percentError);
// Bad variable names, but important for learning.
let MAE = 0; // Mean Absolute Error
let MPE = 0; // Mean Percent Error
let MSE = 0; // Mean Square Error
let MObserved = 0; // Mean of Observed {\bar {y}}, the actual
for (let i = TRAINING_ROUNDS; i < TRAINING_ROUNDS + TEST_ROUNDS; i++) {
    predicted[i] = network.process([i]);
    delta = actual[i] - predicted[i];
    network.adjust(delta, 0.01);
    percentError = Math.abs(actual[i]-predicted[i])/actual[i];
    MPE += percentError
    MSE += (actual[i]-predicted[i])**2
    MAE += Math.abs(actual[i]-predicted[i]);
    MObserved += actual[i];
}
MAE /= TEST_ROUNDS;
MPE /= TEST_ROUNDS;
MSE /= TEST_ROUNDS;

console.log('FINAL TEST Actual: %f, Predicted: %f, Delta: %f, Percent Error: %f', actual[TRAINING_ROUNDS + TEST_ROUNDS-1], predicted[TRAINING_ROUNDS + TEST_ROUNDS-1], delta, percentError);
console.log('MAE: %f', MAE);
console.log('MPE: %f', MPE);
console.log('MSE: %f', MSE);

// Coefficient of Determination, R2
MObserved /= TEST_ROUNDS;
let SStot = 0; // Sum of Squares
let SSres = 0; // Sum of Squares of Residuals
for (let i = TRAINING_ROUNDS; i < TRAINING_ROUNDS + TEST_ROUNDS; i++) {
    SStot += (actual[i] - MObserved)**2;
    SSres += (actual[i] - predicted[i])**2;
}
let R2 = 1.0 - (SSres / SStot); // The closer to 1 the better
console.log('R2: %f', R2);
