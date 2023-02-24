package ANN;

import java.util.stream.IntStream;

import static ANN.INeuralNetwork.Layer.*;
import static ANN.Matrix.*;


public record NeuralNetwork(Matrix... weights) implements INeuralNetwork {


    public Matrix[] feedforward(final Matrix inputs) {
        //generating input layer output
        var var1 = multiply(this.getWeightsIH(), inputs);
        var1 = add(var1, this.getBiasIH());
        var1 = activateMatrix(var1, activationFunctions.get(HIDDEN_1));

        //generating hidden layer output
        var var2 = multiply(this.getWeightsHH(), var1);
        var2 = add(var2, this.getBiasHH());
        var2 = activateMatrix(var2, activationFunctions.get(HIDDEN_2));

        //generating final output   NEW MATRIX RESULTING FROM THE EVOLVING OUTPUT
        var var3 = multiply(this.getWeightsHO(), var2);
        var3 = add(var3, this.getBiasHO());
        var3 = activateMatrix(var3, activationFunctions.get(OUTPUT));
        return new Matrix[]{var1, var2, var3};
    }

    public Matrix guess(final double[] inputs) {
        return feedforward(new Matrix(inputs))[2];
    }

    @Override
    public Matrix guess(Matrix inputs) {
        return feedforward(inputs)[2];
    }

    public void train(Matrix given, Matrix targets) {
        Matrix[] var1 = feedforward(given);
        var var2 = new Matrix(targets.getMatrix());
        //calculate out error
        var var3 = subtract(var2, var1[2]);
        Matrix[] vars1 = gradientDescend(var3, var1[2], var1[1], OUTPUT);
        this.getWeightsHO().set(add(this.getWeightsHO(), vars1[1]));
        this.getBiasHO().set(add(this.getBiasHO(), vars1[0]));
        //calculate hidden error
        var2 = transpose(this.getWeightsHO());
        var3 = multiply(var2, var3);
        vars1 = gradientDescend(var3, var1[1], var1[0], HIDDEN_2);
        this.getWeightsHH().set(add(this.getWeightsHH(), vars1[1]));
        this.getBiasHH().set(add(this.getBiasHH(), vars1[0]));
        //calculate hidden error2
        var2 = transpose(this.getWeightsHH());
        var3 = multiply(var2, vars1[2]);
        vars1 = gradientDescend(var3, var1[0], given, HIDDEN_1);
        this.getWeightsIH().set(add(this.getWeightsIH(), vars1[1]));
        this.getBiasIH().set(add(this.getBiasIH(), vars1[0]));
    }

    public Matrix[] gradientDescend(final Matrix v, final Matrix h, final Matrix in, Layer layers) {
        //calculate gradient
        var hg = Matrix.derivative(h, activationFunctions.get(layers));
        hg = hg.multiply(v);
        hg = hg.multiply(1e-2);
        //calculate and adjust hidden weights
        final var it = transpose(in);
        final var d_ih = multiply(hg, it);
        return new Matrix[]{hg, d_ih, v};
    }

    @Override
    public String toString() {
        return """
                {
                 "weight input -> hidden": {
                  %s
                 },
                 "bias input -> hidden": {
                  %s
                 },
                 "weight hidden -> hidden": {
                  %s
                 },
                 "bias hidden -> hidden": {
                  %s
                 },
                 "weight hidden -> output": {
                  %s
                 },
                 "bias hidden -> output": {
                  %s
                 }
                }""".formatted(getWeightsIH(), getBiasIH(), getWeightsHH(), getBiasHH(), getWeightsHO(), getBiasHO());
    }

    @Override
    public NeuralNetwork clone() {
        return new NeuralNetwork(
                weights[0].clone(),
                weights[1].clone(),
                weights[2].clone(),
                weights[3].clone(),
                weights[4].clone(),
                weights[5].clone()
        );
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        NeuralNetwork that = (NeuralNetwork) o;
        return IntStream.range(0, this.weights().length).
                allMatch(i -> weights()[i].equals(that.weights()[i]));
    }

}
