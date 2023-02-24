package ANN;

import java.util.Map;

import static ANN.IActivationFunction.type.*;
import static ANN.INeuralNetwork.Layer.*;

public interface INeuralNetwork extends Cloneable {


    //TODO:move this to instance field and mutate it in crossover
    Map<Layer, IActivationFunction.type> activationFunctions = Map.of(HIDDEN_1,ARCTAN,HIDDEN_2,TANH,OUTPUT,SIGMOID);


    static INeuralNetwork createNetwork(int in_nodes, int hid_nodes1, int hid_nodes2, int out_nodes, boolean random) {
        Matrix w1, w2, w3, b1, b2, b3;
        w1 = new Matrix(hid_nodes1, in_nodes);
        w2 = new Matrix(hid_nodes2, hid_nodes1);
        w3 = new Matrix(out_nodes, hid_nodes2);
        b1 = new Matrix(hid_nodes1, 1);
        b2 = new Matrix(hid_nodes2, 1);
        b3 = new Matrix(out_nodes, 1);
        if (random) {
            w1 = Matrix.randomize(w1);
            w2 = Matrix.randomize(w2);
            w3 = Matrix.randomize(w3);
            b1 = Matrix.randomize(b1);
            b2 = Matrix.randomize(b2);
            b3 = Matrix.randomize(b3);
        }
        return new NeuralNetwork(w1, b1, w2, b2, w3, b3);
    }


    Matrix[] weights();

    Matrix[] feedforward(Matrix input);
    Matrix guess(Matrix inputs);

    void train(Matrix given, Matrix targets);

    default void train( Data data) {
        train(data.inputM(), data.outputM());
    }

    default Matrix guess(double[] inputs) {
        return guess(new Matrix(inputs));
    }

    default Matrix getWeightsIH() {
        return weights()[0];
    }

    default Matrix getBiasIH() {
        return weights()[1];
    }

    default Matrix getWeightsHH() {
        return weights()[2];
    }

    default Matrix getBiasHH() {
        return weights()[3];
    }

    default Matrix getWeightsHO() {
        return weights()[4];
    }

    default Matrix getBiasHO() {
        return weights()[5];
    }


    enum Layer {
        INPUT, HIDDEN_1, HIDDEN_2, OUTPUT
    }

    final record Data(double[] input, double[] output) {
        Matrix inputM() {
            return new Matrix(input);
        }

        Matrix outputM() {
            return new Matrix(output);
        }
    }
}
