package ANN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static ANN.Matrix.RANDOM_PROVIDER;
import static java.lang.Double.MAX_VALUE;
import static java.lang.Double.MIN_VALUE;
import static java.lang.Math.*;

public interface IActivationFunction {
    double activate(double x);

    static void main(String[] args) {
        Map<IActivationFunction, Range> ranges = new ConcurrentHashMap<>();

        for (IActivationFunction f : Arrays.stream(type.values()).map(type -> type.f).collect(Collectors.toList())) {
            ranges.put(f, new Range(MAX_VALUE, MIN_VALUE));
        }


        for (int i = 0; i < 1000000; i++) {
            if (i % 100000 == 0) System.out.println(i);
            for (int k = 0; k < 8; k++) {
                CompletableFuture.runAsync(() -> {
                    double v = RANDOM_PROVIDER.nextDouble(-1e8, 1e8);

                    for (IActivationFunction f : new ArrayList<>(ranges.keySet())) {
                        Range existing = ranges.get(f);
                        boolean replaceMin = false;
                        boolean replaceMax = false;
                        double a = f.activate(v);
                        if (existing.min() > a) {
                            replaceMin = true;
                        }
                        if (existing.max() < a) {
                            replaceMax = true;
                        }
                        ranges.replace(f, new Range(replaceMin ? a : existing.min(), replaceMax ? a : existing.max()));
                    }
                });
            }

        }


        for (type t : type.values()) {
            Range range = ranges.get(t.f);
            System.out.printf("FUNCTION : %s   { %g  ->  %g  }\n", t, range.min(), range.max());
        }

    }

    interface Derivative {
        double derivative(double x);
    }

    double INFINITY = 1e2;


    enum type {
        IDENTITY(x -> x, x -> 1, new Range(-INFINITY,INFINITY)),

        SIGMOID(x -> 1 / (1 + exp(-x)), x -> x * (1 - x), new Range(0, 1)),

        TANH(Math::tanh, x -> 1 - (x * x), new Range(-1, 1)),

        ARCTAN(Math::atan, x -> 1 / (x * x + 1), new Range(-(PI / 2), (PI / 2))),
        RELU(x -> max(0, x), x -> x < 0 ? 0 : 1, new Range(0, INFINITY)),

        LEAKY_RELU(x -> max(alpha() * x, x), x -> x < 0 ? alpha() : 1, new Range(alpha() * -INFINITY, INFINITY)),

        ELU(x -> x < 0 ? alpha() * (exp(x) - 1) : x, x -> x < 0 ? x + alpha() : 1, new Range(-alpha(), INFINITY)),

        SOFT_PLUS(x -> log(1 + exp(x)), x -> 1 / (1 + exp(-x)), new Range(0, INFINITY));

        private final IActivationFunction f;
        private final Derivative d;
        private final Range range;


        type(IActivationFunction f, Derivative d, Range range) {
            this.f = f;
            this.d = d;
            this.range = range;
        }

        public static double alpha() {
            return .3;
        }

        public double activate(double x) {
            return this.f.activate(x);
        }

        public double derive(double x) {
            return this.d.derivative(x);
        }

        public char getCode() {
            return switch (this) {
                case ARCTAN -> 'A';
                case ELU -> 'E';
                case RELU -> 'R';
                case TANH -> 'T';
                case SIGMOID -> 'S';
                case IDENTITY -> 'I';
                case SOFT_PLUS -> 'P';
                case LEAKY_RELU -> 'L';
            };
        }

        public Range getRange() {
            return this.range;
        }

        @Override
        public String toString() {
            return switch (this) {
                case IDENTITY -> "Identity";
                case SIGMOID -> "Sigmoid";
                case TANH -> "Tanh";
                case ARCTAN -> "Arctan";
                case RELU -> "Relu";
                case LEAKY_RELU -> "LeakyRelu";
                case ELU -> "Elu";
                case SOFT_PLUS -> "SoftPlus";
            };
        }

        public type fromCode(char code) {
            return switch (code) {
                case 'A' -> ARCTAN;
                case 'E' -> ELU;
                case 'R' -> RELU;
                case 'T' -> TANH;
                case 'S' -> SIGMOID;
                case 'I' -> IDENTITY;
                case 'P' -> SOFT_PLUS;
                case 'L' -> LEAKY_RELU;
                default -> throw new IllegalArgumentException();
            };
        }
    }

    record Range(double min, double max) {

    }

}
