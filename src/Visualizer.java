import ANN.IActivationFunction;
import ANN.INeuralNetwork;
import ANN.Matrix;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.*;

import static ANN.IActivationFunction.INFINITY;
import static ANN.INeuralNetwork.Layer.INPUT;
import static ANN.Matrix.RANDOM_PROVIDER;

public class Visualizer {
    public static final Dimension screen = new Dimension(800,800);
    private final JFrame frame = new JFrame();
    private final Visual panel = new Visual();
    private final INeuralNetwork network;

    private final Layer in,hid1,hid2,out;


    public static void main(String[] args) {
        new Visualizer(INeuralNetwork.createNetwork(Settings.NUM_INPUTS,Settings.NUM_HIDDEN,Settings.NUM_HIDDEN,Settings.NUM_OUTPUTS,true)).run();
    }

    public Visualizer(INeuralNetwork network){
        this.network = network;
        this.in = new Layer(0,Settings.NUM_INPUTS, IActivationFunction.type.IDENTITY);
        this.hid1 = new Layer(1,Settings.NUM_HIDDEN,INeuralNetwork.activationFunctions.get(INeuralNetwork.Layer.HIDDEN_1));
        this.hid2 = new Layer(2,Settings.NUM_HIDDEN,INeuralNetwork.activationFunctions.get(INeuralNetwork.Layer.HIDDEN_2));
        this.out = new Layer(3,Settings.NUM_OUTPUTS,INeuralNetwork.activationFunctions.get(INeuralNetwork.Layer.OUTPUT));
        this.panel.fw.put(in,new double[Settings.NUM_INPUTS]);
        this.panel.fw.put(hid1,new double[Settings.NUM_HIDDEN]);
        this.panel.fw.put(hid2,new double[Settings.NUM_HIDDEN]);
        this.panel.fw.put(out,new double[Settings.NUM_OUTPUTS]);
        this.init();
    }

    private void init(){
        frame.setSize(screen);
        panel.setPreferredSize(screen);
        panel.setBackground(Color.BLACK);
        frame.add(panel);
        frame.pack();
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
    private void run() {

        ScheduledExecutorService service = Executors.newSingleThreadScheduledExecutor();
        service.scheduleWithFixedDelay(()-> {
            double[] input = new double[Settings.NUM_INPUTS];
            for (int i = 0; i< Settings.NUM_INPUTS; i++)
                input[i] = RANDOM_PROVIDER.nextDouble(-INFINITY,INFINITY);
            Matrix[] fw = network.feedforward(new Matrix(input));
            this.update(input,fw);
            panel.repaint();
        },300,180, TimeUnit.MILLISECONDS);
    }


    public void update(double[] input, Matrix[] fw){
        panel.fw.replace(in,input);
        double[] wh1 = Arrays.stream(fw[0].getMatrix()).mapToDouble(doubles -> doubles[0]).toArray();
        panel.fw.replace(hid1,wh1);
        double[] wh2 = Arrays.stream(fw[1].getMatrix()).mapToDouble(doubles -> doubles[0]).toArray();
        panel.fw.replace(hid2,wh2);
        double[] wh3 = Arrays.stream(fw[2].getMatrix()).mapToDouble(doubles -> doubles[0]).toArray();
        panel.fw.replace(out,wh3);
    }

    public class Visual extends JPanel {

        Map<Layer,double[]> fw = new ConcurrentHashMap<>();

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;

            drawWeights(g2,network.getWeightsIH().getMatrix(),in,hid1);
            drawWeights(g2,network.getWeightsHH().getMatrix(),hid1,hid2);
            drawWeights(g2,network.getWeightsHO().getMatrix(),hid2,out);

            in.drawLayer( g2, fw.get(in));
            hid1.drawLayer( g2,fw.get(hid1));
            hid2.drawLayer( g2, fw.get(hid2));
            out.drawLayer( g2, fw.get(out));
        }

        public void drawWeights(Graphics2D g, double[][] w,Layer in, Layer out) {
            in.drawWeights(g,w,out);
        }
    }
}
