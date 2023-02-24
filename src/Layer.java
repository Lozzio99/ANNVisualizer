import ANN.IActivationFunction;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;

import static java.awt.Color.WHITE;

public record Layer(int id, int numNodes, IActivationFunction.type type) {
    private static final double startX = 100;
    private static final double startY = 100;
    private static final double r = Settings.NODE_DIAMETER / 2;


    public void drawLayer(Graphics2D g, double[] w) {
        double x = startX + (id * Settings.LAYER_WIDTH);
        double offset = Settings.LAYER_HEIGHT / (this.numNodes - 1);

        g.setColor(WHITE);

        for (double y = startY, i = 0; i < this.numNodes; y += offset, i++) {
            double v = w[(int)i];
            int mapped = (int) map(v, type.getRange().min(), type.getRange().max(), 0, 255);
            g.setColor(new Color(mapped,mapped,mapped));
            var c = new Ellipse2D.Double(x - r, y - r, Settings.NODE_DIAMETER, Settings.NODE_DIAMETER);
            g.fill(c);
            g.setColor(WHITE);
            g.draw(c);
        }

    }

    public void drawWeights(Graphics2D g, double[][] w, Layer out) {
        double x1 = startX + (id * Settings.LAYER_WIDTH);
        double x2 = startX + (out.id * Settings.LAYER_WIDTH);
        double offset1 = Settings.LAYER_HEIGHT / (this.numNodes - 1);
        double offset2 = Settings.LAYER_HEIGHT / (out.numNodes - 1);


        for (double y1 = startY, i = 0; i < this.numNodes; y1 += offset1, i++) {
            for (double y2 = startY, k = 0; k < out.numNodes; y2 += offset2, k++) {

                double v = w[(int)k][(int)i];
                Color c;
                if (v > 0) {
                    int mapped = (int) map(v, 0, 1, 200, 0);
                    c = new Color(255, mapped, mapped);
                } else {
                    int mapped = (int) map(v, -1, 0, 0, 200);
                    c = new Color(mapped, mapped, 255);
                }
                g.setColor(c);
                g.draw(new Line2D.Double(x1, y1, x2, y2));
            }
        }

    }


    public static double map(double x, double inMin, double inMax, double outMin, double outMax) {
        if (x < inMin) return outMin;
        if (x > inMax) return outMax;
        return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
    }

}
