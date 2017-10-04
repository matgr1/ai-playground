package matgr.ai.neuralnet.activation;

import matgr.ai.math.MathFunctions;

public class SigmoidActivationFunction extends ActivationFunction {

    public static final SigmoidActivationFunction INSTANCE = new SigmoidActivationFunction();

    private SigmoidActivationFunction() {
        super("sigmoid");
    }

    @Override
    public double[] defaultParameters() {
        return new double[0];
    }

    @Override
    protected double computeActivation(double x, double[] parameters) {
        return MathFunctions.sigmoid(x);
    }
}
