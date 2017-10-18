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
    protected double computeActivation(double activationInput, double[] parameters) {
        return MathFunctions.sigmoid(activationInput);
    }

    @Override
    protected double computeActivationDerivative(double activationInput, double activationOutput, double[] parameters) {
        return activationOutput * (1 - activationOutput);
    }
}
