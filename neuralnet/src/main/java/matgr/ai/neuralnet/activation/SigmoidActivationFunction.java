package matgr.ai.neuralnet.activation;

import matgr.ai.math.MathFunctions;

public class SigmoidActivationFunction extends ActivationFunction {
    public static final SigmoidActivationFunction instance = new SigmoidActivationFunction();

    private SigmoidActivationFunction() {
        super("Sigmoid", new ParameterMetadata("Activation Response", 1.0));
    }

    @Override
    protected double computeActivation(double x, double[] parameters) {
        double activationResponse = parameters[0];
        return MathFunctions.sigmoid(x / activationResponse);
    }
}
