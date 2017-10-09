package matgr.ai.neuralnet.activation;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class GaussianActivationFunction extends ActivationFunction {

    public static final GaussianActivationFunction INSTANCE = new GaussianActivationFunction();

    private GaussianActivationFunction() {
        super("gaussian");
    }

    @Override
    public double[] defaultParameters() {
        return new double[0];
    }

    @Override
    protected double computeActivation(double activationInput, double[] parameters) {
        return Math.exp(-(activationInput * activationInput) / 2);
    }

    @Override
    protected double computeActivationDerivativeFromOutput(double activationOutput, double[] parameters) {
        throw new NotImplementedException();
    }
}
