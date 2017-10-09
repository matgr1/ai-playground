package matgr.ai.neuralnet.activation;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class ReluActivationFunction extends ActivationFunction {

    public static final ReluActivationFunction INSTANCE = new ReluActivationFunction();

    private ReluActivationFunction() {
        super("ReLU");
    }

    @Override
    public double[] defaultParameters() {
        return new double[0];
    }

    @Override
    protected double computeActivation(double activationInput, double[] parameters) {
        return Math.max(0, activationInput);
    }

    @Override
    protected double computeActivationDerivativeFromOutput(double activationOutput, double[] parameters) {
        throw new NotImplementedException();
    }
}
