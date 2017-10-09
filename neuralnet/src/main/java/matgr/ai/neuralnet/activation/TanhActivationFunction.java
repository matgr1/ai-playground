package matgr.ai.neuralnet.activation;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class TanhActivationFunction extends ActivationFunction {

    public static final TanhActivationFunction INSTANCE = new TanhActivationFunction();

    private TanhActivationFunction() {
        super("tanh");
    }

    @Override
    public double[] defaultParameters() {
        return new double[0];
    }

    @Override
    protected double computeActivation(double activationInput, double[] parameters) {
        return Math.tanh(activationInput);
    }

    @Override
    protected double computeActivationDerivativeFromOutput(double activationOutput, double[] parameters) {
        throw new NotImplementedException();
    }
}
