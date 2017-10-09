package matgr.ai.neuralnet.activation;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class SoftplusActivationFunction extends ActivationFunction {

    public static final SoftplusActivationFunction INSTANCE = new SoftplusActivationFunction();

    private SoftplusActivationFunction() {
        super("softplus");
    }

    @Override
    public double[] defaultParameters() {
        return new double[0];
    }

    @Override
    protected double computeActivation(double activationInput, double[] parameters) {
        return Math.log(1 + Math.exp(activationInput));
    }

    @Override
    protected double computeActivationDerivativeFromOutput(double activationOutput, double[] parameters) {
        throw new NotImplementedException();
    }
}
