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
    protected double computeActivation(double x, double[] parameters) {
        return Math.max(0, x);
    }

    @Override
    protected double computeActivationInverse(double x, double[] parameters) {
        throw new NotImplementedException();
    }

    @Override
    protected double computeActivationDerivative(double x, double[] parameters) {
        throw new NotImplementedException();
    }
}
