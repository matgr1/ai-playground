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
    protected double computeActivation(double x, double[] parameters) {
        return Math.tanh(x);
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
