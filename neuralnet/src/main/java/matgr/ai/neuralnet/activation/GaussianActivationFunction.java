package matgr.ai.neuralnet.activation;

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
    protected double computeActivation(double x, double[] parameters) {
        return Math.exp(-(x * x) / 2);
    }
}
