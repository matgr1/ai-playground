package matgr.ai.neuralnet.activation;

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
}
