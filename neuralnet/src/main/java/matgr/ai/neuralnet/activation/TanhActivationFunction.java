package matgr.ai.neuralnet.activation;

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
}
