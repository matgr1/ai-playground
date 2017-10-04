package matgr.ai.neuralnet.activation;

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
    protected double computeActivation(double x, double[] parameters) {
        return Math.log(1 + Math.exp(x));
    }
}
