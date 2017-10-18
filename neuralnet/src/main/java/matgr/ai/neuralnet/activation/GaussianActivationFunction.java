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
    protected double computeActivation(double activationInput, double[] parameters) {
        return Math.exp(-(activationInput * activationInput) / 2);
    }

    @Override
    protected double computeActivationDerivative(double activationInput, double activationOutput, double[] parameters) {
        return -2.0 * activationInput * activationOutput;
    }
}
