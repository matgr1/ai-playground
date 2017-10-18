package matgr.ai.neuralnet.activation;

public class IdentityActivationFunction extends ActivationFunction {

    public static final IdentityActivationFunction INSTANCE = new IdentityActivationFunction();

    private IdentityActivationFunction() {
        super("pass through");
    }

    @Override
    public double[] defaultParameters() {
        return new double[0];
    }

    @Override
    protected double computeActivation(double activationInput, double[] parameters) {
        return activationInput;
    }

    @Override
    protected double computeActivationDerivative(double activationInput, double activationOutput, double[] parameters) {
        return 1.0;
    }
}
