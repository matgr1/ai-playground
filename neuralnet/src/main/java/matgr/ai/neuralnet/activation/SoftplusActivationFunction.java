package matgr.ai.neuralnet.activation;

public class SoftplusActivationFunction extends ActivationFunction {
    public static final SoftplusActivationFunction instance = new SoftplusActivationFunction();

    private SoftplusActivationFunction() {
        super("Softplus", new ParameterMetadata("Activation Response", 1.0));
    }

    @Override
    protected double computeActivation(double x, double[] parameters) {
        double activationResponse = parameters[0];
        return Math.log(1 + Math.exp(x / activationResponse));
    }
}
