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
    protected double computeActivation(double activationInput, double[] parameters) {
        return Math.tanh(activationInput);
    }

    @Override
    protected double computeActivationDerivative(double activationInput, double activationOutput, double[] parameters) {
        return 1.0 - (activationOutput * activationOutput);
    }
}
