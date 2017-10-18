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
    protected double computeActivation(double activationInput, double[] parameters) {
        return Math.max(0, activationInput);
    }

    @Override
    protected double computeActivationDerivative(double activationInput, double activationOutput, double[] parameters) {

        if (activationInput <= 0) {
            return 0.0;
        }

        return 1.0;
    }
}
