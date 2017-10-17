package matgr.ai.neuralnet;

import matgr.ai.neuralnet.activation.ActivationFunction;

public class Neuron {

    private ActivationFunction activationFunction;
    private double[] activationFunctionParameters;

    public final NeuronType type;

    protected Neuron(NeuronType type,
                     ActivationFunction activationFunction,
                     double... activationFunctionParameters) {

        this.type = type;

        if (canActivate()) {
            setActivationFunction(activationFunction, activationFunctionParameters);
        }
    }

    public static Neuron bias() {
        return new Neuron(NeuronType.Bias, null);
    }

    public static Neuron input() {
        return new Neuron(NeuronType.Input, null);
    }

    public static Neuron hidden(
            ActivationFunction activationFunction,
            double... activationFunctionParameters) {

        return new Neuron(
                NeuronType.Hidden,
                activationFunction,
                activationFunctionParameters);
    }

    public static Neuron output(
            ActivationFunction activationFunction,
            double... activationFunctionParameters) {

        return new Neuron(
                NeuronType.Output,
                activationFunction,
                activationFunctionParameters);
    }

    public static <NeuronT extends Neuron> NeuronT deepClone(NeuronT neuron) {

        @SuppressWarnings("unchecked")
        NeuronT clone = (NeuronT) neuron.deepClone();

        if (clone.getClass() != neuron.getClass()) {
            throw new IllegalArgumentException("Invalid item - clone not overridden correctly in derived class");
        }

        return clone;
    }

    public boolean canActivate() {
        switch (type) {
            case Hidden:
            case Output:
                return true;
            default:
                return false;
        }
    }

    public void setActivationFunction(ActivationFunction activationFunction, double... activationFunctionParameters) {

        if (!canActivate()) {
            throw new IllegalStateException("This neuron type cannot be activated");
        }

        if (null == activationFunction) {
            throw new IllegalArgumentException("activationFunction");
        }

        activationFunction.validateParameters(activationFunctionParameters);

        this.activationFunction = activationFunction;
        this.activationFunctionParameters = activationFunctionParameters;
    }

    public double computeActivation(double x) {

        if (!canActivate()) {
            throw new IllegalStateException("This neuron type cannot be activated");
        }

        return activationFunction.compute(x, activationFunctionParameters);
    }

    public ActivationFunction getActivationFunction() {

        if (!canActivate()) {
            throw new IllegalStateException("This neuron type cannot be activated");
        }

        return activationFunction;
    }

    public double[] getActivationFunctionParameters() {

        if (!canActivate()) {
            throw new IllegalStateException("This neuron type cannot be activated");
        }

        return activationFunctionParameters;
    }

    protected Neuron deepClone() {
        return new Neuron(type, activationFunction, activationFunctionParameters);
    }
}