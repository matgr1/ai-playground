package matgr.ai.neuralnet.cyclic;

import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronType;
import matgr.ai.neuralnet.activation.ActivationFunction;

public class CyclicNeuron extends Neuron {

    public final long id;

    private ActivationFunction activationFunction;
    private double[] activationFunctionParameters;

    protected CyclicNeuron(NeuronType type,
                           long id,
                           ActivationFunction activationFunction,
                           double... activationFunctionParameters) {

        super(type);

        this.id = id;

        if (canActivate()) {
            setActivationFunction(activationFunction, activationFunctionParameters);
        }
    }

    public static CyclicNeuron bias(long id) {
        return new CyclicNeuron(NeuronType.Bias, id, null);
    }

    public static CyclicNeuron input(long id) {
        return new CyclicNeuron(NeuronType.Input, id, null);
    }

    public static CyclicNeuron hidden(
            long id,
            ActivationFunction activationFunction,
            double... activationFunctionParameters) {

        return new CyclicNeuron(
                NeuronType.Hidden,
                id,
                activationFunction,
                activationFunctionParameters);
    }

    public static CyclicNeuron output(
            long id,
            ActivationFunction activationFunction,
            double... activationFunctionParameters) {

        return new CyclicNeuron(
                NeuronType.Output,
                id,
                activationFunction,
                activationFunctionParameters);
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

    public boolean canActivate() {
        switch (type) {
            case Hidden:
            case Output:
                return true;
            default:
                return false;
        }
    }

    @Override
    protected CyclicNeuron deepClone() {
        return new CyclicNeuron(type, id, getActivationFunction(), getActivationFunctionParameters());
    }
}