package matgr.ai.neuralnet.cyclic;

import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronType;
import matgr.ai.neuralnet.activation.ActivationFunction;

public class CyclicNeuron extends Neuron {

    public final long id;

    protected CyclicNeuron(NeuronType type,
                           long id,
                           ActivationFunction activationFunction,
                           double... activationFunctionParameters) {

        super(type, activationFunction, activationFunctionParameters);

        this.id = id;
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

    @Override
    protected CyclicNeuron deepClone() {
        return new CyclicNeuron(type, id, getActivationFunction(), getActivationFunctionParameters());
    }
}