package matgr.ai.neuralnet;

import matgr.ai.neuralnet.activation.ActivationFunction;

public class DefaultNeuronFactory implements NeuronFactory<Neuron> {

    @Override
    public Neuron createBias(long id) {
        return Neuron.bias(id);
    }

    @Override
    public Neuron createInput(long id) {
        return Neuron.input(id);
    }

    @Override
    public Neuron createHidden(long id,
                               ActivationFunction activationFunction,
                               double[] activationFunctionParameters) {
        return Neuron.hidden(id, activationFunction, activationFunctionParameters);
    }

    @Override
    public Neuron createOutput(long id,
                               ActivationFunction activationFunction,
                               double[] activationFunctionParameters) {
        return Neuron.output(id, activationFunction, activationFunctionParameters);
    }
}
