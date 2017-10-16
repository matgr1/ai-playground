package matgr.ai.neuralnet;

import matgr.ai.neuralnet.activation.ActivationFunction;

public class DefaultNeuronFactory implements NeuronFactory<Neuron> {

    @Override
    public Neuron createBias() {
        return Neuron.bias();
    }

    @Override
    public Neuron createInput() {
        return Neuron.input();
    }

    @Override
    public Neuron createHidden(ActivationFunction activationFunction,
                               double[] activationFunctionParameters) {
        return Neuron.hidden(activationFunction, activationFunctionParameters);
    }

    @Override
    public Neuron createOutput(ActivationFunction activationFunction,
                               double[] activationFunctionParameters) {
        return Neuron.output(activationFunction, activationFunctionParameters);
    }
}
