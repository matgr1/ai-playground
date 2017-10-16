package matgr.ai.neuralnet.cyclic;

import matgr.ai.neuralnet.activation.ActivationFunction;

public class DefaultCyclicNeuronFactory implements CyclicNeuronFactory<CyclicNeuron> {

    @Override
    public CyclicNeuron createBias(long id) {
        return CyclicNeuron.bias(id);
    }

    @Override
    public CyclicNeuron createInput(long id) {
        return CyclicNeuron.input(id);
    }

    @Override
    public CyclicNeuron createHidden(long id,
                                     ActivationFunction activationFunction,
                                     double[] activationFunctionParameters) {
        return CyclicNeuron.hidden(id, activationFunction, activationFunctionParameters);
    }

    @Override
    public CyclicNeuron createOutput(long id,
                                     ActivationFunction activationFunction,
                                     double[] activationFunctionParameters) {
        return CyclicNeuron.output(id, activationFunction, activationFunctionParameters);
    }
}
