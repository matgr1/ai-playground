package matgr.ai.neuralnet;

import matgr.ai.neuralnet.activation.ActivationFunction;

public interface NeuronFactory<NeuronT extends Neuron> {

    NeuronT createBias();

    NeuronT createInput();

    NeuronT createHidden(ActivationFunction activationFunction,
                         double[] activationFunctionParameters);

    NeuronT createOutput(ActivationFunction activationFunction,
                         double[] activationFunctionParameters);

}
