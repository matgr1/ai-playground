package matgr.ai.neuralnet;

import matgr.ai.neuralnet.activation.ActivationFunction;

public interface NeuronFactory<NeuronT extends Neuron> {

    NeuronT createBias(long id);

    NeuronT createInput(long id);

    NeuronT createHidden(long id,
                         ActivationFunction activationFunction,
                         double[] activationFunctionParameters);

    NeuronT createOutput(long id,
                         ActivationFunction activationFunction,
                         double[] activationFunctionParameters);

}
