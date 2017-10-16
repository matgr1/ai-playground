package matgr.ai.neuralnet.cyclic;

import matgr.ai.neuralnet.activation.ActivationFunction;

public interface CyclicNeuronFactory<NeuronT extends CyclicNeuron> {

    NeuronT createBias(long id);

    NeuronT createInput(long id);

    NeuronT createHidden(long id,
                         ActivationFunction activationFunction,
                         double[] activationFunctionParameters);

    NeuronT createOutput(long id,
                         ActivationFunction activationFunction,
                         double[] activationFunctionParameters);

}
