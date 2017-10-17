package matgr.ai.neuralnet.feedforward;

import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronState;

public interface LayerActivationFunction {

    <NeuronT extends Neuron> void activate(Iterable<NeuronState<NeuronT>> neurons);

    double computeDerivative(double activationOutput);
}
