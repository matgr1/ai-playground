package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.SizedIterable;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronParameters;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.activation.ActivationFunction;

public class OutputLayer<NeuronT extends Neuron> extends FullyConnectedLayer<NeuronT> {

    public OutputLayer(NeuronFactory<NeuronT> neuronFactory) {

        super(neuronFactory);
    }

    protected OutputLayer(OutputLayer<NeuronT> other) {

        super(other);
    }

    @Override
    protected OutputLayer<NeuronT> deepClone() {
        return new OutputLayer<>(this);
    }

    void setNeurons(SizedIterable<NeuronT> previousLayerNeurons,
                    Iterable<NeuronParameters> neuronParameters) {

        writableNeurons.clear();

        if (neuronParameters != null) {

            for (NeuronParameters parameters : neuronParameters) {

                ActivationFunction activationFunction = parameters.activationFunction;
                double[] activationFunctionParameters = parameters.activationFunctionParameters;

                NeuronT newNeuron = neuronFactory.createHidden(activationFunction, activationFunctionParameters);
                NeuronState<NeuronT> newNeuronState = new NeuronState<>(newNeuron);

                writableNeurons.add(newNeuronState);
            }
        }

        connect(previousLayerNeurons);
    }
}
