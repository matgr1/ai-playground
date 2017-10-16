package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.SizedIterable;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronParameters;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.activation.ActivationFunction;

public class DefaultLayer<NeuronT extends Neuron> extends FullyConnectedLayer<NeuronT> {

    public DefaultLayer(NeuronFactory<NeuronT> neuronFactory) {

        super(neuronFactory);
    }

    protected DefaultLayer(DefaultLayer<NeuronT> other) {

        super(other);
    }

    @Override
    protected DefaultLayer<NeuronT> deepClone() {
        return new DefaultLayer<>(this);
    }

    void setNeurons(SizedIterable<NeuronT> previousLayerNeurons,
                    NeuronLayer<NeuronT> nextLayer,
                    Iterable<NeuronParameters> neuronParameters) {

        if (nextLayer == null) {
            throw new IllegalArgumentException("nextLayer not provided");
        }

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
        nextLayer.connect(neurons());
    }
}
