package matgr.ai.neuralnet.feedforward;

import matgr.ai.neuralnet.Connection;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronMap;
import matgr.ai.neuralnet.NeuronParameters;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.ReadOnlyNeuronMap;
import matgr.ai.neuralnet.activation.ActivationFunction;

public class HiddenNeuronLayer<NeuronT extends Neuron> extends NeuronLayer<NeuronT> {

    final NeuronMap<NeuronT> writableNeurons;

    public final ReadOnlyNeuronMap<NeuronT> neurons;

    public HiddenNeuronLayer(NeuronFactory<NeuronT> neuronFactory) {

        super(neuronFactory);

        this.writableNeurons = new NeuronMap<>();
        this.neurons = new ReadOnlyNeuronMap<>(this.writableNeurons);
    }

    private HiddenNeuronLayer(HiddenNeuronLayer<NeuronT> other) {

        super(other);

        this.writableNeurons = new NeuronMap<>();
        this.neurons = new ReadOnlyNeuronMap<>(writableNeurons);

        for (NeuronState<NeuronT> neuron : other.writableNeurons.values()) {

            NeuronState<NeuronT> neuronClone = neuron.deepClone();
            writableNeurons.addNeuron(neuronClone);
        }
    }

    @Override
    public int neuronCount() {
        return writableNeurons.count();
    }

    @Override
    public Iterable<NeuronT> neurons() {
        return neurons.values();
    }

    @Override
    protected Iterable<NeuronState<NeuronT>> writableNeurons() {
        return writableNeurons.values();
    }

    @Override
    protected HiddenNeuronLayer<NeuronT> deepClone() {
        return new HiddenNeuronLayer<>(this);
    }

    void setNeurons(Iterable<NeuronT> previousLayerNeurons,
                    NeuronLayer<NeuronT> nextLayer,
                    Iterable<NeuronParameters> neuronParameters) {

        if (nextLayer == null) {
            throw new IllegalArgumentException("nextLayer not provided");
        }

        // TODO: could sync this instead of just clearing...
        nextLayer.connectionMap.clear();

        for (NeuronParameters parameters : neuronParameters) {

            ActivationFunction activationFunction = parameters.activationFunction;
            double[] activationFunctionParameters = parameters.activationFunctionParameters;

            long id = writableNeurons.getNextFreeNeuronId();
            NeuronT newNeuron = neuronFactory.createHidden(id, activationFunction, activationFunctionParameters);

            writableNeurons.addNeuron(newNeuron);

            for (NeuronT previousNeuron : previousLayerNeurons) {

                long sourceId = previousNeuron.id;
                long targetId = newNeuron.id;

                NeuronConnectionMap sourceMap = connectionMap.computeIfAbsent(
                        targetId,
                        k -> new NeuronConnectionMap());

                Connection newConnection = new Connection(sourceId, targetId, true, 0.0);
                sourceMap.sourceMap.put(sourceId, newConnection);
            }

            for (NeuronT nextNeuron : nextLayer.neurons()) {

                long sourceId = newNeuron.id;
                long targetId = nextNeuron.id;

                NeuronConnectionMap sourceMap = nextLayer.connectionMap.computeIfAbsent(
                        targetId,
                        k -> new NeuronConnectionMap());

                Connection newConnection = new Connection(sourceId, targetId, true, 0.0);
                sourceMap.sourceMap.put(sourceId, newConnection);
            }
        }
    }
}
