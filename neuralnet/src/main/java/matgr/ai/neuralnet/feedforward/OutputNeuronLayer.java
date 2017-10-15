package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.SelectIterable;
import matgr.ai.neuralnet.Connection;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronParameters;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.activation.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public class OutputNeuronLayer<NeuronT extends Neuron> extends NeuronLayer<NeuronT> {

    final List<NeuronState<NeuronT>> writableNeurons;

    public final Iterable<NeuronT> neurons;

    public OutputNeuronLayer(NeuronFactory<NeuronT> neuronFactory) {

        super(neuronFactory);

        this.writableNeurons = new ArrayList<>();
        this.neurons = new SelectIterable<>(this.writableNeurons, n -> n.neuron);
    }

    private OutputNeuronLayer(OutputNeuronLayer<NeuronT> other) {

        super(other);

        this.writableNeurons = new ArrayList<>();
        this.neurons = new SelectIterable<>(this.writableNeurons, n -> n.neuron);

        for (NeuronState<NeuronT> neuron : other.writableNeurons) {

            NeuronState<NeuronT> neuronClone = neuron.deepClone();
            writableNeurons.add(neuronClone);
        }
    }

    @Override
    public int neuronCount() {
        return writableNeurons.size();
    }

    @Override
    public Iterable<NeuronT> neurons() {
        return neurons;
    }

    @Override
    protected Iterable<NeuronState<NeuronT>> writableNeurons() {
        return writableNeurons;
    }

    @Override
    protected OutputNeuronLayer<NeuronT> deepClone() {
        return new OutputNeuronLayer<>(this);
    }

    void setNeurons(Iterable<NeuronT> previousLayerNeurons,
                    Iterable<NeuronParameters> neuronParameters) {

        long id = 0;

        writableNeurons.clear();

        if (neuronParameters != null) {

            for (NeuronParameters parameters : neuronParameters) {

                ActivationFunction activationFunction = parameters.activationFunction;
                double[] activationFunctionParameters = parameters.activationFunctionParameters;

                NeuronT newNeuron = neuronFactory.createHidden(id++, activationFunction, activationFunctionParameters);
                NeuronState<NeuronT> newNeuronState = new NeuronState<>(newNeuron, 0.0, 0.0);

                writableNeurons.add(newNeuronState);

                for (NeuronT previousNeuron : previousLayerNeurons) {

                    long sourceId = previousNeuron.id;
                    long targetId = newNeuron.id;

                    NeuronConnectionMap sourceMap = connectionMap.computeIfAbsent(
                            targetId,
                            k -> new NeuronConnectionMap());

                    Connection newConnection = new Connection(sourceId, targetId, true, 0.0);
                    sourceMap.sourceMap.put(sourceId, newConnection);
                }
            }
        }
    }
}
