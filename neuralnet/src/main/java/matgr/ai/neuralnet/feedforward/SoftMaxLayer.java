package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.DefaultSizedIterable;
import matgr.ai.common.SizedIterable;
import matgr.ai.common.SizedSelectIterable;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;
import org.apache.commons.collections4.iterators.ArrayIterator;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class SoftMaxLayer<NeuronT extends Neuron> extends NeuronLayer<NeuronT> {

    private final SizedIterable<NeuronT> neurons;
    private final List<NeuronState<NeuronT>> writableNeurons;

    protected SoftMaxLayer(NeuronFactory<NeuronT> neuronFactory) {

        super(neuronFactory);

        this.writableNeurons = new ArrayList<>();
        this.neurons = new SizedSelectIterable<>(this.writableNeurons, n -> n.neuron);
    }

    protected SoftMaxLayer(SoftMaxLayer<NeuronT> other) {

        super(other);

        this.writableNeurons = new ArrayList<>();
        this.neurons = new SizedSelectIterable<>(this.writableNeurons, n -> n.neuron);

        for (NeuronState<NeuronT> neuron : other.writableNeurons) {

            NeuronState<NeuronT> neuronClone = neuron.deepClone();
            writableNeurons.add(neuronClone);
        }
    }

    @Override
    protected SoftMaxLayer<NeuronT> deepClone() {
        return new SoftMaxLayer<>(this);
    }

    @Override
    public int inputCount() {
        return writableNeurons.size();
    }

    @Override
    public int outputCount() {
        return writableNeurons.size();
    }

    @Override
    public SizedIterable<NeuronT> outputNeurons() {
        return neurons;
    }

    @Override
    protected SizedIterable<NeuronState<NeuronT>> outputWritableNeurons() {
        return new DefaultSizedIterable<>(writableNeurons);
    }

    @Override
    void randomizeWeights(RandomGenerator random) {

        // NOTE: nothing to do here... no weights...
    }

    @Override
    void connect(SizedIterable<NeuronT> previousLayerNeurons) {

        // TODO: could sync instead of clear...

        writableNeurons.clear();

        for (int i = 0; i < previousLayerNeurons.size(); i++) {

            NeuronT newNeuron = neuronFactory.createHidden();
            NeuronState<NeuronT> newNeuronState = new NeuronState<>(newNeuron);

            writableNeurons.add(newNeuronState);
        }
    }

    @Override
    void activate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons, double bias) {

        // NOTE: this helps mitigate numerical stability issues... (see here:
        //       https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)

        Iterator<NeuronState<NeuronT>> neurons = writableNeurons.iterator();

        double maxPreSynapse = Double.NEGATIVE_INFINITY;

        for (NeuronState<NeuronT> previousLayerNeuron : previousLayerNeurons) {

            NeuronState<NeuronT> neuron = neurons.next();

            neuron.preSynapse = previousLayerNeuron.postSynapse;
            maxPreSynapse = Math.max(maxPreSynapse, neuron.preSynapse);
        }

        double d = -maxPreSynapse;

        // compute the overall sum (and save the intermediate values)

        double expSum = 0.0;
        double[] exps = new double[writableNeurons.size()];

        int index = 0;

        for (NeuronState<NeuronT> neuron : writableNeurons) {

            double curExp = Math.exp(neuron.preSynapse + d);

            exps[index++] = curExp;
            expSum += curExp;
        }

        // compute the activations

        ArrayIterator<Double> expsIterator = new ArrayIterator<>(exps);

        for (NeuronState<NeuronT> neuron : writableNeurons) {

            double curExp = expsIterator.next();
            neuron.postSynapse = curExp / expSum;
        }
    }

    @Override
    void resetPostSynapseErrorDerivatives(double value) {

        for (NeuronState<NeuronT> neuron : writableNeurons) {

            neuron.postSynapseErrorDerivative = value;
        }
    }

    @Override
    void backPropagate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons, double bias, double learningRate) {

        for (int i = 0; i < writableNeurons.size(); i++) {

            NeuronState<NeuronT> neuron_i = writableNeurons.get(i);

            double dE_dOut_i = neuron_i.postSynapseErrorDerivative;

            for (int j = 0; j < writableNeurons.size(); j++) {

                NeuronState<NeuronT> neuron_j = writableNeurons.get(j);

                // update previous neuron dE/dOut
                // TODO: don't need to compute this on the last pass

                // NOTE: this is softmax derivative for current input/output (see here:
                //       https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
                double dOut_i_dOutPrev_j;

                if (i == j) {

                    dOut_i_dOutPrev_j = neuron_i.postSynapse * (1.0 - neuron_j.postSynapse);

                } else {

                    dOut_i_dOutPrev_j = -neuron_j.postSynapse * neuron_i.postSynapse;
                }

                double dE_dOutPrev_j = dE_dOut_i * dOut_i_dOutPrev_j;

                NeuronState<NeuronT> previousNeuron_j = previousLayerNeurons.get(j);
                previousNeuron_j.postSynapseErrorDerivative += dE_dOutPrev_j;
            }
        }
    }
}
