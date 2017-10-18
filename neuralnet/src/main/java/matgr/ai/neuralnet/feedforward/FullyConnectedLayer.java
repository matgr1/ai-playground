package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.DefaultSizedIterable;
import matgr.ai.common.SizedIterable;
import matgr.ai.common.SizedSelectIterable;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.activation.ActivationFunction;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class FullyConnectedLayer<NeuronT extends Neuron> extends ActivatableLayer<NeuronT> {

    private final SizedIterable<NeuronT> neurons;
    private final List<IncomingConnections> connections;

    private final List<NeuronState<NeuronT>> writableNeurons;

    protected FullyConnectedLayer(NeuronFactory<NeuronT> neuronFactory,
                                  ActivationFunction activationFunction,
                                  double... activationFunctionParameters) {

        super(neuronFactory, activationFunction, activationFunctionParameters);

        this.writableNeurons = new ArrayList<>();
        this.neurons = new SizedSelectIterable<>(this.writableNeurons, n -> n.neuron);

        this.connections = new ArrayList<>();
    }

    protected FullyConnectedLayer(FullyConnectedLayer<NeuronT> other) {

        super(other);

        this.connections = new ArrayList<>();

        this.writableNeurons = new ArrayList<>();
        this.neurons = new SizedSelectIterable<>(this.writableNeurons, n -> n.neuron);

        for (NeuronState<NeuronT> neuron : other.writableNeurons) {

            NeuronState<NeuronT> neuronClone = neuron.deepClone();
            writableNeurons.add(neuronClone);
        }

        for (IncomingConnections connection : other.connections) {

            IncomingConnections copy = connection.deepClone();
            connections.add(copy);
        }
    }

    @Override
    protected FullyConnectedLayer<NeuronT> deepClone() {
        return new FullyConnectedLayer<>(this);
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

        for (IncomingConnections neuronConnections : connections) {

            for (IncomingConnection neuronConnection : neuronConnections.connections) {

                neuronConnection.weight = getRandomWeight(random);
            }

            neuronConnections.biasWeight = getRandomWeight(random);
        }
    }

    void setNeurons(int count) {

        // TODO: could sync instead of clear...

        writableNeurons.clear();

        for (int i = 0; i < count; i++) {

            NeuronT newNeuron = neuronFactory.createHidden();
            NeuronState<NeuronT> newNeuronState = new NeuronState<>(newNeuron);

            writableNeurons.add(newNeuronState);
        }
    }

    @Override
    void connect(SizedIterable<NeuronT> previousLayerNeurons) {

        // TODO: this just needs to make sure the sizes are correct... so don't really need to clear,
        //       just make sure the size of the connections list matches this layers neuron counts
        //       (probably should really do this in setNeurons), and make sure that each entry in the
        //       connections list has the right number of weights (that number being the size of
        //       previousLayerNeurons...)

        connections.clear();

        for (int i = 0; i < writableNeurons.size(); i++) {

            IncomingConnections neuronConnections = new IncomingConnections();
            connections.add(neuronConnections);

            for (NeuronT ignored : previousLayerNeurons) {

                IncomingConnection neuronConnection = new IncomingConnection();
                neuronConnections.connections.add(neuronConnection);
            }
        }
    }

    @Override
    void activate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons, double bias) {

        Iterator<IncomingConnections> connectionsIterator = connections.iterator();

        for (NeuronState<NeuronT> neuron : writableNeurons) {

            neuron.preSynapse = 0.0;

            IncomingConnections neuronConnections = connectionsIterator.next();
            Iterator<IncomingConnection> neuronConnectionIterator = neuronConnections.connections.iterator();

            for (NeuronState<NeuronT> previousNeuron : previousLayerNeurons) {

                IncomingConnection neuronConnection = neuronConnectionIterator.next();

                neuron.preSynapse += previousNeuron.postSynapse * neuronConnection.weight;

                if (Double.isNaN(neuron.preSynapse)) {
                    // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                    //       return a status code from this function)... it should be given
                    //       "input" and "weight" (so it can decide what to do based on input values being
                    //       infinite/NaN/etc... if it fails, then set this to 0.0
                    neuron.preSynapse = 0.0;
                }
            }

            neuron.preSynapse += (neuronConnections.biasWeight * bias);

            if (Double.isNaN(neuron.preSynapse)) {
                // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                //       return a status code from this function)... it should be given
                //       "neuron.incomingBiasWeight" and "bias" (so it can decide what to do based on input values being
                //       infinite/NaN/etc... if it fails, then set this to 0.0
                neuron.preSynapse = 0.0;
            }

            activateNeuron(neuron);
        }
    }

    @Override
    void resetPostSynapseErrorDerivatives(double value) {

        for (NeuronState<NeuronT> neuron : writableNeurons) {

            neuron.postSynapseErrorDerivative = value;
        }
    }

    @Override
    void backPropagate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons,
                       double bias,
                       double learningRate) {

        Iterator<IncomingConnections> connectionsIterator = connections.iterator();

        for (NeuronState<NeuronT> neuron : writableNeurons) {

            IncomingConnections neuronConnections = connectionsIterator.next();
            Iterator<IncomingConnection> neuronConnectionIterator = neuronConnections.connections.iterator();

            double dE_dOut = neuron.postSynapseErrorDerivative;
            double dOut_dIn = computePreSynapseOutputDerivative(neuron);

            double dE_dIn = dE_dOut * dOut_dIn;

            for (NeuronState<NeuronT> previousNeuron : previousLayerNeurons) {

                double dIn_dW = previousNeuron.postSynapse;
                double dE_dW = dE_dIn * dIn_dW;

                IncomingConnection neuronConnection = neuronConnectionIterator.next();
                double currentWeight = neuronConnection.weight;

                // update incoming connection weight
                neuronConnection.weight = currentWeight - (dE_dW * learningRate);

                // update previous neuron dE/dOut
                // TODO: don't need to compute this on the last pass
                double dIn_dOutPrev = currentWeight;
                double dE_dOutPrev = (dE_dIn * dIn_dOutPrev);
                previousNeuron.postSynapseErrorDerivative += dE_dOutPrev;
            }

            // update incoming bias weight
            double dIn_dW_Bias = bias;
            double dE_dW_Bias = dE_dIn * dIn_dW_Bias;

            double currentWeight_Bias = neuronConnections.biasWeight;
            double newWeight_Bias = currentWeight_Bias - (dE_dW_Bias * learningRate);

            neuronConnections.biasWeight = newWeight_Bias;
        }
    }

    protected static class IncomingConnections {

        public double biasWeight;
        public final List<IncomingConnection> connections;

        public IncomingConnections() {

            biasWeight = 0.0;
            connections = new ArrayList<>();
        }

        private IncomingConnections(IncomingConnections other) {

            biasWeight = other.biasWeight;
            connections = new ArrayList<>();

            for (IncomingConnection connection : other.connections) {

                IncomingConnection connectionCopy = connection.deepClone();
                connections.add(connectionCopy);
            }
        }

        public IncomingConnections deepClone() {
            return new IncomingConnections(this);
        }
    }

    protected static class IncomingConnection {

        public double weight;

        public IncomingConnection() {
            weight = 0.0;
        }

        public IncomingConnection(IncomingConnection other) {
            weight = other.weight;
        }

        public IncomingConnection deepClone() {
            return new IncomingConnection(this);
        }
    }
}
