package matgr.ai.neuralnet.cyclic;

import com.google.common.collect.Iterators;
import matgr.ai.math.MathFunctions;
import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.activation.KnownActivationFunctions;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Logger;

public class CyclicNeuralNet<ConnectionT extends Connection, NeuronT extends Neuron> {

    private final static Logger logger;

    static {
        logger = Logger.getLogger(CyclicNeuralNet.class.getName());
    }

    private final NeuronFactory<NeuronT> neuronFactory;
    private final ConnectionFactory<ConnectionT> connectionFactory;

    private final NeuronMap<NeuronT> writableNeurons;
    private final ConnectionMap<ConnectionT> writableConnections;

    private long version;

    public final ReadOnlyNeuronMap<NeuronT> neurons;
    public final ReadOnlyConnectionMap<ConnectionT> connections;

    public CyclicNeuralNet(NeuronFactory<NeuronT> neuronFactory,
                           ConnectionFactory<ConnectionT> connectionFactory,
                           int inputCount,
                           Iterable<OutputNeuronParameters> outputNeuronsParameters) {

        this(neuronFactory, connectionFactory);

        if (null == outputNeuronsParameters) {
            throw new IllegalArgumentException("outputNeuronsParameters not provided");
        }

        NeuronT bias = neuronFactory.createBias(writableNeurons.getNextFreeNeuronId());
        writableNeurons.addNeuron(bias);

        for (int i = 0; i < inputCount; i++) {

            NeuronT inputNeuron = neuronFactory.createInput(writableNeurons.getNextFreeNeuronId());
            writableNeurons.addNeuron(inputNeuron);
        }

        for (OutputNeuronParameters parameters : outputNeuronsParameters) {

            NeuronT outputNeuron = neuronFactory.createOutput(
                    writableNeurons.getNextFreeNeuronId(),
                    parameters.activationFunction,
                    parameters.activationFunctionParameters);

            writableNeurons.addNeuron(outputNeuron);
        }

        this.version = 0;
    }

    protected CyclicNeuralNet(CyclicNeuralNet<ConnectionT, NeuronT> other) {

        this(other.neuronFactory, other.connectionFactory);

        // TODO: can this be relaxed?
        if (other.getClass() != this.getClass()) {
            throw new IllegalArgumentException("Cannot copy graph of a different type");
        }

        for (NeuronState<NeuronT> neuron : other.writableNeurons.values()) {

            NeuronState<NeuronT> neuronClone = neuron.deepClone();
            writableNeurons.addNeuron(neuronClone);
        }

        for (ConnectionT connection : other.writableConnections.values()) {

            ConnectionT connectionClone = Connection.deepClone(connection);
            addConnection(connectionClone);
        }

        this.version = other.version;
    }

    private CyclicNeuralNet(NeuronFactory<NeuronT> neuronFactory,
                            ConnectionFactory<ConnectionT> connectionFactory) {

        if (null == neuronFactory) {
            throw new IllegalArgumentException("neuronFactory not provided");
        }
        if (null == connectionFactory) {
            throw new IllegalArgumentException("connectionFactory not provided");
        }

        this.neuronFactory = neuronFactory;
        this.connectionFactory = connectionFactory;

        this.writableNeurons = new NeuronMap<>();
        this.neurons = new ReadOnlyNeuronMap<>(this.writableNeurons);

        this.writableConnections = new ConnectionMap<>();
        this.connections = new ReadOnlyConnectionMap<>(this.writableConnections);
    }

    public NeuronT biasNeuron() {
        return neurons.getSingle(NeuronType.Bias);
    }

    public long version() {
        return version;
    }

    public static <
            CyclicNeuralNetT extends CyclicNeuralNet<ConnectionT, NeuronT>,
            ConnectionT extends Connection,
            NeuronT extends Neuron>
    CyclicNeuralNetT deepClone(CyclicNeuralNet network) {

        @SuppressWarnings("unchecked")
        CyclicNeuralNetT clone = (CyclicNeuralNetT) network.deepClone();

        if (clone.getClass() != network.getClass()) {
            throw new IllegalArgumentException("Invalid item - clone not overridden correctly in derived class");
        }

        return clone;
    }

    public boolean isConnected(long sourceId, long targetId) {
        return writableConnections.isConnected(sourceId, targetId);
    }

    public NeuronT addHiddenNeuron(ActivationFunction activationFunction, double... activationFunctionParameters) {
        return addHiddenNeuron(null, activationFunction, activationFunctionParameters);
    }

    public NeuronT addHiddenNeuron(long neuronId,
                                   ActivationFunction activationFunction,
                                   double... activationFunctionParameters) {
        return addHiddenNeuron((Long) neuronId, activationFunction, activationFunctionParameters);
    }

    public boolean removeHiddenNeuron(long neuronId) {

        try {

            int incomingCount = writableConnections.getIncomingConnectionCount(neuronId);
            int outgoingCount = writableConnections.getIncomingConnectionCount(neuronId);

            int connectionCount = incomingCount + outgoingCount;

            if (connectionCount > 0) {
                throw new IllegalStateException(
                        String.format("Cannot remove neuron, it is in use by %d connections", connectionCount));
            }

            return writableNeurons.removeNeuron(neuronId);
        } finally {
            version++;
        }
    }

    public ConnectionT addConnection(long sourceNeuronId,
                                     long targetNeuronId,
                                     boolean enabled,
                                     double weight) {

        ConnectionT connection = connectionFactory.createConnection(
                sourceNeuronId,
                targetNeuronId,
                enabled,
                weight);

        addConnection(connection);

        return connection;
    }

    public boolean removeConnection(ConnectionT connection) {

        try {

            return writableConnections.removeConnection(connection);

        } finally {
            version++;
        }
    }

    public List<Double> activateSingle(List<Double> inputSet,
                                       double bias,
                                       int maxStepsPerActivation,
                                       boolean resetStateBeforeActivation) {

        List<List<Double>> inputSets = new ArrayList<>();
        inputSets.add(inputSet);

        return activateSet(inputSets, bias, maxStepsPerActivation, resetStateBeforeActivation);
    }

    public List<Double> activateSet(List<List<Double>> inputSets,
                                    double bias,
                                    int maxStepsPerActivation,
                                    boolean resetStateBeforeActivation) {

        if (inputSets.size() <= 0) {
            throw new IllegalStateException("No input sets provided");
        }

        int inputNeuronCount = writableNeurons.count(NeuronType.Input);
        int outputNeuronCount = writableNeurons.count(NeuronType.Output);

        if (inputNeuronCount <= 0) {
            throw new IllegalStateException("No input neurons in are present in the network");
        }
        if (outputNeuronCount <= 0) {
            throw new IllegalStateException("No output neurons in are present in the network");
        }

        Iterable<NeuronState<NeuronT>> inputNeurons = writableNeurons.values(NeuronType.Input);
        Iterable<NeuronState<NeuronT>> hiddenNeurons = writableNeurons.values(NeuronType.Hidden);
        Iterable<NeuronState<NeuronT>> outputNeurons = writableNeurons.values(NeuronType.Output);

        // initialize state
        if (resetStateBeforeActivation) {

            // TODO: this could maybe be done in one of the other loops?

            for (NeuronState<NeuronT> neuron : writableNeurons.values()) {
                neuron.preSynapse = 0.0;
                neuron.postSynapse = 0.0;
            }
        }

        // pipeline the input sets...
        int numSteps = maxStepsPerActivation + (inputSets.size() - 1);

        // set the bias neuron value
        NeuronState<NeuronT> biasNeuron = writableNeurons.getSingle(NeuronType.Bias);
        biasNeuron.postSynapse = bias;

        boolean completed = false;

        for (int step = 0; step < numSteps; step++) {

            if (step < inputSets.size()) {

                // initialize the current input set
                List<Double> inputs = inputSets.get(step);

                if (inputs.size() != inputNeuronCount) {
                    throw new IllegalArgumentException("Input signal array has an incorrect number of inputs");
                }

                Iterator<Double> inputIterator = inputs.iterator();

                for (NeuronState<NeuronT> neuron : inputNeurons) {

                    double inputValue = inputIterator.next();

                    if (Double.isNaN(inputValue)) {
                        // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                        //       return a status code from this function)...  if it fails, then set this to 0.0
                        inputValue = 0.0;
                    }

                    neuron.postSynapse = inputValue;
                }
            }

            // update all connection sums
            for (Connection connection : connections.values()) {

                if (connection.enabled) {

                    if (connection.weight != 0.0) {

                        NeuronState<NeuronT> sourceNeuron = writableNeurons.get(connection.sourceId);
                        NeuronState<NeuronT> targetNeuron = writableNeurons.get(connection.targetId);

                        targetNeuron.preSynapse += sourceNeuron.postSynapse * connection.weight;

                        if (Double.isNaN(targetNeuron.preSynapse)) {
                            // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                            //       return a status code from this function)... it should be given
                            //       "sourceNeuron.postSynapse" and "connection.weight" (so it can decide what to do
                            //       based on input values being infinite/NaN/etc... if it fails, then set this to 0.0
                            targetNeuron.preSynapse = 0.0;
                        }

                        if (Double.isNaN(targetNeuron.preSynapse)) {
                            throw new IllegalStateException("NaN pre-synapse value");
                        }
                    }
                }
            }

            boolean moreWork = false;

            Iterator<NeuronState<NeuronT>> hiddenAndOutputNeurons =
                    Iterators.concat(hiddenNeurons.iterator(), outputNeurons.iterator());

            // propagate inputs through each hidden and output neuron's activation function
            while (hiddenAndOutputNeurons.hasNext()) {

                NeuronState<NeuronT> neuron = hiddenAndOutputNeurons.next();

                double value = neuron.neuron.computeActivation(neuron.preSynapse);

                if (Double.isNaN(value)) {
                    // NOTE: sigmoid shouldn't produce NaN, so fallback to this one for now...
                    // TODO: pass in some sort of NaN handler (with the ability to completely bail out and return a
                    //       status code from this function)... if it fails, then try this
                    value = KnownActivationFunctions.SIGMOID.compute(neuron.preSynapse);
                }

                if (Double.isNaN(value)) {
                    throw new IllegalStateException("NaN activation output");
                }

                // NOTE: this may only help for very simple networks...
                if (!MathFunctions.fuzzyCompare(value, neuron.postSynapse)) {
                    moreWork = true;
                }

                neuron.postSynapse = value;
                neuron.preSynapse = 0.0;
            }

            if (!moreWork) {
                completed = true;
                break;
            }
        }

        if (!completed) {
            logger.fine("Activation reached max steps without converging");
        }

        // read the outputs from the output neurons
        List<Double> outputs = new ArrayList<>();

        for (NeuronState<NeuronT> neuron : outputNeurons) {
            outputs.add(neuron.postSynapse);
        }

        return outputs;
    }

    protected CyclicNeuralNet<ConnectionT, NeuronT> deepClone() {
        return new CyclicNeuralNet<>(this);
    }

    protected void addConnection(ConnectionT connection) {

        try {

            NeuronState<NeuronT> sourceNeuron = writableNeurons.get(connection.sourceId);
            if (sourceNeuron == null) {
                throw new IllegalArgumentException("Source neuron not found");
            }

            NeuronState<NeuronT> targetNeuron = writableNeurons.get(connection.targetId);
            if (targetNeuron == null) {
                throw new IllegalArgumentException("Target neuron not found");
            }

            if (targetNeuron.neuron.type == NeuronType.Bias) {
                throw new IllegalArgumentException("Cannot target Bias neuron");
            }
            if (targetNeuron.neuron.type == NeuronType.Input) {
                throw new IllegalArgumentException("Cannot target Input neurons");
            }

            writableConnections.addConnection(connection);

        } finally {
            version++;
        }
    }

    private NeuronT addHiddenNeuron(Long neuronId,
                                    ActivationFunction activationFunction,
                                    double... activationFunctionParameters) {

        try {

            NeuronT neuron;

            long idToUse;

            if (null == neuronId) {
                idToUse = writableNeurons.getNextFreeNeuronId();
            } else {
                idToUse = neuronId;
            }

            neuron = neuronFactory.createHidden(
                    idToUse,
                    activationFunction,
                    activationFunctionParameters);

            writableNeurons.addNeuron(neuron);

            return neuron;

        } finally {
            version++;
        }
    }

}
