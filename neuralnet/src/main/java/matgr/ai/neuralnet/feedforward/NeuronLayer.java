package matgr.ai.neuralnet.feedforward;

import matgr.ai.math.RandomFunctions;
import matgr.ai.neuralnet.Connection;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.activation.KnownActivationFunctions;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.HashMap;
import java.util.Map;

public abstract class NeuronLayer<NeuronT extends Neuron> {

    protected final NeuronFactory<NeuronT> neuronFactory;
    protected final Map<Long, NeuronLayer.NeuronConnectionMap> connectionMap;

    protected NeuronLayer(NeuronFactory<NeuronT> neuronFactory) {

        this.neuronFactory = neuronFactory;
        this.connectionMap = new HashMap<>();
    }

    protected NeuronLayer(NeuronLayer<NeuronT> other) {

        if (null == other) {
            throw new IllegalArgumentException("other");
        }

        this.neuronFactory = other.neuronFactory;
        this.connectionMap = new HashMap<>();

        for (Map.Entry<Long, NeuronLayer.NeuronConnectionMap> sourceEntry : other.connectionMap.entrySet()) {

            NeuronLayer.NeuronConnectionMap copy = sourceEntry.getValue().deepClone();
            this.connectionMap.put(sourceEntry.getKey(), copy);
        }
    }

    public static <NeuronLayerT extends NeuronLayer> NeuronLayerT deepClone(NeuronLayerT layer) {

        @SuppressWarnings("unchecked")
        NeuronLayerT clone = (NeuronLayerT) layer.deepClone();

        if (clone.getClass() != layer.getClass()) {
            throw new IllegalArgumentException("Invalid item - clone not overridden correctly in derived class");
        }

        return clone;
    }

    public abstract int neuronCount();

    public abstract Iterable<NeuronT> neurons();

    protected abstract Iterable<NeuronState<NeuronT>> writableNeurons();

    protected abstract NeuronLayer<NeuronT> deepClone();

    void randomizeWeights(RandomGenerator random) {

        for (NeuronLayer.NeuronConnectionMap sourceMap : connectionMap.values()) {

            for (Connection connection : sourceMap.sourceMap.values()) {

                connection.weight = getRandomWeight(random);
            }

            sourceMap.biasWeight = getRandomWeight(random);
        }
    }

    void activate(Iterable<NeuronState<NeuronT>> previousLayerNeurons, double bias) {

        // TODO: this could maybe be done in one of the other loops?
        for (NeuronState<NeuronT> neuron : writableNeurons()) {
            neuron.preSynapse = 0.0;
        }

        for (NeuronState<NeuronT> neuron : writableNeurons()) {

            NeuronConnectionMap sourceMap = connectionMap.get(neuron.neuron.id);

            for (NeuronState<NeuronT> previousNeuron : previousLayerNeurons) {

                Connection connection = sourceMap.sourceMap.get(previousNeuron.neuron.id);

                if (connection.enabled) {

                    double weight = connection.weight;

                    neuron.preSynapse += previousNeuron.postSynapse * weight;

                    if (Double.isNaN(neuron.preSynapse)) {
                        // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                        //       return a status code from this function)... it should be given
                        //       "input" and "weight" (so it can decide what to do based on input values being
                        //       infinite/NaN/etc... if it fails, then set this to 0.0
                        neuron.preSynapse = 0.0;
                    }
                }
            }

            neuron.preSynapse += (sourceMap.biasWeight * bias);

            if (Double.isNaN(neuron.preSynapse)) {
                // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                //       return a status code from this function)... it should be given
                //       "neuron.incomingBiasWeight" and "bias" (so it can decide what to do based on input values being
                //       infinite/NaN/etc... if it fails, then set this to 0.0
                neuron.preSynapse = 0.0;
            }

            ActivationFunction activationFunction = neuron.neuron.getActivationFunction();
            double[] activationFunctionParameters = neuron.neuron.getActivationFunctionParameters();

            neuron.postSynapse = activationFunction.compute(neuron.preSynapse, activationFunctionParameters);

            if (Double.isNaN(neuron.postSynapse)) {
                // NOTE: sigmoid shouldn't produce NaN, so fallback to this one for now...
                // TODO: pass in some sort of NaN handler (with the ability to completely bail out and return a
                //       status code from this function)... if it fails, then try this
                neuron.postSynapse = KnownActivationFunctions.SIGMOID.compute(
                        neuron.preSynapse,
                        KnownActivationFunctions.SIGMOID.defaultParameters());
            }
        }
    }

    Map<Long, Double> backPropagate(double learningRate,
                                    double bias,
                                    Iterable<NeuronState<NeuronT>> previousLayerNeurons,
                                    Map<Long, Double> thisLayer_dE_dOut) {

        Map<Long, Double> previousLayer_dE_dOut = new HashMap<>();

        for (NeuronState<NeuronT> neuron : writableNeurons()) {

            NeuronConnectionMap sourceMap = connectionMap.get(neuron.neuron.id);

            double neuronOutput = neuron.postSynapse;
            double dE_dOut = thisLayer_dE_dOut.get(neuron.neuron.id);

            ActivationFunction activationFunction = neuron.neuron.getActivationFunction();
            double[] activationFunctionParameters = neuron.neuron.getActivationFunctionParameters();

            double dOut_dIn = activationFunction.computeDerivativeFromActivationOutput(
                    neuronOutput,
                    activationFunctionParameters);

            double dE_dIn = dE_dOut * dOut_dIn;

            // update incoming connection weights
            for (NeuronState<NeuronT> previousNeuron : previousLayerNeurons) {

                double dIn_dW = previousNeuron.postSynapse;
                double dE_dW = dE_dIn * dIn_dW;

                Connection connection = sourceMap.sourceMap.get(previousNeuron.neuron.id);
                double currentWeight = connection.weight;

                // TODO: don't need to compute this on the last pass
                // TODO: do this better... maybe a context object with context objects (so don't need to "put" again,
                //       and maybe better parameters to this function
                Double currentPrevLayer_dE_dOut = previousLayer_dE_dOut.get(previousNeuron.neuron.id);

                if (currentPrevLayer_dE_dOut == null) {
                    currentPrevLayer_dE_dOut = 0.0;
                }

                currentPrevLayer_dE_dOut += (dE_dIn * currentWeight);
                previousLayer_dE_dOut.put(previousNeuron.neuron.id, currentPrevLayer_dE_dOut);

                connection.weight = currentWeight - (dE_dW * learningRate);
            }

            // update incoming bias weight
            double dIn_dW_Bias = bias;
            double dE_dW_Bias = dE_dIn * dIn_dW_Bias;

            double currentWeight_Bias = sourceMap.biasWeight;
            double newWeight_Bias = currentWeight_Bias - (dE_dW_Bias * learningRate);

            sourceMap.biasWeight = newWeight_Bias;
        }

        return previousLayer_dE_dOut;
    }

    private static double getRandomWeight(RandomGenerator random) {
        return RandomFunctions.nextDouble(random, -1.0, 1.0);
    }

    protected static class NeuronConnectionMap {
        public double biasWeight;
        public final Map<Long, Connection> sourceMap;

        public NeuronConnectionMap() {
            biasWeight = 0.0;
            sourceMap = new HashMap<>();
        }

        private NeuronConnectionMap(NeuronLayer.NeuronConnectionMap other) {
            biasWeight = other.biasWeight;
            sourceMap = new HashMap<>();

            for (Map.Entry<Long, Connection> sourceEntry : other.sourceMap.entrySet()) {

                Connection copy = Connection.deepClone(sourceEntry.getValue());
                sourceMap.put(sourceEntry.getKey(), copy);
            }
        }

        public NeuronLayer.NeuronConnectionMap deepClone() {
            return new NeuronLayer.NeuronConnectionMap(this);
        }
    }
}
