package matgr.ai.neuralnet.feedforward;

import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.activation.KnownActivationFunctions;

public abstract class ActivatableLayer<NeuronT extends Neuron> extends NeuronLayer<NeuronT> {

    private final ActivationFunction activationFunction;
    private final double[] activationFunctionParameters;

    protected ActivatableLayer(NeuronFactory<NeuronT> neuronFactory,
                               ActivationFunction activationFunction,
                               double... activationFunctionParameters) {
        super(neuronFactory);

        if (null == activationFunction) {
            throw new IllegalArgumentException("activationFunction not provided");
        }
        if (null == activationFunctionParameters) {
            throw new IllegalArgumentException("activationFunctionParameters not provided");
        }

        this.activationFunction = activationFunction;
        this.activationFunctionParameters = activationFunctionParameters;
    }

    protected ActivatableLayer(ActivatableLayer<NeuronT> other) {

        super(other);

        this.activationFunction = other.activationFunction;
        this.activationFunctionParameters = other.activationFunctionParameters;
    }

    protected void activateNeuron(NeuronState<NeuronT> neuron) {

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

    protected double computePreSynapseOutputDerivative(NeuronState<NeuronT> neuron) {

        // TODO: handle NaNs
        double neuronOutput = neuron.postSynapse;
        return activationFunction.computeDerivative(neuronOutput, activationFunctionParameters);
    }
}
