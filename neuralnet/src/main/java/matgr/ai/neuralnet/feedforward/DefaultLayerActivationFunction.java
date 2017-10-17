package matgr.ai.neuralnet.feedforward;

import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.activation.KnownActivationFunctions;

public class DefaultLayerActivationFunction implements LayerActivationFunction {

    public final ActivationFunction function;
    public final double[] parameters;

    public DefaultLayerActivationFunction(ActivationFunction function, double[] parameters) {
        this.function = function;
        this.parameters = parameters;
    }

    @Override
    public <NeuronT extends Neuron> void activate(Iterable<NeuronState<NeuronT>> neurons) {

        for (NeuronState<NeuronT> neuron : neurons) {

            neuron.postSynapse = function.compute(neuron.preSynapse, parameters);

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

    @Override
    public double computeDerivative(double activationOutput) {
        return function.computeDerivative(activationOutput, parameters);
    }
}
