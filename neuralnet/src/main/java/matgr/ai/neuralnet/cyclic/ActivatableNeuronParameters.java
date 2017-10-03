package matgr.ai.neuralnet.cyclic;

import matgr.ai.neuralnet.activation.ActivationFunction;

/**
 * @deprecated
 * Delete, just use OutputNeuronParameters...
 */
@Deprecated
public abstract class ActivatableNeuronParameters {

    public final ActivationFunction activationFunction;

    public final double[] activationFunctionParameters;

    protected ActivatableNeuronParameters(ActivationFunction activationFunction,
                                          double... activationFunctionParameters) {

        activationFunction.validateParameters(activationFunctionParameters);

        this.activationFunction = activationFunction;
        this.activationFunctionParameters = activationFunctionParameters;
    }

}