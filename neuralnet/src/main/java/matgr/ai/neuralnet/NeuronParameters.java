package matgr.ai.neuralnet;

import matgr.ai.neuralnet.activation.ActivationFunction;

public class NeuronParameters {

    public final ActivationFunction activationFunction;

    public final double[] activationFunctionParameters;

    public NeuronParameters(ActivationFunction activationFunction, double... activationFunctionParameters) {

        activationFunction.validateParameters(activationFunctionParameters);

        this.activationFunction = activationFunction;
        this.activationFunctionParameters = activationFunctionParameters;
    }
}