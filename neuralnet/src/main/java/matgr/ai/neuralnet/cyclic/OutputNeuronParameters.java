package matgr.ai.neuralnet.cyclic;

import matgr.ai.neuralnet.activation.ActivationFunction;

public class OutputNeuronParameters {

    public final ActivationFunction activationFunction;

    public final double[] activationFunctionParameters;

    public OutputNeuronParameters(ActivationFunction activationFunction, double... activationFunctionParameters) {

        activationFunction.validateParameters(activationFunctionParameters);

        this.activationFunction = activationFunction;
        this.activationFunctionParameters = activationFunctionParameters;
    }
}