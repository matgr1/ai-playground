package matgr.ai.neuralnet.cyclic;

import matgr.ai.neuralnet.activation.ActivationFunction;

public class OutputNeuronParameters extends ActivatableNeuronParameters {

    public OutputNeuronParameters(ActivationFunction activationFunction, double... activationFunctionParameters) {
        super(activationFunction, activationFunctionParameters);
    }

}