package matgr.ai.neuralnet.cyclic;

import matgr.ai.neuralnet.activation.ActivationFunction;

public class HiddenNeuronParameters extends ActivatableNeuronParameters {

    public final Long nodeId;

    public HiddenNeuronParameters(Long nodeId,
                                  ActivationFunction activationFunction,
                                  double... activationFunctionParameters) {

        super(activationFunction, activationFunctionParameters);

        this.nodeId = nodeId;
    }

}