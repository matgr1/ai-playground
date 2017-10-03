package matgr.ai.neuralnet.activation;

import matgr.ai.neuralnet.cyclic.OutputNeuronParameters;

import java.util.ArrayList;
import java.util.List;

public final class DefaultActivationFunctions {

    // TODO: allow different activation functions ...also, for NEAT the default activationResponse should be 4.9
    //       (see here: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
    // TODO: look here for activation functions: http://blog.otoro.net/2016/05/07/backprop-neat/ (there are a
    //       bunch listed and some explanations in a colorful table)

    public static final ActivationFunction OUTPUT_NODE_ACTIVATION_FUNCTION;
    public static final ActivationFunction HIDDEN_NODE_ACTIVATION_FUNCTION;

    static {
        OUTPUT_NODE_ACTIVATION_FUNCTION = SigmoidActivationFunction.instance;
        HIDDEN_NODE_ACTIVATION_FUNCTION = SigmoidActivationFunction.instance;
    }

    public static List<OutputNeuronParameters> createOutputNeuronParameters(int outputCount, double activationResponse) {
        List<OutputNeuronParameters> parameters = new ArrayList<>();
        for (int i = 0; i < outputCount; i++) {
            parameters.add(new OutputNeuronParameters(OUTPUT_NODE_ACTIVATION_FUNCTION, activationResponse));
        }
        return parameters;
    }
}
