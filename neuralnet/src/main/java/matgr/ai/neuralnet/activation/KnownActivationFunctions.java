package matgr.ai.neuralnet.activation;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class KnownActivationFunctions {

    private static Map<String, ActivationFunction> writableAll;

    public static final SigmoidActivationFunction SIGMOID;
    public static final SoftplusActivationFunction SOFT_PLUS;
    public static final TanhActivationFunction TANH;
    public static final ReluActivationFunction RELU;
    public static final GaussianActivationFunction GAUSSIAN;

    public static final Map<String, ActivationFunction> ALL;

    static {

        writableAll = new HashMap<>();
        ALL = Collections.unmodifiableMap(writableAll);

        SIGMOID = addFunction(SigmoidActivationFunction.INSTANCE);
        SOFT_PLUS = addFunction(SoftplusActivationFunction.INSTANCE);
        TANH = addFunction(TanhActivationFunction.INSTANCE);
        RELU= addFunction(ReluActivationFunction.INSTANCE);
        GAUSSIAN= addFunction(GaussianActivationFunction.INSTANCE);
    }

    private static <TFunction extends ActivationFunction> TFunction addFunction(TFunction function) {
        writableAll.put(function.name, function);
        return function;
    }
}
