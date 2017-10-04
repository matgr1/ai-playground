package matgr.ai.neuralnet.activation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

public abstract class ActivationFunction {

    public final String name;

    public final List<ParameterMetadata> parameters;

    protected ActivationFunction(String name, ParameterMetadata... parameters) {
        this(
                name,
                consumer -> {
                    for (ParameterMetadata parameter : parameters) {
                        consumer.accept(parameter);
                    }
                });
    }

    protected ActivationFunction(String name, Iterable<ParameterMetadata> parameters) {
        this(name, parameters::forEach);
    }

    private ActivationFunction(String name, Consumer<Consumer<ParameterMetadata>> iterateParameters) {
        this.name = name;

        List<ParameterMetadata> writableParameters = new ArrayList<>();
        iterateParameters.accept(writableParameters::add);

        this.parameters = Collections.unmodifiableList(writableParameters);
    }

    public double compute(double x, double... parameters) {
        validateParameters(parameters);
        return computeActivation(x, parameters);
    }

    public void validateParameters(double... parameters) {
        int parameterCount = 0;
        if (parameters != null) {
            parameterCount = parameters.length;
        }

        if (parameterCount != this.parameters.size()) {
            throw new IllegalArgumentException("Incorrect number of parameters supplied");
        }
    }

    public abstract double[] defaultParameters();

    protected abstract double computeActivation(double x, double[] parameters);
}
