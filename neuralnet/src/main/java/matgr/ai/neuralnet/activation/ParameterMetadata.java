package matgr.ai.neuralnet.activation;

public class ParameterMetadata {

    public final String name;
    public final double defaultValue;

    public ParameterMetadata(String name, double defaultValue) {
        this.name = name;
        this.defaultValue = defaultValue;
    }
}
