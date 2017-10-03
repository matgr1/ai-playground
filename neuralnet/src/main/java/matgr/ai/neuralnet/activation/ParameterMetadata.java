package matgr.ai.neuralnet.activation;

public class ParameterMetadata {
    public String name;

    public double defaultValue;

    public ParameterMetadata(String name, double defaultValue) {
        this.name = name;
        this.defaultValue = defaultValue;
    }
}
