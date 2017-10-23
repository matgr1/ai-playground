package matgr.ai.neuralnet;

import java.util.List;

public class TrainingSet {

    public List<Double> inputs;
    public List<Double> expectedOutputs;

    public TrainingSet(List<Double> inputs, List<Double> expectedOutputs) {

        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
    }
}
