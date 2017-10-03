package matgr.ai.neuralnet.feedforward;

public class FeedForwardNeuralNetProperties {

    public final int inputCount;

    public final int outputCount;

    public final int hiddenLayers;

    public final int neuronsPerHiddenLayer;

    public final int hiddenNeurons;

    public final WeightsCount hiddenNeuronWeights;

    public final int outputNeurons;

    public final WeightsCount outputNeuronWeights;

    public final int totalNeurons;

    public final WeightsCount totalNeuronWeights;

    public FeedForwardNeuralNetProperties(
            int inputCount,
            int outputCount,
            int hiddenLayers,
            int neuronsPerHiddenLayer,
            int hiddenNeurons,
            WeightsCount hiddenNeuronWeights,
            int outputNeurons,
            WeightsCount outputNeuronWeights) {

        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.hiddenLayers = hiddenLayers;
        this.neuronsPerHiddenLayer = neuronsPerHiddenLayer;

        this.hiddenNeurons = hiddenNeurons;
        this.hiddenNeuronWeights = hiddenNeuronWeights;
        this.outputNeurons = outputNeurons;
        this.outputNeuronWeights = outputNeuronWeights;

        totalNeurons = hiddenNeurons + outputNeurons;
        totalNeuronWeights = WeightsCount.add(hiddenNeuronWeights, outputNeuronWeights);

    }
}
