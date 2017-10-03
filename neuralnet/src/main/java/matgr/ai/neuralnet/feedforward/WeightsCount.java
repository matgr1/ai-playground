package matgr.ai.neuralnet.feedforward;

public class WeightsCount {

    public final int weights;

    public final int biasWeights;

    public final int total;

    public WeightsCount() {
        this(0, 0);
    }

    public WeightsCount(int weights, int biasWeights) {
        this.weights = weights;
        this.biasWeights = biasWeights;
        total = weights + biasWeights;
    }

    public static WeightsCount add(WeightsCount a, WeightsCount b) {
        return new WeightsCount(a.weights + b.weights, a.biasWeights + b.biasWeights);
    }

    public WeightsCount add(WeightsCount weights) {
        return add(this, weights);
    }

}