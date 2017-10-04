package matgr.ai.neat;

import matgr.ai.genetic.Genome;
import matgr.ai.neuralnet.cyclic.*;

import java.util.UUID;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

// TODO: look here: http://nn.cs.utexas.edu/downloads/papers/stanley.gecco02_1.pdf and
//       here: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
public class NeatGenome implements Genome {

    protected UUID genomeId;

    // TODO: don't expose this?
    public final NeatNeuralNet neuralNet;

    public NeatGenome(int inputCount,
                      Iterable<OutputNeuronParameters> outputNodesParameters) {

        this(
                new NeatNeuralNet(inputCount, outputNodesParameters),
                UUID.randomUUID());
    }

    public NeatGenome(NeatGenome other) {
        this(CyclicNeuralNet.deepClone(other.neuralNet), other.genomeId);
    }

    private NeatGenome(NeatNeuralNet neuralNet, UUID genomeId) {

        this.neuralNet = neuralNet;
        this.genomeId = genomeId;
    }

    @Override
    public NeatGenome deepClone(UUID genomeId) {
        NeatGenome clone = deepClone();
        clone.genomeId = genomeId;
        return clone;
    }

    public NeatGenome deepClone() {
        return new NeatGenome(this);
    }

    @Override
    public UUID genomeId() {
        return genomeId;
    }

    public static double computeDistance(NeatGenome a,
                                         NeatGenome b,
                                         double excessFactor,
                                         double disjointFactor,
                                         double weightFactor,
                                         int baseGenomeSize,
                                         int minGenomeNormalizationSize) {

        // TODO: maybe find a better way of computing the distance? this might start getting a bit low when the
        //       number of connections increases... (or maybe not as long as there are lots of different connections?)
        // TODO: might not matter when pruning is implemented... ? ...can something be detected somehow?
        // TODO: maybe normalizationSize should always be 1?
        // TODO: compare neurons as well? (activation functions/parameters?) ...does this make sense since they aren't
        //       correlated for crossover? (might still be useful for speciation...)

        int sizeA = a.neuralNet.connectionMap().size() - baseGenomeSize;
        int sizeB = b.neuralNet.connectionMap().size() - baseGenomeSize;

        int normalizationSize = Math.max(1, Math.max(sizeA, sizeB) - minGenomeNormalizationSize);
        return computeDistance(a, b, excessFactor, disjointFactor, weightFactor, normalizationSize);
    }

    public static double computeDistance(NeatGenome a,
                                         NeatGenome b,
                                         double excessFactor,
                                         double disjointFactor,
                                         double weightFactor) {
        return computeDistance(a, b, excessFactor, disjointFactor, weightFactor, 1);
    }

    public static void correlate(
            NeatGenome a,
            NeatGenome b,
            Consumer<NeatConnection> onExcess,
            Consumer<NeatConnection> onDisjoint,
            BiConsumer<NeatConnection, NeatConnection> onMatch) {
        correlate(a, b, onExcess, onDisjoint, onExcess, onDisjoint, onMatch);
    }

    public static void correlate(
            NeatGenome a,
            NeatGenome b,
            Consumer<NeatConnection> onAExcess,
            Consumer<NeatConnection> onADisjoint,
            Consumer<NeatConnection> onBExcess,
            Consumer<NeatConnection> onBDisjoint,
            BiConsumer<NeatConnection, NeatConnection> onMatch) {

        SortedConnectionGeneIterator aEnum = new SortedConnectionGeneIterator(a.neuralNet.connectionMap());
        SortedConnectionGeneIterator bEnum = new SortedConnectionGeneIterator(b.neuralNet.connectionMap());

        if (aEnum.isPastEnd()) {

            while (!bEnum.isPastEnd()) {
                onAExcess.accept(bEnum.getCurrent());
                bEnum.increment();
            }

        } else if (bEnum.isPastEnd()) {

            while (!aEnum.isPastEnd()) {
                onAExcess.accept(aEnum.getCurrent());
                aEnum.increment();
            }

        } else {

            while (!aEnum.isPastEnd() || !bEnum.isPastEnd()) {

                if (aEnum.isPastEnd()) {

                    if (null != aEnum.getPrevious()) {

                        long aEndInnovationId = aEnum.getPrevious().innovationNumber;
                        long bInnovationId = bEnum.getCurrent().innovationNumber;

                        if (bInnovationId > aEndInnovationId) {
                            // b excess
                            onBExcess.accept(bEnum.getCurrent());
                        } else {
                            // b disjoint
                            onBDisjoint.accept(bEnum.getCurrent());
                        }

                    } else {

                        // b excess
                        onBExcess.accept(bEnum.getCurrent());

                    }

                    bEnum.increment();

                } else if (bEnum.isPastEnd()) {

                    if (null != bEnum.getPrevious()) {

                        long bEndInnovationId = bEnum.getPrevious().innovationNumber;
                        long aInnovationId = aEnum.getCurrent().innovationNumber;

                        if (aInnovationId > bEndInnovationId) {
                            // a excess
                            onAExcess.accept(aEnum.getCurrent());
                        } else {
                            // a disjoint
                            onADisjoint.accept(aEnum.getCurrent());
                        }

                    } else {

                        // a excess
                        onAExcess.accept(aEnum.getCurrent());

                    }

                    aEnum.increment();

                } else {

                    long aInnovationId = aEnum.getCurrent().innovationNumber;
                    long bInnovationId = bEnum.getCurrent().innovationNumber;

                    if (aInnovationId < bInnovationId) {

                        if (bEnum.isAtStart()) {
                            // a excess
                            onAExcess.accept(aEnum.getCurrent());
                        } else {
                            // a disjoint
                            onADisjoint.accept(aEnum.getCurrent());
                        }

                        aEnum.increment();

                    } else if (bInnovationId < aInnovationId) {

                        if (aEnum.isAtStart()) {
                            // b excess
                            onBExcess.accept(bEnum.getCurrent());
                        } else {
                            // b disjoint
                            onBDisjoint.accept(bEnum.getCurrent());
                        }

                        bEnum.increment();

                    } else {

                        // match
                        onMatch.accept(aEnum.getCurrent(), bEnum.getCurrent());

                        aEnum.increment();
                        bEnum.increment();

                    }
                }

            }

        }
    }

    private static double computeDistance(NeatGenome a,
                                          NeatGenome b,
                                          double excessFactor,
                                          double disjointFactor,
                                          double weightFactor,
                                          int normalizationSize) {

        final int[] disjointCount = {0};
        final int[] excessCount = {0};

        final double[] weightDifferenceSum = {0.0};
        final int[] weightDifferenceCount = {0};

        correlate(
                a,
                b,
                c -> excessCount[0]++,
                c -> disjointCount[0]++,
                (cA, cB) -> {
                    double difference = Math.abs(cA.weight - cB.weight);

                    weightDifferenceSum[0] += difference;
                    weightDifferenceCount[0]++;
                });

        double averageWeightDifference = 0.0;

        if (weightDifferenceCount[0] > 0) {
            averageWeightDifference = weightDifferenceSum[0] / (double) weightDifferenceCount[0];
        }

        double excessDistance = excessFactor * (double) excessCount[0] / (double) normalizationSize;
        double disjointDistance = disjointFactor * (double) disjointCount[0] / (double) normalizationSize;
        double weightDistance = weightFactor * averageWeightDifference;

        return excessDistance + disjointDistance + weightDistance;
    }
}
