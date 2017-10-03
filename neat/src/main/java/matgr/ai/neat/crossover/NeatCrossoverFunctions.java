package matgr.ai.neat.crossover;

import matgr.ai.neat.NeatConnection;
import matgr.ai.neat.NeatGenome;
import matgr.ai.genetic.FitnessItem;
import matgr.ai.genetic.GenomeParents;
import matgr.ai.genetic.SortedGenomeParents;
import matgr.ai.genetic.crossover.CrossoverFunctions;
import matgr.ai.math.RandomFunctions;
import matgr.ai.neat.NeatNeuralNet;
import matgr.ai.neuralnet.cyclic.Neuron;
import matgr.ai.neuralnet.cyclic.NeuronType;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.SortedMap;
import java.util.TreeMap;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

public final class NeatCrossoverFunctions {

    @FunctionalInterface
    public interface CreateNeatGenome<NeatGenomeT> {
        NeatGenomeT create(int inputCount, int outputCount, double activationResponse);
    }

    // TODO: crossover/mutate activation functions/any other parameters of the nodes?
    public static <NeatGenomeT extends NeatGenome> NeatGenomeT crossover(RandomGenerator random,
                                                                         NeatCrossoverSettings settings,
                                                                         GenomeParents<NeatGenomeT> parents,
                                                                         CreateNeatGenome<NeatGenomeT> createGenome) {

        SortedGenomeParents<NeatGenomeT> sortedParents = parents.getSorted(random);

        int inputCount = sortedParents.fittest.item.neuralNet.neurons.count(NeuronType.Input);
        int outputCount = sortedParents.fittest.item.neuralNet.neurons.count(NeuronType.Output);

        NeatGenomeT child = createGenome.<NeatGenomeT>create(
                inputCount,
                outputCount,
                sortedParents.fittest.item.activationResponse);

        SortedMap<Long, GenomeParents<GenomeAndConnection<NeatGenomeT>>> correlatedConnections = correlateConnections(
                sortedParents.fittest,
                sortedParents.other);

        // crossover connections
        for (GenomeParents<GenomeAndConnection<NeatGenomeT>> parentConnections : correlatedConnections.values()) {

            SortedGenomeParents<GenomeAndConnection<NeatGenomeT>> sortedParentConnections =
                    parentConnections.getSorted(random);

            FitnessItem<GenomeAndConnection<NeatGenomeT>> fittest = sortedParentConnections.fittest;
            FitnessItem<GenomeAndConnection<NeatGenomeT>> other = sortedParentConnections.other;

            if (null != other) {

                if (RandomFunctions.testProbability(random, settings.getProbability())) {

                    double newWeight = CrossoverFunctions.crossover(
                            random,
                            settings.getConnectionWeightsCrossoverSettings(),
                            fittest.item.connection.weight,
                            other.item.connection.weight);

                    boolean newEnabled;

                    if (fittest.item.connection.enabled && other.item.connection.enabled) {
                        newEnabled = true;
                    } else {
                        if (random.nextDouble() < settings.getConnectionCrossoverDisableRate()) {
                            newEnabled = false;
                        } else {
                            newEnabled = true;
                        }
                    }

                    addConnection(
                            fittest.item.genome.neuralNet,
                            child.neuralNet,
                            fittest.item.connection.innovationNumber,
                            fittest.item.connection.sourceId,
                            fittest.item.connection.targetId,
                            newWeight,
                            newEnabled);

                } else {

                    addConnection(
                            fittest.item.genome.neuralNet,
                            child.neuralNet,
                            fittest.item.connection.innovationNumber,
                            fittest.item.connection.sourceId,
                            fittest.item.connection.targetId,
                            fittest.item.connection.weight,
                            fittest.item.connection.enabled);

                }

            } else {

                addConnection(
                        fittest.item.genome.neuralNet,
                        child.neuralNet,
                        fittest.item.connection.innovationNumber,
                        fittest.item.connection.sourceId,
                        fittest.item.connection.targetId,
                        fittest.item.connection.weight,
                        fittest.item.connection.enabled);
            }
        }

        return child;
    }

    // TODO: share this with the stuff in NeatGenome that computes the distance if possible (at least get rid
    //       of public "ConnectionMap")
    private static <
            NeatGenomeT extends NeatGenome>
    SortedMap<Long, GenomeParents<GenomeAndConnection<NeatGenomeT>>> correlateConnections(
            FitnessItem<NeatGenomeT> parentA,
            FitnessItem<NeatGenomeT> parentB) {

        SortedMap<Long, GenomeParents<GenomeAndConnection<NeatGenomeT>>> correlated = new TreeMap<>();

        correlate(
                parentA,
                parentB,
                (aOnly) -> {

                    FitnessItem<GenomeAndConnection<NeatGenomeT>> parentAConnection =
                            createGenomeAndConnection(parentA.item, aOnly, parentA.fitness);

                    GenomeParents<GenomeAndConnection<NeatGenomeT>> parentConnections = new GenomeParents<>(
                            parentAConnection,
                            null);

                    correlated.put(aOnly.innovationNumber, parentConnections);
                },
                (bOnly) -> {

                    FitnessItem<GenomeAndConnection<NeatGenomeT>> parentBConnection =
                            createGenomeAndConnection(parentB.item, bOnly, parentB.fitness);

                    GenomeParents<GenomeAndConnection<NeatGenomeT>> parentConnections = new GenomeParents<>(
                            null,
                            parentBConnection);

                    correlated.put(bOnly.innovationNumber, parentConnections);
                },
                (aMatch, bMatch) -> {

                    FitnessItem<GenomeAndConnection<NeatGenomeT>> parentAConnection =
                            createGenomeAndConnection(parentA.item, aMatch, parentA.fitness);

                    FitnessItem<GenomeAndConnection<NeatGenomeT>> parentBConnection =
                            createGenomeAndConnection(parentB.item, bMatch, parentB.fitness);

                    GenomeParents<GenomeAndConnection<NeatGenomeT>> parentConnections = new GenomeParents<>(
                            parentAConnection,
                            parentBConnection);

                    correlated.put(aMatch.innovationNumber, parentConnections);
                });

        return correlated;

    }

    private static <NeatGenomeT extends NeatGenome> void correlate(
            FitnessItem<NeatGenomeT> parentA,
            FitnessItem<NeatGenomeT> parentB,
            Consumer<NeatConnection> onAOnly,
            Consumer<NeatConnection> onBOnly,
            BiConsumer<NeatConnection, NeatConnection> onMatch) {

        NeatGenome.correlate(parentA.item, parentB.item, onAOnly, onAOnly, onBOnly, onBOnly, onMatch);

    }

    private static NeatConnection addConnection(NeatNeuralNet parent,
                                                NeatNeuralNet child,
                                                long innovationNumber,
                                                long sourceNodeId,
                                                long targetNodeId,
                                                double weight,
                                                boolean isEnabled) {

        Neuron sourceNode = child.neurons.get(sourceNodeId);
        if (sourceNode == null) {
            sourceNode = addHiddenNode(parent, child, sourceNodeId);
        }

        Neuron targetNode = child.neurons.get(targetNodeId);
        if (targetNode == null) {
            targetNode = addHiddenNode(parent, child, targetNodeId);
        }

        return child.addConnection(sourceNode.id, targetNode.id, isEnabled, weight, innovationNumber);

    }

    private static Neuron addHiddenNode(NeatNeuralNet parent,
                                        NeatNeuralNet child,
                                        long nodeId) {

        Neuron parentNode = parent.neurons.get(nodeId);

        return child.addHiddenNeuron(
                parentNode.id,
                parentNode.getActivationFunction(),
                parentNode.getActivationFunctionParameters());
    }

    private static <
            NeatGenomeT extends NeatGenome>
    FitnessItem<GenomeAndConnection<NeatGenomeT>> createGenomeAndConnection(NeatGenomeT genome,
                                                                            NeatConnection connection,
                                                                            double fitness) {

        GenomeAndConnection<NeatGenomeT> genomeAndConnection = new GenomeAndConnection<>(
                genome,
                connection);

        return new FitnessItem<>(genomeAndConnection, fitness);
    }

    private static class GenomeAndConnection<NeatGenomeT extends NeatGenome> {

        public final NeatGenomeT genome;
        public final NeatConnection connection;

        public GenomeAndConnection(NeatGenomeT genome, NeatConnection connection) {
            this.genome = genome;
            this.connection = connection;
        }
    }
}
