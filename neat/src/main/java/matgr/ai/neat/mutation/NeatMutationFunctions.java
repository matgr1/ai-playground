package matgr.ai.neat.mutation;

import matgr.ai.neat.NeatConnection;
import matgr.ai.neat.NeatGenome;
import matgr.ai.genetic.mutation.MutationFunctions;
import matgr.ai.math.DiscreteDistribution;
import matgr.ai.math.RandomFunctions;
import matgr.ai.neuralnet.activation.DefaultActivationFunctions;
import matgr.ai.neuralnet.cyclic.Neuron;
import matgr.ai.neuralnet.cyclic.NeuronType;
import matgr.ai.neuralnet.cyclic.ConnectionIds;
import matgr.ai.neuralnet.cyclic.ReadOnlyConnectionMap;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.*;

public class NeatMutationFunctions {

    // TODO: do innovationMap better... make a class so it's not just a bunch of longs
    //       (maps is sourceNodeId->targetNodeId->innovationNumber
    public static void mutate(RandomGenerator random,
                              NeatMutationSettings settings,
                              NeatGenome genome,
                              long currentGeneration,
                              Map<Long, Map<Long, Long>> innovationMap) {
        boolean mutated = false;

        DiscreteDistribution<NeatStructuralMutationType> mutationDistribution = DiscreteDistribution.create(
                settings.getMutationProbabilityProportions().keySet(),
                settings.getMutationProbabilityProportions());

        while (!mutated) {

            if ((mutationDistribution.stats.count < 1) || (mutationDistribution.groupedStats.count < 1)) {
                break;
            }

            NeatStructuralMutationType mutationType = mutationDistribution.sample(random, false);

            switch (mutationType) {

                case AddNode:
                    mutated = addNodeMutation(random, genome, innovationMap);
                    break;

                case RemoveNode:
                    mutated = removeNodeMutation(random, genome);
                    break;

                case AddConnection:
                    mutated = addConnectionMutation(random, settings, genome, innovationMap);
                    break;

                case RemoveConnection:
                    mutated = removeConnectionMutation(random, genome);
                    break;

                case MutateWeight:
                    mutated = adjustWeightsMutation(random, settings, genome, currentGeneration);
                    break;

                default:
                    throw new IllegalArgumentException("Invalid mutation type");

            }

            if (!mutated) {
                mutationDistribution = mutationDistribution.removeOutcome(mutationType);
            }
        }
    }

    private static boolean addNodeMutation(RandomGenerator random,
                                           NeatGenome genome,
                                           Map<Long, Map<Long, Long>> innovationMap) {

        NeatConnection oldConnection = randomConnection(random, genome);

        if (oldConnection != null) {
            // add node by splitting connection

            oldConnection.enabled = false;

            Neuron newNode = genome.neuralNet.addHiddenNeuron(
                    DefaultActivationFunctions.HIDDEN_NODE_ACTIVATION_FUNCTION,
                    genome.activationResponse);

            // TODO: order? max weight first or second?
            addConnection(
                    genome,
                    oldConnection.sourceId,
                    newNode.id,
                    oldConnection.weight,
                    innovationMap);

            addConnection(
                    genome,
                    newNode.id,
                    oldConnection.targetId,
                    oldConnection.weight,
                    innovationMap);

            return true;
        }

        return false;
    }

    private static boolean removeNodeMutation(RandomGenerator random, NeatGenome genome) {

        Set<Long> hiddenNeuronIds = genome.neuralNet.neurons.ids(NeuronType.Hidden);

        if (hiddenNeuronIds.size() > 0) {

            long neuronId = RandomFunctions.selectItem(random, hiddenNeuronIds);

            ReadOnlyConnectionMap<NeatConnection> connections = genome.neuralNet.connections;

            List<NeatConnection> incomingConnections = connections.getIncomingConnections(neuronId);
            List<NeatConnection> outgoingConnections = connections.getOutgoingConnections(neuronId);

            // TODO: what about connections to self? handle those, more may be deletable...

            boolean canRemove = false;

            List<NeatConnection> connectionsToRemove = new ArrayList<>();

            if ((0 == incomingConnections.size()) || (0 == outgoingConnections.size())) {

                // "dead end" node, so all connections can be removed

                // TODO: more nodes may be removable now?
                connectionsToRemove.addAll(incomingConnections);
                connectionsToRemove.addAll(outgoingConnections);

                canRemove = true;

            } else if (incomingConnections.size() == 1) {

                NeatConnection incomingConnection = incomingConnections.get(0);
                connectionsToRemove.add(incomingConnection);

                for (NeatConnection outgoingConnection : outgoingConnections) {

                    connectionsToRemove.add(outgoingConnection);

                    if (!genome.neuralNet.isConnected(incomingConnection.sourceId, outgoingConnection.targetId)) {

                        // TODO: is this valid?
                        double weight = (incomingConnection.weight + outgoingConnection.weight) / 2.0;

                        genome.neuralNet.addConnection(
                                incomingConnection.sourceId,
                                outgoingConnection.targetId,
                                true,
                                weight);
                    }
                }

                canRemove = true;

            } else if (outgoingConnections.size() == 1) {

                NeatConnection outgoingConnection = outgoingConnections.get(0);
                connectionsToRemove.add(outgoingConnection);

                for (NeatConnection incomingConnection : incomingConnections) {

                    connectionsToRemove.add(incomingConnection);

                    if (!genome.neuralNet.isConnected(incomingConnection.sourceId, outgoingConnection.targetId)) {

                        // TODO: is this valid?
                        double weight = (incomingConnection.weight + outgoingConnection.weight) / 2.0;

                        genome.neuralNet.addConnection(
                                incomingConnection.sourceId,
                                outgoingConnection.targetId,
                                true,
                                weight);
                    }
                }

                canRemove = true;
            }

            for (NeatConnection connectionToRemove : connectionsToRemove) {
                genome.neuralNet.removeConnection(connectionToRemove);
            }

            if (canRemove) {
                genome.neuralNet.removeHiddenNeuron(neuronId);
                return true;
            }
        }

        return false;
    }

    private static boolean addConnectionMutation(RandomGenerator random,
                                                 NeatMutationSettings settings,
                                                 NeatGenome genome,
                                                 Map<Long, Map<Long, Long>> innovationMap) {

        ConnectionIds connectionIds = randomUnusedConnection(random, genome);

        if (connectionIds != null) {

            Neuron sourceNode = genome.neuralNet.neurons.get(connectionIds.sourceId);
            Neuron targetNode = genome.neuralNet.neurons.get(connectionIds.targetId);

            double weight = settings.getConnectionWeightsMutationSettings().getRandomValueInRange(random);

            addConnection(genome, sourceNode.id, targetNode.id, weight, innovationMap);

            return true;
        }

        return false;
    }

    private static boolean removeConnectionMutation(RandomGenerator random, NeatGenome genome) {

        // TODO: put back node/connection deletion? also fix probabilities in settings... no RemoveNode
        //       right now...
        // TODO: why are no input->output connections being removed? (could it just be that there are only
        //       so many connections that can be added, so they just end up getting put back? ...try pruning
        //       phases and see what happens...)
        // TODO: no elites? is the proportion too small to ever pick any out of these small species?

        NeatConnection connection = randomConnection(random, genome);

        if (connection != null) {

            genome.neuralNet.removeConnection(connection);

            if (isRemovableNeuron(genome, connection.sourceId)) {

                int sourceIncoming = genome.neuralNet.connections.getIncomingConnectionCount(connection.sourceId);
                int sourceOutgoing = genome.neuralNet.connections.getOutgoingConnectionCount(connection.sourceId);

                int sourceConnectionCount = sourceIncoming + sourceOutgoing;

                if (sourceConnectionCount <= 0) {
                    genome.neuralNet.removeHiddenNeuron(connection.sourceId);
                }
            }

            if (isRemovableNeuron(genome, connection.targetId)) {

                int targetIncoming = genome.neuralNet.connections.getIncomingConnectionCount(connection.targetId);
                int targetOutgoing = genome.neuralNet.connections.getOutgoingConnectionCount(connection.targetId);

                int targetConnectionCount = targetIncoming + targetOutgoing;

                if (targetConnectionCount <= 0) {
                    genome.neuralNet.removeHiddenNeuron(connection.targetId);
                }
            }

            return true;
        }

        return false;
    }

    private static boolean adjustWeightsMutation(RandomGenerator random,
                                                 NeatMutationSettings settings,
                                                 NeatGenome genome,
                                                 long currentGeneration) {

        NeatConnection connection = randomConnection(random, genome);

        if (connection != null) {

            connection.weight = MutationFunctions.mutate(
                    random,
                    settings.getConnectionWeightsMutationSettings(),
                    currentGeneration,
                    connection.weight);

            return true;
        }

        return false;
    }

    private static NeatConnection randomConnection(RandomGenerator random, NeatGenome genome) {

        if (genome.neuralNet.connections.count() > 0) {

            long sourceId = RandomFunctions.selectItem(
                    random,
                    genome.neuralNet.connections.sourceIds());

            long targetId = RandomFunctions.selectItem(
                    random,
                    genome.neuralNet.connections.targetIds(sourceId));

            return genome.neuralNet.connections.getConnection(
                    sourceId,
                    targetId);
        }

        return null;
    }

    private static ConnectionIds randomUnusedConnection(RandomGenerator random, NeatGenome genome) {

        int biasCount = genome.neuralNet.neurons.count(NeuronType.Bias);
        int inputCount = genome.neuralNet.neurons.count(NeuronType.Input);

        int hiddenCount = genome.neuralNet.neurons.count(NeuronType.Hidden);
        int outputCount = genome.neuralNet.neurons.count(NeuronType.Output);

        int sourceOnlyCount = biasCount + inputCount;
        int sourceTargetNodeCount = hiddenCount + outputCount;

        int maxConnections = sourceTargetNodeCount * (sourceOnlyCount + sourceTargetNodeCount);

        if (genome.neuralNet.connections.count() < maxConnections) {

            Set<Long> sourceIds = new HashSet<>();

            sourceIds.addAll(genome.neuralNet.neurons.ids(NeuronType.Bias));
            sourceIds.addAll(genome.neuralNet.neurons.ids(NeuronType.Input));
            sourceIds.addAll(genome.neuralNet.neurons.ids(NeuronType.Hidden));
            sourceIds.addAll(genome.neuralNet.neurons.ids(NeuronType.Output));

            while (sourceIds.size() > 0) {

                long sourceId = RandomFunctions.selectItem(random, sourceIds);

                Set<Long> targetIds = new HashSet<>();

                targetIds.addAll(genome.neuralNet.neurons.ids(NeuronType.Hidden));
                targetIds.addAll(genome.neuralNet.neurons.ids(NeuronType.Output));

                while (targetIds.size() > 0) {

                    long targetId = RandomFunctions.selectItem(random, targetIds);

                    NeatConnection connection = genome.neuralNet.connections.getConnection(sourceId, targetId);

                    if (connection == null) {
                        return new ConnectionIds(sourceId, targetId);
                    }

                    targetIds.remove(targetId);
                }

                sourceIds.remove(sourceId);
            }
        }

        return null;
    }

    private static boolean isRemovableNeuron(NeatGenome genome, long neuronId) {

        Neuron neuron = genome.neuralNet.neurons.get(neuronId);

        if (neuron == null) {
            return false;
        }

        if (neuron.type == NeuronType.Hidden) {
            return true;
        }

        return false;
    }

    // TODO: this should be handled by the NeatGenome itself... so maybe it needs a reference to it's population?
    //       (and also the public addconnection in the base class should be protected?)
    // TODO: is the innovationMap just supposed to be for the generation?
    // TODO: either way, this should at least not be here
    public static void addConnection(NeatGenome genome,
                                     long sourceId,
                                     long targetId,
                                     double weight,
                                     Map<Long, Map<Long, Long>> innovationMap) {

        Long existingInnovationNumber = 0L;
        boolean hasExistingInnovationNumber = false;

        Map<Long, Long> targetMap = innovationMap.get(sourceId);

        if (targetMap != null) {

            existingInnovationNumber = targetMap.get(targetId);

            if (existingInnovationNumber != null) {
                hasExistingInnovationNumber = true;
            }

        } else {

            targetMap = new HashMap<>();
            innovationMap.put(sourceId, targetMap);

        }

        if (hasExistingInnovationNumber) {

            genome.neuralNet.addConnection(sourceId, targetId, true, weight, existingInnovationNumber);

        } else {

            NeatConnection newGene = genome.neuralNet.addConnection(sourceId, targetId, true, weight);
            targetMap.put(targetId, newGene.innovationNumber);

        }

    }
}
