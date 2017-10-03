package matgr.ai.neat.mutation;

import matgr.ai.genetic.mutation.MutationSettings;

import java.util.Map;

public interface NeatMutationSettings {

    double getSexualMutationProbability();

    MutationSettings getConnectionWeightsMutationSettings();

    Map<NeatStructuralMutationType, Double> getMutationProbabilityProportions();

}
