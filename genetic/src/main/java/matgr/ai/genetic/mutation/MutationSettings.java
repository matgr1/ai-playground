package matgr.ai.genetic.mutation;

import matgr.ai.genetic.ValueRangeSettings;

public interface MutationSettings extends ValueRangeSettings{

    MutationType getNumericMutationType();

    double getNonUniformMutationAlpha();
    double getNonUniformMutationGenerationFactor();

    double getMaxMutation();
}
