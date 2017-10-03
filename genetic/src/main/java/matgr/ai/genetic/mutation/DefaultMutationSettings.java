package matgr.ai.genetic.mutation;

import matgr.ai.genetic.DefaultValueRangeSettings;

public class DefaultMutationSettings extends DefaultValueRangeSettings implements MutationSettings {

    public MutationType numericMutationType;
    public double nonUniformMutationAlpha;
    public double nonUniformMutationGenerationFactor;
    public double maxMutation;

    public DefaultMutationSettings(double valueRange,
                                   MutationType numericMutationType,
                                   double nonUniformMutationAlpha,
                                   double nonUniformMutationGenerationFactor,
                                   double maxMutation) {
        super(valueRange);
        this.numericMutationType = numericMutationType;
        this.nonUniformMutationAlpha = nonUniformMutationAlpha;
        this.nonUniformMutationGenerationFactor = nonUniformMutationGenerationFactor;
        this.maxMutation = maxMutation;
    }

    @Override
    public MutationType getNumericMutationType() {
        return numericMutationType;
    }

    @Override
    public double getNonUniformMutationAlpha() {
        return nonUniformMutationAlpha;
    }

    @Override
    public double getNonUniformMutationGenerationFactor() {
        return nonUniformMutationGenerationFactor;
    }

    @Override
    public double getMaxMutation() {
        return maxMutation;
    }
}
