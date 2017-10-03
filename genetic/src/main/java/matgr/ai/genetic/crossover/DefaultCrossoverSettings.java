package matgr.ai.genetic.crossover;

import matgr.ai.genetic.DefaultValueRangeSettings;

public class DefaultCrossoverSettings extends DefaultValueRangeSettings implements CrossoverSettings {

    public CrossoverType numericCrossoverType;

    public DefaultCrossoverSettings(double valueRange, CrossoverType numericCrossoverType) {
        super(valueRange);
        this.numericCrossoverType = numericCrossoverType;
    }

    @Override
    public CrossoverType getNumericCrossoverType() {
        return numericCrossoverType;
    }
}
