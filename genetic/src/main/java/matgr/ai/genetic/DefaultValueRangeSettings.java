package matgr.ai.genetic;

public class DefaultValueRangeSettings implements ValueRangeSettings {

    public double valueRange;

    public DefaultValueRangeSettings(double valueRange) {
        this.valueRange = valueRange;
    }

    @Override
    public double getValueRange() {
        return valueRange;
    }
}
