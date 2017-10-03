package matgr.ai.math;

public class DistributionStats {

    public final int count;

    public final double min;

    public final double max;

    public final double total;

    public final double average;

    public DistributionStats(int count, double min, double max, double total) {

        this.count = count;
        this.min = min;
        this.max = max;
        this.total = total;

        this.average = total / (double) count;

    }

}
