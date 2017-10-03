package matgr.ai.neatsample.minesweepers;

public class MineStatus
{
    public final double distance;

    public final double angle;

    public final double score;

    public MineStatus(double distance, double angle, double score)
    {
        this.distance = distance;
        this.angle = angle;
        this.score = score;
    }
}
