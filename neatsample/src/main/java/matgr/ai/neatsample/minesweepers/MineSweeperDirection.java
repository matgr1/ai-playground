package matgr.ai.neatsample.minesweepers;

import matgr.ai.math.MathFunctions;
import matgr.ai.neatsample.Vector;

public class MineSweeperDirection {
    private static final double DEFAULT_ANGLE = 0.0;

    public double angleRadians;

    public double angleDegrees;

    public Vector vector;

    public MineSweeperDirection() {
        this(DEFAULT_ANGLE, computeVector(DEFAULT_ANGLE));
    }

    private MineSweeperDirection(double angleRadians, Vector vector) {
        this.angleRadians = angleRadians;
        this.angleDegrees = MathFunctions.radiansToDegrees(angleRadians);
        this.vector = vector;
    }

    public MineSweeperDirection rotate(double radians) {
        double newAngleRadians = MathFunctions.wrapAngle(angleRadians + radians);
        Vector newVector = computeVector(newAngleRadians);

        return new MineSweeperDirection(newAngleRadians, newVector);
    }

    private static Vector computeVector(double angleRadians) {
        double newVectorX = Math.cos(angleRadians);
        double newVectorY = Math.sin(angleRadians);

        return new Vector(newVectorX, newVectorY);
    }
}
