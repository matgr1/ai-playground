package matgr.ai.math;

public class MathFunctions {

    private static final double TWO_PI = Math.PI * 2.0;

    public static final double DEFAULT_FUZZY_DELTA = 100000.0;

    public static double radiansToDegrees(double radians) {
        return (radians * 180) / Math.PI;
    }

    public static double degreesToRadians(double degrees) {
        return (degrees * Math.PI) / 180;
    }

    public static double wrapAngle(double radians) {
        return radians - (TWO_PI * Math.floor(radians / TWO_PI));
    }

    public static <T extends Comparable<T>> T clamp(T value, T min, T max) {

        if (value.compareTo(min) < 0) {
            value = min;
        }

        if (value.compareTo(max) > 0) {
            value = max;
        }

        return value;

    }

    public static boolean fuzzyCompare(double a, double b) {

        return fuzzyCompare(a, b, DEFAULT_FUZZY_DELTA);

    }

    public static boolean fuzzyCompare(double a, double b, double fuzzyDelta) {

        if (a == b) {
            return true;
        }

        if ((a == 0.0) || (b == 0.0)) {
            a += 1.0;
            b += 1.0;
        }

        double differenceAbs = Math.abs(a - b);
        double minAbs = Math.min(Math.abs(a), Math.abs(b));

        return (differenceAbs * fuzzyDelta) <= minAbs;

    }

    public static double sigmoid(double d) {
        return 1.0 / (1.0 + Math.exp(-d));
    }
}
