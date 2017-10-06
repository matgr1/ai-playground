package matgr.ai.neatsample.minesweepers;

import matgr.ai.genetic.SpeciesMember;
import matgr.ai.math.DiscreteDistributionItem;
import matgr.ai.math.MathFunctions;
import matgr.ai.math.RandomFunctions;
import matgr.ai.neatsample.Point;
import matgr.ai.neatsample.Size;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.logging.Logger;

public abstract class MineSweeper<GenomeT extends MineSweeperGenome>
        implements SpeciesMember<GenomeT>, DiscreteDistributionItem {

    private final static Logger logger = Logger.getLogger(MineSweeper.class.getName());

    private Point position;
    private MineSweeperDirection direction;

    private MineSweeperFitnessState fitnessState;

    public final MineSweeperSettings settings;

    public final double visionConeDistanceSquared;
    public final double visionConeHalfAngle;

    public final GenomeT genome;

    public MineSweeper(RandomGenerator random, GenomeT genome, MineSweeperSettings settings) {

        this.settings = settings;

        this.fitnessState = new MineSweeperFitnessState(0);

        randomizePositionAndDirection(random, settings.minefieldSize);

        visionConeDistanceSquared = settings.visionConeDistance * settings.visionConeDistance;
        visionConeHalfAngle = settings.visionConeAngle / 2.0;

        this.genome = genome;
    }

    public void initializeOnBirth(Point position, MineSweeperDirection direction, long currentIteration) {

        this.fitnessState = new MineSweeperFitnessState(currentIteration);

        this.position = position;
        this.direction = direction;
    }

    public MineField getMineField() {
        return this.genome.getMineField();
    }

    public Point getPosition() {
        return position;
    }

    public MineSweeperDirection getDirection() {
        return direction;
    }

    public int getExplosionCount() {
        return fitnessState.getExplosionCount();
    }

    public int getClearedCount() {
        return fitnessState.getClearedCount();
    }

    public double getFitness() {
        return fitnessState.getFitness();
    }

    public void update(long currentIteration) {

        if (!fitnessState.isDigesting(settings, currentIteration)) {

            double minAngle = -visionConeHalfAngle;

            double degreesPerDivision = (settings.visionConeDivisions == 1)
                    ? settings.visionConeAngle
                    : settings.visionConeAngle / (settings.visionConeDivisions - 1);

            double halfDegreesPerDivision = degreesPerDivision / 2.0;

            List<MineStatus> closestMines = new ArrayList<>();

            // TODO: check all this
            double startAngle = minAngle;

            MineField mineField = getMineField();

            for (int i = 0; i < settings.visionConeDivisions; i++) {

                double endAngle = startAngle + degreesPerDivision;
                double angleOffset = (endAngle + startAngle) / 2.0;

                List<MineStatus> curClosestMines = getClosestVisibleMines(
                        position,
                        direction,
                        settings.visionConeDistance,
                        visionConeDistanceSquared,
                        halfDegreesPerDivision,
                        angleOffset,
                        mineField,
                        currentIteration,
                        settings.mineGestationPeriod);

                for (int j = 0; j < settings.minesPerVisionConeDivision; j++) {

                    if (j >= curClosestMines.size()) {
                        closestMines.add(new MineStatus(0.0, 0.0, 0.0));
                    } else {
                        closestMines.add(curClosestMines.get(j));
                    }

                }

                startAngle = endAngle;
            }

            SpeedAndRotation rawSpeedAndRotation = computeRawSpeedAndDirection(closestMines);
            double rawSpeed = rawSpeedAndRotation.speed;
            double rawRotation = rawSpeedAndRotation.rotation;

            double speed = computeSpeed(
                    rawSpeed,
                    settings.minSpeedForwards,
                    settings.maxSpeedForwards,
                    settings.minSpeedReverse,
                    settings.maxSpeedReverse);

            double rotation = computeRotation(rawRotation, settings.getMaxTurnRate());

            MineSweeperDirection newDirection = direction.rotate(rotation);
            Point newPosition = position.add(newDirection.vector.multiply(speed));

            //double newPositionX = newPosition.X;
            //double newPositionY = newPosition.Y;
            //if (newPositionX > mineField.size.Width) newPositionX = 0;
            //if (newPositionX < 0) newPositionX = mineField.size.Width;
            //if (newPositionY > mineField.size.Height) newPositionY = 0;
            //if (newPositionY < 0) newPositionY = mineField.size.Height;

            boolean pastEdge = false;

            double newPositionX = newPosition.x;
            double newPositionY = newPosition.y;
            if ((newPositionX > mineField.size.width) || (newPositionX < 0) ||
                    (newPositionY > mineField.size.height) || (newPositionY < 0)) {
                pastEdge = true;
            }

            if (pastEdge) {
                // TODO: allow the quicker turning?
                //// TODO: maybe do actual reflection instead...
                //rotation = computeRotation(rawRotation, settings.MinTurnRateAtEdge, settings.MaxTurnRateAtEdge);
                //
                //newDirection = direction.Rotate(rotation);
                //newPosition = position + (newDirection.Vector * speed);

                newPositionX = MathFunctions.clamp(newPosition.x, 0.0, mineField.size.width);
                newPositionY = MathFunctions.clamp(newPosition.y, 0.0, mineField.size.height);
            }

            position = new Point(newPositionX, newPositionY);
            direction = newDirection;
        }
    }

    public void onMineHit(long currentIteration, boolean explodeyMine) {
        fitnessState.onMineHit(settings, currentIteration, explodeyMine);
    }

    public void onMineNotHit(long currentIteration) {
        fitnessState.onMineNotHit(settings, currentIteration);
    }

    public boolean isVisible(Point location) {

        IsVisibleResult result = isVisible(
                position,
                direction,
                location,
                visionConeDistanceSquared,
                visionConeHalfAngle,
                0.0);

        return result.visible;
    }

    @Override
    public double getValue() {
        return getFitness();
    }

    private void randomizePositionAndDirection(RandomGenerator random, Size minefieldSize) {

        double positionX = RandomFunctions.nextDouble(random, 0.0, minefieldSize.width);
        double positionY = RandomFunctions.nextDouble(random, 0.0, minefieldSize.height);
        position = new Point(positionX, positionY);

        MineSweeperDirection direction = new MineSweeperDirection();
        this.direction = direction.rotate(RandomFunctions.nextDouble(random, 0.0, 2.0 * Math.PI));
    }

    private static List<MineStatus> getClosestVisibleMines(Point organismPosition,
                                                           MineSweeperDirection organismDirection,
                                                           double coneDistance,
                                                           double coneDistanceSquared,
                                                           double coneHalfAngle,
                                                           double coneAngleOffset,
                                                           MineField mineField,
                                                           long currentIteration,
                                                           int mineGestationPeriod) {
        List<MineStatus> mineStatuses = new ArrayList<>();

        for (Mine mine : mineField.mines) {

            IsVisibleResult visible = isVisible(
                    organismPosition,
                    organismDirection,
                    mine.location,
                    coneDistanceSquared,
                    coneHalfAngle,
                    coneAngleOffset);

            if (visible.visible) {

                double score = getMineScore(
                        mine,
                        visible.distance,
                        coneDistance,
                        currentIteration,
                        mineGestationPeriod,
                        false);

                MineStatus status = new MineStatus(visible.distance, visible.angle, score);
                mineStatuses.add(status);
            }
        }

        for (Mine mine : mineField.explodeyMines) {

            IsVisibleResult visible = isVisible(
                    organismPosition,
                    organismDirection,
                    mine.location,
                    coneDistanceSquared,
                    coneHalfAngle,
                    coneAngleOffset);

            if (visible.visible) {

                double score = getMineScore(
                        mine,
                        visible.distance,
                        coneDistance,
                        currentIteration,
                        mineGestationPeriod,
                        true);

                MineStatus status = new MineStatus(visible.distance, visible.angle, score);
                mineStatuses.add(status);
            }
        }

        // TODO: make sure this sorts right
        // TODO: would be better if to pas ALL of mines in the cone... how to do varying number of inputs? or
        //       is there another way to represent them?
        mineStatuses.sort(Comparator.comparingDouble(a -> a.distance));

        return mineStatuses;
    }

    private static IsVisibleResult isVisible(
            Point organismPosition,
            MineSweeperDirection organismDirection,
            Point location,
            double coneDistanceSquared,
            double coneHalfAngle,
            double coneAngleOffset) {

        MineSweeperDirection offsetDirection = organismDirection.rotate(coneAngleOffset);

        Point organismToLocation = location.subtract(organismPosition);
        double distanceToMineSquared = organismToLocation.lengthSquared();

        if (distanceToMineSquared > coneDistanceSquared) {
            return new IsVisibleResult(false, 0, 0, 0);
        }

        // TODO: make sure this is correct
        double dotProduct = offsetDirection.vector.dotProduct(organismToLocation);
        double cosAngle = dotProduct / (offsetDirection.vector.length() * organismToLocation.length());

        // TODO: pass in the cos of the angle
        if (cosAngle < Math.cos(coneHalfAngle)) {
            return new IsVisibleResult(false, 0, 0, 0);
        }

        double distance = organismToLocation.length();
        double angle = MathFunctions.wrapAngle(Math.atan2(organismToLocation.y, organismToLocation.x));

        return new IsVisibleResult(true, distance, distanceToMineSquared, angle);
    }

    private static double getMineScore(Mine mine,
                                       double distanceToMine,
                                       double visionConeDistance,
                                       long currentIteration,
                                       int gestationPeriod,
                                       boolean explodyMine) {

        Long iterationsTilActive = mine.getIterationsUntilActive(currentIteration, gestationPeriod);

        if ((null == iterationsTilActive) || (distanceToMine > visionConeDistance)) {
            return 0.0;
        }

        double distanceScore = 1.0 - MathFunctions.clamp(distanceToMine / visionConeDistance, 0.0, 1.0);

        double lifeScore;

        if (gestationPeriod == 0) {
            lifeScore = 1.0;
        } else {
            lifeScore = 1.0 - MathFunctions.clamp((double) iterationsTilActive / (double) gestationPeriod, 0.0, 1.0);
        }

        double score = distanceScore * lifeScore;

        if (explodyMine) {
            return -score;
        }


        return score;
    }

    private static double computeSpeed(double rawSpeed,
                                       double minSpeedForwards,
                                       double maxSpeedForwards,
                                       double minSpeedReverse,
                                       double maxSpeedReverse) {

        if (Double.isNaN(rawSpeed)) {
            logger.warning("Received NaN raw speed value");
            return 0.0;
        }

        if (Double.isInfinite(rawSpeed)) {

            logger.fine("Received infinite raw speed value");

            if (rawSpeed > 0.0) {
                return maxSpeedForwards;
            }

            if (rawSpeed < 0.0) {
                return -maxSpeedReverse;
            }
        }

        final double rawValueMin = 0.0;
        final double rawValueMax = 1.0;
        final double rawValueRange = rawValueMax - rawValueMin;

        double minSpeed = -maxSpeedReverse;
        double outputRange = maxSpeedForwards - minSpeed;

        double slope = outputRange / rawValueRange;
        double intercept = minSpeed - (slope * rawValueMin);

        double value = (slope * rawSpeed) + intercept;

        if (value >= 0) {
            value = MathFunctions.clamp(value, minSpeedForwards, maxSpeedForwards);
        } else {
            value = MathFunctions.clamp(value, -maxSpeedReverse, -minSpeedReverse);
        }

        return value;
    }

    private static double computeRotation(double rawRotation, double maxRotation) {

        if (Double.isNaN(rawRotation)) {
            logger.warning("Received NaN raw rotation value");
            return 0.0;
        }

        if (Double.isInfinite(rawRotation)) {

            logger.fine("Received infinite raw speed value");

            if (rawRotation > 0.0) {
                return maxRotation;
            }

            if (rawRotation < 0.0) {
                return -maxRotation;
            }
        }

        final double rawValueMin = 0.0;
        final double rawValueMax = 1.0;
        final double rawValueRange = rawValueMax - rawValueMin;

        double minRotation = -maxRotation;
        double outputRange = maxRotation - minRotation;

        double slope = outputRange / rawValueRange;
        double intercept = minRotation - (slope * rawValueMin);

        return (slope * rawRotation) + intercept;
    }

    protected abstract List<Double> activateNeuralNet(List<Double> inputs, double bias);

    private SpeedAndRotation computeRawSpeedAndDirection(List<MineStatus> closestMines) {

        List<Double> inputsList = new ArrayList<>();

        // add position
        inputsList.add(position.x);
        inputsList.add(position.y);

        // add direction
        inputsList.add(direction.vector.x);
        inputsList.add(direction.vector.y);

        for (MineStatus mineStatus : closestMines) {
            // add mine
            inputsList.add(mineStatus.angle);
            inputsList.add(mineStatus.score);
        }

        // TODO: reset or not? also how many steps per activation?
        List<Double> outputs = activateNeuralNet(inputsList, settings.bias);

        double rawSpeed = outputs.get(0);
        double rawRotation = outputs.get(1);

        return new SpeedAndRotation(rawSpeed, rawRotation);
    }

    private static class SpeedAndRotation {
        public final double speed;
        public final double rotation;

        private SpeedAndRotation(double speed, double rotation) {
            this.speed = speed;
            this.rotation = rotation;
        }
    }

    private static class IsVisibleResult {
        public final boolean visible;
        public final double distance;
        public final double distanceSquared;
        public final double angle;

        private IsVisibleResult(boolean visible, double distance, double distanceSquared, double angle) {
            this.visible = visible;
            this.distance = distance;
            this.distanceSquared = distanceSquared;
            this.angle = angle;
        }
    }
}
