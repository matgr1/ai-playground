package matgr.ai.neatsample.minesweepers;

import matgr.ai.math.MathFunctions;

class MineSweeperFitnessState {

    // TODO: make these settings
    private static final double baselineBadFitnessAdjustment = -1.0;
    private static final double badFitnessStreakAdjustment = -0.2;

    // TODO: make these settings
    private static final double baselineGoodFitnessAdjustment = 1.0;
    private static final double goodFitnessStreakAdjustment = 0.2;

    private long lastGoodHit;
    private long lastExplosion;

    private double fitness;

    private int streak;

    private int cleared;
    private int exploded;

    public MineSweeperFitnessState(long currentIteration) {
        lastGoodHit = currentIteration;
        lastExplosion = -1;
        streak = 0;

        cleared = 0;
        exploded = 0;
    }

    public int getExplosionCount() {
        return exploded;
    }

    public int getClearedCount() {
        return cleared;
    }

    public double getFitness() {
        return fitness;
    }

    public void onMineHit(MineSweeperSettings settings, long currentIteration, boolean explodeyMine) {

       if (explodeyMine) {

           exploded++;

           lastExplosion = currentIteration;

           if (!lastHitWasExplosion()) {
               streak = 0;
           }

           double fitnessAdjustment = getBadFitnessAdjustment(streak);
           fitness = fitness + fitnessAdjustment;

           streak++;

       } else {

           cleared++;

           lastGoodHit = currentIteration;

           if (lastHitWasExplosion()) {
               streak = 0;
           }

           double fitnessAdjustment = getGoodFitnessAdjustment(streak);
           fitness = fitness + fitnessAdjustment;

           streak++;
       }

       double fitnessAdjustment = getStarvationFitnessAdjustment(
               currentIteration,
               lastGoodHit,
               settings.getDigestionPeriod(),
               settings.maxStarvationFitnessAdjustment,
               settings.getStarvationPeriod(),
               settings.starvationSteepness);

       fitness = fitness + fitnessAdjustment;
    }

    public void onMineNotHit(MineSweeperSettings settings, long currentIteration) {

       double fitnessAdjustment = getStarvationFitnessAdjustment(
               currentIteration,
               lastGoodHit,
               settings.getDigestionPeriod(),
               settings.maxStarvationFitnessAdjustment,
               settings.getStarvationPeriod(),
               settings.starvationSteepness);

       fitness = fitness + fitnessAdjustment;
    }

    public boolean isDigesting(MineSweeperSettings settings, long currentIteration) {

        int digestionPeriod = settings.getDigestionPeriod();

        long lastHit = Math.max(lastExplosion, lastGoodHit);

        if (lastHit < 0) {
            return false;
        }

        if ((lastHit + digestionPeriod) < currentIteration) {
            return false;
        }

        return true;
    }

   private double getGoodFitnessAdjustment(int streak) {
       return baselineGoodFitnessAdjustment + (streak * goodFitnessStreakAdjustment);
   }

   private double getBadFitnessAdjustment(int streak) {
       return baselineBadFitnessAdjustment + (streak * badFitnessStreakAdjustment);
   }

   private double getStarvationFitnessAdjustment(long currentIteration,
                                                 long lastGoodHit,
                                                 long digestionPeriod,
                                                 double maxStarvationFitnessAdjustment,
                                                 double starvationPeriod,
                                                 double starvationSteepness) {

       if (lastGoodHit < 0) {
           lastGoodHit = 0;
       }

       long timeSinceMealComplete = Math.max(0, currentIteration - (lastGoodHit + digestionPeriod));
       return maxStarvationFitnessAdjustment * MathFunctions.sigmoid(starvationSteepness * (timeSinceMealComplete - starvationPeriod));
   }

   private boolean lastHitWasExplosion() {

       if (lastExplosion < 0) {
           return false;
       }

       return lastExplosion > lastGoodHit;
   }
}
