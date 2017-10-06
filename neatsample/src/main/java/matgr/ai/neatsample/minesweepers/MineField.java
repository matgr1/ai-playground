package matgr.ai.neatsample.minesweepers;

import matgr.ai.math.RandomFunctions;
import matgr.ai.neatsample.Point;
import matgr.ai.neatsample.Size;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MineField {
    private final List<Mine> writableMines;
    private final List<Mine> writableExplodeyMines;

    public Size size;

    public final List<Mine> mines;
    public final List<Mine> explodeyMines;

    public MineField(RandomGenerator random, MineSweeperSettings settings) {
        size = settings.minefieldSize;

        writableMines = new ArrayList<>();
        mines = Collections.unmodifiableList(writableMines);

        writableExplodeyMines = new ArrayList<>();
        explodeyMines = Collections.unmodifiableList(writableExplodeyMines);

        for (int i = 0; i < settings.mineCount; i++) {
            Point location = randomLocation(random);
            Mine mine = new Mine(location, 0);

            writableMines.add(mine);
        }

        int explodeyMineCount = settings.getExplodeyMineCount();

        for (int i = 0; i < explodeyMineCount; i++) {
            Point location = randomLocation(random);
            Mine mine = new Mine(location, 0);

            writableExplodeyMines.add(mine);
        }
    }

    public int isHit(long currentIteration, MineSweeperSettings settings, Point mineSweeperPosition) {
        for (int i = 0; i < mines.size(); i++) {
            Mine mine = mines.get(i);

            double mineRadius = settings.getMineRadius();

            if (mine.isActive(currentIteration, settings.mineGestationPeriod)) {
                if (isHit(mineRadius, mine.location, settings.mineSweeperRadius, mineSweeperPosition)) {
                    return i;
                }
            }
        }

        return -1;
    }

    // public void removeMine(int index){
    //     if ((index < 0) || (index >= mines.size())) {
    //         throw new IllegalArgumentException("index is out of range");
    //     }

    //     writableMines.remove(index);
    // }

    public void replaceMine(RandomGenerator random, int index, long currentIteration) {
        if ((index < 0) || (index >= mines.size())) {
            throw new IllegalArgumentException("index is out of range");
        }

        Point location = randomLocation(random);
        Mine mine = new Mine(location, currentIteration);

        writableMines.set(index, mine);
    }

    public int isExplodeyHit(long currentIteration, MineSweeperSettings settings, Point mineSweeperPosition) {
        for (int i = 0; i < explodeyMines.size(); i++) {
            Mine mine = explodeyMines.get(i);

            double mineRadius = settings.getMineRadius();

            if (mine.isActive(currentIteration, settings.mineGestationPeriod)) {
                if (isHit(mineRadius, mine.location, settings.mineSweeperRadius, mineSweeperPosition)) {
                    return i;
                }
            }
        }

        return -1;
    }

    public void replaceExplodeyMine(RandomGenerator random, int index, long currentIteration) {
        if ((index < 0) || (index >= explodeyMines.size())) {
            throw new IllegalArgumentException("index");
        }

        Point location = randomLocation(random);
        Mine mine = new Mine(location, currentIteration);

        writableExplodeyMines.set(index, mine);
    }

    private static boolean isHit(double mineRadius, Point minePosition, double mineSweeperRadius, Point mineSweeperPosition) {
        Point sweeperToMine = minePosition.subtract(mineSweeperPosition);

        double distance = sweeperToMine.length();

        if (distance < (mineRadius + mineSweeperRadius)) {
            return true;
        }

        return false;
    }

    private Point randomLocation(RandomGenerator random) {
        double locationX = RandomFunctions.nextDouble(random, 0.0, size.width);
        double locationY = RandomFunctions.nextDouble(random, 0.0, size.height);

        return new Point(locationX, locationY);
    }
}
