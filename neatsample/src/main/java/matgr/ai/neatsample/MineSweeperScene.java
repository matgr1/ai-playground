package matgr.ai.neatsample;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import matgr.ai.math.MathFunctions;
import matgr.ai.neatsample.minesweepers.Mine;
import matgr.ai.neatsample.minesweepers.MineField;
import matgr.ai.neatsample.minesweepers.MineSweeper;
import matgr.ai.neatsample.minesweepers.MineSweeperSettings;

import java.util.List;

public class MineSweeperScene {

    private static final Color inactiveMineColor = Color.YELLOW;
    private static final Color mineLineColor = Color.GREEN;

    private static final Color activeMineColor = Color.GREEN;
    private static final Color activeExplodeyMineColor = Color.RED;

    private static final Color sweeperLineColor = Color.GREEN;

    public static void draw(Size canvasSize,
                            GraphicsContext graphics,
                            MineSweeper mineSweeper,
                            MineSweeperSettings settings,
                            long currentIteration) {

        graphics.clearRect(0, 0, canvasSize.width, canvasSize.height);

        MineField mineField = mineSweeper.getMineField();

        if ((canvasSize.width > 0.0) && (canvasSize.width < Double.POSITIVE_INFINITY) &&
                (canvasSize.height > 0.0) && (canvasSize.height < Double.POSITIVE_INFINITY)) {

            // draw mines
            drawMines(
                    canvasSize,
                    graphics,
                    settings,
                    mineField.mines,
                    mineSweeper,
                    settings.highlightMinesInView,
                    inactiveMineColor,
                    activeMineColor,
                    mineLineColor,
                    currentIteration);

            drawMines(
                    canvasSize,
                    graphics,
                    settings,
                    mineField.explodeyMines,
                    mineSweeper,
                    settings.highlightMinesInView,
                    inactiveMineColor,
                    activeExplodeyMineColor,
                    mineLineColor,
                    currentIteration);

            // draw sweepers...

            Vector sizeVector = new Vector(canvasSize.width, canvasSize.height);

            double lineThickness = canvasSize.width * settings.lineWidth();

            double sweeperRadius = canvasSize.width * settings.mineSweeperRadius;
            double sweeperDiameter = sweeperRadius * 2;

            Vector sweeperSizeVector = new Vector(sweeperDiameter, sweeperDiameter);
            Vector sweeperHalfSizeVector = sweeperSizeVector.multiply(0.5);
            Vector negativeSweeperHalfSizeVector = sweeperHalfSizeVector.multiply(-1.0);

            Vector sweeperHeadOffsetVector = new Vector(sweeperRadius, 0);

            Vector sweeperHeadSizeVector = sweeperHalfSizeVector;
            Vector sweeperHeadHalfSizeVector = sweeperHeadSizeVector.multiply(0.5);
            Vector negativeSweeperHeadHalfSizeVector = sweeperHeadHalfSizeVector.multiply(-1.0);

            double visionConeLength = canvasSize.width * settings.visionConeDistance;
            double visionConeAngleDegrees = MathFunctions.radiansToDegrees(settings.visionConeAngle);
            double halfVisionConeAngleDegrees = visionConeAngleDegrees * 0.5;

            Color curSweeperColor = Color.WHITE;

            graphics.save();

            graphics.setFill(curSweeperColor);
            graphics.setStroke(sweeperLineColor);

            graphics.setLineWidth(lineThickness);

            Point sweeperLocationCenter = mineSweeper.getPosition().multiply(sizeVector);

            // draw body
            graphics.save();

            graphics.translate(sweeperLocationCenter.x, sweeperLocationCenter.y);
            graphics.rotate(mineSweeper.getDirection().angleDegrees);
            graphics.translate(negativeSweeperHalfSizeVector.x, negativeSweeperHalfSizeVector.y);

            graphics.fillOval(0, 0, sweeperSizeVector.x, sweeperSizeVector.y);
            graphics.strokeOval(0, 0, sweeperSizeVector.x, sweeperSizeVector.y);

            graphics.restore();

            // draw head
            graphics.save();

            graphics.translate(sweeperLocationCenter.x, sweeperLocationCenter.y);
            graphics.rotate(mineSweeper.getDirection().angleDegrees);
            graphics.translate(sweeperHeadOffsetVector.x, sweeperHeadOffsetVector.y);
            graphics.translate(negativeSweeperHeadHalfSizeVector.x, negativeSweeperHeadHalfSizeVector.y);

            graphics.fillOval(0, 0, sweeperHeadSizeVector.x, sweeperHeadSizeVector.y);
            graphics.strokeOval(0, 0, sweeperHeadSizeVector.x, sweeperHeadSizeVector.y);

            graphics.restore();

            // draw draw vision cone line 1 (for sweeper best only)
            graphics.save();

            graphics.translate(sweeperLocationCenter.x, sweeperLocationCenter.y);
            graphics.rotate(mineSweeper.getDirection().angleDegrees);
            graphics.rotate(halfVisionConeAngleDegrees);

            graphics.strokeLine(0.0, 0.0, visionConeLength, 0.0);

            graphics.restore();

            // draw draw vision cone line 2 (for sweeper best only)
            graphics.save();

            graphics.translate(sweeperLocationCenter.x, sweeperLocationCenter.y);
            graphics.rotate(mineSweeper.getDirection().angleDegrees);
            graphics.rotate(-halfVisionConeAngleDegrees);

            graphics.strokeLine(0.0, 0.0, visionConeLength, 0.0);

            graphics.restore();

            graphics.restore();
        }
    }

    private static void drawMines(Size canvasSize,
                                  GraphicsContext graphics,
                                  MineSweeperSettings settings,
                                  List<Mine> mines,
                                  MineSweeper mineSweeper,
                                  boolean highlightMinesInView,
                                  Color inactiveColor,
                                  Color activeColor,
                                  Color lineColor,
                                  long currentIteration) {

        double mineRadius = canvasSize.width * settings.getMineRadius();
        double mineDiameter = mineRadius * 2;

        Vector mineSizeVector = new Vector(mineDiameter, mineDiameter);
        Vector mineHalfSizeVector = mineSizeVector.multiply(0.5);

        double lineThickness = canvasSize.width * settings.lineWidth();

        Vector sizeVector = new Vector(canvasSize.width, canvasSize.height);

        for (Mine mine : mines) {

            boolean highlightMine = false;

            if (highlightMinesInView) {

                if (mineSweeper.isVisible(mine.location)) {
                    highlightMine = true;
                }
            }

            Color curMineColor;
            if (mine.isActive(currentIteration, settings.mineGestationPeriod)) {
                curMineColor = activeColor;
            } else {
                curMineColor = inactiveColor;
            }

            Color curLineColor = lineColor;
            double curLineThickness = lineThickness;

            if (highlightMine) {
                curMineColor = brighten(curMineColor, 0.2);
                curLineColor = brighten(curLineColor, 0.2);
                curLineThickness *= 2.0;
            }

            graphics.save();

            graphics.setFill(curMineColor);
            graphics.setStroke(curLineColor);

            graphics.setLineWidth(curLineThickness);

            Point mineLocationCenter = mine.location.multiply(sizeVector);
            Point mineLocationTopLeft = mineLocationCenter.subtract(mineHalfSizeVector);

            graphics.translate(mineLocationTopLeft.x, mineLocationTopLeft.y);

            graphics.fillOval(0, 0, mineSizeVector.x, mineSizeVector.y);
            graphics.strokeOval(0, 0, mineSizeVector.x, mineSizeVector.y);

            graphics.restore();
        }
    }

    // TODO: would it be better to change lightness instead of brightness?
    private static Color brighten(Color color, double amount) {

        double curBrightness = color.getBrightness();
        double brightnessFactor = (curBrightness + amount) / curBrightness;

        return color.deriveColor(0, 1.0, brightnessFactor, 1.0);
    }
}
