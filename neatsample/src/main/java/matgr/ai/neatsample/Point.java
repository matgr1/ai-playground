package matgr.ai.neatsample;

public class Point implements CartesianCoordinate<Point> {

    public double x;
    public double y;

    public Point(double x, double y) {
        this.x = x;
        this.y = y;
    }

    @Override
    public double getX() {
        return x;
    }

    @Override
    public void setX(double value) {
        x = value;
    }

    @Override
    public double getY() {
        return y;
    }

    @Override
    public void setY(double value) {
        y = value;
    }

    @Override
    public Point clone() {
        return new Point(x, y);
    }
}