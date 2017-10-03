package matgr.ai.neatsample;

public class Vector implements CartesianCoordinate<Vector> {

    public double x;
    public double y;

    public Vector(double x, double y) {
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
    public Vector clone() {
        return new Vector(x, y);
    }
}