package matgr.ai.neatsample;

public interface CartesianCoordinate<CoordinateT extends CartesianCoordinate<CoordinateT>> extends Cloneable {

    double getX();
    void setX(double value);

    double getY();
    void setY(double value);

    default <OtherT extends CartesianCoordinate> CoordinateT add(OtherT other) {

        CoordinateT result = clone();

        result.setX(this.getX() + other.getX());
        result.setY(this.getY() + other.getY());

        return result;
    }

    default <OtherT extends CartesianCoordinate> CoordinateT subtract(OtherT other) {

        CoordinateT result = clone();

        result.setX(this.getX() - other.getX());
        result.setY(this.getY() - other.getY());

        return result;

    }

    default CoordinateT multiply(double value) {

        CoordinateT result = clone();

        result.setX(this.getX() * value);
        result.setY(this.getY() * value);

        return result;

    }

    default <OtherT extends CartesianCoordinate> CoordinateT multiply(OtherT other) {

        CoordinateT result = clone();

        result.setX(this.getX() * other.getX());
        result.setY(this.getY() * other.getY());

        return result;

    }

    default CoordinateT divide(double value) {

        CoordinateT result = clone();

        result.setX(this.getX() / value);
        result.setY(this.getY() / value);

        return result;

    }

    default <OtherT extends CartesianCoordinate> CoordinateT divide(OtherT other) {

        CoordinateT result = clone();

        result.setX(this.getX() / other.getX());
        result.setY(this.getY() / other.getY());

        return result;

    }

    default double length() {
        return length(null);
    }

    default <OtherT extends CartesianCoordinate> double length(OtherT other) {
        double length = lengthSquared(other);
        return Math.sqrt(length);
    }

    default double lengthSquared() {
        return lengthSquared(null);
    }

    default <OtherT extends CartesianCoordinate> double lengthSquared(OtherT other) {

        CartesianCoordinate tmp = this;

        if (other != null) {
            tmp = other.subtract(this);
        }

        return tmp.dotProduct(tmp);
    }

    default <OtherT extends CartesianCoordinate> double dotProduct(OtherT other) {

        if (other == null) {
            throw new IllegalArgumentException("other coordinate not supplied");
        }

        return (this.getX() * other.getX()) + (this.getY() * other.getY());
    }

    default <OtherT extends CartesianCoordinate> double crossProduct(OtherT other) {

        if (other == null) {
            throw new IllegalArgumentException("other coordinate not supplied");
        }

        return (this.getX() * other.getY()) - (this.getY() * other.getX());
    }

    CoordinateT clone();

}
