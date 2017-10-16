package matgr.ai.neuralnet.feedforward;

public class ConvolutionalLayerSizes {

    public final int inputWidth;
    public final int inputHeight;

    public final int kernelRadiusX;
    public final int kernelRadiusY;

    public final int kernelWidth;
    public final int kernelHeight;

    public final int outputWidth;
    public final int outputHeight;

    public ConvolutionalLayerSizes(int inputWidth,
                                   int inputHeight,
                                   int kernelRadiusX,
                                   int kernelRadiusY) {

        this.kernelRadiusX = kernelRadiusX;
        this.kernelRadiusY = kernelRadiusY;

        this.kernelWidth = (kernelRadiusX * 2) + 1;
        this.kernelHeight = (kernelRadiusY * 2) + 1;

        if (this.kernelWidth > inputWidth) {
            throw new IllegalArgumentException("Kernel inputWidth cannot be larger than inputWidth");
        }
        if (this.kernelHeight > inputHeight) {
            throw new IllegalArgumentException("Kernel inputHeight cannot be larger than inputHeight");
        }

        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;

        this.outputWidth = 1 + (this.inputWidth - this.kernelWidth);
        this.outputHeight = 1 + (this.inputHeight - this.kernelHeight);
    }

    public ConvolutionalLayerSizes deepClone() {
        return new ConvolutionalLayerSizes(inputWidth, inputHeight, kernelRadiusX, kernelRadiusY);
    }
}
