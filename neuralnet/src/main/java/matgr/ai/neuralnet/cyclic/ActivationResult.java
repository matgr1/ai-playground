package matgr.ai.neuralnet.cyclic;

import java.util.List;

public class ActivationResult {

    public final ActivationResultCode resultCode;

    public final List<Double> outputs;

    public ActivationResult(ActivationResultCode resultCode, List<Double> outputs){
        this.resultCode = resultCode;
        this.outputs = outputs;
    }

    public boolean isSuccess(){
        return resultCode == ActivationResultCode.Success;
    }
}
