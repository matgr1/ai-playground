package matgr.ai.neatsample;

import javafx.embed.swing.SwingNode;

public class NonResizableSwingNode extends SwingNode {
    public boolean isResizable() {
        return false;
    }
}