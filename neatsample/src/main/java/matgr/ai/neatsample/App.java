package matgr.ai.neatsample;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class App extends Application {

    public static void main(String[] args) {

        // TODO: without this, JGraphT eventually blows up... find a better way, this is terrible... (also, this may
        //       need to be passed on the commandline if something causes Arrays to be initialized before this is
        //       called... look for errors coming from JGraph about comparisons violating contracts...)
        System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");

        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {

        Parent root = FXMLLoader.load(getClass().getResource("/fxml/App.fxml"));

        primaryStage.setTitle("NEAT");
        primaryStage.setScene(new Scene(root, 300, 275));
        primaryStage.show();
    }
}
