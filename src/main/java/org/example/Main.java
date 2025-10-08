package org.example;

import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.ndarray.*;
import ai.djl.repository.zoo.*;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.*;
import ai.djl.inference.Predictor;

import java.io.IOException;
import java.nio.file.Paths;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

// Run with:
// mvn exec:java -Dexec.mainClass="org.example.Main"

class Input {
    float[][] data;  // shape: [450, 11]
    float[] mask;    // shape: [450]
    float[] y;
    float[] ogpred;


    public Input(float[][] data, float[] mask, float[] y, float[] ogpred) {
        this.data = data;
        this.mask = mask;
        this.y = y;
        this.ogpred = ogpred;
    }
}


class MyTranslator implements Translator<Input, float[][]> {

    @Override
    public NDList processInput(TranslatorContext ctx, Input input) {
        NDManager manager = ctx.getNDManager();

        NDArray x = manager.create(input.data).reshape(1, 450, 11);
        NDArray maskNd = manager.create(input.mask).reshape(1, 450);

        return new NDList(x, maskNd);
    }

    @Override
    public float[][] processOutput(TranslatorContext ctx, NDList list) {
        NDArray output = list.singletonOrThrow();  // shape: (1, 450, C) or (1, 450)
        output = output.squeeze(0);                // remove batch dim -> (450, C)

        float[] flat = output.toFloatArray();
        long[] shape = output.getShape().getShape();
        int dim0 = (int) shape[0];
        int dim1 = shape.length > 1 ? (int) shape[1] : 1;

        float[][] result = new float[dim0][dim1];
        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                result[i][j] = flat[i * dim1 + j];
            }
        }
        return result;
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }
}

public class Main {

      public static Input loadExampleFromText(String path) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        ArrayList<float[]> xList = new ArrayList<>();
        ArrayList<Float> yList = new ArrayList<>();
        ArrayList<Float> maskList = new ArrayList<>();
        ArrayList<Float> probsList = new ArrayList<>();  // optional if you want probs

        // State tracking
        String section = "";
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;

            if (line.equals("x:")) {
                section = "x";
                continue;
            } else if (line.equals("y:")) {
                section = "y";
                continue;
            } else if (line.equals("mask:")) {
                section = "mask";
                continue;
            } else if (line.equals("probs:")) {
                section = "probs";
                continue;
            }

            switch (section) {
                case "x":
                    String[] parts = line.split("\\s+");
                    float[] row = new float[parts.length];
                    for (int i = 0; i < parts.length; i++) {
                        row[i] = Float.parseFloat(parts[i]);
                    }
                    xList.add(row);
                    break;
                case "y":
                    yList.add(Float.parseFloat(line));
                    break;
                case "mask":
                    maskList.add(Float.parseFloat(line));
                    break;
                case "probs":
                    probsList.add(Float.parseFloat(line));
                    break;
            }
        }
        br.close();

        // Convert ArrayList to arrays
        float[][] data = xList.toArray(new float[0][]);
        float[] mask = new float[maskList.size()];
        for (int i = 0; i < mask.length; i++) mask[i] = maskList.get(i);

        float[] y = new float[yList.size()];
        for (int i = 0; i < y.length; i++) y[i] = yList.get(i);
        float[] prob = new float[probsList.size()];
        for (int i = 0; i < prob.length; i++) prob[i] = probsList.get(i);

        return new Input(data, mask, y, prob);
    }

    public static void main(String[] args) {

        MyTranslator myTranslator = new MyTranslator();

        Criteria<Input, float[][]> myCriteria = Criteria.builder()
                .setTypes(Input.class, float[][].class)
                .optModelPath(Paths.get("nets/classifier_torchscript_sector1_weightInTraining.pt"))
                .optEngine("PyTorch")
                .optTranslator(myTranslator)
                .optProgress(new ProgressBar())
                .build();

        System.out.println("Loading model...");

        try (ZooModel<Input, float[][]> model = myCriteria.loadModel();
             Predictor<Input, float[][]> predictor = model.newPredictor()) {

            Input input = loadExampleFromText("nets/example_sector1_weightInTraining.txt");

            System.out.println("Predicting...");
            float[][] preds = predictor.predict(input);

            System.out.println("Got " + preds.length + " predictions (one per hit).");

            for (int i = 0; i < 450; i++) {  // show first 5 hits
                System.out.print("Hit " + i + ": ");
                System.out.printf("y %.4f ", input.y[i]);
                System.out.printf("mask %.4f ", input.mask[i]);
                for (float p : preds[i]) {
                    System.out.printf("pred: %.4f ", p);
                }

                
                System.out.printf("pytorch pred %.4f ", input.ogpred[i]);
                
                System.out.println();
            }

        } catch (IOException | TranslateException | ModelException e) {
            e.printStackTrace();
        }
    }
}
