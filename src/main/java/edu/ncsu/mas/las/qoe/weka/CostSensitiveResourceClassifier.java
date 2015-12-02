package edu.ncsu.mas.las.qoe.weka;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.ncsu.mas.las.qoe.weka.DatasetBuilder.Feature;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.core.Attribute;
import weka.core.Instances;

public class CostSensitiveResourceClassifier {
  private Instances data;

  public static void main(String[] args) throws Exception {
    String csvFilename = args[0];

    DatasetBuilder datasetBldr = new DatasetBuilder();
    datasetBldr.buildDataset(csvFilename);

    CostSensitiveResourceClassifier resClassifier = new CostSensitiveResourceClassifier();
    resClassifier.setDataset(datasetBldr.getDataset());
    resClassifier.preprocess();
    
    resClassifier.classify("[0 1; 1 0]");
    resClassifier.classify("[0 1; 2 0]");
    resClassifier.classify("[0 1; 4.2 0]");
  }

  public void setDataset(Instances data) {
    this.data = data;
  }

  public void preprocess() {
    data.setClassIndex(data.numAttributes() - 1);
    
    List<String> featureNames = new ArrayList<String>();
    featureNames.add(Feature.SERVICE_TIME_AT_CALL.name());
    featureNames.add(Feature.DISPATCHER_PRIORITY.name());
    featureNames.add(Feature.TRUE_PRIORITY.name());
    featureNames.add(Feature.CLASS.name());

    for (int i = 0; i < data.numAttributes(); i++) {
      Attribute att = data.attribute(i);
      if (!featureNames.contains(att.name())) {
        data.deleteAttributeAt(att.index());
      }
    }
  }
  
  public void classifyWithBagging(String costString) throws Exception {
    SMO baseCls = new SMO();
    // baseCls.setOptions(new String[] { "-C", "1.0", "-L", "0.001", "-P", "1.0E-12", "-N", "0", 
    //    "-V", "-1", "-W", "1", "-K", "weka.classifiers.functions.supportVector.PolyKernel" });
    
    Bagging baggingCls = new Bagging();
    baggingCls.setClassifier(baseCls);
    // baggingCls.setOptions(new String[] { "P", "100", "-S", "1", "-num-slots", "1", "-I", "10" });
    
    CostSensitiveClassifier constSensCls = new CostSensitiveClassifier();
    constSensCls.setClassifier(baggingCls);
    constSensCls.setCostMatrix(CostMatrix.parseMatlab(costString));

    Evaluation eval = new Evaluation(data);
    eval.crossValidateModel(constSensCls, data, 10, new Random(1));
    // System.out.println(eval.toSummaryString("\nResults\n======\n", false));

    for (int i = 0; i < 2; i++) {
      System.out.println("Class: " + i + "; Precision: " + eval.precision(i) + "; Recall: "
          + eval.recall(i) + "; F-measure: " + eval.fMeasure(i));
    }
    
    printConfusionMatrix(eval.confusionMatrix());
  }

  public void classify(String costString) throws Exception {
    LibSVM baseCls = new LibSVM();
    
    CostSensitiveClassifier constSensCls = new CostSensitiveClassifier();
    constSensCls.setClassifier(baseCls);
    constSensCls.setCostMatrix(CostMatrix.parseMatlab(costString));

    Evaluation eval = new Evaluation(data);
    eval.crossValidateModel(constSensCls, data, 10, new Random(1));
    // System.out.println(eval.toSummaryString("\nResults\n======\n", false));

    for (int i = 0; i < 2; i++) {
      System.out.println("Class: " + i + "; Precision: " + eval.precision(i) + "; Recall: "
          + eval.recall(i) + "; F-measure: " + eval.fMeasure(i));
    }
    
    printConfusionMatrix(eval.confusionMatrix());
  }
  
  public void printConfusionMatrix(double[][] confusionMatrix) {
    for (int i = 0; i < confusionMatrix.length; i++) {
      for (int j = 0; j < confusionMatrix[i].length; j++) {
        System.out.print(confusionMatrix[i][j] + ", ");
      }
      System.out.println();
    }
  }
}
