package edu.ncsu.mas.las.qoe.weka;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.LinkedHashMap;
import java.util.Map;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class FeatureBuilder {

  public enum Feature {
    AMBULANCE_TYPE, 
    SERVICE_TIME_AT_CALL, 
    TRAVEL_TIME_TO_CALL,
    TRAVEL_TIME_HOSPITAL_TO_HOME,
    TRAVEL_TIME_TO_HOSPITAL,
    QUEUED, // Redundant
    QUEUE_TIME,
    SYMPTOMS,
    CLASS
  }

  private Map<Feature, FastVector> valTypes = new LinkedHashMap<Feature, FastVector>();

  private Instances data;

  public static void main(String[] args) throws IOException {
    String csvFilename = args[0];
    String arffFilename = args[1];

    FeatureBuilder featureBldr = new FeatureBuilder();
    featureBldr.setupAttributes();
    featureBldr.buildFeatures(csvFilename);
    featureBldr.saveArff(arffFilename);
  }

  private void setupAttributes() {
    FastVector atts = new FastVector();

    FastVector ambulanceTypeVals = new FastVector();
    ambulanceTypeVals.addElement("1"); // ALS
    ambulanceTypeVals.addElement("2"); // BLS
    atts.addElement(new Attribute(Feature.AMBULANCE_TYPE.name(), ambulanceTypeVals));
    valTypes.put(Feature.AMBULANCE_TYPE, ambulanceTypeVals);

    atts.addElement(new Attribute(Feature.SERVICE_TIME_AT_CALL.name())); // Numeric
    valTypes.put(Feature.SERVICE_TIME_AT_CALL, null);

    atts.addElement(new Attribute(Feature.TRAVEL_TIME_TO_CALL.name())); // Numeric
    valTypes.put(Feature.TRAVEL_TIME_TO_CALL, null);

    atts.addElement(new Attribute(Feature.TRAVEL_TIME_TO_HOSPITAL.name())); // Numeric
    valTypes.put(Feature.TRAVEL_TIME_TO_HOSPITAL, null);

    atts.addElement(new Attribute(Feature.TRAVEL_TIME_HOSPITAL_TO_HOME.name())); // Numeric
    valTypes.put(Feature.TRAVEL_TIME_HOSPITAL_TO_HOME, null);

    FastVector binaryVals = new FastVector();
    binaryVals.addElement("0");
    binaryVals.addElement("1");
    atts.addElement(new Attribute(Feature.QUEUED.name(), binaryVals));
    valTypes.put(Feature.QUEUED, binaryVals);

    atts.addElement(new Attribute(Feature.QUEUE_TIME.name())); // Numeric
    valTypes.put(Feature.QUEUE_TIME, null);
    
    FastVector symptomsVals = new FastVector();
    symptomsVals.addElement("Sick, non-specific"); 
    symptomsVals.addElement("Heart Problems");
    symptomsVals.addElement("Back pain");
    symptomsVals.addElement("Fire");
    symptomsVals.addElement("Diabetes");
    symptomsVals.addElement("Transport");
    symptomsVals.addElement("Car accident");
    symptomsVals.addElement("Environmental");
    symptomsVals.addElement("Falls");
    symptomsVals.addElement("Psychological");
    symptomsVals.addElement("Bleeding");
    symptomsVals.addElement("Abdominal pain");
    symptomsVals.addElement("Seizure");
    symptomsVals.addElement("Unknown");
    symptomsVals.addElement("Stroke");
    symptomsVals.addElement("Unconscious");
    symptomsVals.addElement("Trauma, non-specific");
    symptomsVals.addElement("Poisoning");
    symptomsVals.addElement("Allergic reaction");
    symptomsVals.addElement("Assault");
    symptomsVals.addElement("Head pain");
    symptomsVals.addElement("Bite");
    symptomsVals.addElement("Choking");
    symptomsVals.addElement("CPR");
    symptomsVals.addElement("Electrical");
    symptomsVals.addElement("Eye");
    symptomsVals.addElement("Gynecological");
    atts.addElement(new Attribute(Feature.SYMPTOMS.name(), symptomsVals));
    valTypes.put(Feature.SYMPTOMS, symptomsVals);

    FastVector classVals = new FastVector();
    classVals.addElement("Correct"); // Correctly classified
    classVals.addElement("Incorrect"); // Misclassified
    atts.addElement(new Attribute(Feature.CLASS.name(), classVals));
    valTypes.put(Feature.CLASS, classVals);

    data = new Instances("one-county-data", atts, 0);
  }

  private void buildFeatures(String csvFilename) throws IOException {
    try (Reader in = new FileReader(csvFilename)) {
      Iterable<CSVRecord> records = CSVFormat.EXCEL.withHeader().parse(in);
      for (CSVRecord record : records) {
        double[] vals = new double[data.numAttributes()];
        int i = 0;

        for (Feature feature : valTypes.keySet()) {
          switch (feature) {
          case AMBULANCE_TYPE:
            vals[i++] = valTypes.get(feature).indexOf(
                record.get("Ambulance Type (1 for ALS 2 for BLS)"));
            break;

          case CLASS:
            String truePriority = record.get("True Priority");
            String dispatchedPriority = record.get("Dispatched Priority");
            if (truePriority.equals(dispatchedPriority)) {
              vals[i++] = valTypes.get(feature).indexOf("Correct");
            } else {
              vals[i++] = valTypes.get(feature).indexOf("Incorrect");
            }
            break;

          case QUEUED:
            vals[i++] = valTypes.get(feature).indexOf(record.get("Queued? (1=yes 0=no)"));
            break;

          case QUEUE_TIME:
            vals[i++] = Double.parseDouble(record.get("Queue Time"));
            break;

          case SERVICE_TIME_AT_CALL:
            vals[i++] = Double.parseDouble(record.get("Service Time At Call"));
            break;

          case TRAVEL_TIME_TO_CALL:
            vals[i++] = Double.parseDouble(record.get("Travel Time To Call"));
            break;
            
          case TRAVEL_TIME_HOSPITAL_TO_HOME:
            vals[i++] = Double.parseDouble(record.get("Travel Time Hospital to Home"));
            break;
            
          case TRAVEL_TIME_TO_HOSPITAL:
            vals[i++] = Double.parseDouble(record.get("Travel Time to Hospital"));
            break;
            
          case SYMPTOMS:
            vals[i++] = valTypes.get(feature).indexOf(record.get("Call \"symptoms\""));
            break;
          }
        }

        data.add(new Instance(1.0, vals));
      }
    }
  }

  private void saveArff(String arffFilename) throws IOException {
    ArffSaver saver = new ArffSaver();
    saver.setInstances(data);
    saver.setFile(new File(arffFilename));
    saver.writeBatch();
  }
}
