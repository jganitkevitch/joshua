package joshua.discriminative.training.risk_annealer.hypergraph.deprecated;

import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Logger;

import joshua.decoder.hypergraph.DiskHyperGraph;
import joshua.decoder.hypergraph.HyperGraph;
import joshua.discriminative.FileUtilityOld;
import joshua.discriminative.feature_related.feature_template.FeatureTemplate;
import joshua.discriminative.training.risk_annealer.hypergraph.FeatureForest;
import joshua.discriminative.training.risk_annealer.hypergraph.RiskAndFeatureAnnotationOnLMHG;


/*
 * This class generates hypergraphs, either from: a hypergraph generated by a decoder on the fly a
 * disk hypergraph a hypergraph in memory
 */

@Deprecated
public class FeatureForestFactory {

  private int baselineLMFeatID;
  private int baselineLMOrder;

  DiskHyperGraph diskHG = null;
  private String diskHGFilePrefix;
  int numSent;

  private boolean saveHGInMemory = true;
  private ArrayList<FeatureForest> hyperGraphs;


  HashMap<String, Integer> featureStringToIntegerMap;
  List<FeatureTemplate> featTemplates;

  boolean notRiskAndFeatureAnnoated = true;// TODO
  boolean useNoEquivAnnotator = true;
  RiskAndFeatureAnnotation riskAnnotator;
  RiskAndFeatureAnnotationOnLMHG riskAnnotatorNoEquiv;


  double[] linearCorpusGainThetas;
  String[] referenceFiles;
  BufferedReader[] refFileReaders;

  static int bleuNgramOrder = 4;

  /** Logger for this class. */
  private static final Logger logger = Logger.getLogger(FeatureForestFactory.class.getName());


  public FeatureForestFactory(int numSent_, String diskHGFilePrefix_, int baselineLMFeatID_,
      int baselineLMOrder_, boolean saveHGInMemory_,
      HashMap<String, Integer> featureStringToIntegerMap_, List<FeatureTemplate> featTemplates_,
      double[] linearCorpusGainThetas_, String[] referenceFiles_) {

    this.numSent = numSent_;
    this.diskHGFilePrefix = diskHGFilePrefix_;

    this.baselineLMFeatID = baselineLMFeatID_;
    this.baselineLMOrder = baselineLMOrder_;

    this.featureStringToIntegerMap = featureStringToIntegerMap_;
    this.featTemplates = featTemplates_;

    this.linearCorpusGainThetas = linearCorpusGainThetas_;

    if (baselineLMOrder >= bleuNgramOrder) {
      useNoEquivAnnotator = true;
      this.riskAnnotatorNoEquiv =
          new RiskAndFeatureAnnotationOnLMHG(baselineLMOrder, baselineLMFeatID,
              linearCorpusGainThetas, featureStringToIntegerMap, featTemplates, true);

    } else {
      useNoEquivAnnotator = false;
      this.riskAnnotator =
          new RiskAndFeatureAnnotation(4, linearCorpusGainThetas, featureStringToIntegerMap,
              featTemplates);
    }

    this.referenceFiles = referenceFiles_;

    // ====== read all the HG into memory and annotate them
    this.saveHGInMemory = saveHGInMemory_;

    if (saveHGInMemory == true) {
      hyperGraphs = new ArrayList<FeatureForest>();
      readHGsIntoMemoryAllAtOnce(numSent);
    }

  }


  public void startLoop() {
    if (saveHGInMemory == false) {
      initDiskReading();
    }
  }

  public void endLoop() {
    if (saveHGInMemory == false) {
      finalizeDiskReading();
    }
  }


  public FeatureForest nextHG(int sentID) {

    // === feature forests
    FeatureForest fForest;

    if (saveHGInMemory == false) {// on disk
      fForest = readOneHGFromDisk();
    } else {// in memory
      fForest = hyperGraphs.get(sentID);
    }

    return fForest;
  }


  private void readHGsIntoMemoryAllAtOnce(int numSent) {
    initDiskReading();
    for (int i = 0; i < numSent; i++) {
      FeatureForest fForest = readOneHGFromDisk();
      hyperGraphs.add(fForest);
    }
    finalizeDiskReading();
  }

  private void initDiskReading() {
    logger.info("initialize reading hypergraphss..............");

    diskHG = new DiskHyperGraph(baselineLMFeatID, true, null); // have model costs stored
    diskHG.initRead(diskHGFilePrefix + ".hg.items", diskHGFilePrefix + ".hg.rules", null);

    // === references files, they are needed only when we want annote the hypergraph with risk
    refFileReaders = new BufferedReader[referenceFiles.length];
    for (int i = 0; i < referenceFiles.length; i++)
      refFileReaders[i] = FileUtilityOld.getReadFileStream(referenceFiles[i], "UTF-8");
  }

  private void finalizeDiskReading() {
    logger.info("finalize reading hypergraphss..............");
    diskHG.closeReaders();

    // === references files
    for (int i = 0; i < referenceFiles.length; i++) {
      FileUtilityOld.closeReadFile(refFileReaders[i]);
    }
  }

  private FeatureForest readOneHGFromDisk() {

    // === reference sentences
    String[] referenceSentences = new String[refFileReaders.length];
    for (int i = 0; i < refFileReaders.length; i++)
      referenceSentences[i] = FileUtilityOld.readLineLzf(refFileReaders[i]);

    // === disk hypergraph
    HyperGraph testHG = diskHG.readHyperGraph();

    if (notRiskAndFeatureAnnoated) {
      if (useNoEquivAnnotator)
        return this.riskAnnotatorNoEquiv.riskAnnotationOnHG(testHG, referenceSentences);
      else
        return this.riskAnnotator.riskAnnotationOnHG(testHG, referenceSentences);
    } else
      return (FeatureForest) testHG;

  }

}
