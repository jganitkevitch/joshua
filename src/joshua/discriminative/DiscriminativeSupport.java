package joshua.discriminative;

import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

import joshua.corpus.Vocabulary;
import joshua.discriminative.feature_related.feature_function.FeatureTemplateBasedFF;
import joshua.discriminative.feature_related.feature_template.EdgeBigramFT;
import joshua.discriminative.feature_related.feature_template.FeatureTemplate;
import joshua.discriminative.feature_related.feature_template.MicroRuleFT;
import joshua.discriminative.feature_related.feature_template.NgramFT;
import joshua.discriminative.feature_related.feature_template.TMFT;
import joshua.discriminative.feature_related.feature_template.TargetTMFT;


public class DiscriminativeSupport {

  static private Logger logger = Logger.getLogger(DiscriminativeSupport.class.getName());


  static public void increaseCount(HashMap<String, Double> tbl, String feat, double increment) {
    Double oldCount = tbl.get(feat);
    if (oldCount != null)
      tbl.put(feat, oldCount + increment);
    else
      tbl.put(feat, increment);
  }

  // LOAD THE MICRO FEATURE RULES AND WEIGHTS
  public static void loadModel(String modelFile, HashMap<String, Double> modelTable,
      Map<String, Integer> rulesIDTable) {

    BufferedReader reader = FileUtilityOld.getReadFileStream(modelFile, "UTF-8");
    modelTable.clear();
    String line;

    // NEWLY ADDED: THE ID OF SPARSE FEATURES
    int feat_id = 0;

    while ((line = FileUtilityOld.readLineLzf(reader)) != null) {
      String[] fds = line.split("\\s+\\|{3}\\s+");
      StringBuffer featNameSB = new StringBuffer();
      for (int i = 0; i < fds.length - 1; i++) {
        featNameSB.append(Vocabulary.id(fds[i]));
        if (i < fds.length - 2) featNameSB.append(" ||| ");
      }
      String featName = featNameSB.toString();

      // obtain abbreviated featName
      if (rulesIDTable != null) {
        Integer id = rulesIDTable.get(featName);
        if (id != null) featName = "r" + id;
      }

      // SAVE THE DISCRIMINATIVE FEATURE NAME + VALUE
      double val = new Double(fds[fds.length - 1]);
      modelTable.put(featName, val);
      // System.out.println(featName+" "+val);

      // NEWLY ADDED
      map_feat_id.put(featName, feat_id);
      logger.info("Saving feature:" + featName + " with id " + feat_id);
      feat_id++;
    }
    FileUtilityOld.closeReadFile(reader);
  }

  public static void loadFeatureSet(String featureSetFile, HashSet<String> featSet) {
    featSet.clear();
    BufferedReader reader = FileUtilityOld.getReadFileStream(featureSetFile, "UTF-8");
    String feat;
    while ((feat = FileUtilityOld.readLineLzf(reader)) != null) {
      featSet.add(feat);
    }
    FileUtilityOld.closeReadFile(reader);
  }


  static public List<String> readFileList(String file) {
    List<String> res = new ArrayList<String>();
    BufferedReader reader = FileUtilityOld.getReadFileStream(file, "UTF-8");
    String line;
    while ((line = FileUtilityOld.readLineLzf(reader)) != null) {
      res.add(line);
    }
    FileUtilityOld.closeReadFile(reader);
    return res;
  }

  // read the sent ids into the hashtable
  public static HashMap<Integer, Boolean> setupDataSelTbl(String fDataSel) {
    if (fDataSel == null) return null;
    HashMap<Integer, Boolean> res = new HashMap<Integer, Boolean>();
    BufferedReader t_reader_data_sel = FileUtilityOld.getReadFileStream(fDataSel, "UTF-8");
    String sentID;
    while ((sentID = FileUtilityOld.readLineLzf(t_reader_data_sel)) != null) {
      res.put(new Integer(sentID), true);
    }
    FileUtilityOld.closeReadFile(t_reader_data_sel);
    return res;
  }



  public static void scaleMapEntries(HashMap<?, Double> map, double scale) {
    for (Map.Entry<?, Double> entry : map.entrySet()) {
      entry.setValue(entry.getValue() * scale);
    }
  }


  // speed issue: assume tbl_feats is smaller than model
  static public double computeLinearCombinationLogP(HashMap<String, Double> featTbl,
      HashMap<String, Double> model) {
    double res = 0;

    // System.out.println("computeLinearCombinationLog````````````````````");

    for (Map.Entry<String, Double> entry : featTbl.entrySet()) {
      String featKey = entry.getKey();
      double featCount = entry.getValue();

      if (model.containsKey(featKey)) {
        double weight = model.get(featKey);
        res += weight * featCount;

        // System.out.println(weight + " " + featCount + " " + featKey);
      } else {
        // logger.info("nonexisit feature: " + featKey);
      }
    }

    return res;
  }



  static public FeatureTemplateBasedFF setupRerankingFeature(int featID, double weight,
      boolean useTMFeat, boolean useLMFeat, boolean useEdgeNgramOnly, boolean useTMTargetFeat,
      boolean useMicroTMFeat, String wordMapFile, int ngramStateID, int baselineLMOrder,
      int startNgramOrder, int endNgramOrder, String featureFile, String modelFile,
      Map<String, Integer> rulesStringToIDTable) {

    boolean useIntegerString = false;
    boolean useRuleIDName = false;
    if (rulesStringToIDTable != null) useRuleIDName = true;

    // ============= restricted feature set
    HashSet<String> restrictedFeatureSet = null;
    if (featureFile != null) {
      restrictedFeatureSet = new HashSet<String>();
      DiscriminativeSupport.loadFeatureSet(featureFile, restrictedFeatureSet);
      // restricted_feature_set.put(HGDiscriminativeLearner.g_baseline_feat_name, 1.0); //should not
      // add the baseline feature
      logger.info("============use  restricted feature set========================");
    }


    // ============= feature templates
    List<FeatureTemplate> featTemplates =
        DiscriminativeSupport.setupFeatureTemplates(useTMFeat, useLMFeat, useEdgeNgramOnly,
            useTMTargetFeat, useMicroTMFeat, wordMapFile, ngramStateID, baselineLMOrder,
            startNgramOrder, endNgramOrder, useIntegerString, useRuleIDName, rulesStringToIDTable,
            restrictedFeatureSet);

    // ================ discriminative reranking model
    HashMap<String, Double> modelTbl = new HashMap<String, Double>();

    // MODEL FILE IS THE micro.rule.feat... FILE

    DiscriminativeSupport.loadModel(modelFile, modelTbl, rulesStringToIDTable);

    return new FeatureTemplateBasedFF(featID, weight, modelTbl, featTemplates, restrictedFeatureSet);
  }


  // TODO: should merge with setupFeatureTemplates in HGMinRiskDAMert
  static public List<FeatureTemplate> setupFeatureTemplates(boolean useTMFeat, boolean useLMFeat,
      boolean useEdgeNgramOnly, boolean useTMTargetFeat, boolean useMicroTMFeat,
      String wordMapFile, int ngramStateID, int baselineLMOrder, int startNgramOrder,
      int endNgramOrder, boolean useIntegerString, boolean useRuleIDName,
      Map<String, Integer> rulesStringToIDTable, Set<String> restrictedFeatureSet) {

    List<FeatureTemplate> featTemplates = new ArrayList<FeatureTemplate>();

    if (useTMFeat == true) {
      FeatureTemplate ft = new TMFT(useIntegerString, useRuleIDName);
      featTemplates.add(ft);
    }

    if (useTMTargetFeat == true) {
      FeatureTemplate ft = new TargetTMFT(useIntegerString);
      featTemplates.add(ft);
    }

    if (useMicroTMFeat) {
      int startOrder = 2;// TODO
      int endOrder = 2;// TODO
      MicroRuleFT microRuleFeatureTemplate =
          new MicroRuleFT(useRuleIDName, startOrder, endOrder, wordMapFile);
      microRuleFeatureTemplate.setupTbl(rulesStringToIDTable, restrictedFeatureSet);
      featTemplates.add(microRuleFeatureTemplate);
    }

    if (useLMFeat == true) {
      FeatureTemplate ft =
          new NgramFT(useIntegerString, ngramStateID, baselineLMOrder, startNgramOrder,
              endNgramOrder);
      featTemplates.add(ft);
    } else if (useEdgeNgramOnly) {// exclusive with use_lm_feat
      FeatureTemplate ft = new EdgeBigramFT(ngramStateID, baselineLMOrder, useIntegerString);
      featTemplates.add(ft);
    }
    logger.info("templates are: " + featTemplates);


    return featTemplates;
  }

  // NEWLY ADDED: TO GET THE FEAT ID OF A FEATURE
  public static Integer getMicroFeatID(String feat_name) {
    if (map_feat_id.containsKey(feat_name))
      return map_feat_id.get(feat_name);
    else {
      System.err.println("ERROR: MICRO FEATURE HAS NEVER BEEN SAVED: " + feat_name);
      return -1;
    }
  }

  private static HashMap<String, Integer> map_feat_id = new HashMap<String, Integer>();
}
