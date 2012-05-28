package joshua.discriminative.training.expbleu;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.logging.Logger;

import joshua.corpus.Vocabulary;
import joshua.decoder.BLEU;
import joshua.decoder.JoshuaDecoder;
import joshua.decoder.ff.FeatureFunction;
import joshua.decoder.hypergraph.DiskHyperGraph;
import joshua.decoder.hypergraph.KBestExtractor;
import joshua.discriminative.FileUtilityOld;
import joshua.discriminative.feature_related.feature_function.FeatureTemplateBasedFF;
import joshua.discriminative.feature_related.feature_template.BaselineFT;
import joshua.discriminative.feature_related.feature_template.FeatureTemplate;
import joshua.discriminative.feature_related.feature_template.IndividualBaselineFT;
import joshua.discriminative.feature_related.feature_template.MicroRuleFT;
import joshua.discriminative.feature_related.feature_template.NgramFT;
import joshua.discriminative.feature_related.feature_template.TMFT;
import joshua.discriminative.feature_related.feature_template.TargetTMFT;
import joshua.discriminative.ranker.HGRanker;
import joshua.discriminative.training.risk_annealer.AbstractMinRiskMERT;
import joshua.discriminative.training.risk_annealer.GradientOptimizer;
import joshua.discriminative.training.risk_annealer.hypergraph.HGAndReferences;
import joshua.discriminative.training.risk_annealer.hypergraph.HGMinRiskDAMert;
import joshua.discriminative.training.risk_annealer.hypergraph.HyperGraphFactory;
import joshua.discriminative.training.risk_annealer.hypergraph.MRConfig;

public class HGMaxExpbleu extends AbstractMinRiskMERT {

  private boolean haveRefereces = true;
  private String hypFilePrefix;
  private String sourceTrainingFile;
  private Integer oralceFeatureID;
  private double curLossScale;
  private JoshuaDecoder joshuaDecoder;
  private ArrayList<FeatureTemplate> featTemplates;
  private MicroRuleFT microRuleFeatureTemplate;
  private boolean useIntegerString = false; // ToDo
  private String curConfigFile;
  private String curFeatureFile;
  private HashMap<String, Integer> featureStringToIntegerMap;
  private String curHypFilePrefix;

  static private Logger logger = Logger.getLogger(HGMinRiskDAMert.class.getSimpleName());

  public HGMaxExpbleu(String configFile, int numTrainingSentence, String[] devRefs,
      String hypFilePrefix, String sourceTrainingFile) {
    super(configFile, numTrainingSentence, devRefs);
    // TODO Auto-generated constructor stub


    // ADD TERMINALS IN THE REF FILE TO THE SYMBOL TABLE
    if (devRefs != null) {
      for (String refFile : devRefs) {
        logger.info("add symbols for file " + refFile);
        addAllWordsIntoSymbolTbl(refFile);
      }
    } else {
      logger.info("Must include reference files in Max Expected Bleu Training");
      System.exit(1);
    }

    // READ IN CONFIGURATION FILE, READ IN FEATURES(SET FEATURE TEMPLATE) AND SET WEIGHTS
    this.initialize();

    this.hypFilePrefix = hypFilePrefix;
    this.sourceTrainingFile = sourceTrainingFile;

    if (MRConfig.oneTimeHGRerank == false) {
      joshuaDecoder = JoshuaDecoder.getUninitalizedDecoder();
      joshuaDecoder.initialize(configFile);
    }

  }

  // SETUP FEATURE RELATED DATA STRUCTURE
  private void initialize() {
    // TODO Auto-generated method stub
    // ===== read configurations
    MRConfig.readConfigFile(this.configFile);

    // ===== initialize the featureTemplates
    // ALL FEATURE TEMPLATES ARE SAVED IN "ArrayList<FeatureTemplate> featTemplates"
    setupFeatureTemplates();

    // ====== initialize featureStringToIntegerMap and weights
    initFeatureMapAndWeights(MRConfig.featureFile);
  }

  // MAP FEATURE NAME TO INTEGERS
  // SAVE FEATURE WEIGHTS TO ARRAY
  private void initFeatureMapAndWeights(String featureFile) {
    // TODO Auto-generated method stub
    featureStringToIntegerMap = new HashMap<String, Integer>();
    List<Double> temInitWeights = new ArrayList<Double>();
    int featID = 0;

    // ==== baseline feature
    if (MRConfig.useBaseline) {
      featureStringToIntegerMap.put(MRConfig.baselineFeatureName, featID++);
      temInitWeights.add(MRConfig.baselineFeatureWeight);
    }

    // ==== individual bs feature
    if (MRConfig.useIndividualBaselines) {
      List<Double> weights = readBaselineFeatureWeights(this.configFile);
      for (int id : MRConfig.baselineFeatIDsToTune) {
        String featName = MRConfig.individualBSFeatNamePrefix + id;

        System.out.println("******** " + featName + " " + id);

        featureStringToIntegerMap.put(featName, featID++);

        double weight = weights.get(id);
        System.out.println("+++++++ " + weight + " " + id);

        temInitWeights.add(weight);
      }
    }

    // ==== features in file
    if (MRConfig.useSparseFeature) {
      BufferedReader reader = FileUtilityOld.getReadFileStream(featureFile, "UTF-8");
      String line;
      while ((line = FileUtilityOld.readLineLzf(reader)) != null) {
        String[] fds = line.split("\\s+\\|{3}\\s+");// feature_key |||
        // feature vale; the
        // feature_key
        // itself may
        // contain "|||"
        StringBuffer featKey = new StringBuffer();
        for (int i = 0; i < fds.length - 1; i++) {
          featKey.append(fds[i]);
          if (i < fds.length - 2) featKey.append(" ||| ");
        }
        double initWeight = new Double(fds[fds.length - 1]);// initial
        // weight

        // ADD SPARSE FEATURE WEIGHTS
        temInitWeights.add(initWeight);
        featureStringToIntegerMap.put(featKey.toString(), featID++);
      }
      FileUtilityOld.closeReadFile(reader);
    }

    // ==== initialize lastWeightVector
    numPara = temInitWeights.size();
    lastWeightVector = new double[numPara];
    for (int i = 0; i < numPara; i++)
      lastWeightVector[i] = temInitWeights.get(i);
  }

  // SETUP FEATURE TEMPLATES FOR ALL THE FEATURES USED IN THE CURRENT SETTING
  private void setupFeatureTemplates() {
    // TODO Auto-generated method stub
    this.featTemplates = new ArrayList<FeatureTemplate>();

    // SETUP THE FEATURE UTIILTY(TEMPLATE) FOR EACH FEATURE TO BE USED

    // BASELINE FEATURES AS A WHOLE
    // ONLY CREATE ONE TEMPLATE
    if (MRConfig.useBaseline) {
      FeatureTemplate ft = new BaselineFT(MRConfig.baselineFeatureName, true);
      featTemplates.add(ft);
    }

    // USE INDIVIDUAL BASELINE FEATURES
    // CREATE TEMPLATE FOR EACH INDIVIDUAL FEATURE
    if (MRConfig.useIndividualBaselines) {
      for (int id : MRConfig.baselineFeatIDsToTune) {
        String featName = MRConfig.individualBSFeatNamePrefix + id;
        FeatureTemplate ft = new IndividualBaselineFT(featName, id, true);
        featTemplates.add(ft);
      }
    }

    // CREATE TEMPLATE FOR SPARSE FEATURES(IF USED)
    if (MRConfig.useSparseFeature) {

      if (MRConfig.useMicroTMFeat) {
        // FeatureTemplate ft = new TMFT(useIntegerString,
        // MRConfig.useRuleIDName);
        this.microRuleFeatureTemplate =
            new MicroRuleFT(MRConfig.useRuleIDName, MRConfig.startTargetNgramOrder,
                MRConfig.endTargetNgramOrder, MRConfig.wordMapFile);
        featTemplates.add(microRuleFeatureTemplate);
      }

      if (MRConfig.useTMFeat) {
        FeatureTemplate ft = new TMFT(useIntegerString, MRConfig.useRuleIDName);
        featTemplates.add(ft);
      }

      if (MRConfig.useTMTargetFeat) {
        FeatureTemplate ft = new TargetTMFT(useIntegerString);
        featTemplates.add(ft);
      }

      if (MRConfig.useLMFeat) {
        FeatureTemplate ft =
            new NgramFT(useIntegerString, MRConfig.ngramStateID, MRConfig.baselineLMOrder,
                MRConfig.startNgramOrder, MRConfig.endNgramOrder);
        featTemplates.add(ft);
      }
    }

    // DISPLAY WHAT FEATURES ARE USED IN THE CURRENT SETTING
    System.out.println("feature template are " + featTemplates.toString());
  }

  @Override
  // DECODE(GENERATE HYPGERGRAPH) FOR TRAINING SET
  public void decodingTestSet(double[] weights, String nbestFile) {
    // TODO Auto-generated method stub

    // HERE baselinecombo MEANING TREAT ALL BASELINE FEATURES AS ONE FEATURE
    /**
     * three scenarios: (1) individual baseline features (2) baselineCombo + sparse feature (3)
     * individual baseline features + sparse features
     */

    // SET THE FEATURE WEIGHTS
    if (MRConfig.useSparseFeature)
      joshuaDecoder.changeFeatureWeightVector(getIndividualBaselineWeights(), this.curFeatureFile);
    else
      joshuaDecoder.changeFeatureWeightVector(getIndividualBaselineWeights(), null);

    // call Joshua decoder to produce an hypergraph using the new weight
    // vector
    joshuaDecoder.decodeTestSet(sourceTrainingFile, nbestFile);

  }

  @Override
  public void mainLoop() {
    // TODO Auto-generated method stub
    /**
     * Here, we need multiple iterations as we do pruning when generate the hypergraph Note that
     * DeterministicAnnealer itself many need to solve an optimization problem at each temperature,
     * and each optimization is solved by LBFGS which itself involves many iterations (of computing
     * gradients)
     * */

    for (int iter = 1; iter <= 10; iter++) {// TODO: change 10 to a config variable

      // ==== re-normalize weights, and save config files
      this.curConfigFile = configFile + "." + iter; // JOSHUA CONFIG FILE
      this.curFeatureFile = MRConfig.featureFile + "." + iter; // MICRO FEAT FILE

      if (MRConfig.normalizeByFirstFeature) normalizeWeightsByFirstFeature(lastWeightVector, 0);

      saveLastModel(configFile, curConfigFile, MRConfig.featureFile, curFeatureFile);
      // writeConfigFile(lastWeightVector, configFile, configFile+"." +
      // iter);

      // ==== re-decode based on the new weights
      // BY DEFAULT, oneTimeHGRerank=false, THEREFORE NEED TO ITERATE
      if (MRConfig.oneTimeHGRerank) {
        this.curHypFilePrefix = hypFilePrefix;
      } else {
        this.curHypFilePrefix = hypFilePrefix + "." + iter;

        // DECODE, AND SAVE THE HG TO THE CURRENT HG FILE
        decodingTestSet(null, curHypFilePrefix);
        System.out.println("Decoded: " + curHypFilePrefix);
      }

      // MAP RULE TO INTEGER
      Map<String, Integer> ruleStringToIDTable =
          DiskHyperGraph.obtainRuleStringToIDTable(curHypFilePrefix + ".hg.rules");

      // try to abbrevate the featuers if possible
      // addAbbreviatedNames(ruleStringToIDTable);

      // micro rule features
      if (MRConfig.useSparseFeature && MRConfig.useMicroTMFeat) {
        this.microRuleFeatureTemplate.setupTbl(ruleStringToIDTable,
            featureStringToIntegerMap.keySet());
      }

      HyperGraphFactory hgFactory =
          new HyperGraphFactory(curHypFilePrefix, referenceFiles, MRConfig.ngramStateID,
              this.haveRefereces);

      // //=====re-compute onebest BLEU
      // if(MRConfig.normalizeByFirstFeature)
      // normalizeWeightsByFirstFeature(lastWeightVector, 0);
      //
      //
      // computeOneBestBLEU(curHypFilePrefix);

      // @todo: check convergency
      // USED TO COMPUTE THE FUNCTION(EXPBLEU) VALUE AND ITS GRADIENT VECTOR
      // WILL BE USED IN runLBFGS() BELOW
      ExpbleuGradientComputer comp =
          new ExpbleuGradientComputer(this.numPara, MRConfig.gainFactor, 1.0, 0.0, false, false,
              this.numTrainingSentence, hgFactory, this.featureStringToIntegerMap,
              this.featTemplates, haveRefereces, MRConfig.maxNumHGInQueue, MRConfig.numThreads);
      // comp.reComputeFunctionValueAndGradient(lastWeightVector);

      GradientOptimizer lbfgsRunner =
          new GradientOptimizer(this.numPara, lastWeightVector, false, comp, MRConfig.useL2Regula,
              MRConfig.varianceForL2, MRConfig.useModelDivergenceRegula, MRConfig.lambda,
              MRConfig.printFirstN);
      lastWeightVector = lbfgsRunner.runLBFGS();
    }

    // final output
    // AFTER ALL ITERATIONS
    if (MRConfig.normalizeByFirstFeature) normalizeWeightsByFirstFeature(lastWeightVector, 0);

    // SAVE THE LAST RESULTS TO config FILES
    saveLastModel(configFile, configFile + ".final", MRConfig.featureFile, MRConfig.featureFile
        + ".final");
    // writeConfigFile(lastWeightVector, configFile, configFile+".final");

    // System.out.println("#### Final weights are: ");
    // annealer.getLBFGSRunner().printStatistics(-1, -1, null, lastWeightVector);
  }


  // OPTIONAL FUNCTION, JUST TO MAKE THE RULE NAMES SIMPLER
  private void addAbbreviatedNames(Map<String, Integer> rulesIDTable) {
    // try to abbrevate the featuers if possible
    if (MRConfig.useRuleIDName) {
      // add the abbreviated feature name into featureStringToIntegerMap

      // System.out.println("size1=" + featureStringToIntegerMap.size());

      for (Entry<String, Integer> entry : rulesIDTable.entrySet()) {
        Integer featureID = featureStringToIntegerMap.get(entry.getKey());
        if (featureID != null) {
          String abbrFeatName = "r" + entry.getValue();// TODO??????
          featureStringToIntegerMap.put(abbrFeatName, featureID);
          // System.out.println("full="+entry.getKey() + "; abbrFeatName="+abbrFeatName +
          // "; id="+featureID);
        }
      }
      // System.out.println("size2=" + featureStringToIntegerMap);
      // System.exit(0);
    }

  }

  // COMPUTE THE AVERAGE BLEU SCORE / LOG_PROB FOR ALL THE 1-BEST TRANSLATIONS
  private void computeOneBestBLEU(String curHypFilePrefix) {
    if (this.haveRefereces == false) return;

    double bleuSum = 0;
    double googleGainSum = 0;
    double modelSum = 0;

    // ==== feature-based feature
    int featID = 999;
    double weight = 1.0;
    HashSet<String> restrictedFeatureSet = null;
    HashMap<String, Double> modelTbl =
        (HashMap<String, Double>) obtainModelTable(this.featureStringToIntegerMap,
            this.lastWeightVector);

    // modelTbl LOOKS LIKE: {oov WDT=0.0, CD IN=0.0, ... }

    // System.out.println("modelTable: " + modelTbl);
    FeatureFunction ff =
        new FeatureTemplateBasedFF(featID, weight, modelTbl, this.featTemplates,
            restrictedFeatureSet);

    // ==== reranker
    List<FeatureFunction> features = new ArrayList<FeatureFunction>();
    features.add(ff);
    HGRanker reranker = new HGRanker(features);

    // ==== kbest
    boolean addCombinedCost = false;
    KBestExtractor kbestExtractor =
        new KBestExtractor(MRConfig.use_unique_nbest, MRConfig.use_tree_nbest, false,
            addCombinedCost, false, true);

    // ==== loop
    HyperGraphFactory hgFactory =
        new HyperGraphFactory(curHypFilePrefix, referenceFiles, MRConfig.ngramStateID, true);
    hgFactory.startLoop();

    // FOR EACH SENTENCE
    for (int sentID = 0; sentID < this.numTrainingSentence; sentID++) {
      HGAndReferences res = hgFactory.nextHG();
      reranker.rankHG(res.hg); // reset best pointer and transition prob
      // RERANK THE HG FOR THIS SENTENCE

      // EXTRACT THE 1-BEST TRANSLATION FROM HG
      String hypSent = kbestExtractor.getKthHyp(res.hg.goalNode, 1, -1, null, null);

      // COMPUTE BLEU FOR THE 1-BEST
      double bleu = BLEU.computeSentenceBleu(res.referenceSentences, hypSent);
      bleuSum += bleu;

      double googleGain =
          BLEU.computeLinearCorpusGain(MRConfig.linearCorpusGainThetas, res.referenceSentences,
              hypSent);
      googleGainSum += googleGain;

      // THIS IS THE LOG_PROBABILITY OF THE 1-BEST
      modelSum += res.hg.bestLogP();
      // System.out.println("logP=" + res.hg.bestLogP() + "; Bleu=" + bleu
      // +"; googleGain="+googleGain);

    }
    hgFactory.endLoop();

    System.out.println("AvgLogP=" + modelSum / this.numTrainingSentence + "; AvgBleu=" + bleuSum
        / this.numTrainingSentence + "; AvgGoogleGain=" + googleGainSum / this.numTrainingSentence
        + "; SumGoogleGain=" + googleGainSum);
  }

  // CREATE modelTbl FOR THE computeOneBestBLEU() FUNCTION
  private Map<String, Double> obtainModelTable(HashMap<String, Integer> featureStringToIntegerMap,
      double[] weightVector) {
    HashMap<String, Double> modelTbl = new HashMap<String, Double>();
    for (Map.Entry<String, Integer> entry : featureStringToIntegerMap.entrySet()) {
      int featID = entry.getValue();
      double weight = lastWeightVector[featID];// last model
      modelTbl.put(entry.getKey(), weight);
    }
    return modelTbl;
  }

  // SAVE THE LASTEST UPDATED WEIGHT TO THE FILE
  // SAVE NEW PARAMS TO FILES "configOutput(eg: joshua.config.2)" and
  // "sparseFeaturesOutput(eg: micro.rule.feat.sup.500.2)"
  private void saveLastModel(String configTemplate, String configOutput,
      String sparseFeaturesTemplate, String sparseFeaturesOutput) {
    // TODO Auto-generated method stub
    if (MRConfig.useSparseFeature) {

      // SAVE joshua.config FILE
      JoshuaDecoder.writeConfigFile(getIndividualBaselineWeights(), configTemplate, configOutput,
          sparseFeaturesOutput);

      // SAVE SPARSE FEATURES FILE
      saveSparseFeatureFile(sparseFeaturesTemplate, sparseFeaturesOutput);

    } else {
      JoshuaDecoder.writeConfigFile(getIndividualBaselineWeights(), configTemplate, configOutput,
          null);
    }
  }

  // SAVE SPARSE FEATURES FILE(IF "useSparseFeature")
  // EG: VB IN ||| -6.935
  private void saveSparseFeatureFile(String fileTemplate, String outputFile) {
    // TODO Auto-generated method stub
    BufferedReader template = FileUtilityOld.getReadFileStream(fileTemplate, "UTF-8");
    BufferedWriter writer = FileUtilityOld.getWriteFileStream(outputFile);
    String line;

    while ((line = FileUtilityOld.readLineLzf(template)) != null) {
      // == construct feature name
      String[] fds = line.split("\\s+\\|{3}\\s+");// feature_key |||
      // feature vale; the
      // feature_key itself
      // may contain "|||"
      StringBuffer featKey = new StringBuffer();
      for (int i = 0; i < fds.length - 1; i++) {
        featKey.append(fds[i]);
        if (i < fds.length - 2) featKey.append(" ||| ");
      }

      // == write the learnt weight
      // double oldWeight = new Double(fds[fds.length-1]);//initial weight

      // GET THE FEATURE ID
      int featID = this.featureStringToIntegerMap.get(featKey.toString());

      // GET THE CORRESPONDING NEW FEATURE WEIGHT
      double newWeight = lastWeightVector[featID];// last model
      // System.out.println(featKey +"; old=" + oldWeight + "; new=" +
      // newWeight);

      // WRITE FEATURE WEIGHTS TO FILE
      FileUtilityOld.writeLzf(writer, featKey.toString() + " ||| " + newWeight + "\n");

      featID++;
    }
    FileUtilityOld.closeReadFile(template);
    FileUtilityOld.closeWriteFile(writer);
  }

  // READ IN INDIVIDUAL BASELINE FEATURE WEIGHTS
  private double[] getIndividualBaselineWeights() {
    double baselineWeight = 1.0;
    if (MRConfig.useBaseline) baselineWeight = getBaselineWeight();

    List<Double> weights = readBaselineFeatureWeights(this.configFile);

    // change the weights we are tunning
    if (MRConfig.useIndividualBaselines) {
      // FOR ALL BASELINE FEATURES
      for (int id : MRConfig.baselineFeatIDsToTune) {
        String featName = MRConfig.individualBSFeatNamePrefix + id; // LIKE "bs0", "bs1", ..."bs9"

        int featID = featureStringToIntegerMap.get(featName);
        weights.set(id, baselineWeight * lastWeightVector[featID]);
      }
    }

    if (MRConfig.lossAugmentedPrune) {
      String featName = MRConfig.individualBSFeatNamePrefix + this.oralceFeatureID;
      if (featureStringToIntegerMap.containsKey(featName)) {
        logger
            .severe("we are tuning the oracle model, must be wrong in specifying baselineFeatIDsToTune");
        System.exit(1);
      }

      weights.set(this.oralceFeatureID, this.curLossScale);
      System.out.println("curLossScale=" + this.curLossScale + "; oralceFeatureID="
          + this.oralceFeatureID);
    }

    double[] res = new double[weights.size()];
    for (int i = 0; i < res.length; i++) {
      res[i] = weights.get(i);
    }
    return res;
  }

  // IF "useBaseline=true", READ IN BASELINE WEIHGTS
  private double getBaselineWeight() {
    String featName = MRConfig.baselineFeatureName;
    int featID = featureStringToIntegerMap.get(featName);
    double weight = lastWeightVector[featID];
    System.out.println("baseline weight is " + weight);

    return weight;
  }

  // ADD THE WORDS IN THE REFERENCE TO THE SYMBOL TABLE
  static public void addAllWordsIntoSymbolTbl(String file) {
    BufferedReader reader = FileUtilityOld.getReadFileStream(file, "UTF-8");
    String line;
    while ((line = FileUtilityOld.readLineLzf(reader)) != null) {
      Vocabulary.addAll(line);
    }
    FileUtilityOld.closeReadFile(reader);
  }

  public static void main(String[] args) {
    if (args.length < 3) {
      System.out.println("Wrong number of parameters!");
      System.exit(1);
    }
    // long start_time = System.currentTimeMillis();
    String joshuaConfigFile = args[0].trim();
    String sourceTrainingFile = args[1].trim();
    String hypFilePrefix = args[2].trim();

    String[] devRefs = null;
    if (args.length > 3) {
      devRefs = new String[args.length - 3];
      for (int i = 3; i < args.length; i++) {
        devRefs[i - 3] = args[i].trim();
        System.out.println("Use ref file " + devRefs[i - 3]);
      }
    }


    int numSentInDevSet = FileUtilityOld.numberLinesInFile(sourceTrainingFile);

    HGMaxExpbleu trainer =
        new HGMaxExpbleu(joshuaConfigFile, numSentInDevSet, devRefs, hypFilePrefix,
            sourceTrainingFile);

    trainer.mainLoop();
  }

}
