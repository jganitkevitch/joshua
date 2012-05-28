package joshua.discriminative.feature_related.feature_function;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import joshua.corpus.Vocabulary;
import joshua.decoder.BLEU;
import joshua.decoder.chart_parser.SourcePath;
import joshua.decoder.ff.DefaultStatefulFF;
import joshua.decoder.ff.lm.NgramExtractor;
import joshua.decoder.ff.state_maintenance.DPState;
import joshua.decoder.ff.tm.Rule;
import joshua.decoder.hypergraph.HGNode;
import joshua.util.io.LineReader;
import joshua.util.io.Reader;
import joshua.util.io.UncheckedIOException;


public class BLEUOracleModel extends DefaultStatefulFF {

  private int startNgramOrder = 1;
  private int endNgramOrder = 4;
  private NgramExtractor ngramExtractor;

  private Reader<String>[] referenceReaders;


  private boolean useIntegerNgram = true;

  private static Logger logger = Logger.getLogger(BLEUOracleModel.class.getName());

  Map<Integer, Map<String, Integer>> tblOfReferenceNgramTbls;
  private int maxSentIDSoFar = -1;// TODO: assume valid sentID start from zero

  /*
   * private static double unigramPrecision = 0.85; private static double precisionDecayRatio = 0.7;
   * private static int numUnigramTokens = 10; private static double[] linearCorpusGainThetas
   * =BLEU.computeLinearCorpusThetas( numUnigramTokens, unigramPrecision, precisionDecayRatio);
   */
  private double[] linearCorpusGainThetas = null;

  public BLEUOracleModel(int ngramStateID, int baselineLMOrder, int featID, double weight,
      String referenceFile, double[] linearCorpusGainThetas) {

    super(ngramStateID, weight, featID);

    if (baselineLMOrder < endNgramOrder)
      logger.severe("baselineLMOrder is too small; baselineLMOrder=" + baselineLMOrder);

    this.ngramExtractor = new NgramExtractor(ngramStateID, useIntegerNgram, baselineLMOrder);

    this.linearCorpusGainThetas = linearCorpusGainThetas;
    logger.info("linearCorpusGainThetas=" + this.linearCorpusGainThetas);

    // setup reference reader
    this.referenceReaders = new LineReader[1];
    LineReader reader = openOneFile(referenceFile);
    this.referenceReaders[0] = reader;
    this.tblOfReferenceNgramTbls = new HashMap<Integer, Map<String, Integer>>();
    logger.info("number of references used is " + referenceReaders.length);
  }


  public BLEUOracleModel(int ngramStateID, int baselineLMOrder, int featID, double weight,
      String[] referenceFiles, double[] linearCorpusGainThetas) {

    super(ngramStateID, weight, featID);

    this.ngramExtractor = new NgramExtractor(ngramStateID, useIntegerNgram, baselineLMOrder);


    this.linearCorpusGainThetas = linearCorpusGainThetas;
    logger.info("linearCorpusGainThetas=" + this.linearCorpusGainThetas);

    // setup reference readers
    this.referenceReaders = new LineReader[referenceFiles.length];
    for (int i = 0; i < referenceFiles.length; i++) {
      LineReader reader = openOneFile(referenceFiles[i]);
      this.referenceReaders[i] = reader;
    }
    this.tblOfReferenceNgramTbls = new HashMap<Integer, Map<String, Integer>>();
    logger.info("number of references used is " + referenceReaders.length);
  }


  /**
   * we do not have sentence-specific estimation
   * */
  public double estimateLogP(Rule rule, int sentID) {
    return 0;
  }

  public double estimateFutureLogP(Rule rule, DPState curDPState, int sentID) {
    // TODO Auto-generated method stub
    return 0;
  }

  public double finalTransitionLogP(HGNode antNode, int spanStart, int spanEnd, SourcePath srcPath,
      int sentID) {
    // TODO Auto-generated method stub
    return 0;
  }

  @Override
  public double reEstimateTransitionLogP(Rule rule, List<HGNode> antNodes, int spanStart,
      int spanEnd, SourcePath srcPath, int sentID) {
    return computeTransitionBleu(rule, antNodes, setupReferenceNgramTable(sentID));
  }

  public double transitionLogP(Rule rule, List<HGNode> antNodes, int spanStart, int spanEnd,
      SourcePath srcPath, int sentID) {
    return computeTransitionBleu(rule, antNodes, setupReferenceNgramTable(sentID));
  }

  // ========================= risk related =============
  synchronized private Map<String, Integer> setupReferenceNgramTable(int sentID) {
    while (this.maxSentIDSoFar < sentID) {
      this.maxSentIDSoFar++;
      try {
        logger.info("open a new sentence with id " + this.maxSentIDSoFar);
        String[] referenceSentences = new String[referenceReaders.length];
        for (int i = 0; i < referenceReaders.length; i++) {
          referenceSentences[i] = referenceReaders[i].readLine();
        }

        if (this.useIntegerNgram) {
          referenceSentences = convertToIntegerString(referenceSentences);
        }

        Map<String, Integer> ngramTable =
            BLEU.constructMaxRefCountTable(referenceSentences, endNgramOrder);
        this.tblOfReferenceNgramTbls.put(this.maxSentIDSoFar, ngramTable);

      } catch (IOException ioe) {
        logger.severe("read references error");
        System.exit(0);
        throw new UncheckedIOException(ioe);
      }
    }
    return this.tblOfReferenceNgramTbls.get(sentID);
  }


  private double computeTransitionBleu(Rule rule, List<HGNode> antNodes,
      Map<String, Integer> refNgramTable) {

    double transitionBLEU = 0;

    if (rule != null) {
      int hypLength = rule.getEnglish().length - rule.getArity();

      /**
       * this statement is most time-consuming
       **/
      HashMap<String, Integer> hyperedgeNgramTable =
          ngramExtractor.getTransitionNgrams(rule, antNodes, startNgramOrder, endNgramOrder);

      transitionBLEU =
          BLEU.computeLinearCorpusGain(linearCorpusGainThetas, hypLength, hyperedgeNgramTable,
              refNgramTable);
    } else {
      // note: hyperedges under goal item does not contribute BLEU, do nothing
    }

    return transitionBLEU;
  }

  private LineReader openOneFile(String file) {
    try {
      return new LineReader(file);
    } catch (IOException ioe) {
      throw new UncheckedIOException(ioe);
    }
  }

  private String[] convertToIntegerString(String[] strSentences) {
    String[] intSentences = new String[strSentences.length];
    int j = 0;
    for (String str : strSentences) {
      int[] ids = Vocabulary.addAll(str);

      StringBuffer intSent = new StringBuffer();;
      for (int i = 0; i < ids.length; i++) {
        intSent.append(ids[i]);
        if (i < ids.length - 1) intSent.append(" ");
      }
      intSentences[j] = intSent.toString();
      // System.out.println("str: " + strSentences[j]);
      // System.out.println("int: " + intSentences[j]);
      j++;
    }
    // convertToStrString(intSentences);
    return intSentences;
  }
}
