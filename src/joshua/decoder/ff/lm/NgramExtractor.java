package joshua.decoder.ff.lm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import joshua.corpus.Vocabulary;
import joshua.decoder.ff.state_maintenance.DPState;
import joshua.decoder.ff.state_maintenance.NgramDPState;
import joshua.decoder.ff.tm.Rule;
import joshua.decoder.hypergraph.HGNode;
import joshua.decoder.hypergraph.HyperEdge;
import joshua.util.Ngram;

public class NgramExtractor {

  private int ngramStateID;
  private int baselineLMOrder;
  private boolean useIntegerNgram;

  private static String START_SYM = "<s>";
  private int START_SYM_ID;
  private static String STOP_SYM = "</s>";
  private int STOP_SYM_ID;

  static private Logger logger = Logger.getLogger(NgramExtractor.class.getSimpleName());

  // default is useIntegerNgram
  public NgramExtractor(int ngramStateID, int baselineLMOrder) {
    this(ngramStateID, true, baselineLMOrder);
  }

  public NgramExtractor(int ngramStateID, boolean useIntegerNgram, int baselineLMOrder) {
    this.ngramStateID = ngramStateID;
    this.useIntegerNgram = useIntegerNgram;
    this.baselineLMOrder = baselineLMOrder;

    this.START_SYM_ID = Vocabulary.id(START_SYM);
    this.STOP_SYM_ID = Vocabulary.id(STOP_SYM);
  }

  /** for generative model, should set startNgramOrder=endNgramOrder */
  // EXTRACT NGRAMS FROM THE HYPEREDGE, RETURN A NGRAM HASHTABLE
  public HashMap<String, Integer> getTransitionNgrams(HyperEdge dt, int startNgramOrder,
      int endNgramOrder) {
    // CALL THE FUNCTION BELOW TO DO THE ACTUAL EXTRACTION
    return getTransitionNgrams(dt.getRule(), dt.getAntNodes(), startNgramOrder, endNgramOrder);
  }

  /** for generative model, should set startNgramOrder=endNgramOrder */
  public HashMap<String, Integer> getTransitionNgrams(Rule rule, List<HGNode> antNodes,
      int startNgramOrder, int endNgramOrder) {
    return computeTransitionNgrams(rule, antNodes, startNgramOrder, endNgramOrder);
  }

  /** does not work for generative model */
  public HashMap<String, Integer> getRuleNgrams(Rule rule, int startNgramOrder, int endNgramOrder) {
    return computeRuleNgrams(rule, startNgramOrder, endNgramOrder);
  }

  /** for generative model, should always set startNgramOrder=1 to allow partional ngram */
  public HashMap<String, Integer> getFutureNgrams(Rule rule, DPState curDPState,
      int startNgramOrder, int endNgramOrder) {
    // TODO: do not consider <s> and </s>
    boolean addStart = false;
    boolean addEnd = false;

    return computeFutureNgrams((NgramDPState) curDPState, startNgramOrder, endNgramOrder, addStart,
        addEnd);
  }

  /** for generative model, should always set startNgramOrder=1 to allow partional ngram */
  public HashMap<String, Integer> getFinalTransitionNgrams(HyperEdge edge, int startNgramOrder,
      int endNgramOrder) {
    return getFinalTransitionNgrams(edge.getAntNodes().get(0), startNgramOrder, endNgramOrder);
  }

  /** for generative model, should always set startNgramOrder=1 to allow partional ngram */
  public HashMap<String, Integer> getFinalTransitionNgrams(HGNode antNode, int startNgramOrder,
      int endNgramOrder) {
    return computeFinalTransitionNgrams(antNode, startNgramOrder, endNgramOrder);
  }



  /**
   * work for both generative and discriminative model
   * */
  // TODO: consider speed up this function

  // EXTRACTING THE NGRAM COUNTS
  private HashMap<String, Integer> computeTransitionNgrams(Rule rule, List<HGNode> antNodes,
      int startNgramOrder, int endNgramOrder) {

    if (baselineLMOrder < endNgramOrder) {
      System.out.println("baselineLMOrder is too small");
      System.exit(0);
    }

    // ==== hyperedges not under "goal item"
    // new ngrams created due to the combination
    HashMap<String, Integer> newNgramCounts = new HashMap<String, Integer>();
    // the ngram that has already been computed
    HashMap<String, Integer> oldNgramCounts = new HashMap<String, Integer>();

    // ENGLISH TRANSLATION COVERED BY THE CURRENT HYPEREDGE
    int[] enWords = rule.getEnglish();

    // a continous sequence of words due to combination; the sequence stops whenever the
    // right-lm-state jumps in (i.e., having eclipsed words)
    // COMBINE WORD SEQUENCE FROM LEFT NODE WITH THOSE FROM RIGHT NODE
    List<Integer> words = new ArrayList<Integer>();

    for (int c = 0; c < enWords.length; c++) {
      int curID = enWords[c];

      if (Vocabulary.nt(curID)) {
        int index = -(curID + 1);

        HGNode antNode = antNodes.get(index); // GET THE LEFT/RIGHT ANTECEDENT NODE OF THIS
                                              // HYPEREDGE
        NgramDPState state = (NgramDPState) antNode.getDPState(this.ngramStateID);

        // System.out.println("lm_feat_is: " + this.lm_feat_id + " ; state is: " + state);
        List<Integer> leftContext = state.getLeftLMStateWords(); // LEFT CORNER WORD
        List<Integer> rightContext = state.getRightLMStateWords(); // RIGHT CORNER WORD

        // SANITY CHECK
        if (leftContext.size() != rightContext.size()) {
          System.out.println("getAllNgrams: left and right contexts have unequal lengths");
          System.exit(1);
        }

        // == find new ngrams created
        for (int t : leftContext)
          words.add(t);

        this.getNgrams(oldNgramCounts, startNgramOrder, endNgramOrder, leftContext);

        if (rightContext.size() >= baselineLMOrder - 1) {// the right and left are NOT overlapping
          this.getNgrams(oldNgramCounts, startNgramOrder, endNgramOrder, rightContext);
          this.getNgrams(newNgramCounts, startNgramOrder, endNgramOrder, words);

          // start a new chunk; the sequence stops whenever the right-lm-state jumps in (i.e.,
          // having eclipsed words)
          words.clear();
          for (int t : rightContext)
            words.add(t);
        }
      } else {// terminal words
        words.add(curID);
      }
    }

    this.getNgrams(newNgramCounts, startNgramOrder, endNgramOrder, words);

    // newNgramCounts HAS BEEN SETUP

    // === now deduct ngram counts
    HashMap<String, Integer> res = new HashMap<String, Integer>();

    for (Map.Entry<String, Integer> entry : newNgramCounts.entrySet()) {
      String ngram = entry.getKey();
      int finalCount = entry.getValue(); // THE finalCount = OLD COUNT + NEW COUNT(IF THE NGRAM IS
                                         // NEW)

      // System.out.println("------------------------------\n"+ngram + " " + finalCount);

      // IF THE TRANSITION NGRAM IS ALREADY IN THE OLD NGRAM COUNT
      if (oldNgramCounts.containsKey(ngram)) {
        finalCount = finalCount - oldNgramCounts.get(ngram);
        // JUST FOR SANITY CHECK - NEW COUNT SHOULDN'T BE LESS THAN THE OLD COUNT
        if (finalCount < 0) {
          logger.warning("negative count for ngram: " + ngram + "; new: " + entry.getValue()
              + "; old: " + oldNgramCounts.get(ngram));
          System.out.println(" rule is: " + rule.toString());

          // JUST TO PRINT OUT ERROR INFO
          for (int i = 0; i < antNodes.size(); i++) {
            HGNode antNode = antNodes.get(i);
            NgramDPState state = (NgramDPState) antNode.getDPState(this.ngramStateID);
            // System.out.println("lm_feat_is: " + this.lm_feat_id + " ; state is: " + state);

            List<Integer> leftContext = state.getLeftLMStateWords();
            for (int wrd : leftContext)
              System.out.print(Vocabulary.word(wrd) + " ");
            System.out.println();

            List<Integer> rightContext = state.getRightLMStateWords();
            for (int wrd : rightContext)
              System.out.print(Vocabulary.word(wrd) + " ");
            System.out.println();

          }
          // System.exit(0);//TODO
        }
      }

      if (finalCount > 0) res.put(ngram, finalCount); // SAVE THE NGRAM COUNT
    }
    return res;
  }

  /**
   * work for both generative and discriminative model
   * */
  private HashMap<String, Integer> computeFinalTransitionNgrams(HGNode antNode,
      int startNgramOrder, int endNgramOrder) {

    if (baselineLMOrder < endNgramOrder) {
      System.out.println("baselineLMOrder is too small");
      System.exit(0);
    }

    HashMap<String, Integer> res = new HashMap<String, Integer>();
    NgramDPState state = (NgramDPState) antNode.getDPState(this.ngramStateID);

    List<Integer> currentNgram = new ArrayList<Integer>();
    List<Integer> leftContext = state.getLeftLMStateWords();
    List<Integer> rightContext = state.getRightLMStateWords();
    if (leftContext.size() != rightContext.size()) {
      System.out.println("computeFinalTransition: left and right contexts have unequal lengths");
      System.exit(1);
    }

    // ============ left context
    currentNgram.add(START_SYM_ID);
    for (int i = 0; i < leftContext.size(); i++) {
      int t = leftContext.get(i);
      currentNgram.add(t);

      if (currentNgram.size() >= startNgramOrder && currentNgram.size() <= endNgramOrder)
        this.getNgrams(res, currentNgram.size(), currentNgram.size(), currentNgram);

      if (currentNgram.size() == baselineLMOrder) {
        currentNgram.remove(0);
      }
    }

    // ============ right context
    // switch context: get the last possible new ngram: this ngram can be <s> a </s>
    int tSize = currentNgram.size();
    for (int i = 0; i < rightContext.size(); i++) {// replace context
      currentNgram.set(tSize - rightContext.size() + i, rightContext.get(i));
    }
    currentNgram.add(STOP_SYM_ID);

    if (currentNgram.size() >= startNgramOrder && currentNgram.size() <= endNgramOrder)
      this.getNgrams(res, currentNgram.size(), currentNgram.size(), currentNgram);

    return res;
  }



  /**
   * TODO: This does not work for a generative model. For example, for a rule: a b x_0 c d, under
   * generative model, we only want ngrams: a; a b; c; c d;, but not b and d
   * 
   * **/
  private HashMap<String, Integer> computeRuleNgrams(Rule rule, int startNgramOrder,
      int endNgramOrder) {

    if (baselineLMOrder < endNgramOrder) {
      System.out.println("baselineLMOrder is too small");
      System.exit(0);
    }

    HashMap<String, Integer> newNgramCounts = new HashMap<String, Integer>();// new ngrams created
                                                                             // due to the
                                                                             // combination

    int[] enWords = rule.getEnglish();
    List<Integer> words = new ArrayList<Integer>();
    for (int c = 0; c < enWords.length; c++) {
      int curWrd = enWords[c];
      if (Vocabulary.nt(curWrd)) {
        this.getNgrams(newNgramCounts, startNgramOrder, endNgramOrder, words);
        words.clear();
      } else {
        words.add(curWrd);
      }
    }
    this.getNgrams(newNgramCounts, startNgramOrder, endNgramOrder, words);
    return newNgramCounts;
  }


  /**
   * TODO: This does not work when addStart == true or addEnd == true. But, if both addStart ==
   * false or addEnd == false, then it works both for discrimnaitve and generative
   **/
  private HashMap<String, Integer> computeFutureNgrams(NgramDPState state, int startNgramOrder,
      int endNgramOrder, boolean addStart, boolean addEnd) {

    if (baselineLMOrder < endNgramOrder) {
      System.out.println("baselineLMOrder is too small");
      System.exit(0);
    }

    HashMap<String, Integer> res = new HashMap<String, Integer>();

    List<Integer> currentNgram = new ArrayList<Integer>();
    List<Integer> leftContext = state.getLeftLMStateWords();
    List<Integer> rightContext = state.getRightLMStateWords();
    if (leftContext.size() != rightContext.size()) {
      System.out.println("computeFinalTransition: left and right contexts have unequal lengths");
      System.exit(1);
    }

    // ============ left context
    if (addStart == true) {// this does not really work
      /**
       * TODO: this does not really work as the new ngrams generated should be different for
       * discriminative or generative model.
       * */
      currentNgram.add(START_SYM_ID);
    }
    // approximate the full-ngram with smaller-order ngrams
    for (int i = 0; i < leftContext.size(); i++) {
      int t = leftContext.get(i);
      currentNgram.add(t);

      if (currentNgram.size() >= startNgramOrder && currentNgram.size() <= endNgramOrder - 1)
        this.getNgrams(res, currentNgram.size(), currentNgram.size(), currentNgram);

      if (currentNgram.size() == baselineLMOrder) {
        currentNgram.remove(0);
      }
    }

    // ============ right context
    // switch context: get the last possible new ngram: this ngram can be <s> a </s>
    if (addEnd == true) {// only when add_end is true, we get a complete ngram, otherwise, all
                         // ngrams in r_state are incomplete and we should do nothing
      /**
       * TODO: this will be different for discriminative or generative model. For example, for
       * discriminative model, we may get new ngrams like "a b </s>""b </s>". But, for generative
       * model, we will get at most one ngram, whose order is baselineLMOrder
       */
    }
    return res;
  }

  private static void getNgrams(HashMap<String, Integer> tbl, int startNgramOrder,
      int endNgramOrder, List<Integer> wrds) {
    Ngram.getNgrams(tbl, startNgramOrder, endNgramOrder, wrds);
  }
}
