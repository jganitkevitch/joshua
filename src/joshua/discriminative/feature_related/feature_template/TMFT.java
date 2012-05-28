package joshua.discriminative.feature_related.feature_template;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import joshua.decoder.ff.tm.Rule;
import joshua.decoder.hypergraph.HGNode;
import joshua.discriminative.DiscriminativeSupport;



public class TMFT extends AbstractFeatureTemplate {

  boolean useIntegerString = true;

  /**
   * we can use the ruleID as feature name
   */
  boolean useRuleIDName = true;
  String prefix = "r";



  public TMFT(boolean useIntegerString, boolean useRuleIDName) {

    this.useIntegerString = useIntegerString;
    this.useRuleIDName = useRuleIDName;

    System.out.println("TM template");
  }


  public void getFeatureCounts(Rule rule, List<HGNode> antNodes,
      HashMap<String, Double> featureTbl, HashSet<String> restrictedFeatureSet, double scale) {
    computeCounts(rule, featureTbl, restrictedFeatureSet, scale);
  }


  public void estimateFeatureCounts(Rule rule, HashMap<String, Double> featureTbl,
      HashSet<String> restrictedFeatureSet, double scale) {
    computeCounts(rule, featureTbl, restrictedFeatureSet, scale);
  }

  private void computeCounts(Rule rule, HashMap<String, Double> featureTbl,
      HashSet<String> restrictedFeatureSet, double scale) {
    if (rule != null) {
      String key = null;
      if (this.useRuleIDName) {
        key = this.prefix + rule.getRuleID();
        // System.out.println("key is " + key + "; And: "
        // +rule.toStringWithoutFeatScores(symbolTbl));System.exit(0);
      } else {
        key = rule.toStringWithoutFeatScores();
      }

      if (restrictedFeatureSet == null || restrictedFeatureSet.contains(key) == true) {
        DiscriminativeSupport.increaseCount(featureTbl, key, scale);
        // System.out.println("key is " + key +"; lhs " + Vocabulary.getWord(rule.getLHS()));

      }

    }


  }


}
