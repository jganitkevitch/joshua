package joshua.tools;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import joshua.corpus.Vocabulary;
import joshua.decoder.JoshuaConfiguration;
import joshua.decoder.ff.tm.Rule;
import joshua.util.FormatUtils;
import joshua.util.PackedGrammarServer;

public class ParaphraseGraph {

  private static final Logger logger = Logger.getLogger(ParaphraseGraph.class.getName());

  private static String enfr_file = null;
  private static String fren_file = null;
  private static String first_feature = null;
  private static String second_feature = null;
  private static int n = -1;
  private static int m = -1;
  private static boolean constrain_lhs = false;
  private static boolean probability = false;
  private static boolean show_labels = false;
  private static boolean thickness = false;

  private static String seed = null;

  public static void main(String[] args) throws FileNotFoundException, IOException {

    for (int i = 0; i < args.length; i++) {
      if ("-enfr".equals(args[i]) && (i < args.length - 1)) {
        enfr_file = args[++i];
      } else if ("-fren".equals(args[i]) && (i < args.length - 1)) {
        fren_file = args[++i];
      } else if ("-ff".equals(args[i]) && (i < args.length - 1)) {
        first_feature = args[++i];
      } else if ("-sf".equals(args[i]) && (i < args.length - 1)) {
        second_feature = args[++i];
      } else if ("-n".equals(args[i]) && (i < args.length - 1)) {
        n = Integer.parseInt(args[++i]);
      } else if ("-m".equals(args[i]) && (i < args.length - 1)) {
        m = Integer.parseInt(args[++i]);
      } else if ("-lhs".equals(args[i])) {
        constrain_lhs = true;
      } else if ("-prob".equals(args[i])) {
        probability = true;
      } else if ("-label".equals(args[i])) {
        show_labels = true;
      } else if ("-thick".equals(args[i])) {
        thickness = true;
      } else if ("-seed".equals(args[i]) && (i < args.length - 1)) {
        seed = args[++i];
      }
    }

    if (seed == null) {
      logger.severe("No seed phrase specified.");
      return;
    }
    if (enfr_file == null) {
      logger.severe("No en-fr grammar specified.");
      return;
    }
    if (fren_file == null) {
      logger.severe("No fr-en grammar specified.");
      return;
    }
    if (first_feature == null) {
      logger.severe("No first label specified.");
      return;
    }
    if (second_feature == null) {
      logger.severe("No second label specified.");
      return;
    }
    if (n <= 0 || m < 0) {
      logger.severe("N and M not specified correctly.");
      return;
    }

    JoshuaConfiguration jc = new JoshuaConfiguration();
    PackedGrammarServer enfr = new PackedGrammarServer(enfr_file, jc);

    List<Rule> first_rules = enfr.get(seed);
    Collections.sort(first_rules, new RuleComparator(first_feature));
    Map<String, List<Link>> first_map = new HashMap<String, List<Link>>();

    List<String> lhs_labels = new ArrayList<String>();
    Map<String, Double> lhs_strengths = new HashMap<String, Double>();

    for (int i = 0; i < first_rules.size() && i < n; ++i) {
      Rule r = first_rules.get(i);
      double value = feature(r, first_feature);
      if (probability)
        value = Math.exp(-value);
      String lhs = (constrain_lhs ? Vocabulary.word(r.getLHS()) : "X");
      if (!first_map.containsKey(lhs)) {
        first_map.put(lhs, new ArrayList<Link>());
        lhs_labels.add(lhs);
        lhs_strengths.put(lhs, (probability ? Math.exp(-feature(r, "p(LHS|f)")) : 0));
      }
      first_map.get(lhs).add(new Link(r.getEnglishWords(), value));
      if (!probability)
        lhs_strengths.put(lhs, lhs_strengths.get(lhs) + value);
    }

    if (constrain_lhs && probability) {
      double sum = 0;
      for (double p : lhs_strengths.values())
        sum += p;
      for (String lhs : lhs_strengths.keySet())
        lhs_strengths.put(lhs, lhs_strengths.get(lhs) / sum);
    }

    Vocabulary.clear();
    PackedGrammarServer fren = new PackedGrammarServer(fren_file, jc);

    System.out.println("digraph {");
    System.out.println("size=\"6,9\"");
    System.out.println("splines=true;\n sep=\"+15,15\";\n overlap=scalexy;\n nodesep=0.6;"
        + "\n node [fontsize=11];");
    System.out.println("node [shape = doublecircle];");
    System.out.println("seed [ label = \"" + seed + "\" ];");
    System.out.println("node [shape = oval];");
    for (int i = 0; i < lhs_labels.size(); ++i) {
      String lhs = lhs_labels.get(i);
      String connect = "seed";
      if (constrain_lhs) {
        System.out.println("lhs_" + i + " [ label = \"" + FormatUtils.cleanNonterminal(lhs)
            + "\" ];");
        System.out.println(link("seed", "lhs_" + i, "p(LHS|f)", lhs_strengths.get(lhs)));
        connect = "lhs_" + i;
      }

      List<Link> first_links = first_map.get(lhs);

      if (probability) {
        double sum = 0;
        for (Link l : first_links)
          sum += l.value;
        for (Link l : first_links)
          l.value /= sum;
      }

      for (int j = 0; j < first_links.size(); ++j) {
        Link l = first_links.get(j);

        System.out.println("fr_" + i + "_" + j + " [ label = \""
            + l.phrase.replaceAll("\"", "\\\\\"") + "\" ];");
        System.out.println(link(connect, "fr_" + i + "_" + j, first_feature, l.value));

        List<Rule> all_second = fren.get(l.phrase);
        List<Rule> second_rules = new ArrayList<Rule>();
        for (Rule r : all_second)
          if (!constrain_lhs || r.getLHS() == Vocabulary.id(lhs))
            second_rules.add(r);
        Collections.sort(second_rules, new RuleComparator(second_feature));
        List<Link> second_links = new ArrayList<Link>();

        for (int k = 0; k < second_rules.size() && k < m; ++k) {
          Rule r = second_rules.get(k);
          double value = feature(r, second_feature);
          if (probability)
            value = Math.exp(-value);
          second_links.add(new Link(r.getEnglishWords(), value));
        }
        if (probability) {
          double sum = 0;
          for (Link p : second_links)
            sum += p.value;
          for (Link p : second_links)
            p.value /= sum;
        }
        for (int k = 0; k < second_links.size(); ++k) {
          Link p = second_links.get(k);
          System.out.println("en_" + i + "_" + j + "_" + k + " [ label = \""
              + p.phrase.replaceAll("\"", "\\\\\"") + "\" ];");
          System.out.println(link("fr_" + i + "_" + j, "en_" + i + "_" + j + "_" + k,
              second_feature, p.value));
        }

      }
    }

    System.out.println("}");
  }

  private static String link(String from, String to, String label, double value) {
    if (probability) {
      if (show_labels)
        return String.format("%s -> %s [ label = \"%s = %.2f\" " + thickness(value) + "];", from,
            to, label, value);
      else
        return String.format("%s -> %s [ label = \"%.2f\" " + thickness(value) + "];", from, to,
            value);
    } else {
      if (show_labels)
        return String.format("%s -> %s [ label = \"%s = %d\" " + thickness(value) + "];", from, to,
            label, (int) value);
      else
        return String.format("%s -> %s [ label = \"%d\" " + thickness(value) + "];", from, to,
            (int) value);
    }
  }

  private static double feature(Rule r, String name) {
    return (r.getFeatureVector().keySet().contains(name) ? r.getFeatureVector().get(name) : 0);
  }

  private static String thickness(double value) {
    if (!thickness)
      return "";
    return "penwidth=" + (probability ? 0.5 + 3 * value : Math.log10(value));
  }

  public static class RuleComparator implements Comparator<Rule> {
    String feature_name;

    public RuleComparator(String fn) {
      feature_name = fn;
    }

    public int compare(Rule a, Rule b) {
      float a_val, b_val;
      try {
        a_val = a.getFeatureVector().get(feature_name);
      } catch (Exception e) {
        a_val = 0;
      }
      try {
        b_val = b.getFeatureVector().get(feature_name);
      } catch (Exception e) {
        b_val = 0;
      }
      return (probability ? 1 : -1) * Float.compare(a_val, b_val);
    }
  };

  public static class Link {

    String phrase;
    double value;

    public Link(String phrase, double value) {
      this.phrase = phrase;
      this.value = value;
    }

    public String getPhrase() {
      return phrase;
    }

    public void setPhrase(String phrase) {
      this.phrase = phrase;
    }

    public double getValue() {
      return value;
    }

    public void setValue(double value) {
      this.value = value;
    }

  }

}
