package joshua.decoder.segment_file;

import joshua.decoder.JoshuaConfiguration;

public class ParseTreeInput extends Sentence {

  public ParseTreeInput(String input, int id, JoshuaConfiguration joshuaConfiguration) {
    super(input, id,joshuaConfiguration);
  }

  // looks_like_parse_tree = sentence.sentence().matches("^\\(+[A-Z]+ .*");

  // private SyntaxTree syntax_tree;

  // ParseTreeInput() {
  // SyntaxTree syntax_tree = new ArraySyntaxTree(sentence.sentence(), Vocabulary);
  // }

  // public int[] int_sentence() {
  // return syntax_tree.getTerminals();
  // }
}
