package joshua.decoder.ff.lm.srilm;

/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version: 1.3.21
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */


class srilmJNI {
  public final static native long new_unsigned_array(int jarg1);
  public final static native void delete_unsigned_array(long jarg1);
  public final static native long unsigned_array_getitem(long jarg1, int jarg2);
  public final static native void unsigned_array_setitem(long jarg1, int jarg2, long jarg3);
  public final static native long initLM(int jarg1, int jarg2, int jarg3);
  public final static native long initVocab(int jarg1, int jarg2);
  public final static native long getIndexForWord(String jarg1);
  public final static native String getWordForIndex(long jarg1);
  public final static native int readLM(long jarg1, String jarg2);
  public final static native float getWordProb(long jarg1, long jarg2, long jarg3);
  public final static native float getProb_lzf(long jarg1, long jarg2, int jarg3, long jarg4);
  public final static native long getBOW_depth(long jarg1, long jarg2, int jarg3);
  public final static native float get_backoff_weight_sum(long jarg1, long jarg2, int jarg3, int jarg4);
  public final static native int getVocab_None();
  public final static native void write_vocab_map(long jarg1, String jarg2);
  public final static native void write_default_vocab_map(String jarg1);
  public final static native String getWordForIndex_Vocab(long jarg1, long jarg2);
  public final static native long getIndexForWord_Vocab(long jarg1, String jarg2);
}
