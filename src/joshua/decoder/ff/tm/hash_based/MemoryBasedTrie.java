/*
 * This file is part of the Joshua Machine Translation System.
 * 
 * Joshua is free software; you can redistribute it and/or modify it under the terms of the GNU
 * Lesser General Public License as published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with this library;
 * if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307 USA
 */
package joshua.decoder.ff.tm.hash_based;

import java.util.Collection;
import java.util.HashMap;

import joshua.decoder.ff.tm.RuleCollection;
import joshua.decoder.ff.tm.Trie;

/**
 * @author Zhifei Li, <zhifei.work@gmail.com>
 * @version $LastChangedDate$
 */
public class MemoryBasedTrie implements Trie {
  MemoryBasedRuleBin ruleBin = null;
  HashMap<Integer, MemoryBasedTrie> childrenTbl = null;


  /* See Javadoc for Trie interface. */
  public MemoryBasedTrie match(int sym_id) {
    if (null == childrenTbl) {
      return null;
    } else {
      return childrenTbl.get(sym_id);
    }
  }

  /* See Javadoc for Trie interface. */
  public boolean hasExtensions() {
    return (null != this.childrenTbl);
  }

  public HashMap<Integer, MemoryBasedTrie> getExtensionsTable() {
    return this.childrenTbl;
  }

  public void setExtensions(HashMap<Integer, MemoryBasedTrie> tbl_children_) {
    this.childrenTbl = tbl_children_;
  }

  /* See Javadoc for Trie interface. */
  public boolean hasRules() {
    return (null != this.ruleBin);
  }


  public void setRuleBin(MemoryBasedRuleBin rb) {
    ruleBin = rb;
  }

  /* See Javadoc for Trie interface. */
  public RuleCollection getRuleCollection() {
    return this.ruleBin;
  }

  /* See Javadoc for Trie interface. */
  public Collection<MemoryBasedTrie> getExtensions() {
    return this.childrenTbl.values();
  }

}
