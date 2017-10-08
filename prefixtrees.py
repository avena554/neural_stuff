"""
Trees of unbounded arrity with labeled edges and no shared labels between the outgoing edges of node.
labeled egdes to children nodes are implemented with python dictionaries.
"""
class DTree:
    #Side effects (no copy of the descendants list) to allow efficient node insertion in intermediate positions. If no children given, initializes as a tree with no descendants.
    def __init__ (self, children=None):
        if(children!=None):
            self._children=children
        else:
            self._children={}
            
    #return the node reached through the edge with given label, or None if there is no such node.
    def child(self, label):
        if label in self._children:
            return self._children[label]
        else:
            return None
        
    def edges(self):
        return self._children

    def labels(self):
        #print(self._children)
        return self._children.keys()

    #add an edge to target node with given label
    #if an edge with that label already exists it is overwritten
    def add_edge(self, label, node):
        self._children[label]=node

    #replace the set of edges with those provided by a dictionary. With side effects (non copy)
    def replace_edges(self, dict):
        self._children=dict


#work in progress
"""
class NodeFunctionI:
    def apply(label, node):
        raise NotImplementedError("Interface method has no implementation")
        
class TreeReducerI:
    def reduce(reducer, arg0, l0, tree):
        raise NotImplementedError("Interface method has no implementation")

    def remember(arg):
        raise NotImplementedError("Interface method has no implementation")

class PreOrderTreeReducer(TreeReducerI):
    def reduce(function, arg, l, tree):
        next_arg = function.remember(function.apply(arg, l, tree))        
        for l in tree.labels():
            reduce(function, next_arg, l, tree.child(l))
"""


"""
Prefix tree: We use the variable-arity edge-labeled trees of DTree to build a structure (Trie, or PrefixTree) allowing insertion and retrieval of a word w in O(|w|). We use a tree of prefixes, in which the concatenation of edges' labels and node factors along a path from the root represent words. A node x corresponds to a common prefix for all its descendants, given by l1.f_x1.l2.f_x2.....l_kf_xk where x=x_k and x1...x_n is the path from the root to x, and f_xk is the factor of x_k.
"""
class PrefixTree(DTree):
    def __init__(self, children=None, factor=""):
        DTree.__init__(self,children)
        self.factor=factor

    
    """
    Walks the tree down a path fixed by a given prefix p, as long as it is feasible, then returns the last visited node n, its parent node p, the label of the edge from p to n, the first index i at which the prefix disagrees with the tree, and an offset o corresponding to the number of character of n's factor that were consumed during the walk as they matched the appropriate part of the prefix (if the last visited node was reached after consuming the character at index i_n in the prefix, then i=i_n+o)
    """
    def _walk_prefix(self, prefix):
        index=0
        offset=0
        factor=self.factor
        c_node=self
        parent=None
        c=None
        #walk down the tree (consuming factors along the way) along the prefix path
        while (index<len(prefix)):
            #if there is no pending factor to consume
            if(offset==len(factor)):
                #read next charater in the prefix
                c=prefix[index]
                #walk down the tree
                child=c_node.child(c)
                if(child!=None):
                    offset=0
                    factor=child.factor
                    parent=c_node
                    c_node=child
                else:
                    #could not find a descendant for the current letter
                    break                    
            else:
                #reads the factor and the prefix simultaneously.
                if(factor[offset]!=prefix[index]):
                    #factor and prefix split ways at the current index
                    break
                offset+=1
            index+=1
            
        return (c_node, parent, c, index, offset)
    
    @staticmethod
    def _insert_missing_prefix(prefix, node, parent, label, index, offset):
        inserted=None
        split_node=node
        #If the prefix to insert disagrees with the tree at some position inside a given factor (means we didn't manage read a complete factor along the prefix--offset < len(node.factor)), we need to split this factor adequately.
        if(offset<len(node.factor)):
            #We insert a new node between the node and its descendants, thus breaking the factor into two parts.
            split_char=node.factor[offset]        
            split_node=PrefixTree(children={split_char:node})
            #split_node.value=node.value
            parent.add_edge(label, split_node)
            #node.value=None
            #now we need to adjust factors
            #the following is maybe not as optimal as it could be regarding string manipulations
            split_node.factor=node.factor[0:offset]
            
            node.factor=node.factor[offset+1:]
            
                    
            inserted=node
        #Otherwise the prefix to insert disagrees with the tree right after a complete factor, i.e., there is no descendent from the prefix represented by node following an edge labelled by prefix[index+1].
        else:
            #insertion will be simply handled by the missing suffix handler code below, as in the splitting case.
            split_node=node
        
            #If we are in the case where a suffix was missing from the tree, insert it at the appropriate position (the other case arrises when the prefix ended before we consumed all of the node's factor. In that case, splitting is enough).
        if((index)<len(prefix)):
            suffix_node=PrefixTree(children={},factor=prefix[index+1:])
            split_node.add_edge(prefix[index], suffix_node)
            inserted=suffix_node

        return inserted
        
        
    """
    look up a given prefix in the tree, if it is absent and f_insert is set to true, inserts it. Returns the corresponding node or None if it was not found nor inserted.
    """
    def _lookup(self, prefix, f_insert=True):
        (node, parent, label, index, offset)=self._walk_prefix(prefix)
        p_node=node
        #if we didn't consume all of the prefix, or we did not finish to consume the last factor, then there is not yet a node representing the prefix in the tree, we must add it.
        if(index!=len(prefix) or offset!=len(node.factor)):
            if(f_insert):
                p_node= PrefixTree._insert_missing_prefix(prefix, node, parent, label, index, offset)
            else:
                p_node=None
        return p_node

"""
Dictionary obtained from the PrefixTree by introducing a special end symbol and decorating the nodes corresponding with prefixes ending with the end symbol with a value
"""
class TrieDict():
    def __init__(self, end_symbol="</$>"):
        self._end_symbol=end_symbol
        self.tree=PrefixTree()
        
    """
    Get the value stored for a given prefix
    """
    def get_value(self,prefix):
        p_node=self.tree._lookup(prefix+self._end_symbol, f_insert=False)
        if(p_node!=None):
            return p_node.value
        else:
            raise KeyError(prefix)
            
    """
    Set the value for a given prefix. If the prefix is not yet in the tree, it is added first.
    """
    def set_value(self, prefix, value):
        p_node=self.tree._lookup(prefix+self._end_symbol, f_insert=True)
        p_node.value=value

    """
    tell wether a node reached though an edge labeled 'label' with factor 'factor' is a terminal node (thus bears a value)
    """
    def _is_terminal(self, label, factor):
        return ((label==self._end_symbol) or (label==self._end_symbol[0] and factor==self._end_symbol[1:])) or factor.endswith(self._end_symbol)
    
    """
    auxiliary to the printing function
    """
    def _aux_str(self, node, pref, label):
        s=node.factor
        if(self._is_terminal(label, node.factor)):
            s+=": "+str(node.value)
        for label in node.labels():
            s+="\n"
            s+=pref+"-->"+label+self._aux_str(node.child(label), pref+"   ", label)
                    
        return s

    """
    represent the tree as a (multiline string)
    """
    def __str__(self):
        return "<$>"+self._aux_str(self.tree, "", "")    


    

    
"""
Interface specifying actions (visit_prefix and visit_terminal) to be performed on every prefix, respectively, on every encountered (possibly intermediary) prefix of the tree and every terminal (that is, bearing a value) prefix of the tree.
"""
"""
class PrefixFunctionI():
    
    def visit_prefix(prefix):
        raise NotImplementedError("interface method has no implementation")
    
    def visit_terminal(prefix, value):
        raise NotImplementedError("Interface method has no implementation")
""" 
"""
Make a node function out of a prefix function, suitable for a depth-first in-order traversal of the prefixes in the tree. 
"""
"""
class PrefixPreorderNodeFunction(NodeFunctionI):
    def __init__(trie_dict, p_fun):
        self.trie = trie_dict
        #use a queue instead
        self.prefix = []
        self.p_fun = p_fun

    def visit(arg, label, node):
        self.prefix.insert(label, len(self.prefix))
        self.p_fun.visit_prefix("".join(self.prefix))
        if(self.trie_dict._is_terminal(node))
        
    def remember(arg):
        return arg

class PreOrderTreeTraversal(TreeTraversalI):

    def __init__():
        self.p = []
        self.l = ''
        
    def accept(visitor, tree):
        prefix="".join(self.p)+self.l+tree.factor
        visitor.visit(prefix)
        if(self._is_terminal(l, tree.factor)):
            visitor.visit_terminal(prefix, tree.value)
            
        self.p.extend(self.l)
        for l in tree.labels():
            self.l=l
            self.accept(visitor, tree.child(l))
                
""" 
 
    
  
