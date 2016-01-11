UNI: qc2200   Name: Qianyuan Chen

1) b.
For all arcs in this dependency graph, we check if there exists an arc overlapping with it, i.e. for arc (a, b), we check if there exists an arc (c, d) where {c<a and a<d<b} or {a<c<b and d>b}. If (c, d) exists, this dependency graph is not projective, otherwise it is. 

1) c.
Projective:
Their bios, which are attached, attest to the excellence of their work and show clearly their promise in both research and teaching.  

Non-projective:
I ate a pizza tonight which contains only cheese and chocolate.


2) b. 
UAS: 0.229038040231 
LAS: 0.125473013344

Report: 
The UAS and LAS are 22.9% and 12.5% respectively using the Swedish dataset. This means that only about 23% of all the dependency relationship were found out by the badfeatures.model, and only about 12% of the dependency arcs within the relationship parsed out were correctly labeled. Compared with the current depencency parser, the performance is much lower than the average and needs to be improved.


3) a.
Feature 1 DIS: distance between the word at the top of stack and the word at start of buffer

Implementation: using the ‘address’ attribute for each token to get the index in the sentence of two words, and calculation the difference as:
    tokens[buffer[0]]['address']-tokens[stack[-1]]['address']

Complexity: o(1) since the ‘address’ attribute has already been stored in the token list object, obtaining and calculating the distance does not need additional search or calculation. However if we do not have index for each word, getting the distance feature will be o(n).

Performance: good, by adding distance alone can improve the LAS about 0.1-0.12, but better to apply with other feature. Also I have tried counting the VERB distance (number of verds between two words) as suggested in the book, but the performance is worth than this.


Feature 2 0_POSTAG: the POSTAG at the top of stack and at the start of buffer

Implementation: using the ‘tag’ attribute for each token to get the POSTAG for each word.

Complexity: o(1) since the ‘tag’ has already been stored in the token list and we can directly get it.

Performance: excellent! Using POSTAG feature alone can improve the LAS about 0.2-0.25. Since the dependency information is closely related to the syntax, the part of speech information will be a great feature for the classifier. Also I have compared the performance between coarse-grained tag and fine-grained tag, and find out that fine-grained tag is a little better than ctag.


Feature 3 1_POSTAG: the POSTAG at the second top of stack and at the second of buffer

Implementation: similar with Feature 2, using ‘tag’ attribute for each token

Complexity: o(1) since the ‘tag’ has already been stored in the token list and we can directly get it.

Performance: good. Adding alone is about 0.1 improvement, and the further 0.02-0.04 after adding Feature 2. This feature is similar with the idea of bigram, which also using the information of surrounding words to help determining the information of this word. So combining Feature 2 and 3, totally 4 tags are taken into consideration for each classification. 


Feature 4 NCHILD: the number of children (dependents) of the word at the top of stack and the start of buffer

Implementation:  counting the number of dependents from the attribute list ‘deps’ stored in the token list, using method len() to count the number

Complexity: o(1) this is equal to the complexity of calculating the length of the list ‘deps’ (len(deps)) which is o(1) in Python. If we don’t have the deps list, the complexity of extracting this feature will be o(n) since we need to traverse the whole sentence to find the dependent.

Performance: good. Applying alone can have about 0.15 improvement on LAS, and combining with the 0_POSTAG feature can have a great increasing in both UAS and LAS.


3) c.
Swedish:
UAS: 0.862178848835 
LAS: 0.737104162517

Danish:
UAS: 0.872255489022 
LAS: 0.777045908184

English:
UAS: 0.911949685535 
LAS: 0.874213836478

3) d.
Complexity: Since the arc-eager shift-reduce parser using the Greedy strategy to make decision, it’s quite fast compared with other dynamic programming based parsers. The arc-eager parser will scan everyone of N tokens in the input sentence, and make a transition based on oracle using o(1) complexity, so the overall complexity is o(N).

Tradeoff: The arc-eager parser will not search for ALL possible states because of the Greedy strategy, so it may lose little accuracy and not finding the global optimal solution.



