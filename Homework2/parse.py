from providedcode import dataset
import sys
from providedcode.transitionparser import TransitionParser
from providedcode.dependencygraph import DependencyGraph

if __name__ == '__main__':

  testdata = []
  for line in sys.stdin:
    sentence = DependencyGraph.from_sentence(line)
    testdata.append(sentence)
  model = sys.argv[1]

  tp = TransitionParser.load(model)
  parsed = tp.parse(testdata)
 
  for p in parsed:
    print(p.to_conll(10).encode('utf-8'))

