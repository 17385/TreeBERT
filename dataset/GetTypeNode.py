import json
import os
import sys

from tqdm import tqdm
from tree_sitter import Language, Parser

CS_LANGUAGE = Language('build_parser/languages_java_py_cs.so', 'c_sharp')
JA_LANGUAGE = Language('build_parser/languages_java_py_cs.so', 'java')
PY_LANGUAGE = Language('build_parser/languages_java_py_cs.so', 'python')

lang = {
    "py" : PY_LANGUAGE,
    "java" : JA_LANGUAGE,
    "cs" : CS_LANGUAGE
}
parser = Parser()

Path = "/home/pretrain_data_code/"
savepath = "data/type_list_"
files= os.listdir(Path)
fail_num = 0
queue = []
code = []
pytypeNode = []
javatypeNode = []
cstypeNode = []

typeNode = {
    "py" : pytypeNode,
    "java" : javatypeNode,
    "cs" : cstypeNode
}

class TailRecurseException(BaseException):
  def __init__(self, args, kwargs):
    self.args = args
    self.kwargs = kwargs

def tail_call_optimized(g):
  """
  This function decorates a function with tail call
  optimization. It does this by throwing an exception
  if it is it's own grandparent, and catching such
  exceptions to fake the tail call optimization.
  
  This function fails if the decorated
  function recurses in a non-tail context.
  """
  def func(*args, **kwargs):
    f = sys._getframe()
    if f.f_back and f.f_back.f_back \
        and f.f_back.f_back.f_code == f.f_code:
      raise TailRecurseException(args, kwargs)
    else:
      while 1:
        try:
          return g(*args, **kwargs)
        except TailRecurseException as e:
          args = e.args
          kwargs = e.kwargs
  func.__doc__ = g.__doc__
  return func


@tail_call_optimized
def getTypeNode(node, code_type):
    queue.extend(node.children)
    type = node.type

    if type not in typeNode[code_type]:
        typeNode[code_type].append(type)
    
    if(len(queue)==0):
        return
    return getTypeNode(queue.pop(0), code_type)


code_num = {"py":100, "java":100, "cs":100}
for index, file in enumerate(files):
    with open(Path + file) as f1:
        s = f1.readlines()
        for line in tqdm(s,
                        desc="file %d" % (index),
                        total=len(s),
                        bar_format="{l_bar}{r_bar}"):

            line = json.loads(line)
            code = line["content"]

            code_type =line['filepath'].split(".")[-1]

            # Set parser language type
            parser.set_language(lang[code_type])

            if (code_type in code_num.keys()):
                if code_num[code_type] > 0:
                    code_num[code_type] -= 1
                else:
                    break
            
            tree = parser.parse(bytes(code, "utf8"))

            try:
                getTypeNode(tree.root_node, code_type)
            except:
                fail_num += 1
                continue

# open save file
for type in typeNode:
    savepath_tmp = savepath + type
    with open(savepath_tmp, "w") as f:
        f.write(" ".join(typeNode[type]))
