# coding=UTF-8
# This Python file uses the following encoding: utf-8

# for pre-train (python and java)

import copy
import json
import json as json
import os
import random
import re
import sys

import numpy as np
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


# github repository information storage path
Path = "/home/pretrain_data_code/"
# Save path of the downloaded file
savepath = "/home/pretrain_data_AST_tmp/"

AST = []
queue = []
parentQueue = []
code = []
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

def getNodeValue(code, start_point, end_point):
    if start_point[0]==end_point[0]:
        value=code[start_point[0]][start_point[1]:end_point[1]]
    else:
        value=""
        value+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            value+=code[i]
        value+=code[end_point[0]][:end_point[1]]   
    return value

@tail_call_optimized
def getAST(node, parentIndex=-1):
    index = len(AST)

    queue.extend(node.children)
    parentQueue.extend([index for _ in range(len(node.children))])

    json_node = {}

    # If there is no child node, value will be taken
    if(len(node.children)==0):
        value = getNodeValue(code, node.start_point, node.end_point)
        json_node["value"] = value

    json_node["type"] = node.type
    json_node["children"] = []
    if(parentIndex!=-1):
        AST[parentIndex]["children"].append(index)

    AST.append(json_node)

    if(len(queue)==0):
        return
    return getAST(queue.pop(0), parentIndex=parentQueue.pop(0))

def ProcessCode(code):
    """
    Delete blank lines
    Processing code blocks, INDENT represents the beginning of the block and DEDENT represents the end of the block.
    """
    lines = code.split("\n")
    codePathList = []
    indentationNum = 0
    recordIndentationNum = []
    for i in range(len(lines)):
        line=lines[i].replace("\n", "")
        rhs = line.lstrip()
        # The indentation of this line is larger than the previous line, which means that a new block of code is starting, add "INDENT" at the beginning of this line.
        num = len(line) - len(rhs)
        if(indentationNum < num):
            tmp = abs((indentationNum-num)/4) * "INDENT " + rhs
        # The indentation number of this line is smaller than the previous line, which means that the block of code is over, add "DEDENT" at the beginning of the previous line.
        elif(indentationNum > num):
            codePathList[i-1] = abs((indentationNum-num)/4) * "DEDENT " + codePathList[i-1]
            tmp =  rhs
        else:
            tmp = rhs
        indentationNum = num
        recordIndentationNum.append(indentationNum)
        # print('.' * (len(line) - len(rhs)) + rhs)
        codePathList.append(tmp)
    # End of Code
    codePathList[-1] = abs((0 - indentationNum)/4) * "DEDENT " + codePathList[-1]
    # 无缩进的行不加NEWLINE, TODO: 这里是不是有点问题
    # codePathList = ["NEWLINE "+codePathList[i] for i in range(len(codePathList)) if (codePathList[i] != "") and recordIndentationNum[i] != 0]
    targetCode = []
    for i in range(len(codePathList)):
        if (codePathList[i] != "") and recordIndentationNum[i] != 0:
            targetCode.append("NEWLINE "+codePathList[i])
        if (codePathList[i] != "") and recordIndentationNum[i] == 0:
            targetCode.append(codePathList[i])
    
    return " ".join(targetCode)

# Remove comments from the code, taking care before converting to AST.
def DeleteComment(s, code_type): 
    if(code_type=="py"): 
        s = re.sub(r'(#.*)', '', s)
        s= re.sub(r'(\'\'\')[\s\S]*?(\'\'\')', "", s, re.S)
        s= re.sub(r'(\"\"\")[\s\S]*?(\"\"\")', "", s, re.S)
    if(code_type=="java") or (code_type=="cs"): 
        s = re.sub(r'(\/\/.*)', '', s)
        s= re.sub(r'(\/\*)[\s\S]*?(\*\/)', "", s, re.S)
    return s

def getPathIndex(AST):
    paths = []
    path = []

    # data = eval(data) #use with dumps to remove u
    # Add root node
    path.append(0)

    GetPath(AST[0], paths, path)
    return paths

def GetPath(node, paths, path):
    '''
    Recursively get the root-terminal path from the parsed AST
    '''
    if len(node["children"])!=0:
        children = node.get('children')
        for i in children:
            child = AST[i]
            path.append(i)
            GetPath(child, paths, path)
        path.pop()
    else:
        tempPath = copy.deepcopy(path)
        path.pop()
        paths.append(tempPath)

def myreplace(matched):
    return " " + matched.group(0) + " "

def TMLM(AST,pathIndex,code):
    decoder_input = []
    decoder_output = []
    AST_mask_nodes = []

    # mask AST at encoder
    for path in pathIndex:
        p = len(path)
        q = np.array(range(p))
        select_node_pro_dis = np.exp(q-p) / np.sum(np.exp(q-p))
        for index, node in enumerate(path):
            if (random.random()<select_node_pro_dis[index])and(random.random()<0.15):
                if (index == len(path)-1):
                    try:
                        mask_node = AST[node].get('value').replace("\n", "").replace("\t", " ").replace("\/?", "").replace("\\", "")
                        mask_node = re.sub(r"[\W]", myreplace, mask_node)
                        mask_node = mask_node.split()
                    except:
                        mask_node = [AST[node].get('type')]

                    AST_mask_nodes.extend(mask_node)
                AST[node]['type'] = '<mask>'
                if 'value' in AST[node].keys():
                    AST[node]['value'] = '<mask>'

    # mask code at decoder
    code = code.replace("\n", "").replace("\t", " ").replace("\/?", "").replace("\\", "")
    code = re.sub(r"[\W]", myreplace, code)
    decoder_output = code.split()
    for index, token in enumerate(decoder_output):
        if(token not in AST_mask_nodes):
            decoder_input.append("<mask>")
        else:
            decoder_input.append(token)

    return AST, decoder_input, decoder_output

def NOP(AST):
    if random.random() > 0.5:
        return AST, 1
    else:
        position = random.sample(range(0,len(AST)),2)
        AST[position[0]]['type'], AST[position[1]]['type'] = AST[position[1]]['type'], AST[position[0]]['type']
        if ('value' in AST[position[0]].keys()) and ('value' in AST[position[1]].keys()):
            AST[position[0]]['value'], AST[position[1]]['value'] = AST[position[1]]['value'], AST[position[0]]['value']
        return AST, 0

def getNodePosEmCoeff(AST):
    AST[0]['coeff'] = 0.5
    for parent in AST:
        children = parent.get('children')
        if len(children)>0:
            children = parent.get('children')
            c  = len(children)
            for i, child in enumerate(children):
                coeff = (c-i)/(c+1.0)
                AST[child]['coeff'] = coeff
    return AST

def truncatCode(code, max_code_len):
    trunc_code = []
    token_num = 0
    code = code.split('\n')
    for line in code:
        tmp_tokens = []
        line = line.split()
        for token in line:
            tmp_tokens.append(token)
            token_num +=1
            if token_num>200:
                break
        str = " ".join(tmp_tokens)
        # Determine if it is a blank line
        if not len(line):
            continue
        trunc_code.append(str)
        if token_num>200:
            break
    return trunc_code

def ParseToASTPath(Path, savepath, max_code_len=200, max_node_num=20, max_path_num=100):
    files= os.listdir(Path)
    fail_num = 0    

    for index, file in enumerate(files):
        # num = 0
        with open(Path + file) as f1:
            
            # open save file
            savepath_tmp = savepath + "pretrain_data_%d" % index
            f = open(savepath_tmp, "w")
            s = f1.readlines()
            for line in tqdm(s,
                            desc="file %d" % (index),
                            total=len(s),
                            bar_format="{l_bar}{r_bar}"):
                # if(num>10):
                #     break
                # num += 1

                global code
                line = json.loads(line)
                try:
                    code = line["content"]
                    code_type =line['filepath'].split(".")[-1]
                except:
                    fail_num += 1
                    continue
                parser.set_language(lang[code_type])

                # Processing Code
                code = DeleteComment(code,code_type)
                # target_code = ProcessCode(target_code)
                code = truncatCode(code, max_code_len)
                
                # code = "public void RemovePresentationFormat(){MutableSection s = (MutableSection)FirstSection;s.RemoveProperty(PropertyIDMap.PID_PRESFORMAT);}"
                tree = parser.parse(bytes("\n".join(code), "utf8"))
                target_code = "\n".join(code)

                # Processing AST into json format, and list AST containing all ASTs
                global AST
                AST = []
                try:
                    getAST(tree.root_node, parentIndex=-1)
                except:
                    fail_num += 1
                    continue

                # Construct pre-training tasks
                pathIndex = getPathIndex(AST)
                if len(pathIndex)<2:
                    fail_num += 1
                    continue
                AST, decoder_input, decoder_output = TMLM(AST,pathIndex,target_code)
                AST, is_ast_order = NOP(AST)

                # Get the node position embedding coefficient
                getNodePosEmCoeff(AST)

                # Extracting paths from the AST
                paths = []
                coeffs = []
                for path in pathIndex:
                    tmp_path = []
                    coeff = []
                    for index, node in enumerate(path):
                        coeff.append(AST[node]['coeff'])
                        if('value' in AST[node].keys()):
                            node = AST[node].get('value').replace("\n", "").replace("\t", " ").replace("\/?", "").replace(" ", "_").replace("\\", "")
                        else:
                            node = AST[node].get('type')
                        tmp_path.append(node)
                        
                    paths.append(tmp_path[:max_node_num])
                    coeffs.append(coeff[:max_node_num])

                # Save the pre-trained dataset constructed for TMLM and NOP to a file.
                output = {"lan_type": code_type,
                        "encoder_input": paths[:max_path_num],
                        "decoder_input": decoder_input[:max_code_len],
                        "decoder_output": decoder_output[:max_code_len],
                        "is_ast_order": is_ast_order,
                        "node_pos_em_coeff": coeffs[:max_path_num]}
                f.write(json.dumps(output))
                f.write("\n")
            f.close()

    
    print("fail number:", fail_num)


if __name__ == "__main__":
    ParseToASTPath(Path,savepath,max_code_len=200, max_node_num=20, max_path_num=100)
