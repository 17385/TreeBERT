import sys
import json as json
import ast
import sys
import json
import copy
import re
from tqdm import tqdm

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError:
        return False
    return True

def PrintUsage():
    sys.stderr.write("""
Usage:
    parse_python.py <file>

""")
    exit(1)

def read_file_to_string(filename):
    f = open(filename, 'rt')
    s = f.read()
    f.close()
    return s

def parseCode(code):
    tree = ast.parse(code)
    
    json_tree = []
    def gen_identifier(identifier, node_type = 'identifier'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        json_node['value'] = identifier
        return pos
    
    def traverse_list(l, node_type = 'list'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        children = []
        for item in l:
            children.append(traverse(item))
        if (len(children) != 0):
            json_node['children'] = children
        return pos
        
    def traverse(node):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = type(node).__name__
        children = []
        if isinstance(node, ast.Name):
            json_node['value'] = node.id
        elif isinstance(node, ast.Num):
            json_node['value'] = unicode(node.n)
        elif isinstance(node, ast.Str):
            json_node['value'] = node.s.decode('utf-8')
        elif isinstance(node, ast.alias):
            json_node['value'] = unicode(node.name)
            if node.asname:
                children.append(gen_identifier(node.asname))
        elif isinstance(node, ast.FunctionDef):
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ClassDef):
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                json_node['value'] = unicode(node.module)
        elif isinstance(node, ast.Global):
            for n in node.names:
                children.append(gen_identifier(n))
        elif isinstance(node, ast.keyword):
            json_node['value'] = unicode(node.arg)
        

        # Process children.
        if isinstance(node, ast.For):
            children.append(traverse(node.target))
            children.append(traverse(node.iter))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.If) or isinstance(node, ast.While):
            children.append(traverse(node.test))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.With):
            children.append(traverse(node.context_expr))
            if node.optional_vars:
                children.append(traverse(node.optional_vars))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.TryExcept):
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.handlers, 'handlers'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.TryFinally):
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.finalbody, 'finalbody'))
        elif isinstance(node, ast.arguments):
            children.append(traverse_list(node.args, 'args'))
            children.append(traverse_list(node.defaults, 'defaults'))
            if node.vararg:
                children.append(gen_identifier(node.vararg, 'vararg'))
            if node.kwarg:
                children.append(gen_identifier(node.kwarg, 'kwarg'))
        elif isinstance(node, ast.ExceptHandler):
            if node.type:
                children.append(traverse_list([node.type], 'type'))
            if node.name:
                children.append(traverse_list([node.name], 'name'))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.ClassDef):
            children.append(traverse_list(node.bases, 'bases'))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.decorator_list, 'decorator_list'))
        elif isinstance(node, ast.FunctionDef):
            children.append(traverse(node.args))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.decorator_list, 'decorator_list'))
        else:
            # Default handling: iterate over children.
            # ast.iter_child_nodes:Yield all direct child nodes of node, that is, all fields that are nodes and all items of fields that are lists of nodes.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr_context) or isinstance(child, ast.operator) or isinstance(child, ast.boolop) or isinstance(child, ast.unaryop) or isinstance(child, ast.cmpop):
                    # Directly include expr_context, and operators into the type instead of creating a child.
                    json_node['type'] = json_node['type'] + type(child).__name__
                else:
                    children.append(traverse(child))
                
        if isinstance(node, ast.Attribute):
            children.append(gen_identifier(node.attr, 'attr'))
                
        if (len(children) != 0):
            json_node['children'] = children
        return pos
    
    traverse(tree)
    return json.dumps(json_tree, separators=(',', ':'), ensure_ascii=False)

def SaveToFile(filename, contents):
    f = open(filename, 'w')
    for index, content in enumerate(contents):
        try:
            f.write(content)
            f.write('\n')
        except:
            print(index)
            print(content)
    f.close()

def GetPaths(AST):
    paths = []
    path = []
    global data 
    data = json.loads(AST)
    # data = eval(data)
    path.append(data[0].get("type"))

    GetPath(data[0], paths, path)
    return paths

def ReadCodePathList(filename, fileNum=100000):
    f =open(filename, "r")
    line = f.readline()
    codePathList = []
    while(fileNum > 0):
        line=line.strip('\n')
        codePathList.append(line)
        fileNum = fileNum - 1
        line = f.readline()
    f.close()
    return codePathList


def GetPath(node, paths, path):
    
    if node.has_key('children'):
        children = node.get('children') ##chiledren list 
        for i in children:
            child = data[i]
            if child.has_key('value'):
                path.append(((child.get('value')).replace('\n',' ')).replace('\t',' '))
            elif child.has_key('type'):
                path.append(child.get('type'))
            GetPath(child, paths, path)
        path.pop()
    else:
        tempPath = copy.deepcopy(path)
        path.pop()
        paths.append("|".join(tempPath))


def ProcessCode(code):
    lines = code.split("\n")
    codePathList = []
    indentationNum = 0
    recordIndentationNum = []
    for i in range(len(lines)):
        line=lines[i].replace("\n", "")
        rhs = line.lstrip()
        num = len(line) - len(rhs)
        if(indentationNum < num):
            tmp = abs((indentationNum-num)/4) * "INDENT " + rhs
        elif(indentationNum > num):
            codePathList[i-1] = abs((indentationNum-num)/4) * "DEDENT " + codePathList[i-1]
            tmp =  rhs
        else:
            tmp = rhs
        indentationNum = num
        recordIndentationNum.append(indentationNum)
        # print('.' * (len(line) - len(rhs)) + rhs)
        codePathList.append(tmp)
 
    codePathList[-1] = abs((0 - indentationNum)/4) * "DEDENT " + codePathList[-1]
    targetCode = []
    for i in range(len(codePathList)):
        if (codePathList[i] != "") and recordIndentationNum[i] != 0:
            targetCode.append("NEWLINE "+codePathList[i])
        if (codePathList[i] != "") and recordIndentationNum[i] == 0:
            targetCode.append(codePathList[i])
    
    return " ".join(targetCode)


def DeleteComment(s):   
    s = re.sub(r'(#.*)', '', s)
    s= re.sub(r'(\'\'\')[\s\S]*?(\'\'\')', "", s, re.S)
    s= re.sub(r'(\"\"\")[\s\S]*?(\"\"\")', "", s, re.S)
    return s

def processCodeFile(codeFilePath, ASTSavePath, PathSavePath, targetCodeSavePath):
    ASTS = []
    codePathList = ReadCodePathList(codeFilePath, fileNum=10000)
    targetCodes = []
    uncompileNum = 0
    for codePath in tqdm(codePathList):
        try:
            f = open(codePath, 'rt')
            s = f.read()
            f.close()
        
            sourceCode = DeleteComment(s)
            targetCode = ProcessCode(sourceCode)
            AST = parseCode(sourceCode)
            ASTS.append(AST)   
            targetCodes.append(targetCode)
        except:
              uncompileNum = uncompileNum + 1
        
    
    print("uncompile file num:", uncompileNum)

    SaveToFile(targetCodeSavePath, targetCodes)
    SaveToFile(ASTSavePath, ASTS)

    f = open(ASTSavePath, "r")
    pathsTrainData = []
    line = f.readline()
    while line:
        paths = GetPaths(line)
        pathsTrainData.append("\t".join(paths))
        line = f.readline()
    f.close()

    SaveToFile(PathSavePath, pathsTrainData)

def processCodeSnippet(codePath, PathSavePath, targetCommentSavePath):
    def getCode(filename):
        f =open(filename, "r")
        line = f.readline()
        code = []
        comment = []
        while(line):
            line = line.split("\t")
            code.append(line[3])
            comment.append(line[2])
            line = f.readline()
        f.close()
        return code, comment
    
    codes, comments = getCode(codePath)
    ASTS = []
    targetCodes = []
    label = []
    num = 0.0
    total_num = 0.0
    for index, code in enumerate(codes):
        total_num = total_num + 1.0
        sourceCode = DeleteComment(code)
        sourceCode = sourceCode.replace("\\n","\n")

        try:
            AST = parseCode(unicode(sourceCode))
            if len(AST)> 200  > 50:
                ASTS.append(unicode(AST)) 
                label.append(comments[index])
        except:
            num = num + 1

    print(num)       
    print("error rate", num/total_num)

    SaveToFile(targetCommentSavePath, label)

    pathsTrainData = []
    for line in ASTS:
        paths = GetPaths(line)
        pathsTrainData.append("\t".join(paths))

    SaveToFile(PathSavePath, pathsTrainData)


def processCodeSum(codePath, PathSavePath, targetCommentSavePath):
    def getCode(filename):
        f =open(filename, "r")
        line = f.readline()
        code = []
        comment = []
        while(line):
            line = json.loads(line)
            if line["label"].find("main") == -1:
                code.append(line["function"])
                comment.append(line["label"])
            line = f.readline()
        f.close()
        return code, comment
    
    codes, comments = getCode(codePath)
    ASTS = []
    label = []
    num = 0.0
    total_num = 0.0
    for index, code in enumerate(codes):
        total_num = total_num + 1.0
        sourceCode = DeleteComment(code)
        sourceCode = sourceCode.replace("\t","")        
        try:
            AST = parseCode(unicode(sourceCode))
            ASTS.append(unicode(AST)) 
            label.append(comments[index])
        except:
            num = num + 1

    print(num)       
    print("error rate", num/total_num)

    SaveToFile(targetCommentSavePath, label)
    pathsTrainData = []
    for line in ASTS:
        paths = GetPaths(line)
        pathsTrainData.append("\t".join(paths))

    SaveToFile(PathSavePath, pathsTrainData)
    

if __name__ == "__main__":
    
    # processCodeFile(CODE_PATH, AST_SAVE_PATH, PATHS_TRAIN_DATA_SAVE_PATH, TARGET_CODE_SAVE_PATH)

    # processCodeSnippet(CODE_PATH_1, PATHS_TRAIN_DATA_SAVE_PATH_1, TARGET_COMMENT_SAVE_PATH_1)

    # processCodeSum(CODE_PATH_2, PATHS_TRAIN_DATA_SAVE_PATH_2, TARGET_COMMENT_SAVE_PATH_2)
    pass
