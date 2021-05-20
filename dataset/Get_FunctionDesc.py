import os
import time
import linecache
import re
import json
import chardet

pySrcListFilePath = './python100k_train.txt'
pySrcDetailFileDir = '/data.tar/'
pySrcRealyFilePath = ''
saveJsonLogPath = './result100_1.json'

func_name_indexList = []
cur_func_name = ''
cur_func_body_index = 0
func_name_List = []
replace_def_sign=[]
func_BodyContent = []

pattern_def1 = re.compile(r'^def (.*)\(.+\):$')
pattern_def2 = re.compile(r'^def (.*)\(.*$')
pattern_class1 = re.compile(r'^class (.*)\(.+\):$')
pattern_class2 = re.compile(r'^class (.*)\(.*:$')


def getCode(s):
    def RepleaceName(matched):
        return matched.group("space")+""+matched.group("def") + "___" + matched.group("parm")

    code = re.sub(r'(?P<space>\s*)(?P<def>def )(?P<name>.*)(?P<parm>\(.*\))', RepleaceName, s)
    return code


def Match_Def(str):
    global func_name_List, cur_func_name
    if (pattern_def1.search(str) == None and pattern_def2.search(str) == None):
        return False
    else:
        func_name = pattern_def2.findall(str)
        func_name_List.append(func_name[0])
        print(func_name_List)
        return True


def Match_Class(str):
    if (pattern_class1.search(str) == None and pattern_class2.search(str) == None):
        return False
    else:
        return True


def Get_FuncBodyName(pyFile):
    global func_name_indexList, func_BodyContent, cur_func_name, cur_func_body_index,replace_def_sign

    with open(pyFile, 'rb') as f:
        cur_encoding = chardet.detect(f.read())['encoding']

    countLines = len(open(pyFile, 'r', encoding=cur_encoding).readlines())

    for i in range(1, countLines + 1):
        readline = linecache.getline(pyFile, i)#.strip()

        if Match_Def(readline) == True:
            func_name_indexList.append(i)
            cur_func_body_index = i + 1
            replace_def_sign.append(getCode(readline))

    total_funcs = len(func_name_List)

    for j in range(total_funcs):
        cur_func_name = func_name_List[j]
        func_BodyContent.clear()
        func_BodyContent.append('\t')
        func_BodyContent.append(replace_def_sign[j])
        try:
            last = func_name_indexList[j + 1]
        except IndexError as e:
            last = countLines + 1
        for k in range(func_name_indexList[j] + 1, last):
            readlineTemp = linecache.getline(pyFile, k)
            if Match_Class(readlineTemp.strip()) == True:
                break
            elif readlineTemp.find('):') != -1:
                continue
            elif readlineTemp.isspace() == True or len(readlineTemp) == 0:
                continue
            else:
                func_BodyContent.append('\t')
                func_BodyContent.append(readlineTemp)

        result_func = {
            "function": ''.join(func_BodyContent),
            "label": cur_func_name
        }
        with open(saveJsonLogPath, 'a+', encoding='gbk') as f:
            json.dump(result_func, f)
            f.write('\n')
        f.close()


def EumPyPath():
    if not os.path.isfile(pySrcListFilePath):
        return
    countSrcPy = len(open(pySrcListFilePath, 'r', encoding='utf-8').readlines())
    for i in range(0, countSrcPy + 1):
        func_name_indexList.clear()
        func_name_List.clear()

        strLine = linecache.getline(pySrcListFilePath, i).rstrip()
        pySrcRealyFilePath = '{}{}'.format(pySrcDetailFileDir, strLine)
        print('----------------------------------', i)
        if not os.path.isfile(pySrcRealyFilePath):
            print('numb {} line missing py src file.....'.format(i))
            continue
        try:
            Get_FuncBodyName(pySrcRealyFilePath)
        except UnicodeDecodeError as e:
            continue
        time.sleep(0.1)
    pass


def Sorting_json():
    with open(saveJsonLogPath, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) > 200:
                with open('./result100_2.json', 'a+', encoding='utf-8')as ff:
                    ff.write(line)
                ff.close()
            else:
                print(line)


if __name__ == '__main__':
    # EumPyPath()
    # Sorting_json()
    pass
