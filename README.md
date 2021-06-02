# TreeBERT
This is an implementation of the model described in: **TreeBERT: A Pre-Trained Model Based on Abstract Syntax Tree for Programming**.   
TreeBERT is a tree-based pre-trained model for improving programming language-oriented generation tasks. To utilize the tree structure, TreeBERT represents the AST corresponding to the code as a set of composition paths and introduces node position embedding. The model is trained by tree masked language modeling (TMLM) and node order prediction (NOP) with a hybrid objective. 

The pre-trained TreeBERT can be applied to a wide range of program-oriented generation tasks and without extensive modification architecture. We have currently applied TreeBERT to two PL-oriented generation tasks: code summarization and code documentation.
## Requirements
* Python2 (for processing the code into an AST)
* Python3
* Numpy
* Pytorch 1.7.0
* Tqdm

## Pre-training Data Ready
The pre-training dataset we use is the Python and Java pre-training corpus published by [CuBERT](https://github.com/google-research/google-research/tree/master/cubert).   
By running `dataset\ParseTOASTPath.py` you can transform the code snippet into an AST, extract the paths from the root node to the terminal nodes in the AST, and standardize the format of the target code snippet.

## Fine-tuning Data Ready
### Code Summarization
In the code summarization task, we further evaluate TreeBERT on two datasets, Python and Java, where the python dataset uses [py150](https://www.sri.inf.ethz.ch/py150) , and the Java dataset uses [java-small](https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz), [java-med](https://s3.amazonaws.com/code2seq/datasets/java-med.tar.gz), and [java-large](https://s3.amazonaws.com/code2seq/datasets/java-large.tar.gz).
These data sets can be processed into the form required by code summarization by running `dataset\Get_FunctionDesc.py`.
```
{
    "function": Function, its function name is replaced by "__",
    "label": Function Name
}
```
### Code Documentation
We use the Java dataset provided by [DeepCom](https://github.com/xing-hu/DeepCom/blob/master/data.7z) to fine-tune our model in the code documentation task.
## Pre-training
#### 1. Create vocab:
```
python dataset/vocab.py -sc data/ProcessedData/SourcePath.txt -tc data/ProcessedData/TargetCode.txt -o data/vocab_2000.small --num_merges 2000
```
#### 2.Training TCBERT using GPU (no test):
```
python __main__.py -d data/ProcessedData -v  data/ProcessedData/vocab_2000.small  -o  output/tcbert.model --with_cuda True
```
## Adapting TreeBERT to Downstream Tasks
#### Code Summarization
```
finetune -d data/fine_tune_CodeDoc \
-s data/DeepCom(python)/FineTunePath_10.txt\
-t data/DeepCom(python)/FineTuneTargetComment_10.txt\
-m output/tcbert.model.ep0\
-sd data/DeepCom(python)/FineTunePath_10.txt\
-td data/DeepCom(python)/FineTuneTargetComment_10.txt\
-v data/vocab.small\
-o output/CodeDoc/CodeDoc.model\
--with_cuda True
```
#### Code Documentation
```
finetune -d data/fine_tune_CodeSum\
-s  data/py150_code_sum/CodeSumPath_20.txt\
-t  data/py150_code_sum/CodeSumComment_20.txt\
-m output/tcbert.model.ep0\
-sd  data/py150_code_sum/CodeSumPath_20.txt
-td  data/py150_code_sum/CodeSumComment_20.txt\
-v  data/vocab.small
-o  output/CodeSum/CodeSum.model\
--with_cuda True
```
