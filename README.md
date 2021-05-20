# TreeBERT

This is an implementation of the model described in:**TreeBERT: A Pre-Trained Model Based on Abstract Syntax Tree for Programming**.   
TreeBERT is a programming language (PL) oriented AST path-based pre-training model that learns code representations through a mixture of Path Masked Language Modeling and Node Order Prediction training targets in order to fully learn the syntactic structure of programs and improve the model generation capabilities.The PMLM and NOP mixed target training process is schematically shown below.
![](figure\model-arch.png)  
TreeBERT can be applied directly to PL downstream generation tasks without modification by fine-tune.  
## Requirements
* Python2 (for processing the code into an AST)
* Python3
* Numpy
* Pytorch 1.7.0
* Tqdm

## Pre-training Data Ready
The pre-training dataset we use is the Python and Java pre-training corpus published by [CuBERT](https://github.com/google-research/google-research/tree/master/cubert).   
By running `dataset\ParseTOASTPath.py` you can transform the code fragment into an AST, extract the path from the root node to the terminal node in the AST, and standardize the format of the target code fragment.

## Fine-tuning Data Ready
In the code summarization task, we further evaluate TreeBERT on two datasets, Python and Java, where the python dataset uses [py150](https://www.sri.inf.ethz.ch/py150) , and the Java dataset uses [java-small](https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz), [java-med](https://s3.amazonaws.com/code2seq/datasets/java-med.tar.gz), and [java-large](https://s3.amazonaws.com/code2seq/datasets/java-large.tar.gz).
These data sets can be processed into the form required by the code summarization task by running `dataset\Get_FunctionDesc.py`.
```
{
    "function": Function, its function name is replaced by "__",
    "label": Function Name
}
```
## Pre-training
#### 1. Create vocab:
```
python dataset/vocab.py -sc data/ProcessedData/SourcePath.txt -tc data/ProcessedData/TargetCode.txt -o data/vocab_2000.small --num_merges 2000
```
#### 2.Training TCBERT using GPU (no test):
```
python __main__.py -d data/ProcessedData -v  data/ProcessedData/vocab_2000.small  -o  output/tcbert.model --with_cuda True
```
## Fine-tuning
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




If you use the code released through this repository, please cite the following paper:
```

```