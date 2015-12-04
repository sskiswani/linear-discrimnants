# CAP5638 Project 2<br />Classification Using Linear Discriminant Functions and Boosting Algorithms
Python 3 implementation of various two-/multi- class linear discriminant functions through perceptron-like algorithms,
that uses boosting algorithms to create accurate classifiers using linear discriminant functions.

The UCI Wine and USPS Handwritten Digits data sets are included in the `./bin/` directory.

The implementation results detailed in the report (`./rpt/report.pdf`) can be recreated using the command `python3 run.py`. 
The `run.py` script will also accept the classification method, training data, and classification data as command-line arguments,
in the following format:

```
python3 run.py <classification_method> <path/to/training_data> <path/to/classification_data>
```

See the usage section, or use the `--help` (or `-h`) flag for additional information.

**Dependencies** 

- numpy v1.10

**Optional Dependencies**

- scikit-learn v0.17 (SVMs will be excluded if not satisfied)
- matplotlib v1.5 (plots will be omitted from output if not satisfied)

## Usage
Execution of `run.py` without any arguments will generate the results described in the report, which amounts to both training and
running each classification method on the both of the data sets and outputting the results. `run.py` can also take command-line arguments
in the following fashion:

```
python3 run.py <classification_method> <path/to/training_data_set> <path/to/classification_data_set>
```

The `classification_method` argument refers to the following keywords (or the method number):

- Basic two-class classification (Method 1)
    - `single` for the Fixed-Increment Single-Sample Perceptron Algorithm (Method 1a)
    - `batch` for Batch Relaxation with Margin (Method 1b)
- Multi-class classification (Method 2)
    - `rest`  for the one-against-the-rest method (Method 2a)
    - `other` for the one-against-the-other method (Method 2b)
- `ada` to use Adaboost to create strong classifiers. (Method 3)
- `svm` for support vector machines (Method 4)
- `kern` (Method 5)
- `samme` (Method 6)

If no training and/or classification file is provided, the method is used on both of the provided data sets. 
For example:

- `python3 run.py 1a` will run the Fixed-Increment Single-Sample Perceptron Algorithm on both data sets (equivalent to `python3 run.py single`).
- `python3 run.py ada ./train_set.txt ./test_set.txt` will use `./train_set.txt` to train AdaBoost strong classifiers and use `./test_set.txt` to test their accuracy (equivalent to `python3 run.py 3 ./train_set.txt ./test_set.txt`).

Data sets should be in the same format as those in the `./bin/` folder, i.e.:

```
<class_number> <x1> <x2> <xn>
<class_number> <x1> <x2> <xn>
...
```

---

## Classification Methods
The following classification methods are available:

1. **Basic two-class classification using perceptron algorithms**
    - Abstractly, the problem is as follows. Given n labeled training samples, `D = {(x1,L1), (x2, L2), ..., (xn, Ln)}`, when Li is +1 / -1, implement Algorithm 4 (Fixed-Increment Single-Sample Perceptron Algorithm) and Algorithm 8 (Batch Relaxation with Margin) of Chapter 5 in the textbook.
2. **Multi-classclassification:**
    - Use the basic two-class perceptron algorithms to solve multi-class classification problems by using the one-against-the-rest and one-against-the-other methods. Note that you need to handle ambiguous cases properly.
3. **Adaboost to create strong classifiers:**
    - Implement Algorithm 1 (AdaBoost) in Chapter 9 of the textbook to create a strong classifier using the above linear discriminant functions.

### Extra Credit Classifiers
4. **Support vector machines:**
    - By using an available quadratic programming optimizer or an SVM library, implement a training and classification  algorithm for support vector machines. Then use your algorithm on the USPS dataset. Document the classification accuracy and compare the results with that from the two basic algorithms.
    - Implemented using [scikit-learn SVMs](http://scikit-learn.org/stable/modules/svm.html#svm)
5. **Kernel method for linear discriminant functions:**
    - Given a kernel function, derive the kernel-version of Algorithm 4 and implement the algorithm, and then apply it on the given wine and USPS datasets. Document the classification accuracy and compare the results with that from the two basic algorithms without kernels. Use the polynomial function of degree three as the kernel function; optionally, you can use other commonly used kernel functions.
6. **Multiple-class linear machines and multiple-class boosting:**
    - Use the Kesler’s construction to train a linear machine for multi-class classification and then use the SAMME algorithm to boost its performance on the training set. Apply the algorithm on both datasets and classify the corresponding test samples in the test sets. Document the classification accuracy and compare the results with that from the one-against-the-rest and one-against-the- other algorithms.

### References
The textbook referred to above is **Pattern Classification** by Richard O. Duda, Peter E. Hart, and David G. Stork, *2nd edition*.

Attributions and details for the UCI wine and USPS handwritten digit data sets are located in their corresponding readme files (`./bin/wine_readme.txt` and `./bin/handwriting_readme.txt`).