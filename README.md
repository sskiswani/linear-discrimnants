# Classification Using Linear Discriminant Functions and Boosting Algorithms
Python 3 implementation of various two-/multi- class linear discriminant functions through perceptron-like algorithms,
that uses boosting algorithms to create accurate classifiers using linear discriminant functions.

The UCI Wine and USPS Handwritten Digits data sets are included in the `./bin/` directory.

The implementation results detailed in the report (`./rpt/report.pdf`) can be recreated using the `run.py` script. 
The `run.py` script will accepts the classification method, training data, and classification data as command-line arguments,
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
The `run.py` file provides a way to interact with the module.
```
python3 run.py <classification_method> <path/to/training_data_set> <path/to/classification_data_set> -s STRATEGY -r TRAINING_RULE
```

The `classification_method` argument refers to the following keywords:

- `single` for basic two-class classification
- `multi` for multicategory classification
- `ada` to use Adaboost to create strong classifiers. (Method 3)

The `training_rule` option is for choosing the Perceptron learning rule:

- `fixed` for the Fixed-Increment Single-Sample Perceptron Algorithm (Method 1a)
- `batch` for Batch Relaxation with Margin (Method 1b)

The `strategy` option is for multicategory classification only, and corresponds to:
 
 - `rest` for one-against-the-rest classification
 - `other` for one-agains-the-other classification
 
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