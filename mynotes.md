# Notes for EE514 - Data Analytics and Machine Learning

24-Sep-18

Slides available in YouTube: <https://www.youtube.com/watch?v=uPXv79UltKk&index=2&t=0s&list=PLRmSzDzkNJgIGahWmlRV-OclD6-FowK8F>

- basic definitions
  - data analytics
  - machine learning
  - data science

- Practical applications of the technologies taught in this module
  - recommender systems
  - clickstream analysis
  - image recognition
  - video processing
  - automatic speech recognition
  - predictive analysis
  - NLP-sentiment analysis

Module overview (ref: YouTube Slides):

- part 1
  - data management
  - storage
  - visualization
- part 2
  - machine learning
  - predictive analysis
- intro to python for data analytics
- data summarization
- unsupervised ML
- supervised ML principles
- supervised ML algorithms
- representation learning and deep learning

- **_TODO_**:
  - mathematics
    - Linear algebra
    - calculus (multivariate)
    - probability

Kevin contact:
room s361

Grading:

- 25% continuous assessment
  - individual projects
  - visualization and predictive analysis
  - report, code etc to be submitted

- 75% exam

## Introduction to Python

- basics
  - variables and types
  - type conversions
  - basic operations
    - ** for power (^) = x**2
  - containers
    - list
      - mutable - can be changed
      - indexed from 0
      - can contain objs of any type
      => x = []
      - last item in list x[-1]
      - slicing
        - first three items = x[0:3] 0r x[:3]
  - dict
    - like hashmaps or associative arrays
    - key -> value map
    => y = {'key' : value}
  - also has tuples and sets

    - to leave a block empty, use keyword "pass"
    - functions
      - can handle multiple arguments - variadic arguments (*args)
        - def func_name (*args)
    - **_TODO_**: OO aspects in python
    - constructors inside classes
      - special method
      - \__name__
      - will have to use the keyword "self" explicitly as the first argument to the methods in the class
    - module is the .py file - can import to different file
    - package
      - collection of modules
        - create folder with a name
        - add \__init__.py file
    - **_TODO_**: exceptions
    - always prefer to use try-except block
      - example when trying to check if the file exists, use try-except instead of if-else
    - list comprehensions
      - can avoid loops
    - **_TODO_**: generators
      - "yield" keyword
      - transfer control from inside a loop to the caller
    - standard and external libraries

    - Example - Wikipedia scraper

## Numerical programming with numpy

- optimized for numerical computation
- single data type per array
- **_TODO_**: numpy methods
- stride is the third index in the list - can be used to specify the depth
  - negative stride can be used for reversal
- matrix arithmetic operations
  - uses python operations
  - matrix multiplications - use @
  - '*' does element wise multiplications
- linear algebra -np.linalg
- **_TODO_**: eigen values, matrix, stacking, concatenation
- **NEXT CLASS**: Mathematics refresher for ML

---------------------------------------------------------------------------------------------------------------

1-Oct-2018

## Mathematics for ML

- Sets
  - cardinality is the number of element in the set
    - |x| or '#x'

- vectors
  - ordered list of numbers OR columns
  - transpose of column vector is a row vector
  - Dot product - multiply corresponding components and add products
    - also called Inverse product or Scalar product
    - alternative notation $X^TY$
- **_TODO_**:
  - Norms:
    - $||X|| = sqrt(x^Tx)$
    - most common norm in Euclidean norm or L₂ norm
    - Norms and distance metrics

- Matrices
  - rectangular array of numbers
  - transpose:
    - $(A+B)^T = A^T + B^T$
    - identity
    - inverse

- **_TODO_**:
  - matrix multiplication
  - matrix-vector multiplication
  - matrix-matrix multiplication
  - rules of matrix multiplication
    - examples
  - square matrix
    - nonsingular/invertible/nondegenerate
    - singular/degenerate
  - computing inverses
  - quadratic forms
  - positive definiteness
    - symmetric positive definite (SPD)
  - eigen values and eigen vectors
    - eigenvalue decomposition
  - derivatives
    - exponential
    - Logarithms
    - Trigonometry
  - partial derivatives
  - gradient (inverse triangle symbol)
  - multivariable calculus
    - functional vs. function
  - matrix calculus
    - rules for matrix calculus - find gradients of functionals
  - Optimization
    - minimizing or maximizing
    - gradient descent
    - optimizing probabilities
      - log probabilities

#### Note quiz in loop

Data Interpretation, Management and storage:

- Datasets
  - types of data
    - Series (1D)
    - Tables
    - Trees
    - Graphs
    - Text
    - Multimedia
  - two types of information in Datasets
    - metadata - data about data. usually contains semantics (meaning)
    - data
      - actual value
  - scales of measurement - four (Stevens, 1946)
    - nominal
      - just 'name'
      - also called "categorial variables"
      - valid operations
        - = and !=
      - permissible statistics
        - constructors
        - modes
        - contingency correlation
    - ordinal
      - rank-ordered
      - valid operations
        - =, !=, <, >, <=, >=
      - permissible statistics
        - median
        - percentiles
        - spearman correlation
      - quantitative variables
        - interval
          - distance between attributes does have meaning but there is no absolute zero
          - examples
            - Date
            - temperature in degrees F
            - geometric point
          - valid operations
            - =, != ........
      - ratio
        - same as interval but with a meaningful absolute zero
        - examples
          - length, mass, temperature in Kelvin
          - age, weight, weight
        - valid operations
          - .............
  - hierarchy of scales of measurement
  - data as vectors
    - bivariate - 2D space
    - multivariate - multidimensional space (ND)
    - data as Matrices
      - stack quantitative data vectors as matrices
      - data in row, type in column - for python
      - one-hot encoding
        - build "vocabulary" - assign values for unique data
        - assign 1 in matrix when the value appears
        - will be useful
          - to calculate the occurrences
          - most used words
          - least used words
          - average use of a word
        - used in
          - similarity of the documents
          - cluster similar documents
          - topic modelling (PCA/LSI)
          - visualization (t-PSE)
    - multimedia data
      - images can be represented as
        - 3D tensors
        - functions
        - vectors
        - compressed (JPEG, PNG)
        - automatically extracted features
      - Videos
        - same as image but with time
    - data storage and I/O
      - files
      - databases
      - streams
      - structured data
        - tables
        - graphs
        - hierarchies
        - relational databases
      - semi-structured and unstructured
        - natural language plain text
        - HTML files
        - word docs
    - file formats
      - binary formats
        - types
          - 32 bit float (4 bytes)
          - 16 bit integer (2 bytes)
        - properties
          - compact
          - high performance I/O
          - not human readable
          - need to worry about integer sizes, endiness (little-indian, big-indian), signed/unsigned
        - plain text formats (ASCII and Unicode)
          - ASCII formats for tabular data
            - CSV
            - FWF (Fixed Width Format)
              - same as CSV but with standard width for columns

---------------------------------------------------------------------------------------------------------------