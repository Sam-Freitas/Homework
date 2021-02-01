# Objectives

The learning objectives of this assignment are to:
1. get familiar with the TensorFlow Keras framework for training neural networks.
2. experiment with the various hyper-parameter choices of feedforward networks.

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.7 or higher)](https://www.python.org/downloads/)
* [tensorflow (version 2.3)](https://www.tensorflow.org/)
* [pytest](https://docs.pytest.org/)

# Check out a new branch

Go to the repository that GitHub Classroom created for you,
`https://github.com/ua-ista-457/feedforward-networks-<your-username>`, where
`<your-username>` is your GitHub username, and
[create a branch through the GitHub interface](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
Please name the branch `solution`.

Then, clone the repository to your local machine and checkout the branch you
just created:
```
git clone -b solution https://github.com/ua-ista-457/feedforward-networks-<your-username>.git
```
You are now ready to begin working on the assignment.

# Write your code

You will implement several feedforward neural networks using the
[TensorFlow Keras API](https://www.tensorflow.org/guide/keras/).
You should read the documentation strings (docstrings) in each of methods in
`nn.py`, and implement the methods as described.
Write your code below the docstring of each method;
**do not delete the docstrings**.

The following objects and functions may come in handy:
* [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
* [Sequential.compile](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile)
* [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
* [Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
* [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)

# Test your code

Tests have been provided for you in the `test_nn.py` file.
The tests show how each of the methods is expected to be used.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
platform darwin -- Python 3.7.2, pytest-6.0.1, py-1.9.0, pluggy-0.13.1
rootdir: .../feedforward-networks-<your-username>
collected 4 items

test_nn.py FFFF                                                          [100%]

=================================== FAILURES ===================================
...
=========================== 4 failed in 7.19 seconds ===========================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
platform darwin -- Python 3.7.4, pytest-5.0.1, py-1.8.0, pluggy-0.12.0
rootdir: .../feedforward-networks-<your-username>
collected 4 items

test_nn.py
8.2 RMSE for baseline on Auto MPG
6.2 RMSE for deep on Auto MPG
3.9 RMSE for wide on Auto MPG
.
65.0% accuracy for baseline on del.icio.us
68.7% accuracy for relu on del.icio.us
66.9% accuracy for tanh on del.icio.us
.
18.2% accuracy for baseline on UCI-HAR
93.8% accuracy for dropout on UCI-HAR
91.7% accuracy for no dropout on UCI-HAR
.
75.4% accuracy for baseline on census income
78.8% accuracy for early on census income
78.6% accuracy for late on census income
.                                                          [100%]

============================== 4 passed in 23.30s ==============================
```
**Warning**: The performance of your models may change somewhat from run to run,
especially when moving from one machine to another, since neural network models
are randomly initialized.
A correct solution to this assignment should pass the tests on any machine.
Make sure that the tests are passing on GitHub!
If you see that they are failing on GitHub even though they are passing on your
local machine, you will likely need to change your code.
Read the build log on GitHub to see if you have any coding errors;
otherwise, try different hyper-parameters for your model.

# Submit your code

As you are working on the code, you should regularly `git commit` to save your
current changes locally and `git push` to push all saved changes to the remote
repository on GitHub.

To submit your assignment,
[create a pull request on GitHub](https://help.github.com/articles/creating-a-pull-request/#creating-the-pull-request)
where the "base" branch is ``master``, and the "compare" branch is ``solution``.
Once you have created the pull request, go to the "Checks" tab and make sure all
your tests are passing.
Then go to the "Files changed" tab, and make sure that you have only changed
the `nn.py` file and that all your changes look as you would expect them to.
**Do not merge the pull request.**

Your instructional team will grade the code of this pull request, and provide
you feedback in the form of comments on the pull request.

# Grading

Assignments will be graded primarily on their ability to pass the tests that
have been provided to you.
Assignments that pass all tests will receive at least 80% of the possible
points.
To get the remaining 20% of the points, make sure that your code is using
appropriate data structures, existing library functions are used whenever
appropriate, code duplication is minimized, variables have meaningful names,
complex pieces of code are well documented, etc.
