# Objectives

The learning objectives of this assignment are to:
1. implement feed-forward prediction for a single layer neural network 
2. implement training via back-propagation for a single layer neural network 

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.7 or higher)](https://www.python.org/downloads/)
* [numpy](http://www.numpy.org/)
* [pytest](https://docs.pytest.org/)
* [pytest-timeout](https://pypi.org/project/pytest-timeout/)

# Check out a new branch

Go to the repository that GitHub Classroom created for you,
`https://github.com/ua-ista-457/back-propagation-<your-username>`, where
`<your-username>` is your GitHub username, and
[create a branch through the GitHub interface](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
Please name the branch `solution`.

Then, clone the repository to your local machine and checkout the branch you
just created:
```
git clone -b solution https://github.com/ua-ista-457/back-propagation-<your-username>.git
```
You are now ready to begin working on the assignment.

# Write your code

You will implement a simple single-layer neural network with sigmoid activations
everywhere.
This will include making predictions with a network via forward-propagation, and
training the network via gradient descent, with gradients calculated using
back-propagation.

You should read the documentation strings (docstrings) in each of methods in
`nn.py`, and implement the methods as described.
Write your code below the docstring of each method;
**do not delete the docstrings**.

The following objects and functions may come in handy:
* [numpy.ndarray.dot](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.dot.html)
* [numpy.ndarray.T](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.T.html)
* [numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
* [scipy.special.expit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html)

# Test your code

Tests have been provided for you in the `test_nn.py` file.
The tests show how each of the methods is expected to be used.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
platform darwin -- Python 3.7.2, pytest-4.1.1, py-1.7.0, pluggy-0.8.1
rootdir: .../back-propagation-<your-username>
plugins: timeout-1.4.2
collected 5 items

test_nn.py FFFFF                                                         [100%]

=================================== FAILURES ===================================
...
=========================== 5 failed in 0.58 seconds ===========================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
platform darwin -- Python 3.7.4, pytest-5.0.1, py-1.8.0, pluggy-0.12.0
rootdir: .../back-propagation-<your-username>
plugins: timeout-1.3.3
collected 5 items

test_nn.py .....                                                         [100%]

=========================== 5 passed in 0.29 seconds ===========================
```

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
