# Objectives

The learning objectives of this assignment are to:
1. learn the TensorFlow Keras APIs for convolutional and recurrent neural
   networks.
2. explore the space of hyper-parameters for convolutional and recurrent
   networks.

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
`https://github.com/ua-ista-457/convolutional-and-recurrent-networks-<your-username>`, where
`<your-username>` is your GitHub username, and
[create a branch through the GitHub interface](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
Please name the branch `solution`.

Then, clone the repository to your local machine and checkout the branch you
just created:
```
git clone -b solution https://github.com/ua-ista-457/convolutional-and-recurrent-networks-<your-username>.git
```
You are now ready to begin working on the assignment.

# Write your code

You will implement several convolutional and recurrent neural networks using the
[TensorFlow Keras API](https://www.tensorflow.org/guide/keras/).
You should read the documentation strings (docstrings) in each of methods in
`nn.py`, and implement the methods as described.
Write your code below the docstring of each method;
**do not delete the docstrings**.

The following objects and functions may come in handy:
* [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
* [Sequential.compile](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile)
* [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
* [Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding)
* [Conv1D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D)
* [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
* [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)
* [GlobalMaxPool1D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPool1D)
* [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten)
* [SimpleRNN](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN)
* [LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
* [GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
* [Bidirectional](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional)

# Test your code

Tests have been provided for you in the `test_nn.py` file.
The tests show how each of the methods is expected to be used.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
platform darwin -- Python 3.7.2, pytest-6.0.1, py-1.9.0, pluggy-0.13.1
rootdir: .../convolutional-and-recurrent-networks-<your-username>
collected 4 items

test_nn.py FFFF                                                          [100%]

=================================== FAILURES ===================================
...
=========================== 4 failed in 4.40 seconds ===========================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
platform darwin -- Python 3.7.2, pytest-6.0.1, py-1.9.0, pluggy-0.13.1
rootdir: .../convolutional-and-recurrent-<your-username>
collected 4 items

test_nn.py
0.6 RMSE for RNN on toy problem
.
89.0% accuracy for CNN on MNIST sample
.
88.9% accuracy for RNN on Youtube comments
.
85.4% accuracy for CNN on Youtube comments
.                                                                                                               [100%]

========================== 4 passed in 51.79 seconds ===========================
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
