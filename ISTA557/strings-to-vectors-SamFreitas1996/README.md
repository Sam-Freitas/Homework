# Objectives

The learning objectives of this assignment are to:
1. practice Python programming skills and use of numpy arrays
2. get familiar with submitting assignments on GitHub Classroom

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.7 or higher)](https://www.python.org/downloads/)
* [numpy](http://www.numpy.org/)
* [pytest](https://docs.pytest.org/)
* [pytest-timeout](https://pypi.org/project/pytest-timeout/)

If you have not used Git, Python, or Numpy before, this would be a good time to
go through some tutorials:

* [git tutorial](https://try.github.io/)
* [Python tutorial](https://docs.python.org/3/tutorial/)
* [numpy tutorial](https://docs.scipy.org/doc/numpy/user/quickstart.html)

You can find many other tutorials for these tools online.

# Check out a new branch

Before you start editing any code, you will need to create a new branch in your
GitHub repository to hold your work.
This is the first step of the
[standard GitHub workflow](https://guides.github.com/introduction/flow/):

1. Create a branch
2. Add commits
3. Open a Pull Request
4. Discuss and review your code

In this class, the first three steps are your responsibility;
the fourth step is how the instructional team will grade your assignment.
Note that you *must create a branch* (you cannot simply commit to master) or you
will not be able to create a pull request as required for grading.

First, go to the repository that GitHub Classroom created for you,
`https://github.com/ua-ista-457/strings-to-vectors-<your-username>`, where
`<your-username>` is your GitHub username, and
[create a branch through the GitHub interface](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
Please name the branch `solution`.

Then, clone the repository to your local machine and checkout the branch you
just created:
```
git clone -b solution https://github.com/ua-ista-457/strings-to-vectors-<your-username>.git
```
You are now ready to begin working on the assignment.

# Write your code

You will implement an `Index` that associates objects with integer indexes.
This is a very common setup step in training neural networks, which require that
everything be expressed as numbers, not objects.

A template for the `Index` class has been provided to you in the file `nn.py`.
In the template, each method has only a documentation string, with no code in
the body of the method yet.
For example, the `objects_to_indexes` method looks like:
```python
def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
    """
    Returns a vector of the indexes associated with the input objects.

    For objects not in the vocabulary, `start-1` is used as the index.

    :param object_seq: A sequence of objects.
    :return: A 1-dimensional array of the object indexes.
    """
```

You should read the documentation strings (docstrings) in each of methods in
`nn.py`, and implement the methods as described.
Write your code below the docstring of each method;
**do not delete the docstrings**.

The following objects and functions may come in handy:
* [dict](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)
* [numpy.array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)
* [numpy.full](https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html)
* [numpy.zeros](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html)
* [numpy.stack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.stack.html)
* [numpy.nonzero](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html)

# Test your code

Tests have been provided for you in the `test_nn.py` file.
The tests show how each of the methods of `Index` is expected to be used.
For example, the `test_indexes` test method looks like:

```python
@pytest.mark.timeout(1)
def test_indexes():
    vocab = ["four", "three", "", "two", "one"]
    objects = ["one", "", "four", "four"]
    indexes = np.array([4, 2, 0, 0])

    index = nn.Index(vocab)
    assert_array_equal(index.objects_to_indexes(objects), indexes)
    assert index.indexes_to_objects(indexes) == objects

    index = nn.Index(vocab, start=1)
    assert_array_equal(index.objects_to_indexes(objects), indexes + 1)
    assert index.indexes_to_objects(indexes + 1) == objects
```
This tests that your code correctly associates indexes with an input vocabulary
``"four", "three", "", """two", "one"``, that it can convert back and forth
between objects and indexes, and that it can handle indexing that starts from a
number other than 0.

To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
platform darwin -- Python 3.7.4, pytest-5.0.1, py-1.8.0, pluggy-0.12.0
rootdir: .../strings-to-vectors-<your-username>
plugins: timeout-1.3.3
collected 8 items

test_nn.py FFFFFFFF                                                      [100%]

=================================== FAILURES ===================================
_________________________________ test_indexes _________________________________

    @pytest.mark.timeout(1)
    def test_indexes():
        vocab = ["four", "three", "", "two", "one"]
        objects = ["one", "", "four", "four"]
        indexes = np.array([4, 2, 0, 0])

        index = nn.Index(vocab)
>       assert_array_equal(index.objects_to_indexes(objects), indexes)

test_nn.py:21:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

actual = None, desired = array([4, 2, 0, 0])

    def assert_array_equal(actual, desired):
>       assert type(actual) is np.ndarray
E       AssertionError: assert <class 'NoneType'> is <class 'numpy.ndarray'>
E        +  where <class 'NoneType'> = type(None)
E        +  and   <class 'numpy.ndarray'> = np.ndarray

test_nn.py:10: AssertionError
...
=========================== 8 failed in 0.49 seconds ===========================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
platform darwin -- Python 3.7.4, pytest-5.0.1, py-1.8.0, pluggy-0.12.0
rootdir: .../strings-to-vectors-<your-username>
plugins: timeout-1.3.3
collected 8 items

test_nn.py ........                                                      [100%]

=========================== 8 passed in 1.10 seconds ===========================
```


# Submit your code

As you are working on the code, you should regularly `git commit` to save your
current changes locally.
You should also regularly `git push` to push all saved changes to the remote
repository on GitHub.
Make a habit of checking the GitHub page for your repository to make
sure your changes have been correctly pushed there.
You should also check the "commits" page for the branch `solution` on GitHub.
There should be a green check mark beside your last commit, showing that your
code passes all of the given tests.
If there is a red X instead, your code is still failing some of the tests.
If there is a yellow icon, the tests have not completed running; check back in
a little while.
If there is no green, yellow, or red mark, make sure that the branch is named
`solution` and all your commits are on that branch; Travis-CI is configured to
only run tests on the `solution` branch.

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
