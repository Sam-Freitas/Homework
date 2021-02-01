from typing import Sequence, Any

import numpy as np
from pytictoc import TicToc

import sys

def assert_array_equal(actual, desired):
    assert type(actual) is np.ndarray
    np.testing.assert_array_equal(actual, desired)


t = TicToc()
t.tic()



vocab = [chr(i) for i in range(sys.maxunicode + 1)]
object_seq = list('schön día \U0010ffff' * 100)
print('vocab creation')
t.toc()


# objects_to_binary_vector
# given objects get vector 
start=0

no_repeat_vocab = list(dict.fromkeys(vocab))
myDict = dict(list(enumerate(no_repeat_vocab,start)))
print('dictionary creation')
t.toc()
dict_values = list(myDict.values())
dict_keys = list(myDict.keys())  
print('dictionary value creation')
t.toc()
vector = np.zeros(dict_keys[-1]+1)

k=0
# first put them where they belong
# then remove the blank spaces

# for i in range(len(dict_values)):
vector = np.zeros(dict_keys[-1]+1)
# first put them where they belong
# then remove the blank spaces
for i in range(len(object_seq)):
    try: 
        # get the specific object
        thisObject = object_seq[i]
        # find where the dictionary value is and put is as 1
        vector[dict_values.index(thisObject)] = 1
    except:
        # if the .index() fails then it does not exist and then gets passed on 
        # as a zero, but that was already made so pass works 
        pass
print('objects to binary vector')
t.toc()



assert vector[ord('í')] == 1
assert vector[ord('\U0010ffff')] == 1
assert vector[ord('o')] == 0
print('assertion commands')
t.toc()
specific_objects = [None]*int(sum(vector.tolist()))

j=0

true_indexes = np.where(vector == 1)

for i in true_indexes[0]:
    if vector[dict_keys[i]]:
        specific_objects[j]=dict_values[i]
        j=j+1
    else:
        pass


assert specific_objects == list(' acdhnsíö\U0010ffff')
print('binary_vector_to_objects and final assert')
t.toc()
print('end')