from typing import Sequence, Any

import numpy as np

import json



class Index:

    def __init__(self, vocab: Sequence[Any], start=0):
        """
        Assigns an index to each unique item in the `vocab` iterable,
        with indexes starting from `start`.

        Indexes should be assigned in order, so that the first unique item in
        `vocab` has the index `start`, the second unique item has the index
        `start + 1`, etc.
        """
        # global myDict 
        # global self.dict_values
        # global self.dict_keys
        # create global doctionary and each subsequent value

        # use .fromkeys() to remove any repeats

        # no_repeat_vocab = list(dict.fromkeys(vocab))
        # create the dictionary variable myDict
        myDict = dict(list(enumerate(list(dict.fromkeys(vocab)),start)))
        self.dict = myDict

        self.dict_values = list(myDict.values())
        self.dict_keys = list(myDict.keys()) 
        # # and just for ease of use get the values and keys from the dictionary 
        # # these are global because i didnt want to type them every time 
        # self.dict_values = list(myDict.values())
        # self.dict_keys = list(myDict.keys())  
    



    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array of the object indexes.
        """
        # objects = ["one", "", "four", "four"]
        # initalize an array 
        object_indexes = np.zeros(len(object_seq))

        # iterate through each object_seq then convert it into the index of the correcponding value
        # then use that inital index as from the keys to create a loop index for the return object 
        for i in range(len(object_seq)):
            # create a single value for each object_seq value
            thisObject_seq=object_seq[i]
            try:
                # find the corresponding index to the single value
                thisDict_value=self.dict_values.index(thisObject_seq)
                # find the representing keys value
                thisIndex = self.dict_keys[thisDict_value]
                # convert to an array and store 
                object_indexes[i] = np.array(thisIndex)
            except:
                # if the corresponding index does not exist use thisIndex is the starting value 
                #       of the dict minus one (will sometimes give negative)
                thisIndex = self.dict_keys[0]-1
                object_indexes[i] = np.array(thisIndex)

        # return the new object indexes
        return(object_indexes)



    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        If the sequences are not all of the same length, shorter sequences will
        have padding added at the end, with `start-1` used as the pad value.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array of the object indexes.
        """
        # find the number of rows and columns that are necessary 
        num_rows = len(object_seq_seq)
        # find the longest row in a "jagged array"
        num_cols = int(np.max([len(l) for l in object_seq_seq]))

        # initalize the final matrix with just zeros
        new_matrix = np.zeros((num_rows, num_cols))

        # iterate through i and j
        for i in range(num_rows):
            for j in range(num_cols):
                try:
                    # if the object[i][j] exists in self.dict_values, use its index to get the specific
                    # key that represents it
                    new_matrix[i][j] = self.dict_keys[self.dict_values.index(object_seq_seq[i][j])]
                except:
                    # and if the object does not exist in the index
                    # use the pad value (start)-1 
                    new_matrix[i][j] = self.dict_keys[0]-1

        return(new_matrix)

    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """

        # initalize the vector 
        vector = np.zeros(self.dict_keys[-1]+1)

        for i in range(len(object_seq)):
            try: 
                # get the specific object
                # thisObject = object_seq[i]
                # find where the dictionary value is and put is as 1
                vector[self.dict_values.index(object_seq[i]) + self.dict_keys[0]] = 1
            except:
                # if the .index() fails then it does not exist and then gets passed on 
                # as a zero, but that was already made so pass works 
                pass

        return(vector)



    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array, where each row in the array corresponds
                 to a row in the input, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """
        # find the number of rows and columns necessary 
        num_rows = len(object_seq_seq)
        num_cols = self.dict_keys[-1]+1

        # initalize a new np zeros matrix of size
        new_matrix = np.zeros((num_rows,num_cols))

        for i in range(num_rows):
            # initalize the blank row 
            thisRow = np.zeros((num_cols))
            for j in range(num_cols):

                try: 
                    # for every object, attempt to find it in the dictionary then find its index and set =1
                    thisRow[self.dict_keys[self.dict_values.index(object_seq_seq[i][j])]] = np.array(1)
                except:
                    pass
            # assign the new row to the row 
            new_matrix[i,:] = thisRow

        return(new_matrix)

    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_vector: A 1-dimensional array of indexes
        :return: A sequence of objects, one for each index.
        """
        # create a blank list
        seq_objects =[]

        # iterate through the list of the given vector
        for i in range(len(index_vector)):
            try:
                # if the index vector exists in the list
                thisIndex = self.dict_keys.index(index_vector[i])
                # append it in that specific place
                seq_objects.append(self.dict_values[thisIndex])
            except:
                pass
        
        return(seq_objects)






    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_matrix: A 2-dimensional array of indexes
        :return: A sequence of sequences of objects, one for each index.
        """
        # find number of rows and columns
        num_rows = len(index_matrix)
        num_cols = len(index_matrix[0])

        # create a blank list
        new_objects = []

        for i in range(num_rows):
            # create a temporary blank row list 
            nested_list = []
            for j in range(num_cols):
                try:
                    # try to append the a new value to the list if it exists from the given matrix
                    nested_list.append(self.dict_values[self.dict_keys.index(index_matrix[i][j])])
                except:
                    pass
            # each row then gets appended to the last to create 2D list arrays 
            new_objects.append(nested_list)
        
        return(new_objects)

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param vector: A 1-dimensional binary array
        :return: A sequence of objects, one for each nonzero index.
        """
        # create a blank object list of a specific length
        specific_objects = [None]*int(sum(vector.tolist()))

        # initalize start variables 
        j=0
        for i in range(len(self.dict_keys)):
            # iterate through the keys
            if vector[self.dict_keys[i]]:
                # place the object where it should be and update the iteration
                specific_objects[j]=self.dict_values[i]
                j=j+1
            else:
                pass


        return(specific_objects)

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param binary_matrix: A 2-dimensional binary array
        :return: A sequence of sequences of objects, one for each nonzero index.
        """
        # create the number of rows and columns
        num_rows = len(binary_matrix)
        num_cols = len(binary_matrix[0])

        # create a blank list
        new_objects = []

        for i in range(num_rows):
            # create a temp blank row 
            thisRow =[]
            for j in range(num_cols):
                # if the binary matrix
                if binary_matrix[i][j]:
                    # then append the list with the proper value
                    thisRow.append(self.dict_values[self.dict_keys.index(j)])
                else:
                    pass
            # append the new rows to the previous ones to create the 2D list
            new_objects.append(thisRow)

        return(new_objects)

