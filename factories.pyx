cimport numpy as np
from network cimport Network
from useful_functions cimport UsefulFunctions
from libc.stdlib cimport rand, srand, RAND_MAX
cdef extern from "limits.h":
    int INT_MAX
from libc.time cimport time
srand(time(NULL))
import cython
import numpy as np
from cython.parallel import prange

cdef UsefulFunctions UF
UF = UsefulFunctions()


cdef class Factory():

    '''
    Factory that returns networks.
    '''

cdef class GrandNetworkFactory_FullInput_NoInter(Factory):

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void create_neurons_genes(self,
                                   int number_of_neurons,
                                   int number_of_input_neurons,
                                   int number_of_output_neurons,
                                   int[:] activation_functions,
                                   double MUTATION_RATE):

        assert number_of_neurons >= number_of_input_neurons,'Number of input neurons\
        is larger or equal to number of all neurons'
        assert number_of_neurons >= number_of_output_neurons,'Number of output\
        neurons is larger or equal to number of all neurons'

        cdef Py_ssize_t i

        cdef int length_of_activation_f = activation_functions.size
        self.neurons_genes = np.empty((number_of_neurons, 4),
                                    dtype = np.int32 )
        self.mutation_genes = np.empty((number_of_neurons),
                                    dtype = np.float64)

        for i in prange(number_of_neurons, nogil=True):

            self.neurons_genes[i][0] = i

            # This part of code assigns neuron type to a particular neuron dependend
            # on its order number
            self.neurons_genes[i][1] = (0 * (i<number_of_input_neurons)
                                + 1 * (i >= number_of_input_neurons and
                                        i < number_of_neurons -
                                        number_of_output_neurons)
                                + 2 * (i >= number_of_input_neurons and
                                        i >= number_of_neurons -
                                        number_of_output_neurons))

            self.neurons_genes[i][2] = 0     # Since this is the 0'th iteration
            self.neurons_genes[i][3] = activation_functions[UF.randint(0,length_of_activation_f)]

            self.mutation_genes[i] = MUTATION_RATE

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void create_connections_genes(self,
                                       int number_of_neurons,
                                       int number_of_input_neurons,
                                       int number_of_output_neurons,
                                       int[:] input_indexes,
                                       int[:] output_indexes,
                                       double WEIGHT,
                                       double MUTATION_RATE):

        cdef Py_ssize_t i

        self.connections_genes = np.empty((number_of_input_neurons+number_of_output_neurons, 6),
                                        dtype = np.float64)
        cdef int number_of_available_inputs = input_indexes.size
        cdef int number_of_available_outputs = output_indexes.size
        for i in prange(number_of_input_neurons, nogil = True):
            self.connections_genes[i][0] = input_indexes[UF.randint(0, number_of_available_inputs)]

            #how input indexes are assigned to neurons. previously number of inputs should
            #have matched number of input neurons. Now they are assigned randomly

            self.connections_genes[i][1] = i/1.
            self.connections_genes[i][2] = WEIGHT
            self.connections_genes[i][3] = 1.
            self.connections_genes[i][4] = 0.                                   # Since this is the 0'th iteration
            self.connections_genes[i][5] = MUTATION_RATE

        for i in prange(number_of_output_neurons, nogil = True):
            self.connections_genes[i+number_of_input_neurons][0] = (i+number_of_neurons-number_of_output_neurons) /1.
            self.connections_genes[i+number_of_input_neurons][1] = output_indexes[UF.randint(0, number_of_available_outputs)]
            self.connections_genes[i+number_of_input_neurons][2] = WEIGHT
            self.connections_genes[i+number_of_input_neurons][3] = 1.
            self.connections_genes[i+number_of_input_neurons][4] = 0.                                   # Since this is the 0'th iteration
            self.connections_genes[i+number_of_input_neurons][5] = MUTATION_RATE


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void create_matrix(self, int number_of_inputs):

        self.matrix = np.zeros((self.neurons_genes.shape[0], self.neurons_genes.shape[0]+\
                                number_of_inputs), dtype = np.float64)
        self.matrix = self.initialize_connections_no_reordering(self.matrix,
                                                                self.connections_genes,
                                                                number_of_inputs)

    cdef Network get_network(self, int order_number):

        cdef Network A
        A = Network()
        A.params_setter(order_number,
                        self.neurons_genes,
                        self.mutation_genes,
                        self.connections_genes,
                        self.matrix)
        return A

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double [:,:]  initialize_connections_no_reordering(self,
                                                            double [:,:] matrix,
                                                            double [:,:] connections_genes,
                                                            int number_of_inputs ) nogil :

        cdef Py_ssize_t i
        cdef int from_
        cdef int to_
        cdef int input_index
        cdef int length

        with gil: length = connections_genes.shape[0]

        for i in range(length):
            if connections_genes[i][3]==1 and connections_genes[i][1] >= 0: # checks activity status of a connection and whether it's not an output connection

                # if it is input -> input neuron connection
                if connections_genes[i][0]<0:

                    with gil:
                    # since numeration starts with -1
                        from_ = abs(int(connections_genes[i][0]))-1
                        to_ = int(connections_genes[i][1])

                    matrix[to_][from_] = connections_genes[i][2]

                # if it is neuron -> neuron connection
                elif connections_genes[i][0]>=0:

                    with gil:
                        from_ = int(connections_genes[i][0]) + number_of_inputs     # Since there are inputs before neurons in every row
                        to_ = int(connections_genes[i][1])

                    matrix[to_][from_] = connections_genes[i][2]
        return matrix


cdef class NetworkFactory(Factory):

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void create_neurons_genes_through_mating(self,
                                                  Network parent_1,
                                                  Network parent_2):

        cdef Py_ssize_t i, j

        cdef int length_parent_1 = parent_1.neurons_genes.shape[0]
        cdef int length_parent_2 = parent_2.neurons_genes.shape[0]
        cdef int width = parent_1.neurons_genes.shape[1]

        self.first_parent = parent_1.network_order_number
        self.second_parent = parent_2.network_order_number

        cdef int number_of_neurons_in_parents = length_parent_1+length_parent_2
        cdef int the_longest_one, the_shortes_one, longer_index

        cdef double R = rand() / (RAND_MAX * 1.0)
        cdef int number_of_neurons_in_child

        if length_parent_1>=length_parent_2:
            the_longest_one = length_parent_1
            the_smallest_one = length_parent_2
            longer_index = 1

        else:
            the_longest_one = length_parent_2
            the_smallest_one = length_parent_1
            longer_index = 2


        if number_of_neurons_in_parents%2 == 0:
            number_of_neurons_in_child = number_of_neurons_in_parents//2\
                                                + UF.randint(0,3) - 1

        else:
            number_of_neurons_in_child = (number_of_neurons_in_parents+\
                                        1*(R >= 0.5) - 1*(R < 0.5))//2 \
                                        + UF.randint(0,3)-1


        self.created_neurons_genes = np.empty((number_of_neurons_in_child, 4),
                                    dtype = np.int32)
        self.matches = np.empty((number_of_neurons_in_child,4),
                                    dtype = np.int32)
        self.created_mutation_genes = np.empty((number_of_neurons_in_child),
                                    dtype = np.float64)

        for i in prange(number_of_neurons_in_child, nogil=True):

            self.matches[i][0] = UF.randint(0, the_longest_one)      # old neuron || 1/2 parent || new neuron || used status, its index ([i]) must be identical to 'new neuron'
            self.matches[i][2] = i
            self.matches[i][3] = 0

            if self.matches[i][0]<the_smallest_one:

                if rand() / (RAND_MAX * 1.0) <0.5:

                    self.matches[i][1] = 1
                    for j in prange(width):

                        if j == 0:
                            self.created_neurons_genes[i][j] = i
                        else:
                            self.created_neurons_genes[i][j] = parent_1.neurons_genes[self.matches[i][0]][j]

                    self.created_mutation_genes[i] = parent_1.mutation_genes[self.matches[i][0]]

                else:

                    self.matches[i][1] = 2
                    for j in prange(width):

                        if j == 0:
                           self.created_neurons_genes[i][j] = i
                        else:
                            self.created_neurons_genes[i][j] = parent_2.neurons_genes[self.matches[i][0]][j]

                    self.created_mutation_genes[i] = parent_2.mutation_genes[self.matches[i][0]]


            elif self.matches[i][0]>=the_smallest_one:

                self.matches[i][1] = longer_index

                if longer_index == 1:

                    for j in prange(width):
                        if j == 0:
                           self.created_neurons_genes[i][j] = i
                        else:
                            self.created_neurons_genes[i][j] = parent_1.neurons_genes[self.matches[i][0]][j]

                    self.created_mutation_genes[i] = parent_1.mutation_genes[self.matches[i][0]]

                elif longer_index == 2:

                    for j in prange(width):
                        if j == 0:
                           self.created_neurons_genes[i][j] = i
                        else:
                            self.created_neurons_genes[i][j] = parent_2.neurons_genes[self.matches[i][0]][j]

                    self.created_mutation_genes[i] = parent_2.mutation_genes[self.matches[i][0]]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void create_connections_genes_through_mating(self,
                                                      Network parent_1,
                                                      Network parent_2):

        cdef Py_ssize_t i, j, k, l

        cdef int length_parent_1 = parent_1.connections_genes.shape[0]
        cdef int length_parent_2 = parent_2.connections_genes.shape[0]

        cdef int neurons_length_parent_1 = parent_1.neurons_genes.shape[0]
        cdef int neurons_length_parent_2 = parent_2.neurons_genes.shape[0]
        cdef int width = parent_1.connections_genes.shape[1]

        cdef int length_of_matches = self.matches.shape[0]
        cdef int index_of_new_neuron_in_matches
        cdef int index_of_old_neuron

        cdef double[:,:] long_connections_genes = np.empty(((length_of_matches+1)*length_of_matches, width), dtype=np.float64)

        for i in prange((length_of_matches+1)*length_of_matches, nogil = True):
            long_connections_genes[i][3] =-1.

        for i in prange(length_of_matches, nogil = True):

            index_of_new_neuron_in_matches = self.created_neurons_genes[i][0]
            if self.matches[index_of_new_neuron_in_matches][1] == 1:

                index_of_old_neuron = self.matches[index_of_new_neuron_in_matches][0]

                for j in prange(length_parent_1):

                    if parent_1.connections_genes[j][3] != -1.:

                        if parent_1.connections_genes[j][0] == index_of_old_neuron:

                            if parent_1.connections_genes[j][1] >= 0:
                                for k in prange(length_of_matches):

                                    if self.matches[k][0] == parent_1.connections_genes[j][1] and self.matches[k][3] != 1 :
                                        if self.matches[k][1] == 1:

                                            long_connections_genes[i *(length_of_matches+1) + j][0] = index_of_new_neuron_in_matches/1.
                                            long_connections_genes[i *(length_of_matches+1) + j][1] = self.matches[k][2]/1.
                                            long_connections_genes[i *(length_of_matches+1) + j][2] = parent_1.connections_genes[j][2]
                                            long_connections_genes[i *(length_of_matches+1) + j][3] = parent_1.connections_genes[j][3]
                                            long_connections_genes[i *(length_of_matches+1) + j][4] = parent_1.connections_genes[j][4]
                                            long_connections_genes[i *(length_of_matches+1) + j][5] = parent_1.connections_genes[j][5]

                                            self.matches[k][3] = 1


                            if parent_1.connections_genes[j][1] < 0:
                                long_connections_genes[i *(length_of_matches+1) + length_of_matches][0] = index_of_new_neuron_in_matches/1.
                                long_connections_genes[i *(length_of_matches+1) + length_of_matches][1] = parent_1.connections_genes[j][1]
                                long_connections_genes[i *(length_of_matches+1) + length_of_matches][2] = parent_1.connections_genes[j][2]
                                long_connections_genes[i *(length_of_matches+1) + length_of_matches][3] = parent_1.connections_genes[j][3]
                                long_connections_genes[i *(length_of_matches+1) + length_of_matches][4] = parent_1.connections_genes[j][4]
                                long_connections_genes[i *(length_of_matches+1) + length_of_matches][5] = parent_1.connections_genes[j][5]



                        if parent_1.connections_genes[j][0] < 0:
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][0] = parent_1.connections_genes[j][0]
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][1] = index_of_new_neuron_in_matches/1.
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][2] = parent_1.connections_genes[j][2]
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][3] = parent_1.connections_genes[j][3]
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][4] = parent_1.connections_genes[j][4]
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][5] = parent_1.connections_genes[j][5]

            elif self.matches[index_of_new_neuron_in_matches][1] == 2:

                index_of_old_neuron = self.matches[index_of_new_neuron_in_matches][0]

                for j in prange(length_parent_2):

                    if parent_2.connections_genes[j][3] != -1.:

                        if parent_2.connections_genes[j][0] == index_of_old_neuron:

                            if parent_2.connections_genes[j][1] >= 0:
                                for k in prange(length_of_matches):

                                    if self.matches[k][0] == parent_2.connections_genes[j][1] and self.matches[k][3] != 1 :
                                        if self.matches[k][1] == 2:

                                            long_connections_genes[i *(length_of_matches+1) + j][0] = index_of_new_neuron_in_matches/1.
                                            long_connections_genes[i *(length_of_matches+1) + j][1] = self.matches[k][2]/1.
                                            long_connections_genes[i *(length_of_matches+1) + j][2] = parent_2.connections_genes[j][2]
                                            long_connections_genes[i *(length_of_matches+1) + j][3] = parent_2.connections_genes[j][3]
                                            long_connections_genes[i *(length_of_matches+1) + j][4] = parent_2.connections_genes[j][4]
                                            long_connections_genes[i *(length_of_matches+1) + j][5] = parent_2.connections_genes[j][5]

                                            self.matches[k][3] = 1

                                            continue
                            elif parent_2.connections_genes[j][1] < 0:
                                long_connections_genes[i *(length_of_matches+1) + j][0] = index_of_new_neuron_in_matches/1.
                                long_connections_genes[i *(length_of_matches+1) + j][1] = parent_2.connections_genes[j][1]
                                long_connections_genes[i *(length_of_matches+1) + j][2] = parent_2.connections_genes[j][2]
                                long_connections_genes[i *(length_of_matches+1) + j][3] = parent_2.connections_genes[j][3]
                                long_connections_genes[i *(length_of_matches+1) + j][4] = parent_2.connections_genes[j][4]
                                long_connections_genes[i *(length_of_matches+1) + j][5] = parent_2.connections_genes[j][5]

                        if parent_2.connections_genes[j][0] < 0:
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][0] = parent_2.connections_genes[j][0]
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][1] = index_of_new_neuron_in_matches/1.
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][2] = parent_2.connections_genes[j][2]
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][3] = parent_2.connections_genes[j][3]
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][4] = parent_2.connections_genes[j][4]
                            long_connections_genes[i *(length_of_matches+1) + length_of_matches][5] = parent_2.connections_genes[j][5]

            for k in prange(length_of_matches):
                self.matches[k][3] = 0

        self.created_connections_genes = UF.twodim_genes_cleaner(long_connections_genes,
                                                                 3,
                                                                 np.float64,
                                                                 0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void mutate_neurons_genes(self,
                                   double prb_deleteion,
                                   int [:] activation_functions,
                                   int [:] input_indexes,
                                   int [:] output_indexes,
                                   int iteration,
                                   double WEIGHT,
                                   double MUTATION):

        cdef Py_ssize_t i
        cdef int length_of_neurons_genes = self.created_neurons_genes.shape[0]
        cdef int width_of_neurons_genes = self.created_neurons_genes.shape[1]

        cdef int length_of_activation_f = activation_functions.size

        cdef int neurons_to_be_added = int(prb_deleteion * length_of_neurons_genes) + 1 *(0 >= int(prb_deleteion * length_of_neurons_genes))

        for i in prange(length_of_neurons_genes, nogil = True):

            if prb_deleteion >= rand() / (RAND_MAX * 1.0):

                with gil:
                    self.created_neurons_genes, self.created_connections_genes,  self.created_mutation_genes = \
                    self.delete_neuron(self.created_neurons_genes,
                                       self.created_mutation_genes,
                                       self.created_connections_genes,
                                       i,
                                       i)
            else:
                if self.created_mutation_genes[i] >= rand() / (RAND_MAX * 1.0):
                    self.created_neurons_genes[i][3] = activation_functions[UF.randint(0,length_of_activation_f)]

                if self.created_mutation_genes[i] >= rand() / (RAND_MAX * 1.0):
                    UF.change_gene_weigth(self.created_mutation_genes, i)


        self.neurons_genes = np.empty((length_of_neurons_genes+neurons_to_be_added, width_of_neurons_genes), dtype = np.int32)

        self.mutation_genes = np.empty((length_of_neurons_genes+neurons_to_be_added), dtype = np.float64)

        self.neurons_genes, self.mutation_genes,self.connections_genes = \
                                self.add_neurons(self.neurons_genes,
                                                 self.created_neurons_genes,
                                                 self.mutation_genes,
                                                 self.created_mutation_genes,
                                                 self.created_connections_genes,
                                                 neurons_to_be_added,
                                                 input_indexes,
                                                 output_indexes,
                                                 iteration,
                                                 activation_functions,
                                                 WEIGHT,
                                                 MUTATION)

        self.mutation_genes = UF.onedim_genes_cleaner(self.mutation_genes,
                                                      np.float64,
                                                      -1)
        self.neurons_genes = UF.twodim_genes_cleaner(self.neurons_genes,
                                                     0,
                                                     np.int32,
                                                     -1)
        self.connections_genes = UF.twodim_genes_cleaner(self.connections_genes,
                                                         3,
                                                         np.float64,
                                                         -1)



    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void mutate_connections_genes(self,
                                       double prb_deleteion,
                                       int iteration):

        cdef Py_ssize_t i, j
        cdef int length_of_connections_genes = self.connections_genes.shape[0]
        cdef int width_of_connections_genes=  self.connections_genes.shape[1]

        cdef int number_of_connections_to_be_added  = int(prb_deleteion * length_of_connections_genes) + 1 *(0 >= int(prb_deleteion * length_of_connections_genes))

        for i in range(length_of_connections_genes):

            if self.connections_genes[i][3] != -1.:

                if prb_deleteion >= rand() / (RAND_MAX * 1.0) and (self.connections_genes[i][0] >= 0 and self.connections_genes[i][1] >= 0):
                    # extra check added to prevent input and ouput connections from deletion
                    for j in prange(width_of_connections_genes, nogil = True):
                        self.connections_genes[i][j] = -1.

                    continue
                else:
                    if self.connections_genes[i][5] >= rand() / (RAND_MAX * 1.0):
                        self.connection_change(self.neurons_genes,
                                          self.connections_genes,
                                          i)

                    if self.connections_genes[i][5] >= rand() / (RAND_MAX * 1.0):
                        UF.change_gene_weigth(self.connections_genes[i], 5)

        self.connections_genes = self.add_connection(self.neurons_genes,
                                                     self.connections_genes,
                                                     number_of_connections_to_be_added,
                                                     iteration)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void create_matrix(self, int number_of_inputs):

        self.matrix = np.zeros((self.neurons_genes.shape[0], self.neurons_genes.shape[0]+\
                                number_of_inputs), dtype = np.float64)
        cdef Py_ssize_t i
        cdef int length = self.neurons_genes.shape[0]
        cdef dict matches = {}

        for i in range(length):
            matches[self.neurons_genes[i][0]] = i

        self.connections_genes = UF.twodim_genes_cleaner(self.connections_genes, 3, np.float64, -1)

        self.matrix = self.initialize_connections_with_reordering(self.matrix,
                                               self.connections_genes,
                                               number_of_inputs,
                                               matches,
                                               self.neurons_genes)

    cdef Network get_network(self, int order_number):

        cdef Network A
        A = Network()
        A.params_setter(order_number,
                        self.neurons_genes,
                        self.mutation_genes,
                        self.connections_genes,
                        self.matrix)
        return A


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef delete_neuron(self,
                       int [:,:] neurons_genes,
                       double [:] mutation_genes,
                       double [:,:]  connections_genes,
                       int order_number_delete,
                       int order_number_connection):

       cdef Py_ssize_t i, j

       cdef int neurons_genes_length = neurons_genes.shape[0]
       cdef int neurons_genes_width = neurons_genes.shape[1]

       cdef int connections_genes_length = connections_genes.shape[0]
       cdef int connections_genes_width = connections_genes.shape[1]

       cdef int mutation_genes_length = mutation_genes.shape[0]

       for j in prange(neurons_genes_width, nogil = True):
          neurons_genes[order_number_delete][j] = -1
       mutation_genes[order_number_delete] = -1.


       for i in prange(connections_genes_length, nogil=True):
           if connections_genes[i][0] == order_number_connection or connections_genes[i][1] == order_number_connection:

               for j in prange(connections_genes_width):
                    connections_genes[i][j] = -1.

       return  neurons_genes, connections_genes, mutation_genes


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef add_neurons(self,
                     int[:,:] neurons_genes,
                     int[:,:] pre_mutation_neurons_genes,
                     double[:] mutation_genes,
                     double[:] pre_mutation_mutation_genes,
                     double[:, :] pre_mutation_connections_genes,
                     int neurons_to_add,
                     int[:] input_indexes,
                     int[:] output_indexes,
                     int iteration,
                     int[:] activation_functions,
                     double weight,
                     double mutation_rate):

       # All types of neurons may be added (0,1,2)

       cdef Py_ssize_t i, j

       cdef int old_neurons_length = pre_mutation_neurons_genes.shape[0]
       cdef int neurons_width = pre_mutation_neurons_genes.shape[1]

       cdef int old_connections_length = pre_mutation_connections_genes.shape[0]
       cdef int connections_width = pre_mutation_connections_genes.shape[1]

       cdef int length_of_activation_f = activation_functions.size

       cdef int[:] types_of_added_neurons = np.empty(neurons_to_add, dtype = np.int32)
       cdef int input_neurons_counter = 0
       cdef int output_neurons_counter = 0

       for i in prange(neurons_to_add, nogil = True):
           types_of_added_neurons[i] = UF.randint(0,3)   # Here type of added neurons may be changed e.g. randint(1,3)

       for i in range(neurons_to_add):
           if types_of_added_neurons[i] == 0:
               input_neurons_counter += 1
           if types_of_added_neurons[i] == 2:
               output_neurons_counter += 1

       for i in prange(old_neurons_length, nogil = True):
           for j in prange(neurons_width):
               neurons_genes[i][j] = pre_mutation_neurons_genes[i][j]
           mutation_genes[i] = pre_mutation_mutation_genes[i]


       cdef double [:,:] connections_genes = np.empty((old_connections_length+input_neurons_counter+output_neurons_counter, connections_width), dtype = np.float64)

       for i in prange(old_connections_length, nogil = True):
           for j in prange(connections_width):
               connections_genes[i][j] = pre_mutation_connections_genes[i][j]

       cdef int counter = 0
       cdef int number_of_available_inputs = input_indexes.size
       cdef int number_of_available_outputs = output_indexes.size
       for i in range(neurons_to_add):

           neurons_genes[i+old_neurons_length][0] = i+old_neurons_length
           neurons_genes[i+old_neurons_length][1] = types_of_added_neurons[i]
           neurons_genes[i+old_neurons_length][2] = iteration
           neurons_genes[i+old_neurons_length][3] = activation_functions[UF.randint(0,length_of_activation_f)]
           mutation_genes[i+old_neurons_length] = weight

           if types_of_added_neurons[i] == 0:

               connections_genes[counter+old_connections_length][0] = input_indexes[UF.randint(0,number_of_available_inputs)]/1.
               connections_genes[counter+old_connections_length][1] = (i+old_neurons_length)/1.
               connections_genes[counter+old_connections_length][2] = weight
               connections_genes[counter+old_connections_length][3] = 1.
               connections_genes[counter+old_connections_length][4] = iteration
               connections_genes[counter+old_connections_length][5] = mutation_rate

               counter += 1

           if types_of_added_neurons[i] == 2:
               connections_genes[counter+old_connections_length][0] = (i+old_neurons_length)/1.
               connections_genes[counter+old_connections_length][1] = output_indexes[UF.randint(0, number_of_available_outputs)]/1.
               connections_genes[counter+old_connections_length][2] = weight
               connections_genes[counter+old_connections_length][3] = 1.
               connections_genes[counter+old_connections_length][4] = iteration
               connections_genes[counter+old_connections_length][5] = mutation_rate

       return neurons_genes, mutation_genes, connections_genes


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double [:,:] initialize_connections_with_reordering(self,
                                                             double [:,:] matrix,
                                                             double [:,:] connections_genes,
                                                             int number_of_inputs,
                                                             dict matches,
                                                             int[:, :] neurons_genes) nogil:

       cdef Py_ssize_t i
       cdef int from_
       cdef int to_
       cdef int input_index

       cdef int length = connections_genes.shape[0]

       for i in range(length):
           if connections_genes[i][3]==1 and connections_genes[i][1] >= 0: # checks activity status of a connection and whether it's not an output connection


           # if it is input -> input neuron connection
               if connections_genes[i][0]<0:

                   with gil:
                   # since numeration starts with -1
                       from_ = abs(int(connections_genes[i][0]))-1
                       to_ = int(connections_genes[i][1])

                       assert to_ in matches, f'No such neuron {to_, np.asarray(connections_genes[i]),matches}'
                       to_ = matches[to_]

                   matrix[to_][from_] = connections_genes[i][2]

               elif connections_genes[i][0] >= 0:

                   with gil:

                       from_ = int(connections_genes[i][0])
                       to_ = int(connections_genes[i][1])

                       assert from_ in matches, f'No such neuron {from_, np.asarray(connections_genes), matches}'
                       assert to_ in matches, f'No such neuron {to_, np.asarray(connections_genes), matches}'

                       from_ = matches[from_] + number_of_inputs
                       to_ = matches[to_]

                   matrix[to_][from_] = connections_genes[i][2]

       return matrix

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void connection_change(self,
                                int [:,:] neurons_genes,
                                double [:,:] connections_genes,
                                int order_number):

       cdef Py_ssize_t i,j

       cdef int length = neurons_genes.shape[0]
       cdef int width = neurons_genes.shape[1]
       cdef int what_parameter_is_changed = UF.randint(0, width)

       cdef int random_neuron = neurons_genes[UF.randint(0, length)][0]

       if what_parameter_is_changed == 0:

           if connections_genes[order_number][what_parameter_is_changed] >= 0:
               connections_genes[order_number][what_parameter_is_changed] = \
                                               random_neuron /1.
            # same block for output neurons



       elif what_parameter_is_changed == 1:

           if connections_genes[order_number][what_parameter_is_changed] >= 0:
               connections_genes[order_number][what_parameter_is_changed] = \
                                               random_neuron /1.

       elif what_parameter_is_changed == 2:

           UF.change_gene_weigth(connections_genes[order_number], what_parameter_is_changed)

       elif what_parameter_is_changed == 3:

           if connections_genes[order_number][what_parameter_is_changed] == 0.:
               connections_genes[order_number][what_parameter_is_changed] = 1.

           else:
               connections_genes[order_number][what_parameter_is_changed] = 0.

       # other parameters cannot be changed due to their 'nature'

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:,:] add_connection(self,
                                    int [:,:] neurons_genes,
                                    double[:,:] connections_genes,
                                    int number_of_connections_to_be_added,
                                    int iteration):

       cdef Py_ssize_t i, j
       cdef int connections_genes_length = connections_genes.shape[0]
       cdef int connections_genes_width = connections_genes.shape[1]

       cdef int neurons_genes_length = neurons_genes.shape[0]

       cdef double[:,:] new_connections_genes
       cdef double mean_weight = 0.
       cdef double mean_mutation_rate = 0.

       new_connections_genes = \
                      np.empty((connections_genes_length+number_of_connections_to_be_added, connections_genes_width), dtype = np.float64)

       for i in prange(connections_genes_length, nogil = True):

           for j in prange(connections_genes_width):
               new_connections_genes[i][j] = connections_genes[i][j]

       for i in range(connections_genes_length):

           if new_connections_genes[i][3] != -1:
               mean_weight = mean_weight+new_connections_genes[i][2]/connections_genes_length
               mean_mutation_rate = mean_mutation_rate + new_connections_genes[i][5]/connections_genes_length

       for i in prange(number_of_connections_to_be_added, nogil = True):
           new_connections_genes[i+connections_genes_length][0] = neurons_genes[UF.randint(0, neurons_genes_length)][0]/1.
           new_connections_genes[i+connections_genes_length][1] = neurons_genes[UF.randint(0, neurons_genes_length)][0]/1.
           new_connections_genes[i+connections_genes_length][2] = mean_weight
           new_connections_genes[i+connections_genes_length][3] = 1.       # Neurons are added in active state
           new_connections_genes[i+connections_genes_length][4] = iteration
           new_connections_genes[i+connections_genes_length][5] = mean_mutation_rate


       return new_connections_genes
