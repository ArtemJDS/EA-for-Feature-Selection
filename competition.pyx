cimport numpy as np
from network cimport Network
from useful_functions cimport twodim_genes_cleaner, onedim_genes_cleaner, randint, change_gene_weigth, sqroot, round
from factories cimport NetworkFactory
from libc.stdlib cimport rand, srand, RAND_MAX
cdef extern from "limits.h":
    int INT_MAX
from libc.time cimport time
srand(time(NULL))
import cython
import numpy as np
from cython.parallel import prange


class Exctinction(Exception):
    pass

cdef class Mating:

    cdef void set_efficiency_limit(self, double limit):
        self.limit = limit

    cdef void set_neuron_prb_del(self, double neuron_prb_del):
        self.neuron_prb_del = neuron_prb_del

    cdef void set_connection_prb_del(self, double connection_prb_del):
        self.connection_prb_del = connection_prb_del

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_activation_functions(self, int[:] activation_functions):

        cdef Py_ssize_t i
        cdef int length = activation_functions.size
        self.activation_functions = np.empty(length, dtype = np.int32)

        for i in prange(length, nogil = True):
            self.activation_functions[i] = activation_functions[i]

    cdef void set_number_of_inputs(self, int number_of_inputs):
        self.number_of_inputs = number_of_inputs

    cdef void set_iteration(self, int iteration):
        self.iteration = iteration

    cdef void set_default_weigth(self, double weight):
        self.weight = weight

    cdef void set_mutation_rate(self, double mutation_rate):
        self.mutation_rate = mutation_rate

    cdef void set_order_number(self, int init_order_number):
        self.init_order_number = init_order_number

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_input_indexes(self, int[:] input_indexes):

        cdef Py_ssize_t i
        cdef int length = input_indexes.size
        self.input_indexes = np.empty(length, dtype = np.int32)

        for i in prange(length, nogil = True):
            self.input_indexes[i] = input_indexes[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_output_indexes(self, int[:] output_indexes):

        cdef Py_ssize_t i
        cdef int length = output_indexes.size
        self.output_indexes = np.empty(length, dtype = np.int32)

        for i in prange(length, nogil = True):
            self.output_indexes[i] = output_indexes[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_mutation_params(self,
                                  double neuron_prb_del,
                                  double connection_prb_del,
                                  int[:] activation_functions,
                                  int number_of_inputs,
                                  int iteration,
                                  int init_order_number,
                                  int [:] input_indexes,
                                  int [:] output_indexes,
                                  double WEIGHT,
                                  double MUTATION_RATE):

        self.neuron_prb_del = neuron_prb_del
        self.connection_prb_del = connection_prb_del

        cdef Py_ssize_t i

        cdef int length = activation_functions.size
        self.activation_functions = np.empty(length, dtype = np.int32)
        for i in prange(length, nogil = True):
            self.activation_functions[i] = activation_functions[i]

        self.number_of_inputs = number_of_inputs
        self.iteration = iteration
        self.init_order_number = init_order_number
        self.weight = WEIGHT
        self.mutation_rate = MUTATION_RATE

        length = input_indexes.size
        self.input_indexes = np.empty(length, dtype = np.int32)

        for i in prange(length, nogil = True):
            self.input_indexes[i] = input_indexes[i]

        length = output_indexes.size
        self.output_indexes = np.empty(length, dtype = np.int32)

        for i in prange(length, nogil = True):
            self.output_indexes[i] = output_indexes[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int mating(self, Network[:] networks, NetworkFactory factory):

        cdef Py_ssize_t i

        cdef int number_of_networks = networks.size
        cdef int deleted_networks = 0
        cdef int parent_1 = -1
        cdef int parent_2 = -1
        cdef int rand_n
        cdef int counter = 0

        cdef double [:] efficiencies = np.empty(number_of_networks, dtype = np.float64) # create this to use gil inside cycle
                                                                                        # otherwise cannot
        cdef int above_limit_counter = 0         # to check for extinction
        for i in range(number_of_networks):
            efficiencies[i] = networks[i].efficiency
            if efficiencies[i] >= self.limit:
                above_limit_counter += 1

        if above_limit_counter < 2:
            raise Exctinction('Too few networks with efficiency above limit')



        for i in range(number_of_networks):

            if efficiencies[i] < self.limit:
                    with nogil:
                        while parent_1 == -1:
                            rand_n = randint(0, number_of_networks)
                            if efficiencies[rand_n] >= self.limit :
                                parent_1 = rand_n

                        while parent_2 == -1:
                            rand_n = randint(0, number_of_networks)
                            if efficiencies[rand_n] >= self.limit:
                                parent_2 = rand_n

                    networks[i] = self.create_new_network(factory,
                                                          networks[parent_1],
                                                          networks[parent_2],
                                                          self.init_order_number + counter)
                    counter += 1

        return self.init_order_number + counter

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int mating_quantile(self, Network[:] networks, NetworkFactory factory):

        cdef Py_ssize_t i

        cdef int number_of_networks = networks.size
        cdef int deleted_networks = 0
        cdef int parent_1 = -1
        cdef int parent_2 = -1
        cdef int rand_n
        cdef int counter = 0

        cdef double [:] efficiencies = np.empty(number_of_networks, dtype = np.float64) # create this to use gil inside cycle
                                                                                        # otherwise cannot

        cdef int above_limit_counter = 0         # to check for extinction
        for i in range(number_of_networks):
            efficiencies[i] = networks[i].efficiency


        cdef double quantile = np.quantile(efficiencies, self.limit)

        for i in range(number_of_networks):
            if efficiencies[i] >= quantile:
                above_limit_counter += 1

        if above_limit_counter < 2:
            raise Exctinction('Too few networks with efficiency above limit')

        if above_limit_counter != number_of_networks:
            for i in range(number_of_networks):

                if efficiencies[i] < quantile:
                        with nogil:
                            while parent_1 == -1:
                                rand_n = randint(0, number_of_networks)
                                if efficiencies[rand_n] >= self.limit :
                                    parent_1 = rand_n

                            while parent_2 == -1:
                                rand_n = randint(0, number_of_networks)
                                if efficiencies[rand_n] >= self.limit:
                                    parent_2 = rand_n

                        networks[i] = self.create_new_network(factory,
                                                            networks[parent_1],
                                                            networks[parent_2],
                                                            self.init_order_number + counter)
                        counter += 1
        else:
            for i in range(number_of_networks):
                if randint(0,1) == 0:
                    with nogil:
                        while parent_1 == -1:
                            rand_n = randint(0, number_of_networks)
                            if efficiencies[rand_n] >= self.limit :
                                parent_1 = rand_n

                        while parent_2 == -1:
                            rand_n = randint(0, number_of_networks)
                            if efficiencies[rand_n] >= self.limit:
                                parent_2 = rand_n

                    networks[i] = self.create_new_network(factory,
                                                        networks[parent_1],
                                                        networks[parent_2],
                                                        self.init_order_number + counter)
                    counter += 1

        return self.init_order_number + counter


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Network create_new_network(self,
                                    NetworkFactory factory,
                                    Network parent_1,
                                    Network parent_2,
                                    int order_number,
                                    ):

        cdef Network new_network

        factory.create_neurons_genes_through_mating(parent_1,
                                                    parent_2)

        factory.create_connections_genes_through_mating(parent_1,
                                                        parent_2)

        factory.mutate_neurons_genes(self.neuron_prb_del,
                                     self.activation_functions,
                                     self.input_indexes,
                                     self.output_indexes,
                                     self.iteration,
                                     self.weight,
                                     self.mutation_rate)

        factory.mutate_connections_genes(self.connection_prb_del, self.iteration)
        factory.create_matrix(self.number_of_inputs)
        new_network = factory.get_network(order_number)

        return new_network
