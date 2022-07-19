cimport numpy as np
from network cimport Network
from useful_functions cimport UsefulFunctions
from factories cimport NetworkFactory
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

    cdef set_iteration(self, int iteration):
        self.iteration = iteration

    cdef set_default_weigth(self, double weight):
        self.weight = weight

    cdef set_mutation_rate(self, double mutation_rate):
        self.mutation_rate = mutation_rate

    cdef set_order_number(self, int init_order_number):
        self.init_order_number = init_order_number

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_mutation_params(self,
                                  double neuron_prb_del,
                                  double connection_prb_del,
                                  int[:] activation_functions,
                                  int number_of_inputs,
                                  int iteration,
                                  int init_order_number,
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int mating(self, Network[:] networks):

        cdef Py_ssize_t i

        cdef int number_of_networks = networks.size
        cdef int deleted_networks = 0
        cdef int parent_1 = -1
        cdef int parent_2 = -1
        cdef int rand_n
        cdef int counter = 0

        cdef NetworkFactory factory
        factory = NetworkFactory()

        cdef double [:] efficiencies = np.empty(number_of_networks, dtype = np.float64) # create this to use gil inside cycle
                                                                                        # otherwise cannot
        for i in range(number_of_networks):
            efficiencies[i] = networks[i].efficiency


        for i in range(number_of_networks):

            if efficiencies[i] < self.limit:
                    with nogil:
                        while parent_1 == -1:
                            rand_n = UF.randint(0, number_of_networks)
                            if efficiencies[rand_n] >= rand() / RAND_MAX:
                                parent_1 = rand_n

                        while parent_2 == -1:
                            rand_n = UF.randint(0, number_of_networks)
                            if efficiencies[rand_n] >= rand() / RAND_MAX:
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
                                     self.number_of_inputs,
                                     self.iteration,
                                     self.weight,
                                     self.mutation_rate)

        factory.mutate_connections_genes(self.connection_prb_del, self.iteration)
        factory.create_matrix(self.number_of_inputs)
        new_network = factory.get_network(order_number)

        return new_network
