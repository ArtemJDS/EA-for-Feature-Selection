from network cimport Network
from factories cimport NetworkFactory


cdef class Mating:
    '''
    This class implements mating process.

    All networks with efficiency lower than limit are excluded from mating.
    For every such network a new one is created via mating of two networks.

    Parents are chosen randomly
    1. Pick a random network.
    2. If it's efficiency is greater than limit than compare random value from
    0 to 1 with the efficiency.
    3. If this value falls between 0 and this efficiency - be it a parent
    4. Should step 2 or step 3 be failed, chose another network and repeat those
    steps

    All necessary params for mating and mutation must be set

    Main function (mating) also returns new last order_number
    '''

    cdef double limit
    cdef double neuron_prb_del
    cdef double connection_prb_del
    cdef int[:] activation_functions
    cdef int number_of_inputs
    cdef int iteration
    cdef double weight
    cdef double mutation_rate
    cdef int[:] input_indexes
    cdef int[:] output_indexes
    cdef int init_order_number

    cdef void set_efficiency_limit(self, double limit)

    cdef void set_neuron_prb_del(self, double neuron_prb_del)

    cdef void set_connection_prb_del(self, double connection_prb_del)

    cdef void set_activation_functions(self, int[:] activation_functions)

    cdef void set_number_of_inputs(self, int number_of_inputs)

    cdef void set_iteration(self, int iteration)

    cdef void set_default_weigth(self, double weight)

    cdef void set_mutation_rate(self, double mutation_rate)

    cdef void set_order_number(self, int order_number)

    cdef void set_input_indexes(self, int[:] input_indexes)

    cdef void set_output_indexes(self, int[:] output_indexes)

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
                                  double MUTATION_RATE)

    cdef int mating(self, Network[:] networks)

    cdef Network create_new_network(self,
                                    NetworkFactory factory,
                                    Network parent_1,
                                    Network parent_2,
                                    int order_number)
