from network cimport Network

cdef class Factory():


    cdef public int first_parent
    cdef public int second_parent

    cdef public int [:,:] neurons_genes
    cdef public double [:] mutation_genes
    cdef public double [:,:] connections_genes
    cdef public double [:,:] matrix



cdef class GrandNetworkFactory_FullInput_NoInter(Factory):

    '''
    Subclass that returns so-called GrandNetworks

    Those are the first generation of networks and that's why they
    must be initialized from scratch instead of mating function

    Initializes connections from inputs to input neurons
    (every input neurons has connection to a corresponding input)

    Neurons and connections get equal mutation rates

    '''


    cdef void create_neurons_genes(self,
                                   int number_of_neurons,
                                   int number_of_input_neurons,
                                   int number_of_output_neurons,
                                   int[:] activation_functions,
                                   double MUTATION_RATE)

    '''
    Creates genomes of neurons (neurons_genes and mutation_genes)


    activation_functions: array of functions
                         assigned randomly

                       e.g.
                       neuron_0 takes 0 - sigmoid AF
                       neuron_1 takes 3 - ReLu AF


    Assignes:
                   neurons_genes: ndarray, each array containing data
                                   about individual neurons

                   each gene contains :
                           [
                           order_number: int (network-specific numeration)
                           type : int that specifies type
                               (0 - input, 1 - interneuron, 2 - output neuron)

                           iteration : set to 0 in this function because here they
                                       are created for the first time

                           activation_function : int of (from numeration of act_fuctions)
                           ]

                   mutation_genes: ndarray of floats that represents  how probable
                                           a mutation in the gene is
                                   (this is separeted to keep int type of
                                   neurons_genes. In connections_genes on the other
                                   hand mutation_genes is inside the common ndarray
                                   because connections_genes has double type)

    '''

    cdef void create_connections_genes(self,
                                       int number_of_neurons,
                                       int number_of_input_neurons,
                                       int number_of_output_neurons,
                                       int[:] input_indexes,
                                       int[:] output_indexes,
                                       double WEIGHT,
                                       double MUTATION_RATE)

    '''
    Creates genomes of connections.

    Assignes:
                  connections_genes: ndarray of genes each containing data
                                      about connections
                  where each gene:

                    [

                    from_neuron : int, (if input -> inp neuron connection is
                                        specified then this int goes with
                                        minus sign)
                                        that's why numeration of inputs MUST
                                        START WITH -1 (not -0)
                    to_neuron : int

                    weight : float

                    active : int (0 - not active, 1 - active, -1 - deleted)

                    creation_iteration: int (this is the 0'th iteration that's
                                        why value is set to 0)

                    mutation_exposure: float that represents how probable
                                       a mutation in the gene is
                         ]

    '''


    cdef void create_matrix (self, int number_of_inputs)

    cdef Network get_network(self, int order_number)


    cdef double [:,:]  initialize_connections_no_reordering(self,
                                                            double [:,:] matrix,
                                                            double [:,:] connections_genes,
                                                            int number_of_inputs ) nogil

    #Helper function. Adds connections from connections_genes to a matrix


cdef class NetworkFactory(Factory):

    '''
    Subclass that returns common Network

    Takes two parents and creates a new network on their genes and then mutates

    '''
    cdef public int [:,:] matches
    cdef public int [:,:] created_neurons_genes
    cdef public double [:] created_mutation_genes
    cdef public double [:,:] created_connections_genes

    cdef void create_neurons_genes_through_mating(self,
                                                  Network parent_1,
                                                  Network parent_2)

    '''

    Length of new genes are  the mean of length of two parents +
      random value between -1 and 1

    '''

    cdef void create_connections_genes_through_mating(self,
                                                      Network parent_1,
                                                      Network parent_2)
    '''
    Here a rather difficult procedure takes place.

    1. Take neuron from self.created_neurons_genes
    2. Find its old counterpart (from self.matches)
    3. Find connections (that have from_neuron equal to the counterpart)
    4. For every connection if its to_neuron-counterpart is in new network then
    add it to the networks
    4.1 If it is input -> input neuron connection also add it

    '''

    cdef void mutate_neurons_genes(self,
                                   double prb_deleteion,
                                   int [:] activation_functions,
                                   int [:] input_indexes,
                                   int [:] output_indexes,
                                   int iteration,
                                   double WEIGHT,
                                   double MUTATION)

    '''
    Performs deletion and addition of neurons.

    If neuron gets deleted all from and to connections get deleted too.
    All types of neurons may be added.
    If input neuron is added than a corresponding connection is also added.

    '''

    cdef void mutate_connections_genes(self,
                                       double prb_deleteion,
                                       int iteration)

    '''
    Performs deletion and addition of connections.

    '''

    cdef void create_matrix(self, int number_of_inputs)

    cdef Network get_network(self, int order_number)

    cdef delete_neuron(self,
                       int [:,:] neurons_genes,
                       double [:] mutation_genes,
                       double [:,:]  connections_genes,
                       int order_number_delete,
                       int order_number_connection)


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
                     double mutation_rate)

    cdef  double [:,:] initialize_connections_with_reordering(self,
                                                              double [:,:] matrix,
                                                              double [:,:] connections_genes,
                                                              int number_of_inputs,
                                                              dict matches,
                                                              int[:, :] neurons_genes
                                                              ) nogil
    #Helper function. Adds connections from connections_genes to a matrix

    cdef void connection_change(self,
                                int [:,:] neurons_genes,
                                double [:,:] connections_genes,
                                int order_number)

    cdef double[:,:] add_connection(self,
                                    int [:,:] neurons_genes,
                                    double[:,:] connections_genes,
                                    int number_of_connections_to_be_added,
                                    int iteration)
