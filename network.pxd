
cdef class Network:
    cdef readonly int network_order_number
    cdef readonly int first_parent
    cdef readonly int second_parent

    cdef readonly int [:,:] neurons_genes
    cdef readonly double [:] mutation_genes
    cdef readonly double [:,:] connections_genes
    cdef readonly double [:,:] matrix

    cdef public double efficiency

    '''

    Network params:

      network_order_number: variable that keeps track of all created networks
                            (unique)
      first_parent, second_parent: to track genealogy of a network

      neurons_genes, mutation_genes: genes that specify properties of nodes
                                     (they are separated to preserve types)

      connections_genes: genes that specify properties of connections


      matrix: activity matrix used in the environment

      efficiency: how well the network is doing

    '''

    cdef void params_setter(self,
                            int order_number,
                            int [:,:] neurons_genes,
                            double [:] mutation_genes,
                            double [:,:] connections_genes,
                            double [:,:] matrix) nogil
