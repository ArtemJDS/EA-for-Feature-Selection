from cython.parallel import prange
import cython
import numpy as np

cdef class Network:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void params_setter(self,
                            int order_number,
                            int [:,:] neurons_genes,
                            double [:] mutation_genes,
                            double [:,:] connections_genes,
                            double [:,:] matrix) nogil:

        self.network_order_number = order_number

        cdef Py_ssize_t i, j
        cdef int length, width

        with gil:
                length = neurons_genes.shape[0]
                width = neurons_genes.shape[1]

                self.neurons_genes = np.empty((length, width), dtype = np.int32 )
                self.mutation_genes = np.empty(length, dtype = np.float64 )

        for i in range(length):
            for j in range(width):
              self.neurons_genes[i][j] = neurons_genes[i][j]

            self.mutation_genes[i] = mutation_genes[i]

        with gil:
                length = connections_genes.shape[0]
                width = connections_genes.shape[1]
                self.connections_genes = np.empty((length, width), dtype = np.float64 )

        for i in range(length):
            for j in range(width):
              self.connections_genes[i][j] = connections_genes[i][j]

        with gil:
                length = matrix.shape[0]
                width = matrix.shape[1]
                self.matrix = np.empty((length, width), dtype = np.float64 )

        for i in range(length):
            for j in prange(width):
              self.matrix[i][j] = matrix[i][j]

        self.efficiency = 0.
