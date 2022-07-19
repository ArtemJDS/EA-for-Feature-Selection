import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport rand, srand, RAND_MAX
cdef extern from "limits.h":
    int INT_MAX
from libc.time cimport time
srand(time(NULL))


cdef class UsefulFunctions:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef numeric[:,:] twodim_genes_cleaner(self,
                                           numeric[:,:] genes,
                                           int position_ord_n,
                                           type T,
                                           int a):

        cdef Py_ssize_t i, j

        cdef int active_counter = 0
        cdef int length = genes.shape[0]
        cdef int width = genes.shape[1]

        cdef numeric[:,:] new_genes

        cdef int r

        if a == 0:

            for i in range(length):
                if (genes[i][position_ord_n] == 1 or genes[i][position_ord_n] == 0):
                    active_counter = active_counter+1
            new_genes = np.empty((active_counter, width), dtype = T)

            r = 0
            for i in range(length):
                if genes[i][position_ord_n] == 1 or genes[i][position_ord_n] == 0:

                    for j in prange(width, nogil = True):
                        new_genes[r][j] = genes[i][j]
                    r = r+1

        if a == -1:

            for i in range(length):
                if genes[i][position_ord_n] != -1:
                    active_counter = active_counter+1
            new_genes = np.empty((active_counter, width), dtype = T)

            r = 0
            for i in range(length):
                if genes[i][position_ord_n] != -1:

                    for j in prange(width, nogil = True):
                        new_genes[r][j] = genes[i][j]
                    r = r+1

        return new_genes



    cdef numeric[:] onedim_genes_cleaner(self,
                                         numeric[:] genes,
                                         type T,
                                         int a):

        cdef Py_ssize_t i
        cdef int active_counter = 0
        cdef int length = genes.size

        cdef numeric[:] new_genes

        cdef int r
        if a == -1:

            for i in range(length):
                if genes[i] != -1:
                    active_counter = active_counter+1
            new_genes = np.empty(active_counter, dtype = T)

            r = 0
            for i in range(length):
                if genes[i] != -1:
                    new_genes[r] = genes[i]
                    r = r+1

        if a == 0:

            for i in range(length):
                if genes[i] == 0 or genes[i] == 1:
                    active_counter = active_counter+1
            new_genes = np.empty(active_counter, dtype = T)

            r = 0
            for i in range(length):
                if genes[i] == 0 or genes[i] == 1:
                    new_genes[r] = genes[i]
                    r = r+1
        return new_genes



    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int randint(self,
                     int lower,
                     int upper) nogil:
        return (rand() % (upper-lower))


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void change_gene_weigth(self,
                                 double[:] weights,
                                 int index) nogil:

        # Helper function. Changes weight and limits it by 0.03 and 0.97
        # to left space for ptentially useful mutations

        weights[index]  = (weights[index]+rand() / (RAND_MAX * 1.0)*(0.05-(-0.05))-0.05) * \
                          (0.03<weights[index]<0.97) + (0.97 * (weights[index]>0.97)) +\
                          (0.03 * (weights[index]<0.03))
