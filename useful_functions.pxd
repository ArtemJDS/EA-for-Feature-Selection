
ctypedef fused numeric:
    double
    int

cdef class UsefulFunctions:

    cdef numeric[:,:] twodim_genes_cleaner(self,
                                           numeric[:,:] genes,
                                           int position_ord_n,
                                           type T,
                                           int a)
      # Helper function. Takes 2-d genes, check position_ord_n in every array subarray.
      # If a = 0, than genes[i][position_ord_n] must be either 0 or 1
      # IF a = -1 than genes[i][position_ord_n] must not -1
      # Same with position_ord_n2
      # It genes[i][position_ord_n] fits these conditions than genes[i] stays
      # otherwise it gets deleted


    cdef numeric[:] onedim_genes_cleaner(self,
                                         numeric[:] genes,
                                         type T,
                                         int a)
      # Same as twodim_genes_cleaner but with 1-d array


    cdef int randint(self,
                     int lower,
                     int upper) nogil
      # Helper function. Same as np.random.randint but faster
      # Only works for non-negative values
      # randint(0,2) returns either 0 or 1

    cdef void change_gene_weigth(self,
                                 double[:] weights,
                                 int index) nogil
        # Helper function. Changes weight and limits it by 0.03 and 0.97
        # to left space for ptentially useful mutations
