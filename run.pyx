import numpy as np
cimport numpy as np
from cython.parallel import prange
import cv2
import cython
from useful_functions cimport twodim_genes_cleaner, onedim_genes_cleaner, randint, change_gene_weigth, sqroot, round
from activation_functions cimport linear, logistic, tanh, ReLU
from network cimport Network
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
srand(time(NULL))


cdef class Enviroment:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def create_enviroment(self, **kwargs):

        cdef Py_ssize_t i, j, k

        x = cv2.imread(kwargs['1'])
        cdef int heigth = x.shape[0]     # there is always the 1'st
        cdef int width = x.shape[0]
        cdef int number_of_params = len(kwargs)

        cdef unsigned char [:,:,:] img = np.empty((heigth, width, 4), dtype = np.ubyte)
        self.enviroment = np.empty((heigth, width, number_of_params), dtype = np.float64)

        for i in range(number_of_params):
            img = cv2.imread(kwargs[str(i+1)])

            for j in prange(heigth, nogil = True):
                for k in prange(width):
                    self.enviroment[j][k][i] = img[j][k][1]/255.


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void run(self, int number_of_iterations, int order_number) nogil :
        cdef Py_ssize_t i
        cdef int length_of_input
        with gil:
            length_of_input = self.input.size - self.network.matrix.shape[0]

        for i in range(length_of_input):
            self.input[i] = self.common_input(i)

        with gil:
            self.efficiency_storage[order_number] = 0.
            for i in range(number_of_iterations):
                self.dynamics()
                self.efficiency_storage[order_number] += self.current_concentration(0)

            if self.efficiency_storage[order_number] > self.max_normalize:
                self.max_normalize = self.efficiency_storage[order_number]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef str check_zero_efficiency(self, int order_number):
        cdef Py_ssize_t i
        cdef int length_of_neurons = self.network.neurons_genes.shape[0]
        cdef int count_inputs, count_outputs = 0

        for i in range(length_of_neurons):
            if self.network.neurons_genes[i][1] == 0:
                count_inputs = 1
            if self.network.neurons_genes[i][1] == 2:
                count_outputs = 1
            if count_inputs == count_outputs == 1:
              return 'both input and output exist'

        self.efficiency_storage[order_number] = 0.
        return 'set efficiency to zero'


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_network(self, Network network):
        self.network = network


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_output_status(self) nogil:
        cdef Py_ssize_t i, j
        cdef int length_of_neurons
        cdef int length_of_connections
        with gil:
            length_of_neurons = self.network.matrix.shape[0]
            length_of_connections = self.network.connections_genes.shape[0]
            self.output_status = np.empty(length_of_neurons, dtype = np.int32)

        for i in range(length_of_neurons):
            if self.network.neurons_genes[i][1] == 2:

                for j in range(length_of_connections):
                    if self.network.connections_genes[j][0] == self.network.neurons_genes[i][0] and self.network.connections_genes[j][1] < 0:
                        with gil: self.output_status[i] = int(self.network.connections_genes[j][1])
                        break
            else:
                self.output_status[i] = 0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_input(self):
      self.input = np.empty(self.network.matrix.shape[1], dtype = np.float64)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_position(self, int x, int y) nogil:
        self.x = x
        self.y = y


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_number_of_outputs(self, int number_of_outputs) nogil:
        self.number_of_outputs = number_of_outputs


    cdef void set_send(self):
        self.send = np.empty(self.number_of_outputs, dtype = np.float64)


    cdef void set_result(self):
        cdef int number_of_neurons = self.network.matrix.shape[0]
        self.result = np.zeros(number_of_neurons, dtype = np.float64)


    cdef void set_efficiency_storage(self, int length):
      self.max_normalize = 0.
      self.efficiency_storage = np.empty(length, dtype = np.float64)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void dynamics(self) nogil:
      cdef Py_ssize_t i
      cdef int length_of_input
      cdef int number_of_neurons
      with gil:
          number_of_neurons = self.network.matrix.shape[0]
          length_of_input = self.input.size - number_of_neurons
          self.send = np.zeros(self.number_of_outputs, dtype = np.float64)

      for i in range(number_of_neurons):
          self.input[i + length_of_input] = self.result[i]

      with gil:
          self.result = np.matmul(self.network.matrix, self.input)

      for i in range(number_of_neurons):
          self.apply_activation_functions(i)

      with gil:
          for i in range(number_of_neurons):
              if self.output_status[i] != 0:
                  self.send[self.output_status[i] * -1 - 1] += self.result[i]

      for i in range(self.number_of_outputs):
          self.common_output(i)

      for i in range(length_of_input):
          self.input[i] = self.common_input(i)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void common_output(self, int a) nogil:

        if a == 0:
            if self.x +  round(sqroot(self.send[0])) < self.enviroment.shape[1] and self.x +  round(sqroot(self.send[0])) >= 0 :
                self.x += round(sqroot(self.send[0]))

        elif a == 1:
            if self.x - round(sqroot(self.send[1])) < self.enviroment.shape[1] and self.x - round(sqroot(self.send[1])) >= 0:
                self.x -= round(sqroot(self.send[1]))

        elif a == 2:
            if self.y + round(sqroot(self.send[2])) < self.enviroment.shape[0] and self.y + round(sqroot(self.send[2])) >= 0:
                self.y += round(sqroot(self.send[2]))

        elif a == 3:
            if self.y - round(sqroot(self.send[3])) < self.enviroment.shape[0] and self.y - round(sqroot(self.send[3])) >= 0 :
                self.y -= round(sqroot(self.send[3]))


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double common_input(self, int a) nogil:
        if a == 0:
                if self.x + 1 < self.enviroment.shape[1] and self.x + 1 >= 0 :
                    return self.enviroment[self.y][self.x + 1][0]
                else:
                    return 0.
        elif a == 1:
                if self.x - 1 < self.enviroment.shape[1] and self.x -1 >= 0:
                    return self.enviroment[self.y][self.x - 1][0]
                else:
                    return 0.
        elif a == 2:

                if self.y + 1 < self.enviroment.shape[0] and self.y + 1 >= 0:
                    return self.enviroment[self.y + 1][self.x][0]
                else:
                    return 0.
        elif a == 3:
                if self.y - 1 < self.enviroment.shape[0] and self.y - 1 >= 0 :
                    return self.enviroment[self.y - 1][self.x][0]
                else:
                    return 0.


    cdef void output_1(self) nogil:
        if self.x +  round(sqroot(self.send[0])) < self.enviroment.shape[1] and self.x +  round(sqroot(self.send[0])) >= 0 :
            self.x += round(sqroot(self.send[0]))


    cdef void output_2(self) nogil:
        if self.x - round(sqroot(self.send[1])) < self.enviroment.shape[1] and self.x - round(sqroot(self.send[1])) >= 0:
            self.x -= round(sqroot(self.send[1]))


    cdef void output_3(self) nogil:
        if self.y + round(sqroot(self.send[2])) < self.enviroment.shape[0] and self.y + round(sqroot(self.send[2])) >= 0:
            self.y += round(sqroot(self.send[2]))


    cdef void output_4(self) nogil:
        if self.y - round(sqroot(self.send[3])) < self.enviroment.shape[0] and self.y - round(sqroot(self.send[3])) >= 0 :
            self.y -= round(sqroot(self.send[3]))


    cdef double input_1(self) nogil:
        if self.x + 1 < self.enviroment.shape[1] and self.x + 1 >= 0 :
            return self.enviroment[self.y][self.x + 1][0]
        else:
            return 0.


    cdef double input_2(self) nogil:
        if self.x - 1 < self.enviroment.shape[1] and self.x -1 >= 0:
            return self.enviroment[self.y][self.x - 1][0]
        else:
            return 0.


    cdef double input_3(self) nogil:
        if self.y + 1 < self.enviroment.shape[0] and self.y + 1 >= 0:
            return self.enviroment[self.y + 1][self.x][0]
        else:
            return 0.


    cdef double input_4(self) nogil:
        if self.y - 1 >= 0 and self.y - 1 < self.enviroment.shape[0]:
            return self.enviroment[self.y - 1][self.x][0]
        else:
            return 0.


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double current_concentration(self, int substance) nogil:
        return  self.enviroment[self.y][self.x][substance]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void apply_activation_functions(self, int i) nogil:

        cdef int a_f = self.network.neurons_genes[i][3]

        if a_f == 0:
            self.result[i] = linear(self.result[i])
        if a_f == 1:
            self.result[i] = logistic(self.result[i])
        if a_f == 2:
            self.result[i] = tanh(self.result[i])
        if a_f == 3:
            self.result[i] = ReLU(self.result[i])


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void normalize_efficiency_storage(self, int length, double limit, double p):
      cdef Py_ssize_t i
      if self.max_normalize != 0:

          for i in prange(length, nogil = True):
              self.efficiency_storage[i] = self.efficiency_storage[i] / self.max_normalize
      else:
          for i in prange(length, nogil = True):
              self.efficiency_storage[i] = (rand() / (RAND_MAX * 1.0) < p) * limit
