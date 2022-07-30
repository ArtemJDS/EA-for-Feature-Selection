cdef extern from "math.h":
    double exp(double x) nogil

cdef inline double linear(double x) nogil:
    return x

cdef inline double logistic(double x) nogil:
    return 1/(1+exp(x))

cdef inline double tanh(double x) nogil:
    return 2/(1 + exp(-2*x)) - 1

cdef inline double ReLU(double x) nogil:
    return 0 * (x <= 0) + x * (x > 0)
