from network cimport Network


cdef class Enviroment:

    '''
    This class creates enviroment from given maps (create_enviroment method) and
    runs given network in this enviroment.

    dynamics() runs one step.
    run() runs given number of steps using dynamics method.

    For each epoch ( i.e. iteration through all networks) must be called
    set_efficiency_storage() and create_enviroment().

    For each network run must be called set_network(), check_zero_efficiency().
    If the latter returns 'both input and output exist' then set_output_status(),
    set_input(), set_position() (usually with random position), set_number_of_outputs(),
    set_send(), set_result(). Then called run with number of iterations given.

    After all networks has run normalize_efficiency_storage() is to be called and
    efficiencies of network to be set this way (array[i].efficiency = Env.efficiency_storage[i]).

    Efficiency is calculated based on current concentration. Should several concentrations
    be calculated current_concentration(int) method has to be called several times
    with different int (dependend on order number of substance to be calculated, see below).

    Enviroment is 3-d array. 0-axis is y coordinate, 1-axis is x coordinate and
    2 - axis is different substances (for each substance a new map (image) must be
    passe to create_enviroment meethod).

    For each output methods take each output neuron, apply activation function
    and then apply sqroot() with round() (the latter rounds to the closer int)
    If it results in crossing the boarder then zero is returned.

    Input methods return concentration of substance in the place around (for each
    new place, e.g. diagonal, or substance new method must be created). Unless it's
    within boarder return zero.
    '''

    cdef public Network network
    cdef public double[:,:,:] enviroment
    cdef public double [:,:] matrix
    cdef public int [:] output_status
    cdef public double[:] input
    cdef public double[:] send
    cdef public double[:] result
    cdef public int x, y
    cdef public int number_of_outputs
    cdef public double efficiency
    cdef public double[:] efficiency_storage
    cdef public double max_normalize

    cdef void run(self, int number_of_iterations, int order_number) nogil

    cdef str check_zero_efficiency(self, int order_number)

    cdef void set_network(self, Network network)

    cdef void set_output_status(self) nogil

    cdef void set_input(self)

    cdef void set_number_of_outputs(self, int number_of_outputs) nogil

    cdef void set_position(self, int x, int y) nogil

    cdef void set_send(self)

    cdef void set_result(self)

    cdef void set_efficiency_storage(self, int length)

    cdef void dynamics(self) nogil

    cdef void common_output(self, int a) nogil

    cdef double common_input(self, int a) nogil

    cdef void output_1(self) nogil

    cdef void output_2(self) nogil

    cdef void output_3(self) nogil

    cdef void output_4(self) nogil

    cdef double input_1(self) nogil

    cdef double input_2(self) nogil

    cdef double input_3(self) nogil

    cdef double input_4(self) nogil

    cdef double current_concentration(self, int substance) nogil

    cdef void apply_activation_functions(self, int i) nogil

    cdef void normalize_efficiency_storage(self, int length, double limit, double p)
