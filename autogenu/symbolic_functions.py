import sympy


def diff_scalar_func(scalar_func, var):
    """ Calculate partial derivative of a scalar-valued function. 

        Args:
            scalar_func: A symbolic scalar function.
            var: A symbolic scalar or a symbolic vector.

        Returns: 
            Partial derivative of scalar_func with respect to var. If var is a 
            vector, Returns Jacobian.
    """
    return [sympy.diff(scalar_func, var[i]) for i in range(len(var))]


def diff_vector_func(vector_func, var):
    """ Calculate partial derivative of a vector-valued function.

        Args:
            scalar_func: A symbolic scalar function.
            var: A symbolic scalar or a symbolic vector.

        Returns: 
            Partial derivative of scalar_func with respect to var. If var is a 
            vector, Returns Jacobian.
    """
    return [[sympy.diff(vector_func[i], var[j]) for j in range(len(var))] for i in range (len(vector_func))]


def transpose(mat):
    return [[mat[row][col] for row in range(len(mat))] for col in range(len(mat[0]))]


def vector_dot_vector(vec1, vec2):
    assert len(vec1) == len(vec2)
    return sum(vec1[i] * vec2[i] for i in range(len(vec1)))


def matrix_dot_vector(mat, vec):
    assert len(mat[0]) == len(vec)
    return [sum(mat[i][j] * vec[j] for j in range(len(mat[0]))) for i in range(len(mat))]


def matrix_dot_matrix(mat1, mat2):
    assert len(mat1[0]) == len(mat2)
    return [[sum(mat1[i][k] * mat2[k][j] for k in range(len(mat1[0]))) for j in range(len(mat2[0]))] for i in range(len(mat1))]

def matrix_to_array(mat):
    array = []
    for vec in mat:
        array += vec
    return array


def simplify(func):
    """ Simplifies a scalar-valued or vector-valued function.

        Args:
            func: A symbolic functions.
    """
    if type(func) == list:
        for i in range(len(func)):
            func[i] = sympy.simplify(sympy.nsimplify(func[i]))
    else:
        func = sympy.simplify(sympy.nsimplify(func))