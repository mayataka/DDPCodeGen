import linecache
import subprocess
import platform
from enum import Enum, auto

import sympy

from autogenu import symbolic_functions as symfunc


class SolverType(Enum):
    ContinuationGMRES = auto()
    MultipleShootingCGMRES = auto()
    MSCGMRESWithInputSaturation = auto()

class AutoGenU(object):
    """ Automatic C++ code generator for the C/GMRES methods. 

        Args: 
            model_name: The name of the NMPC model. The directory having this 
                name is made and C++ source files are generated in the 
                directory.
            dimx: The dimension of the state of the NMPC model. 
            dimu: The dimension of the control input of the NMPC model. 
    """
    def __init__(self, model_name, dimx, dimu):
        assert isinstance(model_name, str), 'The frst argument must be strings!'
        assert dimx > 0, 'The second argument must be positive integer!'
        assert dimu > 0, 'The third argument must be positive integer!'
        self.__model_name = model_name
        self.__dimx = dimx        
        self.__dimu = dimu        
        self.__scalar_vars = []
        self.__array_vars = []
        self.__is_function_set = False
        self.__is_solver_paramters_set = False
        self.__is_initialization_set = False
        self.__is_simulation_set = False

    def define_t(self):
        """ Returns symbolic scalar variable 't'.
        """
        return sympy.Symbol('t')

    def define_x(self):
        """ Returns symbolic vector variable 'x' whose size is dimx.
        """
        return sympy.symbols('x[0:%d]' %(self.__dimx))

    def define_u(self):
        """ Returns symbolic vector variable 'u' whose size is dimu.
        """
        return sympy.symbols('u[0:%d]' %(self.__dimu))

    def define_scalar_var(self, scalar_var_name):
        """ Returns symbolic variable whose name is scalar_var_name. The name of 
            the variable is memorized.

            Args:
                scalar_var_name: Name of the scalar variable.
        """
        assert isinstance(scalar_var_name, str), 'The input must be strings!'
        scalar_var = sympy.Symbol(scalar_var_name)
        self.__scalar_vars.append([scalar_var, scalar_var_name, 0])
        return scalar_var

    def define_scalar_vars(self, *scalar_var_name_list):
        """ Returns symbolic variables whose names are given by 
            scalar_var_name_list. The names of the variables are memorized.

            Args:
                scalar_var_name_list: Names of the scalar variables.
        """
        scalar_vars = []
        for scalar_var_name in scalar_var_name_list:
            assert isinstance(scalar_var_name, str), 'The input must be list of strings!'
            scalar_var = sympy.Symbol(scalar_var_name)
            self.__scalar_vars.append([scalar_var, scalar_var_name, 0])
            scalar_vars.append(scalar_var)
        return scalar_vars

    def define_array_var(self, array_var_name, dim):
        """ Returns symbolic vector variable whose names is array_var_name and 
            whose dimension is dim. The names of the variable is memorized.

            Args:
                array_var_name: Name of the array variable.
                dim: Dimension of the array variable.
        """
        assert isinstance(array_var_name, str), 'The first argument must be strings!'
        assert dim > 0, 'The second argument must be positive integer!'
        array_var = sympy.symbols(array_var_name+'[0:%d]' %(dim))
        self.__array_vars.append([array_var, array_var_name, []])
        return array_var

    def set_scalar_var(self, scalar_var_name, scalar_value):
        """ Set the value of the scalar variable you defied. 

            Args:
                scalar_var_name: Name of the scalar variable.
                scalar_value: Value of the scalar variable.
        """
        assert isinstance(scalar_var_name, str), 'The first argument must be strings!'
        for defined_scalar_var in self.__scalar_vars:
            if scalar_var_name[0] == defined_scalar_var[1]:
                defined_scalar_var[2] = scalar_value

    def set_scalar_vars(self, *scalar_var_name_and_value_list):
        """ Set the values of the scalar variables you defied. 

            Args:
                scalar_var_name_and_value_lis: A list composed of the name of 
                the scalar variable and value of the scalar variable.
        """
        for var_name_and_value in scalar_var_name_and_value_list:
            for defined_scalar_var in self.__scalar_vars:
                if var_name_and_value[0] == defined_scalar_var[1]:
                    defined_scalar_var[2] = var_name_and_value[1]
    
    def set_array_var(self, var_name, values):
        """ Set the value of the array variable you defied. 

            Args:
                var_name: Name of the arrray variable.
                values: Values of the arry variable. The size must be the 
                    dimension of the array variable.
        """
        assert isinstance(var_name, str), 'The first argument must be strings!'
        for defined_array_var in self.__array_vars:
            if var_name == defined_array_var[1]:
                if len(defined_array_var[0]) == len(values):
                    defined_array_var[2] = values

    def set_functions(self, f, l, phi):
        """ Sets functions that defines the optimal control problem.

            Args: 
                f: The state equation. The dimension must be dimx.
                L: The stage cost.
                phi: The terminal cost.
        """
        assert len(f) > 0 
        assert len(f) == self.__dimx, "Dimension of f must be dimx!"
        x = sympy.symbols('x[0:%d]' %(self.__dimx))
        u = sympy.symbols('u[0:%d]' %(self.__dimu))
        dtau = sympy.Symbol('dtau')
        self.__f = f
        F = [x[i] + dtau * f[i] for i in range(self.__dimx)]
        self.__F = F
        l = dtau * l
        self.__l = l
        self.__lx = symfunc.diff_scalar_func(l, x)
        self.__lu = symfunc.diff_scalar_func(l, u)
        self.__lxx = symfunc.diff_vector_func(self.__lx, x)
        self.__lux = symfunc.diff_vector_func(self.__lu, x)
        self.__luu = symfunc.diff_vector_func(self.__lu, u)
        self.__phi = phi
        self.__phix = symfunc.diff_scalar_func(phi, x)
        self.__phixx = symfunc.diff_vector_func(self.__phix, x)
        Vx = sympy.symbols('Vx[0:%d]' %(self.__dimx))
        Fx = symfunc.diff_vector_func(F, x)
        Fu = symfunc.diff_vector_func(F, u)
        Fx_trans = symfunc.transpose(Fx)
        Fu_trans = symfunc.transpose(Fu)
        self.__FxVx = symfunc.matrix_dot_vector(Fx_trans, Vx)
        self.__FuVx = symfunc.matrix_dot_vector(Fu_trans, Vx)
        Vxx_array = sympy.symbols('Vxx[0:%d]' %(self.__dimx**2))
        Vxx = [[Vxx_array[self.__dimx*i+j] for j in range(self.__dimx)] for i in range(self.__dimx)]
        self.__FxVxxFx = symfunc.matrix_dot_matrix(Fx_trans, symfunc.matrix_dot_matrix(Vxx, Fx))
        self.__FuVxxFx = symfunc.matrix_dot_matrix(Fu_trans, symfunc.matrix_dot_matrix(Vxx, Fx))
        self.__FuVxxFu = symfunc.matrix_dot_matrix(Fu_trans, symfunc.matrix_dot_matrix(Vxx, Fu))
        VxF = sum(F[i] * Vx[i] for i in range(self.__dimx))
        VxFx = symfunc.diff_scalar_func(VxF, x)
        VxFu = symfunc.diff_scalar_func(VxF, u)
        self.__VxFxx = symfunc.diff_vector_func(VxFx, x)
        self.__VxFux = symfunc.diff_vector_func(VxFu, x)
        self.__VxFuu = symfunc.diff_vector_func(VxFu, u)
        assert len(self.__FxVxxFx) == len(self.__VxFxx)
        assert len(self.__FxVxxFx[0]) == len(self.__VxFxx[0])
        assert len(self.__FuVxxFx) == len(self.__VxFux)
        assert len(self.__FuVxxFx[0]) == len(self.__VxFux[0])
        assert len(self.__FuVxxFu) == len(self.__VxFuu)
        assert len(self.__FuVxxFu[0]) == len(self.__VxFuu[0])
        self.__is_function_set = True

    def set_solver_parameters(
            self, T_f, alpha, N
        ):
        """ Sets parameters of the NMPC solvers based on the C/GMRES method. 

            Args: 
                T_f, alpha: Parameter about the length of the horizon of NMPC.
                    The length of the horzion at time t is given by 
                    T_f * (1-exp(-alpha*t)).
                N: The number of the grid for the discretization
                    of the horizon of NMPC.
                finite_difference_increment: The small positive value for 
                    finitei difference approximation used in the FD-GMRES. 
                zeta: A stabilization parameter of the C/GMRES method. It may 
                    work well if you set as zeta=1/sampling_period.
                kmax: Maximam number of the iteration of the Krylov 
                    subspace method for the linear problem. 
        """
        assert T_f > 0
        assert alpha > 0
        assert N > 0
        self.__T_f = T_f
        self.__alpha = alpha
        self.__N = N
        self.__is_solver_paramters_set = True

    def set_solution_initial_guess(self, solution_initial_guess):
        """ Set parameters for the initialization of the C/GMRES solvers. 

            Args: 
                solution_initial_guess: The initial guess of the solution of the 
                    initialization. 
        """
        assert len(solution_initial_guess) == self.__dimu
        self.__solution_initial_guess = solution_initial_guess 
        self.__is_initialization_set = True

    def set_simulation_parameters(
            self, initial_time, initial_state, simulation_time, sampling_time
        ):
        """ Set parameters for numerical simulation. 

            Args: 
                initial_time: The time parameter at the beginning of the 
                    simulation. 
                initial_state: The state of the system at the beginning of the 
                    simulation. 
                simulation_time: The length of the numerical simulation. 
                sampling_time: The sampling period of the numerical simulation. 
        """
        assert len(initial_state) == self.__dimx, "The dimension of initial_state must be dimx!"
        assert simulation_time > 0
        assert sampling_time > 0
        self.__initial_time = initial_time 
        self.__initial_state = initial_state
        self.__simulation_time = simulation_time 
        self.__sampling_time = sampling_time
        self.__is_simulation_set = True

    def generate_source_files(self, use_simplification=False, use_cse=False):
        """ Generates the C++ source file in which the equations to solve the 
            optimal control problem are described. Before call this method, 
            set_functions() must be called.

            Args: 
                use_simplification: The flag for simplification. If True, the 
                    Symbolic functions are simplified. Default is False.
                use_cse: The flag for common subexpression elimination. If True, 
                    common subexpressions are eliminated. Default is False.
        """
        assert self.__is_function_set, "Symbolic functions are not set!. Before call this method, call set_functions()"
        self.__make_model_dir()
        if use_simplification:
            symfunc.simplify(self.__f)
            symfunc.simplify(self.__F)
            symfunc.simplify(self.__FxVx)
            symfunc.simplify(self.__FuVx)
            symfunc.simplify(self.__FxVxxFx)
            symfunc.simplify(self.__FuVxxFx)
            symfunc.simplify(self.__FuVxxFu)
            symfunc.simplify(self.__VxFxx)
            symfunc.simplify(self.__VxFux)
            symfunc.simplify(self.__VxFuu)
            symfunc.simplify(self.__l)
            symfunc.simplify(self.__lx)
            symfunc.simplify(self.__lu)
            symfunc.simplify(self.__lxx)
            symfunc.simplify(self.__lux)
            symfunc.simplify(self.__luu)
            symfunc.simplify(self.__phi)
            symfunc.simplify(self.__phix)
            symfunc.simplify(self.__phixx)
        f_model_h = open('models/'+str(self.__model_name)+'/ocp_model.hpp', 'w')
        f_model_h.writelines([
""" 
#ifndef CDDP_OCP_MODEL_H
#define CDDP_OCP_MODEL_H

#define _USE_MATH_DEFINES

#include <cmath>


namespace cddp {

class OCPModel {
private:
"""
        ])
        f_model_h.write(
            '  static constexpr int dimx_ = '+str(self.__dimx)+';\n'
        )
        f_model_h.write(
            '  static constexpr int dimu_ = '
            +str(self.__dimu)+';\n'
        )
        f_model_h.write('\n')
        f_model_h.writelines([
            '  static constexpr double '+scalar_var[1]+' = '
            +str(scalar_var[2])+';\n' for scalar_var in self.__scalar_vars
        ])
        f_model_h.write('\n')
        for array_var in self.__array_vars:
            f_model_h.write(
                '  double '+array_var[1]+'['+str(len(array_var[0]))+']'+' = {'
            )
            for i in range(len(array_var[0])-1):
                f_model_h.write(str(array_var[2][i])+', ')
            f_model_h.write(str(array_var[2][len(array_var[0])-1])+'};\n')
        f_model_h.writelines([
"""

public:

  // Computes the dynamics f(t, x, u).
  // t : time parameter
  // x : state vector
  // u : control input vector
  // dx : the value of f(t, x, u)
  void dynamics(const double t, const double dtau, const double* x, 
                const double* u, double* dx) const;

  // Computes the state equation F(t, x, u).
  // t : time parameter
  // x : state vector
  // u : control input vector
  // dx : the value of f(t, x, u)
  void stateEquation(const double t, const double dtau, const double* x, 
                     const double* u, double* F) const;

  // Computes the partial derivative of terminal cost with respect to state, 
  // i.e., dphi/dx(t, x).
  // t    : time parameter
  // x    : state vector
  // u    : control input vector
  void stageCostDerivatives(const double t, const double dtau, const double* x, 
                            const double* u, double* lx, double* lu, 
                            double* lxx, double* lux, double* luu) const;

  // Computes the partial derivative of terminal cost with respect to state, 
  // i.e., dphi/dx(t, x).
  // t    : time parameter
  // x    : state vector
  // phix : the value of dphi/dx(t, x)
  void terminalCostDerivatives(const double t, const double* x, double* phix, 
                               double* phixx) const;

  // Computes the partial derivative of terminal cost with respect to state, 
  // i.e., dphi/dx(t, x).
  // t    : time parameter
  // x    : state vector
  // u    : control input vector
  void dynamicsDerivatives(const double t, const double dtau, const double* x, 
                           const double* u, const double* Vx, const double* Vxx, 
                           double* fxVx, double* fuVx, double* fxVxxfx, 
                           double* fuVxxfx, double* fuVxxfu, double* Vxfxx, 
                           double* Vxfux, double* Vxfuu) const;

  // Returns the dimension of the state.
  int dimx() const;

  // Returns the dimension of the contorl input.
  int dimu() const;
};

} // namespace cddp


#endif // CDDP_OCP_MODEL_H
""" 
        ])
        f_model_h.close()
        f_model_c = open('models/'+self.__model_name+'/ocp_model.cpp', 'w')
        f_model_c.writelines([
""" 
#include "ocp_model.hpp"


namespace cddp {

void OCPModel::dynamics(const double t, const double dtau, const double* x, 
                        const double* u, double* dx) const {
""" 
        ])
        self.__write_function(f_model_c, self.__f, 'dx', "=", use_cse)
        f_model_c.writelines([
""" 
}

void OCPModel::stateEquation(const double t, const double dtau, const double* x, 
                             const double* u, double* F) const {
""" 
        ])
        self.__write_function(f_model_c, self.__F, 'F', "=", use_cse)
        f_model_c.writelines([
""" 
}

void OCPModel::stageCostDerivatives(const double t, const double dtau, 
                                    const double* x, const double* u, 
                                    double* lx, double* lu, double* lxx, 
                                    double* lux, double* luu) const {
"""
        ])
        self.__write_multiple_functions(
            f_model_c, use_cse, "+=", [self.__lx, 'lx'], [self.__lu, 'lu'], 
            [symfunc.matrix_to_array(self.__lxx), 'lxx'], 
            [symfunc.matrix_to_array(self.__lux), 'lux'], 
            [symfunc.matrix_to_array(self.__luu), 'luu']
        )
        f_model_c.writelines([
""" 
}


void OCPModel::terminalCostDerivatives(const double t, const double* x, 
                                       double* phix, double* phixx) const {
"""
        ])
        self.__write_multiple_functions(
            f_model_c, use_cse, "=", [self.__phix, 'phix'], 
            [symfunc.matrix_to_array(self.__phixx), 'phixx']
        )
        f_model_c.writelines([
""" 
}

void OCPModel::dynamicsDerivatives(const double t, const double dtau, 
                                   const double* x, const double* u, 
                                   const double* Vx, const double* Vxx, 
                                   double* fxVx, double* fuVx, double* fxVxxfx, 
                                   double* fuVxxfx, double* fuVxxfu, 
                                   double* Vxfxx, double* Vxfux, 
                                   double* Vxfuu) const {
"""
        ])
        self.__write_multiple_functions(
            f_model_c, use_cse, "+=", [self.__FxVx, 'fxVx'], 
            [self.__FuVx, 'fuVx'],
            [symfunc.matrix_to_array(self.__FxVxxFx), 'fxVxxfx'],
            [symfunc.matrix_to_array(self.__FuVxxFx), 'fuVxxfx'],
            [symfunc.matrix_to_array(self.__FuVxxFu), 'fuVxxfu'],
            [symfunc.matrix_to_array(self.__VxFxx), 'Vxfxx'],
            [symfunc.matrix_to_array(self.__VxFux), 'Vxfux'],
            [symfunc.matrix_to_array(self.__VxFuu), 'Vxfuu']
        )
        f_model_c.writelines([
""" 
}

int OCPModel::dimx() const {
  return dimx_;
}

int OCPModel::dimu() const {
  return dimu_;
}

} // namespace cgmres

""" 
        ])
        f_model_c.close() 


    def generate_main(self):
        """ Generates main.cpp that defines NMPC solver, set parameters for the 
            solver, and run numerical simulation. Befire call this method,
            set_solver_type(), set_solver_parameters(), 
            set_initialization_parameters(), and set_simulation_parameters(),
            must be called!
        """
        assert self.__is_solver_paramters_set, "Solver parameters are not set! Before call this method, call set_solver_parameters()"
        assert self.__is_initialization_set, "Initialization parameters are not set! Before call this method, call set_initialization_parameters()"
        assert self.__is_simulation_set, "Simulation parameters are not set! Before call this method, call set_simulation_parameters()"
        """ Makes a directory where the C++ source files are generated.
        """
        f_main = open('models/'+str(self.__model_name)+'/main.cpp', 'w')
        f_main.writelines([
"""
#include <string>

#include "NMPC.hpp"
#include "simulator.hpp"


int main() {
    """
        ])
        f_main.write('  const int N = '+str(self.__N)+';\n')
        f_main.write('  const double T_f = '+str(self.__T_f)+';\n')
        f_main.write('  const double alpha = '+str(self.__alpha)+';\n')
        f_main.write('  const double dt = '+str(0.001)+';\n')
        f_main.write('  cddp::NMPC<'+str(self.__dimx)+', '+str(self.__dimu)+'> nmpc(T_f, alpha, N, dt);\n')
        f_main.write('  double x0['+str(self.__dimx)+'] = {')
        for i in range(self.__dimx-1):
            f_main.write(str(self.__initial_state[i])+', ')
        f_main.write(str(self.__initial_state[self.__dimx-1])+'};\n')
        f_main.write('  double u0['+str(self.__dimu)+'] = {')
        for i in range(self.__dimu-1):
            f_main.write(str(self.__solution_initial_guess[i])+', ')
        f_main.write(str(self.__solution_initial_guess[self.__dimu-1])+'};\n')
        f_main.write('  nmpc.setControlInput(u0);\n')
        f_main.write('  const double simulation_time = '+str(self.__simulation_time)+';\n')
        f_main.write('  const double sampling_time = '+str(self.__sampling_time)+';\n')
        f_main.write('  const std::string save_dir = "simulation_result";\n')
        f_main.write('  const std::string save_file_name = "'+str(self.__model_name)+'";\n')
        f_main.write('  cddp::simulation<cddp::NMPC<'+str(self.__dimx)+', '+str(self.__dimu)+'>>(nmpc, x0, simulation_time, sampling_time, save_dir, save_file_name);\n')
        f_main.writelines([
"""
  return 0;
}
"""
        ])
        f_main.close()
    
    def generate_cmake(self):
        f_cmake= open('models/'+str(self.__model_name)+'/CMakeLists.txt', 'w')
        f_cmake.writelines([
"""
cmake_minimum_required(VERSION 3.1)
project(cddp CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3")

set(MODEL_DIR ${PROJECT_SOURCE_DIR})
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../include)
set(SRC_DIR ${PROJECT_SOURCE_DIR}/../../src)

add_library(
    ocp_model
    STATIC
    ${MODEL_DIR}/ocp_model.cpp
)

add_library(
    memory_manager
    STATIC
    ${SRC_DIR}/memory_manager.cpp
)
target_include_directories(
    memory_manager
    PRIVATE
    ${INCLUDE_DIR}
)

add_executable(
    a.out 
    ${MODEL_DIR}/main.cpp
    ${SRC_DIR}/simulation_data_saver.cpp
)
target_include_directories(
    a.out
    PRIVATE
    ${MODEL_DIR}
    ${INCLUDE_DIR}
)
target_link_libraries(
    a.out
    PRIVATE
    ocp_model
    memory_manager
)
target_compile_options(
    a.out
    PRIVATE
    -O3
)
"""
        ])
        f_cmake.close()

    def build(self, generator='Auto', remove_build_dir=False):
        """ Builds execute file to run numerical simulation. 

            Args: 
                generator: An optional variable for Windows user to choose the
                    generator. If 'MSYS', then 'MSYS Makefiles' is used. If 
                    'MinGW', then 'MinGW Makefiles' is used. The default value 
                    is 'Auto' and the generator is selected automatically. If 
                    sh.exe exists in your PATH, MSYS is choosed, and otherwise 
                    MinGW is used. If different value from 'MSYS' and 'MinGW', 
                    generator is selected automatically.
                remove_build_dir: If true, the existing build directory is 
                    removed and if False, the build directory is not removed.
                    Need to be set True is you change CMake configuration, e.g., 
                    if you change the generator. The default value is False.
        """
        subprocess.run(
            ['mkdir', 'build'], 
            cwd='models/'+self.__model_name, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        subprocess.run(
            ['mkdir', 'simulation_result'], 
            cwd='models/'+self.__model_name+'/build',
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        proc = subprocess.Popen(
            ['cmake', '..', '-DCMAKE_BUILD_TYPE=Release'], 
            cwd='models/'+self.__model_name+'/build', 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT
        )
        for line in iter(proc.stdout.readline, b''):
            print(line.rstrip().decode("utf8"))
        print('\n')
        proc = subprocess.Popen(
            ['cmake', '--build', '.'], 
            cwd='models/'+self.__model_name+'/build', 
            stdout = subprocess.PIPE, 
            stderr = subprocess.STDOUT
        )
        for line in iter(proc.stdout.readline, b''):
            print(line.rstrip().decode("utf8"))
        print('\n')

    def run_simulation(self):
        """ Run numerical simulation. Call after build() succeeded.
        """
        if platform.system() == 'Windows':
            subprocess.run(
                ['rmdir', '/q', '/s', 'simulation_result'], 
                cwd='models/'+self.__model_name, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                shell=True
            )
            proc = subprocess.Popen(
                ['main.exe'], 
                cwd='models/'+self.__model_name+'/build', 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                shell=True
            )
            for line in iter(proc.stdout.readline, b''):
                print(line.rstrip().decode("utf8"))
        else:
            subprocess.run(
                ['rm', '-rf', 'simulation_result'], 
                cwd='models/'+self.__model_name, 
                stdout = subprocess.PIPE, 
                stderr = subprocess.PIPE, 
                shell=True
            )
            proc = subprocess.Popen(
                ['./a.out'], 
                cwd='models/'+self.__model_name+'/build', 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT
            )
            for line in iter(proc.stdout.readline, b''):
                print(line.rstrip().decode("utf8"))


    def __write_function(
            self, writable_file, function, return_value_name, 
            return_operator="=", use_cse=True
        ):
        """ Write input symbolic function onto writable_file. The function's 
            return value name must be set. use_cse is optional.

            Args: 
                writable_file: A writable file, i.e., a file streaming that is 
                    already opened as writing mode.
                function: A symbolic function wrote onto the writable_file.
                return_value_name: The name of the return value.
                use_cse: If true, common subexpression elimination is used. If 
                    False, it is not used.
        """
        if use_cse:
            func_cse = sympy.cse(function)
            for i in range(len(func_cse[0])):
                cse_exp, cse_rhs = func_cse[0][i]
                writable_file.write(
                    '  double '+sympy.ccode(cse_exp)
                    +' = '+sympy.ccode(cse_rhs)+';\n'
                )
            for i in range(len(func_cse[1])):
                writable_file.write(
                    '  '+return_value_name+'[%d] '%i+return_operator+' '
                    +sympy.ccode(func_cse[1][i])+';\n'
                )
        else:
            writable_file.writelines(
                ['  '+return_value_name+'[%d] '%i+return_operator+' '
                +sympy.ccode(function[i])+';\n' for i in range(len(function))]
            )

    def __write_multiple_functions(
            self, writable_file, use_cse=True, return_operator="=",
            *functions_and_return_value_names
        ):
        """ Write input symbolic function onto writable_file. The function's 
            return value name must be set. use_cse is optional.

            Args: 
                writable_file: A writable file, i.e., a file streaming that is 
                    already opened as writing mode.
                function: A symbolic function wrote onto the writable_file.
                return_value_name: The name of the return value.
                use_cse: If true, common subexpression elimination is used. If 
                    False, it is not used.
        """
        if use_cse:
            united_func = []
            for func_and_return_value_name in functions_and_return_value_names:
                united_func.extend(func_and_return_value_name[0])
            united_func_cse = sympy.cse(united_func)
            for i in range(len(united_func_cse[0])):
                cse_exp, cse_rhs = united_func_cse[0][i]
                writable_file.write(
                    '  double '+sympy.ccode(cse_exp)
                    +' = '+sympy.ccode(cse_rhs)+';\n'
                )
            num_funcs = len(functions_and_return_value_names)
            total_func_dim = 0
            for i in range(num_funcs):
                dim_func = len(functions_and_return_value_names[i][0])
                return_value_name = functions_and_return_value_names[i][1]
                for j in range(dim_func):
                    if united_func_cse[1][total_func_dim+j] != 0:
                        writable_file.write(
                            '  '+return_value_name+'[%d] '%j+return_operator+' '
                            +sympy.ccode(united_func_cse[1][total_func_dim+j])+';\n'
                        )
                total_func_dim += dim_func
        else:
            for func_and_return_value_name in functions_and_return_value_names:
                func = func_and_return_value_name[0]
                return_value_name = func_and_return_value_name[1]
                writable_file.writelines(
                    ['  '+return_value_name+'[%d] '%i+return_operator+' '
                    +sympy.ccode(func[i])+';\n' for i in range(len(func))]
                )

    def __make_model_dir(self):
        """ Makes a directory where the C source files of OCP models are 
            generated.
        """
        if platform.system() == 'Windows':
            subprocess.run(
                ['mkdir', 'models'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                shell=True
            )
            subprocess.run(
                ['mkdir', self.__model_name], 
                cwd='models', 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                shell=True
            )
        else:
            subprocess.run(
                ['mkdir', 'models'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            subprocess.run(
                ['mkdir', self.__model_name], 
                cwd='models',
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )

    def __remove_build_dir(self):
        """ Removes a build directory. This function is mainly for Windows 
            users with MSYS.
        """
        if platform.system() == 'Windows':
            subprocess.run(
                ['rmdir', '/q', '/s', 'build'], 
                cwd='models/'+self.__model_name, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                shell=True
            )
        else:
            subprocess.run(
                ['rm', '-r', 'build'],
                cwd='models/'+self.__model_name, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )