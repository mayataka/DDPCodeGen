 
#ifndef OCP_MODEL_H
#define OCP_MODEL_H

#define _USE_MATH_DEFINES

#include <cmath>


namespace cddp {

class OCPModel {
private:
  static constexpr int dimx_ = 4;
  static constexpr int dimu_ = 1;

  static constexpr double m1 = 0.2;
  static constexpr double m2 = 0.7;
  static constexpr double l1 = 0.3;
  static constexpr double l2 = 0.3;
  static constexpr double d1 = 0.15;
  static constexpr double d2 = 0.257;
  static constexpr double J1 = 0.006;
  static constexpr double J2 = 0.051;
  static constexpr double g = 9.80665;

  double q[4] = {1, 1, 0.1, 0.1};
  double q_terminal[4] = {1, 1, 0.1, 0.1};
  double x_ref[4] = {M_PI, 0, 0, 0};
  double r[1] = {0.1};


public:

  // Computes the state equation f(t, x, u).
  // t : time parameter
  // x : state vector
  // u : control input vector
  // dx : the value of f(t, x, u)
  void stateEquation(const double t, const double dtau, const double* x, 
                     const double* u, double* dx) const;

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


#endif // OCP_MODEL_H
