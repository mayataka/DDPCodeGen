 
#ifndef CDDP_OCP_MODEL_H
#define CDDP_OCP_MODEL_H

#define _USE_MATH_DEFINES

#include <cmath>


namespace cddp {

class OCPModel {
private:
  static constexpr int dimx_ = 4;
  static constexpr int dimu_ = 1;

  static constexpr double m_c = 2;
  static constexpr double m_p = 0.2;
  static constexpr double l = 0.5;
  static constexpr double g = 9.80665;
  static constexpr double u_min = -15;
  static constexpr double u_max = 15;
  static constexpr double u_eps = 0.001;

  double q[4] = {2.5, 10, 0.01, 0.01};
  double q_terminal[4] = {2.5, 10, 0.01, 0.01};
  double x_ref[4] = {0, M_PI, 0, 0};
  double r[1] = {1};


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
