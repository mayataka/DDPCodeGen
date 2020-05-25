 
#ifndef CDDP_OCP_MODEL_H
#define CDDP_OCP_MODEL_H

#define _USE_MATH_DEFINES

#include <cmath>


namespace cddp {

class OCPModel {
private:
  static constexpr int dimx_ = 12;
  static constexpr int dimu_ = 6;

  static constexpr double m = 1.44;
  static constexpr double l = 0.23;
  static constexpr double k = 1.6e-09;
  static constexpr double Ixx = 0.0348;
  static constexpr double Iyy = 0.0459;
  static constexpr double Izz = 0.0977;
  static constexpr double gamma = 0.01;
  static constexpr double g = 9.80665;
  static constexpr double z_ref = 5;
  static constexpr double u_min = 0.144;
  static constexpr double u_max = 6;
  static constexpr double epsilon = 0.01;

  double q[12] = {1, 1, 1, 0.01, 0.01, 0, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001};
  double q_terminal[12] = {1, 1, 1, 0.01, 0.01, 0, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001};
  double r[6] = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01};


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
