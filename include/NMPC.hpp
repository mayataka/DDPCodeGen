#ifndef CDDP_NMPC_HPP_
#define CDDP_NMPC_HPP_

#include <cmath>
#include <assert.h>

#include "DDP.hpp"
#include "memory_manager.hpp"


namespace cddp {

template <int dimx, int dimu>
class NMPC {
public:
  NMPC(const double T_f, const double alpha, const int N, 
       const double dt) 
    : ddp_(N), 
      T_f_(T_f), 
      alpha_(alpha),
      horizon_eps_(1.0e-06),
      dt_(dt), 
      x1_(memorymanager::NewVector(dimx)),
      dimx_(dimx),
      dimu_(dimu) {
    assert(N > 0);
    assert(T_f_ > 0);
    assert(alpha_ > 0);
    assert(dt_ > 0);
  }

  ~NMPC() {
    memorymanager::DeleteVector(x1_);
  }

  void updateSolution(const double t, const double* x) {
    ddp_.setHorizonLength(getHorizonLength(t));
    ddp_.rolloutState(t, x);
    ddp_.computeBackwardPass(t, x);
    ddp_.computeForwardPass(t, x);
  }

  void updateSolutionWithContinuation(const double t, const double* x, 
                                      const double sampling_period) {
    ddp_.setHorizonLength(getHorizonLength(t));
    ddp_.predictState(t, x, dt_, x1_);
    ddp_.rolloutState(t, x);
    ddp_.computeBackwardPass(t, x);
    ddp_.computeForwardPass(t, x);
  }


  void getControlInput(double* u) const {
    ddp_.getInitialControlInput(u);
  }

  void setControlInput(const double* u) {
    ddp_.setControlInput(u);
  }

  double getHorizonLength(const double t) const {
    return T_f_ * (1.0-std::exp(-alpha_*(t+horizon_eps_)));
  }

  // Returns the dimension of the state.
  int dim_x() const {
    return dimx_;
  }

  // Returns the dimension of the contorl input.
  int dim_u() const {
    return dimu_;
  }
  
private:
  DDP<dimx, dimu> ddp_;
  double T_f_, alpha_, horizon_eps_, dt_;
  double *x1_;
  const int dimx_, dimu_;

};
  
} // namespace cddp




#endif // CDDP_NMPC_HPP_