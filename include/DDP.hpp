#ifndef DDP_H
#define DDP_H

#include <iostream>
#include <assert.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include "ocp_model.hpp"
#include "memory_manager.hpp"


namespace cddp {

template<int dimx, int dimu>
class DDP {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DDP(const unsigned int N) 
    : ocp_model_(),
      dimx_(dimx), 
      dimu_(dimu), 
      N_(N), 
      horizon_length_(0), 
      dtau_(0),
      u_(memorymanager::NewMatrix(N, dimu)),
      x_(memorymanager::NewMatrix(N, dimx)),
      x1_(memorymanager::NewVector(dimx)),
      Qx_(memorymanager::NewMatrix(N, dimx)),
      Qu_(memorymanager::NewMatrix(N, dimu)),
      Qxx_(memorymanager::NewMatrix(N, dimx*dimx)),
      Qux_(memorymanager::NewMatrix(N, dimu*dimx)),
      Quu_(memorymanager::NewMatrix(N, dimu*dimu)),
      Vx_(memorymanager::NewMatrix(N, dimx)),
      Vxx_(memorymanager::NewMatrix(N, dimx*dimx)),
      Kx_(memorymanager::NewMatrix(N, dimu*dimx)),
      k_(memorymanager::NewMatrix(N, dimu)),
      Q_inv_() {
    assert(N > 0);
    assert(dimx == ocp_model_.dimx());
    assert(dimu == ocp_model_.dimu());
    Q_inv_.setZero();
  }

  ~DDP() {
    memorymanager::DeleteMatrix(u_);
    memorymanager::DeleteMatrix(x_);
    memorymanager::DeleteVector(x1_);
    memorymanager::DeleteMatrix(Qx_);
    memorymanager::DeleteMatrix(Qu_);
    memorymanager::DeleteMatrix(Qxx_);
    memorymanager::DeleteMatrix(Qux_);
    memorymanager::DeleteMatrix(Quu_);
    memorymanager::DeleteMatrix(Vx_);
    memorymanager::DeleteMatrix(Vxx_);
    memorymanager::DeleteMatrix(Kx_);
    memorymanager::DeleteMatrix(k_);
  }


  void rolloutState(const double t, const double* x) {
    ocp_model_.stateEquation(t, dtau_, x, u_[0], x_[0]);
    for (int i=1; i<N_-1; ++i) {
      ocp_model_.stateEquation(t+i*dtau_, dtau_, x_[i-1], u_[i], x_[i]);
    }
  }

  void computeBackwardPass(const double t, const double* x) {
    Eigen::Map<Eigen::VectorXd>(Qx_[0], N_*dimx).setZero();
    Eigen::Map<Eigen::VectorXd>(Qu_[0], N_*dimu).setZero();
    Eigen::Map<Eigen::VectorXd>(Qxx_[0], N_*dimx*dimx).setZero();
    Eigen::Map<Eigen::VectorXd>(Qux_[0], N_*dimx*dimu).setZero();
    Eigen::Map<Eigen::VectorXd>(Quu_[0], N_*dimu*dimu).setZero();
    ocp_model_.terminalCostDerivatives(t+horizon_length_, x_[N_-1], Vx_[N_-1],  
                                       Vxx_[N_-1]);
    for (int i=N_-1; i>0; --i) {
      ocp_model_.stageCostDerivatives(t+i*dtau_, dtau_, x_[i-1], u_[i], Qx_[i], 
                                      Qu_[i], Qxx_[i], Qux_[i], Quu_[i]);
      ocp_model_.dynamicsDerivatives(t+i*dtau_, dtau_, x_[i-1], u_[i], Vx_[i], 
                                     Vxx_[i], Qx_[i], Qu_[i], Qxx_[i], Qux_[i], 
                                     Quu_[i], Qxx_[i], Qux_[i], Quu_[i]);
      Q_inv_ = Eigen::Map<Eigen::Matrix<double, dimu, dimu>>(Quu_[i]).inverse();
      Eigen::Map<Eigen::MatrixXd>(Kx_[i], dimu, dimx)
          = - Q_inv_ * Eigen::Map<Eigen::Matrix<double, dimu, dimx>>(Qux_[i]);
      Eigen::Map<Eigen::VectorXd>(k_[i], dimu)
          = - Q_inv_ * Eigen::Map<Eigen::Matrix<double, dimu, 1>>(Qu_[i]);
      Eigen::Map<Eigen::MatrixXd>(Vxx_[i-1], dimx, dimx) 
          = Eigen::Map<Eigen::Matrix<double, dimx, dimx>>(Qxx_[i])
            + Eigen::Map<Eigen::Matrix<double, dimu, dimx>>(Kx_[i]).transpose()
              * Eigen::Map<Eigen::Matrix<double, dimu, dimu>>(Quu_[i])
              * Eigen::Map<Eigen::Matrix<double, dimu, dimx>>(Kx_[i])
            + Eigen::Map<Eigen::Matrix<double, dimu, dimx>>(Qux_[i]).transpose()
              * Eigen::Map<Eigen::Matrix<double, dimu, dimx>>(Kx_[i])
            + Eigen::Map<Eigen::Matrix<double, dimu, dimx>>(Kx_[i]).transpose()
              * Eigen::Map<Eigen::Matrix<double, dimu, dimx>>(Qux_[i]);
      Eigen::Map<Eigen::VectorXd>(Vx_[i-1], dimx) 
          = Eigen::Map<Eigen::Matrix<double, dimx, 1>>(Qx_[i])
            + Eigen::Map<Eigen::Matrix<double, dimu, dimx>>(Kx_[i]).transpose()
              * Eigen::Map<Eigen::Matrix<double, dimu, dimu>>(Quu_[i])
              * Eigen::Map<Eigen::Matrix<double, dimu, 1>>(k_[i])
            + Eigen::Map<Eigen::Matrix<double, dimu, dimx>>(Qux_[i]).transpose()
              * Eigen::Map<Eigen::Matrix<double, dimu, 1>>(k_[i])
            + Eigen::Map<Eigen::Matrix<double, dimu, dimx>>(Kx_[i]).transpose()
              * Eigen::Map<Eigen::Matrix<double, dimu, 1>>(Qu_[i]);
    }
    ocp_model_.stageCostDerivatives(t, dtau_, x, u_[0], Qx_[0], 
                                    Qu_[0], Qxx_[0], Qux_[0], Quu_[0]);
    ocp_model_.dynamicsDerivatives(t, dtau_, x, u_[0], Vx_[0], 
                                    Vxx_[0], Qx_[0], Qu_[0], Qxx_[0], Qux_[0], 
                                  Quu_[0], Qxx_[0], Qux_[0], Quu_[0]);
    Q_inv_ = Eigen::Map<Eigen::Matrix<double, dimu, dimu>>(Quu_[0]).inverse();
    Eigen::Map<Eigen::VectorXd>(k_[0], dimu)
        = - Q_inv_ * Eigen::Map<Eigen::Matrix<double, dimu, 1>>(Qu_[0]);
  }

  void computeForwardPass(const double t, const double* x) {
    Eigen::Map<Eigen::Matrix<double, dimu, 1>>(u_[0]) 
        += Eigen::Map<Eigen::Matrix<double, dimu, 1>>(k_[0]);
    ocp_model_.stateEquation(t, dtau_, x, u_[0], x1_);
    for (int i=1; i<N_; ++i) {
      Eigen::Map<Eigen::VectorXd>(u_[i], dimu) 
          += Eigen::Map<Eigen::Matrix<double, dimu, 1>>(k_[i]) 
            + Eigen::Map<Eigen::Matrix<double, dimu, dimx>>(Kx_[i])
              * (Eigen::Map<Eigen::Matrix<double, dimx, 1>>(x1_)
                 -Eigen::Map<Eigen::Matrix<double, dimx, 1>>(x_[i-1]));
      Eigen::Map<Eigen::VectorXd>(x_[i-1], dimx_) 
          = Eigen::Map<Eigen::Matrix<double, dimx, 1>>(x1_, dimx_);
      ocp_model_.stateEquation(t+i*dtau_, dtau_, x_[i-1], u_[i], x1_);
    }
  }

  void setControlInput(const double* u) {
    for (int i=0; i<N_; ++i) {
      Eigen::Map<Eigen::Matrix<double, dimu, 1>>(u_[i]) 
          = Eigen::Map<const Eigen::Matrix<double, dimu, 1>>(u);
    }
  }

  void printSolution() {
    for (int i=0; i<N_; ++i) {
      std::cout << Eigen::Map<Eigen::Matrix<double, dimu, 1>>(u_[i]).transpose() << std::endl;
    }
  }

  void setHorizonLength(const double horizon_length) {
    horizon_length_ = horizon_length;
    dtau_ = horizon_length / N_;
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
  OCPModel ocp_model_;
  const unsigned int dimx_, dimu_, N_;
  double horizon_length_, dtau_;
  double **u_, **x_, *x1_;
  double **Qx_, **Qu_, **Qxx_, **Qux_, **Quu_;
  double **Vx_, **Vxx_;
  double **Kx_, **k_;
  Eigen::Matrix<double, dimu, dimu> Q_inv_;

};

} // namespace cddp


#endif // DDP_H