#include "DDP.hpp"

#include "memory_manager.hpp"
#include <eigen3/Eigen/LU>


namespace cddp {



void DDP::rolloutState(const double t, const double* x) {
  ocp_model_.stateEquation(t, dtau_, u_[0], x, x_[0]);
  for (int i=1; i<N_; ++i) {
    ocp_model_.stateEquation(t+i*dtau_, dtau_, u_[i], x_[i-1], x_[i]);
  }
}


void DDP::computeBackwardPass(const double t, const double* x) {
}


void DDP::computeForwardPass(const double t, const double* x) {
  Eigen::Map<Eigen::VectorXd>(u_[0], dimu_) 
      += Eigen::Map<Eigen::VectorXd>(k_[0], dimu_);
  ocp_model_.stateEquation(t, dtau_, u_[0], x, x1_);
  for (int i=1; i<N_; ++i) {
    Eigen::Map<Eigen::VectorXd>(u_[i], dimu_) 
      += Eigen::Map<Eigen::VectorXd>(k_[i], dimu_)
         + Eigen::Map<Eigen::MatrixXd>(Kx_[i], dimx_, dimu_) 
           * (Eigen::Map<Eigen::VectorXd>(x1_, dimx_)-Eigen::Map<Eigen::VectorXd>(x_[i-1], dimx_));
    Eigen::Map<Eigen::VectorXd>(x_[i-1], dimx_) 
        = Eigen::Map<Eigen::VectorXd>(x1_, dimx_);
    ocp_model_.stateEquation(t+i*dtau_, dtau_, u_[i], x_[i-1], x1_);
  }
}


void DDP::setHorizonLength(const double horizon_length) {
  horizon_length_ = horizon_length;
  dtau_ = horizon_length / N_;
}


int DDP::dimx() const {
  return dimx_; 
}


int DDP::dimu() const {
  return dimu_; 
}

};
