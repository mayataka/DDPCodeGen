#include <iostream>
#include <chrono>
#include <string>

#include "ocp_model.hpp"
#include "NMPC.hpp"
#include "simulator.hpp"


int main() {
  const int N = 100;
  const double T_f = 2;
  const double alpha = 1;
  const double dt = 0.001;
  cddp::NMPC<4, 1> nmpc(T_f, alpha, N, dt);

  double *x0 = cddp::memorymanager::NewVector(nmpc.dim_x());
  // Eigen::Map<Eigen::VectorXd>(x0, nmpc.dim_x()) = Eigen::VectorXd::Random(nmpc.dim_x());
  const double simulation_time = 10;
  const double sampling_period = 0.001;
  const std::string save_dir = "simulation_result";
  const std::string save_file_name = "cartpole";
  cddp::simulation<cddp::NMPC<4, 1>>(nmpc, x0, simulation_time, sampling_period, 
                                     save_dir, save_file_name);
  
  cddp::memorymanager::DeleteVector(x0);
  return 0;
}