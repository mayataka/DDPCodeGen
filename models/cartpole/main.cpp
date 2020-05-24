#include <iostream>
#include <chrono>

#include "ocp_model.hpp"
#include "DDP.hpp"
#include "memory_manager.hpp"


int main() {
  const int N = 100;
  const double horizon_length = 0.001;
  cddp::DDP<4, 1> ddp(N);
  ddp.setHorizonLength(horizon_length);
  const double t = 0;
  double *x = cddp::memorymanager::NewVector(ddp.dim_x());
  Eigen::Map<Eigen::VectorXd>(x, ddp.dim_x()) = Eigen::VectorXd::Random(ddp.dim_x());

  std::chrono::system_clock::time_point start_clock, end_clock;

  const int num_ddp = 100000;
  start_clock = std::chrono::system_clock::now();
  for (int i=0; i<num_ddp; ++i) {
    ddp.rolloutState(t, x);
    ddp.computeBackwardPass(t, x);
    ddp.computeForwardPass(t, x);
  }
  end_clock = std::chrono::system_clock::now();
  ddp.printSolution();
  double total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock-start_clock).count();
  total_time *= 1e-06;
  double average_time = total_time / num_ddp;
  std::cout << "average computational time per iteration is " << average_time << std::endl;
  
  cddp::memorymanager::DeleteVector(x);
  return 0;
}