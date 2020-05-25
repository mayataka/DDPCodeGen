
#include <string>

#include "NMPC.hpp"
#include "simulator.hpp"


int main() {
      const int N = 100;
  const double T_f = 2.0;
  const double alpha = 1.0;
  const double dt = 0.001;
  cddp::NMPC<4, 1> nmpc(T_f, alpha, N, dt);
  double x0[4] = {0, 0, 0, 0};
  double u0[1] = {0};
  nmpc.setControlInput(u0);
  const double simulation_time = 10;
  const double sampling_time = 0.001;
  const std::string save_dir = "simulation_result";
  const std::string save_file_name = "cartpole";
  cddp::simulation<cddp::NMPC<4, 1>>(nmpc, x0, simulation_time, sampling_time, save_dir, save_file_name);

  return 0;
}
