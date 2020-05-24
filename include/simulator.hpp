#ifndef CDDP_SIMULATOR_HPP_
#define CDDP_SIMULATOR_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include <eigen3/Eigen/Core>

#include "ocp_model.hpp"
#include "simulation_data_saver.hpp"


namespace cddp {

template <class NMPCSolver>
void simulation(NMPCSolver& nmpc, const double* x0, const double simulation_time, 
                const double sampling_period, const std::string save_dir, 
                const std::string savefile_name) {
  OCPModel ocp_model;
  double x[ocp_model.dimx()], x1[ocp_model.dimx()], u[ocp_model.dimu()];
  std::chrono::system_clock::time_point start_clock, end_clock;

  std::string savefile_header = save_dir + "/" + savefile_name;
  std::ofstream state_data(savefile_header + "_x.dat"), 
    control_input_data(savefile_header + "_u.dat"), 
    conditions_data(savefile_header + "_conditions.dat");

  double total_time = 0;
  Eigen::Map<Eigen::VectorXd>(x, ocp_model.dimx()) 
      = Eigen::Map<const Eigen::VectorXd>(x0, ocp_model.dimx());
  nmpc.getControlInput(u);

  std::cout << "Start simulation" << std::endl;
  for (double t=sampling_period; t<simulation_time; t+=sampling_period) {
    // Saves the current datas.
    SaveSimulationData(ocp_model.dimx(), ocp_model.dimu(), state_data, 
                       control_input_data, t, x, u);

    // Computes the next state vector using the 4th Runge-Kutta-Gill method.
    ocp_model.stateEquation(t, sampling_period, x, u, x1);

    // Updates the solution and measure the computational time of the update.
    start_clock = std::chrono::system_clock::now();
    nmpc.updateSolution(t, x, sampling_period);
    nmpc.getControlInput(u);
    end_clock = std::chrono::system_clock::now();

    // Converts the computational time to seconds.
    double step_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_clock-start_clock).count();
    step_time *= 1e-06;
    total_time += step_time;

    // Updates the state.
    Eigen::Map<Eigen::VectorXd>(x, ocp_model.dimx()) 
        = Eigen::Map<Eigen::VectorXd>(x1, ocp_model.dimx());
  }

  // cout the simulation conditions.
  std::cout << "End simulation\n" 
      << "Total CPU time for control update: " << total_time << " [sec]\n" 
      << "sampling time: " << sampling_period << " [sec]" << "\n" 
      << "CPU time for per control update: " 
      << total_time/((int)(simulation_time/sampling_period)) 
      << " [sec]" << std::endl;

  // Save simulation conditions.
  conditions_data << "simulation name: " << savefile_name << "\n"
      << "simulation time: " << simulation_time << " [sec]\n"
      << "Total CPU time for control update: " << total_time << " [sec]\n"
      << "sampling time: " << sampling_period << " [sec]\n"
      << "CPU time for per control update: " 
      << total_time/((int)(simulation_time/sampling_period)) 
      << " [sec]\n";

  state_data.close();
  control_input_data.close();
  conditions_data.close();
}

} // namespace cgmres

#endif // CDDP_SIMULATOR_HPP_