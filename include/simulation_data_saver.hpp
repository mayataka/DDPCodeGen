#ifndef CDDP_SIMULATION_DATA_SAVER_HPP_ 
#define CDDP_SIMULATION_DATA_SAVER_HPP_

#include <iostream>
#include <fstream>

namespace cddp {

// Saves state_vec, contorl_input_vec, and error_norm to file streams.
void SaveSimulationData(const int dimx, const int dimu, std::ofstream& x_data, 
                        std::ofstream& u_data, const double t, const double* x, 
                        const double* u);

} // namespace cddp 

#endif // CDDP_SIMULATION_DATA_SAVER_HPP_