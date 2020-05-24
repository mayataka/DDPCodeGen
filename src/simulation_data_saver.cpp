#include "simulation_data_saver.hpp"


namespace cddp {

void SaveSimulationData(const int dimx, const int dimu, std::ofstream& x_data, 
                        std::ofstream& u_data, const double t, const double* x, 
                        const double* u) {
  for (int i=0; i<dimx; i++) {
    x_data << x[i] << " ";
  }
  x_data << "\n";
  for (int i=0; i<dimu; i++) {
    u_data << u[i] << " ";
  }
  u_data << "\n";
}

} // namespace cddp