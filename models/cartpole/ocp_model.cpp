 
#include "ocp_model.hpp"


namespace cddp {

void OCPModel::stateEquation(const double t, const double dtau, const double* x, 
                             const double* u, double* dx) const {
  double x0 = cos(x[1]);
  double x1 = l*pow(x[1], 2);
  double x2 = sin(x[1]);
  double x3 = m_p*x2;
  double x4 = dtau/(m_c + m_p*pow(x2, 2));
  dx[0] = dtau*x[2] + x[0];
  dx[1] = dtau*x[3] + x[1];
  dx[2] = x4*(u[0] + x3*(g*x0 + x1)) + x[2];
  dx[3] = x[3] + x4*(-g*x2*(m_c + m_p) - u[0]*x0 - x0*x1*x3)/l;
 
}

void OCPModel::stageCostDerivatives(const double t, const double dtau, 
                                    const double* x, const double* u, 
                                    double* lx, double* lu, double* lxx, 
                                    double* lux, double* luu) const {
  double x0 = dtau*q[0];
  double x1 = dtau*q[1];
  double x2 = dtau*q[2];
  double x3 = dtau*q[3];
  double x4 = dtau*r[0];
  lx[0] += (1.0/2.0)*x0*(2*x[0] - 2*x_ref[0]);
  lx[1] += (1.0/2.0)*x1*(2*x[1] - 2*x_ref[1]);
  lx[2] += (1.0/2.0)*x2*(2*x[2] - 2*x_ref[2]);
  lx[3] += (1.0/2.0)*x3*(2*x[3] - 2*x_ref[3]);
  lu[0] += u[0]*x4;
  lxx[0] += x0;
  lxx[5] += x1;
  lxx[10] += x2;
  lxx[15] += x3;
  luu[0] += x4;
 
}


void OCPModel::terminalCostDerivatives(const double t, const double* x, 
                                       double* phix, double* phixx) const {
  phix[0] = (1.0/2.0)*q_terminal[0]*(2*x[0] - 2*x_ref[0]);
  phix[1] = (1.0/2.0)*q_terminal[1]*(2*x[1] - 2*x_ref[1]);
  phix[2] = (1.0/2.0)*q_terminal[2]*(2*x[2] - 2*x_ref[2]);
  phix[3] = (1.0/2.0)*q_terminal[3]*(2*x[3] - 2*x_ref[3]);
  phixx[0] = q_terminal[0];
  phixx[5] = q_terminal[1];
  phixx[10] = q_terminal[2];
  phixx[15] = q_terminal[3];
 
}

void OCPModel::dynamicsDerivatives(const double t, const double dtau, 
                                   const double* x, const double* u, 
                                   const double* Vx, const double* Vxx, 
                                   double* fxVx, double* fuVx, double* fxVxxfx, 
                                   double* fuVxxfx, double* fuVxxfu, 
                                   double* Vxfxx, double* Vxfux, 
                                   double* Vxfuu) const {
  double x0 = sin(x[1]);
  double x1 = cos(x[1]);
  double x2 = g*x1;
  double x3 = l*pow(x[1], 2);
  double x4 = m_p*(x2 + x3);
  double x5 = x0*x4;
  double x6 = u[0] + x5;
  double x7 = pow(x0, 2);
  double x8 = m_p*x7;
  double x9 = m_c + x8;
  double x10 = dtau/pow(x9, 2);
  double x11 = x10*x6;
  double x12 = m_p*x0;
  double x13 = 2*x1;
  double x14 = x12*x13;
  double x15 = 2*l;
  double x16 = x15*x[1];
  double x17 = g*x0;
  double x18 = x16 - x17;
  double x19 = x1*x4 + x12*x18;
  double x20 = 1.0/x9;
  double x21 = dtau*x20;
  double x22 = -x11*x14 + x19*x21;
  double x23 = 1.0/l;
  double x24 = u[0]*x1;
  double x25 = m_c + m_p;
  double x26 = x17*x25;
  double x27 = x1*x12;
  double x28 = x27*x3;
  double x29 = x23*(-x24 - x26 - x28);
  double x30 = x10*x29;
  double x31 = pow(x1, 2);
  double x32 = m_p*x31;
  double x33 = u[0]*x0 - x16*x27 - x2*x25 - x3*x32 + x3*x8;
  double x34 = x21*x23;
  double x35 = -x14*x30 + x33*x34;
  double x36 = x1*x34;
  double x37 = Vxx[0]*dtau;
  double x38 = Vxx[1]*dtau;
  double x39 = Vxx[12]*x35 + Vxx[4] + Vxx[8]*x22;
  double x40 = Vxx[10]*x22 + Vxx[14]*x35 + Vxx[6];
  double x41 = Vxx[11]*x22 + Vxx[15]*x35 + Vxx[7];
  double x42 = Vxx[13]*x35 + Vxx[5] + Vxx[9]*x22;
  double x43 = Vxx[8] + x37;
  double x44 = Vxx[2]*dtau;
  double x45 = Vxx[10] + x44;
  double x46 = Vxx[3]*dtau;
  double x47 = Vxx[11] + x46;
  double x48 = Vxx[9] + x38;
  double x49 = Vxx[12] + Vxx[4]*dtau;
  double x50 = Vxx[14] + Vxx[6]*dtau;
  double x51 = Vxx[15] + Vxx[7]*dtau;
  double x52 = Vxx[13] + Vxx[5]*dtau;
  double x53 = 2*x11;
  double x54 = 8*dtau*pow(m_p, 2)*x31*x7/pow(x9, 3);
  double x55 = 4*x10*x27;
  double x56 = 2*x30;
  double x57 = 4*l*x[1];
  fxVx[0] += Vx[0];
  fxVx[1] += Vx[1] + Vx[2]*x22 + Vx[3]*x35;
  fxVx[2] += Vx[0]*dtau + Vx[2];
  fxVx[3] += Vx[1]*dtau + Vx[3];
  fuVx[0] += Vx[2]*x21 - Vx[3]*x36;
  fxVxxfx[0] += Vxx[0];
  fxVxxfx[1] += Vxx[1] + Vxx[2]*x22 + Vxx[3]*x35;
  fxVxxfx[2] += Vxx[2] + x37;
  fxVxxfx[3] += Vxx[3] + x38;
  fxVxxfx[4] += x39;
  fxVxxfx[5] += x22*x40 + x35*x41 + x42;
  fxVxxfx[6] += dtau*x39 + x40;
  fxVxxfx[7] += dtau*x42 + x41;
  fxVxxfx[8] += x43;
  fxVxxfx[9] += x22*x45 + x35*x47 + x48;
  fxVxxfx[10] += dtau*x43 + x45;
  fxVxxfx[11] += dtau*x48 + x47;
  fxVxxfx[12] += x49;
  fxVxxfx[13] += x22*x50 + x35*x51 + x52;
  fxVxxfx[14] += dtau*x49 + x50;
  fxVxxfx[15] += dtau*x52 + x51;
  fuVxxfx[0] += -x1*x20*x23*x46 + x20*x44;
  fuVxxfx[1] += x21*x40 - x36*x41;
  fuVxxfx[2] += x21*x45 - x36*x47;
  fuVxxfx[3] += x21*x50 - x36*x51;
  fuVxxfu[0] += x21*(Vxx[10]*x21 - Vxx[14]*x36) - x36*(Vxx[11]*x21 - Vxx[15]*x36);
  Vxfxx[5] += Vx[2]*(-x19*x55 + x21*(m_p*x13*x18 + x12*(x15 - x2) - x5) - x32*x53 + x53*x8 + x54*x6) + Vx[3]*(-x23*x33*x55 + x29*x54 - x32*x56 + x34*(-x15*x27 + x24 + x26 + 4*x28 - x32*x57 + x57*x8) + x56*x8);
  Vxfux[1] += -Vx[2]*x10*x14 + Vx[3]*x0*x34 + 2*Vx[3]*x10*x12*x23*x31;
 
}

int OCPModel::dimx() const {
  return dimx_;
}

int OCPModel::dimu() const {
  return dimu_;
}

} // namespace cgmres

