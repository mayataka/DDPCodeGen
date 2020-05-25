 
#include "ocp_model.hpp"


namespace cddp {

void OCPModel::dynamics(const double t, const double dtau, const double* x, 
                        const double* u, double* dx) const {
  double x0 = pow(l1, 2);
  double x1 = m2*x0;
  double x2 = l1*m2;
  double x3 = d2*x2;
  double x4 = x3*cos(x[1]);
  double x5 = pow(d2, 2);
  double x6 = J2 + m2*x5;
  double x7 = J1 + pow(d1, 2)*m1;
  double x8 = 1.0/(x1 + 2.0*x4 + x6 + x7);
  double x9 = d2*g;
  double x10 = m2*x9*sin(x[0] + x[1]);
  double x11 = d1*m1;
  double x12 = x11 + x2;
  double x13 = g*sin(x[0]);
  double x14 = x3*sin(x[1]);
  double x15 = 2.0*x[1];
  double x16 = 0.5*l1;
  double x17 = pow(m2, 2)*x5;
  double x18 = x16*x17;
  double x19 = pow(x[2], 2.0);
  double x20 = x[2]*x[3];
  double x21 = pow(x[3], 2.0);
  dx[0] = x[2];
  dx[1] = x[3];
  dx[2] = x8*(u[0] - x10 - x12*x13 + 2.0*x14*x[3]*(x[2] + 0.5*x[3]));
  dx[3] = x8*(-g*x18*sin(x15 + x[0]) - u[0]*(x4 + x6) - x0*x17*(x19 + x20 + 0.5*x21)*sin(x15) - x10*(0.5*x1 - x11*x16 + x7) + 0.5*x12*x2*x9*sin(x[0] - x[1]) + x13*(J2*x11 + m2*(J2*l1 + x11*x5) + x18) - x14*(2.0*J2*x20 + J2*x21 + m2*(x0*x19 + x5*pow(x[2] + x[3], 2.0)) + x19*(J2 + x7)))/x6;
 
}

void OCPModel::stateEquation(const double t, const double dtau, const double* x, 
                             const double* u, double* F) const {
  double x0 = d2*g;
  double x1 = m2*x0*sin(x[0] + x[1]);
  double x2 = d1*m1;
  double x3 = l1*m2;
  double x4 = x2 + x3;
  double x5 = g*sin(x[0]);
  double x6 = d2*x3;
  double x7 = x6*sin(x[1]);
  double x8 = pow(l1, 2);
  double x9 = m2*x8;
  double x10 = x6*cos(x[1]);
  double x11 = pow(d2, 2);
  double x12 = J2 + m2*x11;
  double x13 = J1 + pow(d1, 2)*m1;
  double x14 = dtau/(2.0*x10 + x12 + x13 + x9);
  double x15 = 2.0*x[1];
  double x16 = 0.5*l1;
  double x17 = pow(m2, 2)*x11;
  double x18 = x16*x17;
  double x19 = pow(x[2], 2.0);
  double x20 = x[2]*x[3];
  double x21 = pow(x[3], 2.0);
  F[0] = dtau*x[2] + x[0];
  F[1] = dtau*x[3] + x[1];
  F[2] = x14*(u[0] - x1 - x4*x5 + 2.0*x7*x[3]*(x[2] + 0.5*x[3])) + x[2];
  F[3] = x[3] + x14*(-g*x18*sin(x15 + x[0]) - u[0]*(x10 + x12) + 0.5*x0*x3*x4*sin(x[0] - x[1]) - x1*(x13 - x16*x2 + 0.5*x9) - x17*x8*(x19 + x20 + 0.5*x21)*sin(x15) + x5*(J2*x2 + m2*(J2*l1 + x11*x2) + x18) - x7*(2.0*J2*x20 + J2*x21 + m2*(x11*pow(x[2] + x[3], 2.0) + x19*x8) + x19*(J2 + x13)))/x12;
 
}

void OCPModel::stageCostDerivatives(const double t, const double dtau, 
                                    const double* x, const double* u, 
                                    double* lx, double* lu, double* lxx, 
                                    double* lux, double* luu) const {
  double x0 = dtau*q[0];
  double x1 = dtau*q[1];
  double x2 = dtau*q[2];
  double x3 = dtau*q[3];
  double x4 = -u[0] + u_max;
  double x5 = u[0] - u_min;
  lx[0] += (1.0/2.0)*x0*(2*x[0] - 2*x_ref[0]);
  lx[1] += (1.0/2.0)*x1*(2*x[1] - 2*x_ref[1]);
  lx[2] += (1.0/2.0)*x2*(2*x[2] - 2*x_ref[2]);
  lx[3] += (1.0/2.0)*x3*(2*x[3] - 2*x_ref[3]);
  lu[0] += dtau*(r[0]*u[0] - 1/x5 + 1.0/x4);
  lxx[0] += x0;
  lxx[5] += x1;
  lxx[10] += x2;
  lxx[15] += x3;
  luu[0] += dtau*(r[0] + pow(x5, -2) + pow(x4, -2));
 
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
  double x0 = x[0] + x[1];
  double x1 = d2*g;
  double x2 = m2*x1;
  double x3 = x2*cos(x0);
  double x4 = -x3;
  double x5 = d1*m1;
  double x6 = l1*m2;
  double x7 = x5 + x6;
  double x8 = g*cos(x[0]);
  double x9 = x4 - x7*x8;
  double x10 = pow(l1, 2);
  double x11 = m2*x10;
  double x12 = d2*x6;
  double x13 = x12*cos(x[1]);
  double x14 = 2.0*x13;
  double x15 = pow(d2, 2);
  double x16 = m2*x15;
  double x17 = J2 + x16;
  double x18 = pow(d1, 2)*m1;
  double x19 = J1 + x18;
  double x20 = dtau/(x11 + x14 + x17 + x19);
  double x21 = Vx[2]*x20;
  double x22 = 0.5*l1;
  double x23 = pow(m2, 2);
  double x24 = x15*x23;
  double x25 = x22*x24;
  double x26 = 2.0*x[1];
  double x27 = x26 + x[0];
  double x28 = g*cos(x27);
  double x29 = x[0] - x[1];
  double x30 = 0.5*x1*x6*x7;
  double x31 = x30*cos(x29);
  double x32 = 0.5*x11;
  double x33 = x19 - x22*x5 + x32;
  double x34 = -x3*x33;
  double x35 = J2*x5 + m2*(J2*l1 + x15*x5) + x25;
  double x36 = -x25*x28 + x31 + x34 + x35*x8;
  double x37 = 1.0/x17;
  double x38 = x20*x37;
  double x39 = Vx[3]*x38;
  double x40 = x[2] + 0.5*x[3];
  double x41 = x14*x40;
  double x42 = x4 + x41*x[3];
  double x43 = x2*sin(x0);
  double x44 = g*sin(x[0]);
  double x45 = x44*x7;
  double x46 = sin(x[1]);
  double x47 = x12*x46;
  double x48 = 2.0*x47;
  double x49 = x40*x48;
  double x50 = x49*x[3];
  double x51 = dtau*(u[0] - x43 - x45 + x50);
  double x52 = 0.5*J1 + 0.5*J2 + x13 + 0.5*x16 + 0.5*x18 + x32;
  double x53 = pow(x52, -2);
  double x54 = x47*x53;
  double x55 = 0.5*x54;
  double x56 = x20*x42 + x51*x55;
  double x57 = 1.0*x24;
  double x58 = l1*x57;
  double x59 = pow(x[2], 2.0);
  double x60 = x[2]*x[3];
  double x61 = pow(x[3], 2.0);
  double x62 = x59 + x60 + 0.5*x61;
  double x63 = 2.0*x15;
  double x64 = x23*x63;
  double x65 = x10*x64*cos(x26);
  double x66 = 2.0*J2;
  double x67 = J2 + x19;
  double x68 = x[2] + x[3];
  double x69 = J2*x61 + m2*(x10*x59 + x15*pow(x68, 2.0)) + x59*x67 + x60*x66;
  double x70 = u[0]*x47 - x13*x69 - x28*x58 - x31 + x34 - x62*x65;
  double x71 = g*sin(x27);
  double x72 = x25*x71;
  double x73 = x30*sin(x29);
  double x74 = x33*x43;
  double x75 = x10*sin(x26);
  double x76 = x24*x75;
  double x77 = x62*x76;
  double x78 = x35*x44;
  double x79 = x47*x69;
  double x80 = -u[0]*(x13 + x17) - x72 + x73 - x74 - x77 + x78 - x79;
  double x81 = dtau*x55;
  double x82 = x37*x81;
  double x83 = x38*x70 + x80*x82;
  double x84 = x20*x[3];
  double x85 = x48*x84 + 1;
  double x86 = 2.0*pow(x[2], 1.0);
  double x87 = x86 + x[3];
  double x88 = pow(x68, 1.0);
  double x89 = m2*(x10*x86 + x63*x88) + x66*x[3] + x67*x86;
  double x90 = -x47*x89 - x76*x87;
  double x91 = 1.0*x[3];
  double x92 = x47*x91 + x49;
  double x93 = pow(x[3], 1.0);
  double x94 = 1.0*x93 + x[2];
  double x95 = 2.0*x16;
  double x96 = x66*x93 + x66*x[2] + x88*x95;
  double x97 = -x47*x96 - x76*x94;
  double x98 = x38*x97 + 1;
  double x99 = -J2 - x13 - x16;
  double x100 = Vxx[10]*x20;
  double x101 = x36*x38;
  double x102 = Vxx[14]*x101 + Vxx[2] + x100*x9;
  double x103 = x102*x20;
  double x104 = Vxx[11]*x20;
  double x105 = Vxx[15]*x101 + Vxx[3] + x104*x9;
  double x106 = x20*x9;
  double x107 = Vxx[0] + Vxx[12]*x101 + Vxx[8]*x106;
  double x108 = Vxx[13]*x101 + Vxx[1] + Vxx[9]*x106;
  double x109 = x38*x90;
  double x110 = Vxx[10]*x56 + Vxx[14]*x83 + Vxx[6];
  double x111 = x110*x20;
  double x112 = Vxx[11]*x56 + Vxx[15]*x83 + Vxx[7];
  double x113 = Vxx[12]*x83 + Vxx[4] + Vxx[8]*x56;
  double x114 = Vxx[13]*x83 + Vxx[5] + Vxx[9]*x56;
  double x115 = Vxx[10]*x85 + Vxx[14]*x109 + Vxx[2]*dtau;
  double x116 = x115*x20;
  double x117 = Vxx[11]*x85 + Vxx[15]*x109 + Vxx[3]*dtau;
  double x118 = Vxx[0]*dtau + Vxx[12]*x109 + Vxx[8]*x85;
  double x119 = Vxx[13]*x109 + Vxx[1]*dtau + Vxx[9]*x85;
  double x120 = Vxx[14]*x98 + Vxx[6]*dtau + x100*x92;
  double x121 = x120*x20;
  double x122 = Vxx[15]*x98 + Vxx[7]*dtau + x104*x92;
  double x123 = x20*x92;
  double x124 = Vxx[12]*x98 + Vxx[4]*dtau + Vxx[8]*x123;
  double x125 = Vxx[13]*x98 + Vxx[5]*dtau + Vxx[9]*x123;
  double x126 = x38*x99;
  double x127 = -x73 + x74;
  double x128 = x81*x9;
  double x129 = x38*(x58*x71 + x73 + x74);
  double x130 = x36*x82;
  double x131 = 1.0*dtau*x54;
  double x132 = 0.5*x13*x53;
  double x133 = x10*pow(x46, 2)*x57;
  double x134 = x133/pow(x52, 3);
  double x135 = dtau*x37*x80;
  double x136 = Vx[2]*(dtau*x133*x53*x[3] + x14*x84);
  double x137 = x38*(-x13*x89 - x65*x87);
  double x138 = x82*x90;
  double x139 = x13*x91 + x41;
  double x140 = x81*x92;
  double x141 = Vx[3]*(x38*(-x13*x96 - x65*x94) + x82*x97);
  double x142 = x21*x48;
  double x143 = -x47*(x66 + x95);
  double x144 = x142 + x39*(x143 - x76);
  fxVx[0] += Vx[0] + x21*x9 + x36*x39;
  fxVx[1] += Vx[1] + Vx[2]*x56 + Vx[3]*x83;
  fxVx[2] += Vx[0]*dtau + Vx[2]*x85 + x39*x90;
  fxVx[3] += Vx[1]*dtau + Vx[3]*x98 + x21*x92;
  fuVx[0] += x21 + x39*x99;
  fxVxxfx[0] += x101*x105 + x103*x9 + x107;
  fxVxxfx[1] += x102*x56 + x105*x83 + x108;
  fxVxxfx[2] += dtau*x107 + x102*x85 + x105*x109;
  fxVxxfx[3] += dtau*x108 + x103*x92 + x105*x98;
  fxVxxfx[4] += x101*x112 + x111*x9 + x113;
  fxVxxfx[5] += x110*x56 + x112*x83 + x114;
  fxVxxfx[6] += dtau*x113 + x109*x112 + x110*x85;
  fxVxxfx[7] += dtau*x114 + x111*x92 + x112*x98;
  fxVxxfx[8] += x101*x117 + x116*x9 + x118;
  fxVxxfx[9] += x115*x56 + x117*x83 + x119;
  fxVxxfx[10] += dtau*x118 + x109*x117 + x115*x85;
  fxVxxfx[11] += dtau*x119 + x116*x92 + x117*x98;
  fxVxxfx[12] += x101*x122 + x121*x9 + x124;
  fxVxxfx[13] += x120*x56 + x122*x83 + x125;
  fxVxxfx[14] += dtau*x124 + x109*x122 + x120*x85;
  fxVxxfx[15] += dtau*x125 + x121*x92 + x122*x98;
  fuVxxfx[0] += x103 + x105*x126;
  fuVxxfx[1] += x111 + x112*x126;
  fuVxxfx[2] += x116 + x117*x126;
  fuVxxfx[3] += x121 + x122*x126;
  fuVxxfu[0] += x126*(Vxx[15]*x126 + x104) + x20*(Vxx[14]*x126 + x100);
  Vxfxx[0] += x21*(x43 + x45) + x39*(x127 + x72 - x78);
  Vxfxx[1] += Vx[2]*(x128 + x20*x43) + Vx[3]*(x129 + x130);
  Vxfxx[4] += Vx[2]*x128 + Vx[3]*x129 + Vx[3]*x130 + x21*x43;
  Vxfxx[5] += Vx[2]*(x131*x42 + x132*x51 + x134*x51 + x20*(x43 - x50)) + Vx[3]*(x131*x37*x70 + x132*x135 + x134*x135 + x38*(l1*x64*x71 + u[0]*x13 + x127 + 4.0*x77 + x79));
  Vxfxx[6] += Vx[3]*x137 + Vx[3]*x138 + x136;
  Vxfxx[7] += Vx[2]*x140 + x139*x21 + x141;
  Vxfxx[9] += Vx[3]*(x137 + x138) + x136;
  Vxfxx[10] += x39*(-x47*(2.0*J1 + m2*(2.0*x10 + x63) + 2.0*x18 + x66) - x64*x75);
  Vxfxx[11] += x144;
  Vxfxx[13] += Vx[2]*(x139*x20 + x140) + x141;
  Vxfxx[14] += x144;
  Vxfxx[15] += x142 + x39*(x143 - 1.0*x76);
  Vxfux[1] += Vx[2]*x81 + Vx[3]*x82*x99 + x39*x47;
 
}

int OCPModel::dimx() const {
  return dimx_;
}

int OCPModel::dimu() const {
  return dimu_;
}

} // namespace cgmres

