#include <../include/IKFoM_toolkit/esekfom/esekfom.hpp>

typedef MTK::SO3<double> SO3;
typedef MTK::vect<3, double> vect3;
typedef MTK::S2<double, 98, 10, 3> S2;
MTK_BUILD_MANIFOLD(state,
    ((S2, grav))
    ((vect3, pos))
    ((SO3, rot))
    ((vect3, vel))
    ((vect3, bg))
    ((vect3, ba))
    ((SO3, offset_R_C_B))
    ((vect3, offset_P_C_B))
);

MTK_BUILD_MANIFOLD(input,
    ((vect3, a))
    ((vect3, omega))
);

MTK_BUILD_MANIFOLD(measurement,
    ((SO3, R_cr))
    ((vect3, P_cr))
);

Eigen::Matrix<double, 24, 1> f(state &s, input &i) {
    Eigen::Matrix<double, 24, 1> s_rate;
    Eigen::Matrix<double, 3, 3> omega_hat = MTK::hat(vect3(i.omega - s.bg));
    Eigen::Matrix<double, 3, 1> zero31(0.0, 0.0, 0.0);
    s_rate.block(0,0,3,1) = - omega_hat * s.pos - s.vel;
    s_rate.block(3,0,3,1) = - omega_hat * s.vel + s.grav.vec + i.a - s.ba;
    s_rate.block(6,0,3,1) = - vect3(s.rot.toRotationMatrix().inverse() * vect3(i.omega - s.bg));
    s_rate.block(9,0,3,1) = zero31;
    s_rate.block(12,0,3,1) = zero31;
    s_rate.block(15,0,3,1) = zero31;
    s_rate.block(18,0,3,1) = zero31;
    s_rate.block(21,0,3,1) = zero31;
    return s_rate;
}

Eigen::Matrix<double, 24, 23> df_dx(state &s, input &i){
    Eigen::Matrix<double, 24, 23> pf_px;
    Eigen::Matrix<double, 3, 3> zero33;
    Eigen::Matrix<double, 3, 2> zero32;
    Eigen::Matrix<double, 2, 3> zero23;
    Eigen::Matrix<double, 3, 3> eye33;
    zero33 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    zero32 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    zero23 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    eye33 << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix<double, 3, 1> omega = vect3(i.omega - s.bg);
    Eigen::Matrix<double, 3, 3> omega_hat = MTK::hat(vect3(omega));
    Eigen::Matrix<double, 3, 3> grav_hat;
    Eigen::Matrix<double, 3, 2> grav_B;
    Eigen::Matrix<double, 3, 3> pos_hat = MTK::hat(s.pos);
    Eigen::Matrix<double, 3, 3> vel_hat = MTK::hat(s.vel);
    Eigen::Matrix<double, 3, 3> R_inverse = s.rot.toRotationMatrix().inverse();
    s.grav.S2_hat(grav_hat);
    s.grav.S2_Bx(grav_B);
    pf_px.block(0,0,3,23) = (omega_hat * grav_hat * grav_B, zero33, zero33, zero33, -grav_hat, zero33, zero33, zero33, zero33);
    pf_px.block(3,0,3,23) = (zero32, -omega_hat, zero33, -eye33, -pos_hat, zero33, zero33, zero33);
    pf_px.block(6,0,3,23) = (zero32, zero33, -MTK::hat(vect3(R_inverse * omega)), zero33, R_inverse, zero33, zero33, zero33);
    pf_px.block(9,0,3,23) = (-grav_hat * grav_B, zero33, zero33, -omega_hat, -vel_hat, -eye33, zero33, zero33);
    for (int i=0; i<4; i++)
    {
        pf_px.block(12+3*i,0,3,23) = (zero32, zero33, zero33, zero33, zero33, zero33, zero33, zero33);
    }
    return pf_px;
}

Eigen::Matrix<double, 24, 12> df_dw(state &s, input &i){
    Eigen::Matrix<double, 24, 12> pf_pw;
    Eigen::Matrix<double, 3, 3> zero33;
    Eigen::Matrix<double, 3, 3> eye33;
    zero33 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    eye33 << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    Eigen::Matrix<double, 3, 3> grav_hat;
    Eigen::Matrix<double, 3, 3> pos_hat = MTK::hat(s.pos);
    Eigen::Matrix<double, 3, 3> vel_hat = MTK::hat(s.vel);
    Eigen::Matrix<double, 3, 3> R_inverse = s.rot.toRotationMatrix().inverse();
    s.grav.S2_hat(grav_hat);
    pf_pw.block(0,0,3,12) = (-grav_hat, zero33, zero33, zero33);
    pf_pw.block(3,0,3,12) = (-pos_hat, zero33, zero33, zero33);
    pf_pw.block(6,0,3,12) = (R_inverse, zero33, zero33, zero33);
    pf_pw.block(9,0,3,12) = (-vel_hat, -eye33, zero33, zero33);
    pf_pw.block(12,0,3,12) = (zero33, zero33, eye33, zero33);
    pf_pw.block(15,0,3,12) = (zero33, zero33, zero33, eye33);
    pf_pw.block(18,0,3,12) = (zero33, zero33, zero33, zero33);
    pf_pw.block(21,0,3,12) = (zero33, zero33, zero33, zero33);
    return pf_pw;
}

measurement h(state &s, bool &valid){
    measurement mear;
    Eigen::Matrix<double, 3, 3> R_cb = s.offset_R_C_B.toRotationMatrix();
    Eigen::Matrix<double, 3, 3> R_br = s.rot.toRotationMatrix();
    mear.R_cr = R_cb * R_br;
    mear.P_cr = R_cb * s.pos + s.offset_P_C_B;
    return mear;
}

Eigen::Matrix<double, 6, 23> dh_dx(state &s, bool &valid){
    Eigen::Matrix<double, 6, 23> ph_px;
    Eigen::Matrix<double, 3, 3> zero33;
    Eigen::Matrix<double, 3, 2> zero32;
    Eigen::Matrix<double, 3, 3> eye33;
    zero33 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    zero32 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    eye33 << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    Eigen::Matrix<double, 3, 3> R_inverse = s.rot.toRotationMatrix().inverse();
    Eigen::Matrix<double, 3, 3> pos_hat = MTK::hat(s.pos);
    Eigen::Matrix<double, 3, 3> R_cb = s.offset_R_C_B.toRotationMatrix();
    ph_px.block(0,0,3,23) = (zero32, zero33, eye33, zero33, zero33, zero33, R_inverse, zero33);
    ph_px.block(3,0,3,23) = (zero32, R_cb, zero33, zero33, zero33, zero33, -R_cb * pos_hat, eye33);
    return ph_px;
}

Eigen::Matrix<double, 6, 6> dh_dv(state &s, bool &valid){
    Eigen::Matrix<double, 6, 6> ph_pv;
    Eigen::Matrix<double, 3, 3> zero33;
    Eigen::Matrix<double, 3, 3> eye33;
    zero33 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    eye33 << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    ph_pv.block(0,0,3,6) = (eye33, zero33);
    ph_pv.block(3,0,3,6) = (zero33, eye33);
    return ph_pv; 
}