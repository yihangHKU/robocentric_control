#include <../include/IKFoM_toolkit/esekfom/esekfom.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PointStamped.h>

typedef MTK::SO3<double> SO3;
typedef MTK::vect<3, double> vect3;
typedef MTK::S2<double, 981, 100, 3> S2_grav;
typedef MTK::S2<double, 100, 100, 2> S2_line;

MTK_BUILD_MANIFOLD(state,
            ((S2_grav, grav))
            ((S2_line, line))
            ((vect3, pos))
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
            ((vect3, P_cr))
            ((S2_line, line_cr))
            );

vect3 SO3ToEuler(const SO3 &orient) 
{
	Eigen::Matrix<double, 3, 1> _ang;
	Eigen::Vector4d q_data = orient.coeffs().transpose();
	//scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
	double sqw = q_data[3]*q_data[3];
	double sqx = q_data[0]*q_data[0];
	double sqy = q_data[1]*q_data[1];
	double sqz = q_data[2]*q_data[2];
	double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
	double test = q_data[3]*q_data[1] - q_data[2]*q_data[0];

	if (test > 0.49999*unit) { // singularity at north pole
	
		_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
	if (test < -0.49999*unit) { // singularity at south pole
		_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
		
	_ang <<
			std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
			std::asin (2*test/unit),
			std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	vect3 euler_ang(temp, 3);
		// euler_ang[0] = roll, euler_ang[1] = pitch, euler_ang[2] = yaw
	return euler_ang;
}

Eigen::Matrix<double, 24, 1> f(state &s, const  input &i) {
    Eigen::Matrix<double, 24, 1> s_rate;
    Eigen::Matrix<double, 3, 3> omega_hat = MTK::hat(vect3(i.omega - s.bg));
    Eigen::Matrix<double, 3, 1> zero31(0.0, 0.0, 0.0);
    s_rate.block(0,0,3,1) = - (i.omega - s.bg);
    s_rate.block(3,0,3,1) = - (i.omega - s.bg);
    s_rate.block(6,0,3,1) = - omega_hat * s.pos - s.vel;
    s_rate.block(9,0,3,1) = - omega_hat * s.vel + s.grav.vec + i.a - s.ba;
    s_rate.block(12,0,3,1) = zero31;
    s_rate.block(15,0,3,1) = zero31;
    s_rate.block(18,0,3,1) = zero31;
    s_rate.block(21,0,3,1) = zero31;
    return s_rate;
}

Eigen::Matrix<double, 24, 22> df_dx(state &s, const input &i){
    Eigen::Matrix<double, 24, 22> pf_px;
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
    s.grav.S2_hat(grav_hat);
    s.grav.S2_Bx(grav_B);
    pf_px.block(0,0,3,22) << zero32, zero32, zero33, zero33, eye33, zero33, zero33, zero33;
    pf_px.block(3,0,3,22) << zero32, zero32, zero33, zero33, eye33, zero33, zero33, zero33;
    pf_px.block(6,0,3,22) << zero32, zero32, -omega_hat, -eye33, -pos_hat, zero33, zero33, zero33;
    pf_px.block(9,0,3,22) << -grav_hat * grav_B, zero32, zero33, -omega_hat, -vel_hat, -eye33, zero33, zero33;
    for (int i=0; i<4; i++)
    {
        pf_px.block(12+3*i,0,3,22) << zero32, zero32, zero33, zero33, zero33, zero33, zero33, zero33;
    }
    return pf_px;
}

Eigen::Matrix<double, 24, 12> df_dw(state &s, const input &i){
    Eigen::Matrix<double, 24, 12> pf_pw;
    Eigen::Matrix<double, 3, 3> zero33;
    Eigen::Matrix<double, 3, 3> eye33;
    zero33 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    eye33 << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix<double, 3, 3> grav_hat;
    Eigen::Matrix<double, 3, 3> pos_hat = MTK::hat(s.pos);
    Eigen::Matrix<double, 3, 3> vel_hat = MTK::hat(s.vel);
    s.grav.S2_hat(grav_hat);
    pf_pw.block(0,0,3,12) << eye33, zero33, zero33, zero33;
    pf_pw.block(3,0,3,12) << eye33, zero33, zero33, zero33;
    pf_pw.block(6,0,3,12) << -pos_hat, zero33, zero33, zero33;
    pf_pw.block(9,0,3,12) << -vel_hat, -eye33, zero33, zero33;
    pf_pw.block(12,0,3,12) << zero33, zero33, eye33, zero33;
    pf_pw.block(15,0,3,12) << zero33, zero33, zero33, eye33;
    pf_pw.block(18,0,3,12) << zero33, zero33, zero33, zero33;
    pf_pw.block(21,0,3,12) << zero33, zero33, zero33, zero33;
    return pf_pw;
}

measurement h(state &s, bool &valid){
    measurement mear;
    Eigen::Matrix<double, 3, 3> R_cb = s.offset_R_C_B.toRotationMatrix();
    mear.P_cr = R_cb * s.pos + s.offset_P_C_B;
    mear.line_cr.vec = R_cb * s.line.vec;
    return mear;
}

Eigen::Matrix<double, 5, 22> dh_dx(state &s, bool &valid){
    Eigen::Matrix<double, 5, 22> ph_px;
    Eigen::Matrix<double, 3, 3> zero33;
    Eigen::Matrix<double, 3, 2> zero32;
    Eigen::Matrix<double, 2, 3> zero23;
    Eigen::Matrix<double, 2, 2> zero22;
    Eigen::Matrix<double, 3, 3> eye33;
    zero23 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    zero22 << 0.0, 0.0, 0.0, 0.0;
    zero33 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    zero32 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    eye33 << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix<double, 3, 3> pos_hat = MTK::hat(s.pos);
    Eigen::Matrix<double, 3, 3> R_cb = s.offset_R_C_B.toRotationMatrix();
    Eigen::Matrix<double, 3, 3> line_hat;
    Eigen::Matrix<double, 3, 2> line_B;
    Eigen::Matrix<double, 3, 3> line_cr_hat;
    Eigen::Matrix<double, 3, 2> line_cr_B;
    s.line.S2_hat(line_hat);
    s.line.S2_Bx(line_B);
    S2_line line_cr;
    line_cr.vec = R_cb * s.line.vec;
    line_cr.S2_hat(line_cr_hat);
    line_cr.S2_Bx(line_cr_B);
    ph_px.block(0,0,3,22) << zero32, zero32, R_cb, zero33, zero33, zero33, -R_cb * pos_hat, eye33;
    ph_px.block(3,0,2,22) << zero22, -1/line_cr.length/line_cr.length * line_cr_B.transpose() * line_cr_hat * R_cb * line_hat * line_B , zero23, zero23, zero23, zero23, -1/line_cr.length/line_cr.length * line_cr_B.transpose() * line_cr_hat * R_cb * line_hat, zero23; 
    return ph_px;
}

Eigen::Matrix<double, 5, 5> dh_dv(state &s, bool &valid){
    Eigen::Matrix<double, 5, 5> ph_pv;
    Eigen::Matrix<double, 3, 3> zero33;
    Eigen::Matrix<double, 3, 2> zero32;
    Eigen::Matrix<double, 2, 3> zero23;
    Eigen::Matrix<double, 2, 2> zero22;
    Eigen::Matrix<double, 3, 3> eye33;
    Eigen::Matrix<double, 3, 3> R_cb = s.offset_R_C_B.toRotationMatrix();
    S2_line u;
    u.vec = R_cb * s.line.vec;
    Eigen::Matrix<double, 3, 3> u_hat;
    Eigen::Matrix<double, 3, 2> u_B;
    u.S2_hat(u_hat);
    u.S2_Bx(u_B);
    zero23 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    zero22 << 0.0, 0.0, 0.0, 0.0;
    zero32 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    zero33 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    eye33 << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    ph_pv.block(0,0,3,5) << eye33, zero32;
    ph_pv.block(3,0,2,5) << zero23, -1/u.length/u.length * u_B.transpose() * u_hat * u_hat * u_B;
    return ph_pv; 
}

void predict_log(std::ofstream &fout_pre, state &s_log, double &step)
{
    Eigen::Vector3d euler_offset = SO3ToEuler(s_log.offset_R_C_B);
    double angle = s_log.grav.vec.dot(s_log.line.vec)/ s_log.grav.length / s_log.line.length;
    fout_pre << step <<  " " << s_log.pos.transpose() << " " << s_log.vel.transpose() \
    << " " << s_log.bg.transpose() << " " << s_log.ba.transpose()<< " " << s_log.grav.vec.transpose() \
    << " " << s_log.offset_P_C_B.transpose() << " " << euler_offset.transpose() << " "  \
    << " " << s_log.line.vec.transpose() << " " << angle << std::endl;
}

void update_log(std::ofstream &fout_out, state &s_log, measurement &measure, double &step)
{
    Eigen::Vector3d euler_offset = SO3ToEuler(s_log.offset_R_C_B);
    double angle = s_log.grav.vec.dot(s_log.line.vec)/ s_log.grav.length / s_log.line.length;
    fout_out << step  << " " << s_log.pos.transpose() << " " << s_log.vel.transpose() \
    << " " << s_log.bg.transpose() << " " << s_log.ba.transpose()<< " " << s_log.grav.vec.transpose() \
    << " " << s_log.offset_P_C_B.transpose() << " " << euler_offset.transpose() << " " << s_log.line.vec.transpose()\
    << " " <<  measure.P_cr.transpose() << " " << measure.line_cr.vec.transpose() << " " << angle << std::endl;
}

void measure_receive(measurement &measure, geometry_msgs::PoseArray::ConstPtr &gap_measure)
{   
    Eigen::Matrix<double, 3, 1> p_c_r;
    Eigen::Matrix<double, 3, 1> p2_c_r;
    p_c_r << gap_measure->poses[0].position.x, gap_measure->poses[0].position.y, gap_measure->poses[0].position.z;
    p2_c_r << gap_measure->poses[1].position.x, gap_measure->poses[1].position.y, gap_measure->poses[1].position.z;
    measure.P_cr = p_c_r;
    measure.line_cr.vec = (p2_c_r - p_c_r).normalized();
}

void state_init(state &init_state, geometry_msgs::PoseArray::ConstPtr &gap_measure)
{
    Eigen::Matrix<double, 3, 1> p_c_r;
    Eigen::Matrix<double, 3, 1> p2_c_r;
    p_c_r << gap_measure->poses[0].position.x, gap_measure->poses[0].position.y, gap_measure->poses[0].position.z;
    p2_c_r << gap_measure->poses[1].position.x, gap_measure->poses[1].position.y, gap_measure->poses[1].position.z;
    init_state.pos = init_state.offset_R_C_B.toRotationMatrix().inverse() * (p_c_r - init_state.offset_P_C_B);
    init_state.line.vec = init_state.offset_R_C_B.toRotationMatrix().inverse() * (p2_c_r - p_c_r).normalized();
}

void topic_pub(state &s, geometry_msgs::PoseStamped &pose, geometry_msgs::PointStamped &point2)
{
    pose.pose.position.x = s.pos[0];
    pose.pose.position.y = s.pos[1];
    pose.pose.position.z = s.pos[2];
}