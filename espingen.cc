#include <Eigen/Dense>

#ifndef ESPINGEN
#define ESPINGEN

using namespace Eigen;
using namespace std;
typedef std::complex<double> complexg;
const complexg iii(0,1);
double myrand(void) { return (double) rand() / (double) RAND_MAX; }

Vector3d random_unit_vector(void) { 
   double phi = 2.0 * M_PI * myrand();
   double z = 2.0 * myrand() - 1.0;
   double r = sqrt(1.0 - z*z);

   Vector3d u; 
   u[2] = z;
   u[1] = r * sin(phi);
   u[0] = r * cos(phi);
   return u;
}


class Rotation {

    Matrix3d euler_matrix_z1x2z3(double z1, double x2, double z3) { 
       double cosz1 = cos(z1);
       double sinz1 = sin(z1);
       Matrix3d Z1;
       Z1 << cosz1, -sinz1, 0.0,
	 sinz1, cosz1, 0.0,
	 0.0, 0.0, 1.0;

       double cosx2 = cos(x2);
       double sinx2 = sin(x2);
       Matrix3d X2;
       X2 << 1.0, 0.0, 0.0,
	 0.0, cosx2, -sinx2,
	 0.0, sinx2, cosx2;


       double cosz3 = cos(z3);
       double sinz3 = sin(z3);
       Matrix3d Z3;
       Z3 << cosz3, -sinz3, 0.0,
	 sinz3, cosz3, 0.0,
	 0.0, 0.0, 1.0;

       return Z1 * X2 * Z3;
    }

    void mat2euler(const Matrix3d &M, Vector3d &vec) {
       std::numeric_limits<double> double_limit;
       double sy_thresh = double_limit.min() * 4.;
       double sy = sqrt(M(2,0)*M(2,0) + M(2,1)*M(2,1));
       double z1, x2, z3;
       if (sy > sy_thresh) { 
	  x2 = acos(M(2,2));
	  z1 = atan2(M(0,2), -M(1,2));
	  z3 = atan2(M(2,0), M(2,1));
       } else {
	  x2 = 0;
	  z3 = 0;
	  z1 = atan2(M(1,0), M(1,1));
       }
       vec << z1, x2, z3;
    }

    Vector3d angles;
    Matrix3d M;

    void init_from_angles(double z1, double x2, double z3) { 
       angles << z1, x2, z3;
       M = euler_matrix_z1x2z3(z1, x2, z3);
    }

    void init_from_matrix(const Matrix3d &Mnew) {
       M = Mnew;
       mat2euler(M, angles);
    }

public : 

    const Matrix3d &matrix(void) const {
       return M;
    }

    const double matrix(int i, int j) const { 
       return M(i, j);
    }
  
    const Matrix3d &matrix(const Matrix3d &Mnew) {
       init_from_matrix(Mnew);
       return M;
    }

    Rotation(void) { 
       init_from_angles(0, 0, 0);
    }

    Rotation(double z1, double x2, double z3) { 
       init_from_angles(z1, x2, z3);
    }

    Rotation(const Vector3d &v) { 
       init_from_angles(v[0], v[1], v[2]);
    }

    Rotation(const Matrix3d &Mnew) { 
       init_from_matrix(Mnew);
    }

    Rotation& operator = (const Matrix3d &Mnew) { 
       init_from_matrix(Mnew);
       return *this;
    }

    const Vector3d &euler_angles(void) {
       return angles;
    }

    const Vector3d &euler_angles(double z1, double x2, double z3) {
       init_from_angles(z1, x2, z3);
       return angles;
    }

    const Vector3d &euler_angles(const Vector3d &v) {
       init_from_angles(v[0], v[1], v[2]);
       return angles;
    }        


    // allows assignements like 
    // Rotation R =  (Rotation::Y(phi) * Rotation::X(theta)).eval();
    static Matrix3d X(double angle) { 
       Vector3d ux;
       ux << 1.0, 0.0, 0.0;
       Matrix3d R;
       R = AngleAxis<double> ( angle, ux);
       return R;
    }

    static Matrix3d Y(double angle) { 
       Vector3d uy;
       uy << 0.0,1.0,0.0;
       Matrix3d R;
       R = AngleAxis<double> ( angle, uy);
       return R;
    }

    static Matrix3d Z(double angle) { 
       Vector3d uz;
       uz << 0.0,0.0,1.0;
       Matrix3d R;
       R = AngleAxis<double> ( angle, uz);
       return R;
    }

    void angle_and_direction(double theta, const Vector3d &v) { 
       M = AngleAxis<double> ( theta, v );
       mat2euler(M, angles);
    }

    void angle_phi_uz(double theta1, double phi1, double uz1) { 
       Vector3d u1;
       u1 << cos(phi1) * sqrt(1. - uz1*uz1), sin(phi1) * sqrt(1. - uz1*uz1), uz1;
       angle_and_direction(theta1, u1);
    }

    void random(void) { 
      double theta = acos( 2.0 * myrand() - 1.0 );
      double phi1 = 2.0 * M_PI * myrand();
      double phi2 = 2.0 * M_PI * myrand();      
      init_from_angles(phi1, theta, phi2);
    }
};


#endif

