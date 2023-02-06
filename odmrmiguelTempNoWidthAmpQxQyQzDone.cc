#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <nlopt.hpp>

// g++ -I../espin odmrmiguelTempNoWidth.cc -O3 -lnlopt
// at home 
// g++ -I/home/alexei/Cprograms/phys/espin odmrmiguelTempNoWidth.cc -O3 -lnlopt
// ./a.out 
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <vector>
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <algorithm>
#include <sstream>
#include <string>
#include <Eigen/Dense>

const double MHztoGauss(0.357);
const double opt_exp(1.);


double mAtoGauss(double ib) {
   return 1.149 * ib + 11.;
}


#include "espindense.cc"

typedef Matrix<double, 1, 1> Vector1d;
typedef Matrix<double, 5, 1> Vector5d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 7, 1> Vector7d;
typedef Matrix<double, 8, 1> Vector8d;
typedef Matrix<double, 9, 1> Vector9d;

enum IndexOrder { MZ = 0, MY = 1, MX = 2 };

void load_transition(std::string source_filename, std::vector< std::vector<double> > &data, int &rows, int &colmin, int &colmax) { 
   std::ifstream infile(source_filename.c_str());    
   std::string line;

   rows = 0;
   colmax = 0;
   colmin = -1;
   while (std::getline(infile, line)) {  
     std::istringstream iss(line);
     std::vector< double > linedata;

     float value;
     while (iss >> value) {
        linedata.push_back(value);
     }
     int column_number = linedata.size();
     if (column_number > 0) {
        data.push_back(linedata);
    	rows++;
	    if (column_number > colmax) {
	        colmax = column_number;
	    }
	    if (column_number < colmin || colmin < 0) {
	       colmin = column_number;
	    }
     }     
   }
   std::cout << "# loaded " << rows << std::endl;
}


Vector3d vector_cross(Vector3d a, Vector3d b) { 
    Vector3d z;
    z << a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0);
    return z;
}


struct Tetracene {
   Matrix3d c;

   Vector3d c1(void) const { return c.col(0); }
   Vector3d c2(void) const { return c.col(1); }
   Vector3d c3(void) const { return c.col(2); }

   Tetracene(Matrix3d c_init) { c = c_init; } 

   Vector3d r(void) const { return (c1() + c2())/2.0; }
   Vector3d ux(void) const { return (c2() - c1()).normalized(); }
   Vector3d uy(void) const { return (c3() - c1()).normalized(); }
   Vector3d uz(void) const { return vector_cross( ux(), uy() ); }
  
   Matrix3d rot (void) const { 
      Matrix3d R; 
      R.col(0) = ux();
      R.col(1) = uy();
      R.col(2) = uz();
      return R;
   }
};


struct TIPS_Lattice { 
   static const double a(void)  { return 1.4221; }  // in nm
   static const double b(void) { return 1.5133; } 
   static const double c(void) { return 1.6845; } 

   static const Matrix3d M(void) { return (Vector3d() << a(), b(), c()).finished().asDiagonal(); }
   static const Matrix3d invM(void) { return (Vector3d() << 1.0/a(), 1.0/b(), 1.0/c()).finished().asDiagonal(); }

   static const Tetracene t1;
   static const Tetracene t2;
   static const Tetracene t3;
   static const Tetracene t4;

   static const Matrix3d shift_by_vector(double x0, double x1, double x2) {
      Matrix3d T;
      T << x0, x0, x0, 
	   x1, x1, x1,      
	   x2, x2, x2;
      return M() * T;
   }

   static const Vector3d lattice_vector(int nx, int ny, int nz) { 
     Vector3d l;
     l << a() * (double) nx, b() * (double) ny, c() * (double) nz;
      return l;
   }

   static const Tetracene t(int n) { 
      if (n == 1) return t1;
      else if (n == 2) return t2;
      else if (n == 3) return t3;
      else return t4;
   }
  
   TIPS_Lattice(void) 
   {
   }
};

const Tetracene TIPS_Lattice::t1 ( TIPS_Lattice::M() * (Matrix3d() << 0.5215, 0.4657, 0.5612, 
							              0.5962, 0.5239, 0.5846,
							              0.5746, 0.4736, 0.4217).finished().transpose() );

const Tetracene TIPS_Lattice::t2 ( (Vector3d() << -1.0, -1.0, 1.0).finished().asDiagonal() * t1.c + TIPS_Lattice::shift_by_vector(1.5, 1.0, -0.5) );

const Tetracene TIPS_Lattice::t3 ( (Vector3d() << 1.0, -1.0, -1.0).finished().asDiagonal() * t1.c + TIPS_Lattice::shift_by_vector(-0.5, 1.5, 1.0) );

const Tetracene TIPS_Lattice::t4 ( (Vector3d() << -1.0, 1.0, -1.0).finished().asDiagonal() * t1.c + TIPS_Lattice::shift_by_vector(1.0, -0.5, 1.5) );


struct DataForODMR {
   std::vector< std::vector< std::vector<double> > > data;
   std::vector< double > weights;
   Vector3d uB;
   TripletPair triplet_pair;
   TripletPair::SpinMatrix evecZFH;
   SingleQuintet quintet;
};

class AllowedRange {
   double Bmin;
   double Bmax;
   bool limit_is_set;
public:
   AllowedRange(void) {
      limit_is_set = false;
   }

   void set_limits(double bmin, double bmax) { 
      Bmin = bmin;
      Bmax = bmax;
      limit_is_set = true;
   }

   double upper(void) { return Bmax; } 
   double lower(void) { return Bmin; } 
  
   bool inside_range(double b) {
      if (limit_is_set) return (b >= Bmin) && (b <= Bmax);
      else return true;
   }
};

AllowedRange Blimit;

// typedef Vector6d VectorPL;
// typedef Vector8d VectorGen;

typedef Vector3d VectorPL;
typedef Vector3d VectorGen;

struct FitParameters {
   VectorPL PL;
   VectorGen Gen;
   TripletPair::SpinMatrix evecZFH;
   std::vector < TripletPair::SpinMatrix > quintet_rot_matrix_sq; 

   double itauspin;
   double itaurec;
   double theta;
   double rf;
   double alpha;
   std::vector < std::string > PL_rate_names;
   std::vector < std::string > Gen_rate_names;
 
  FitParameters(const DataForODMR &odata) : itauspin( 4.0 ),
			 alpha( 1.0 ),
			 // PL.Fz2 first part of the name is for readabilty only .Fz2 gives the type of matrix element
			 PL_rate_names( { "PL.Qz2", "PL.Qy2", "PL.Qx2" } ),   
			 Gen_rate_names( { "Gen.Qz2", "Gen.Qy2", "Gen.Qx2" } )
  {
      evecZFH = odata.evecZFH;
      quintet_rot_matrix_sq.resize(3);

      TripletPair::SpinMatrix M;
      M = odata.triplet_pair.Sz_rot(odata.quintet.S.rot);
      quintet_rot_matrix_sq[MZ] = M * M;
      M = odata.triplet_pair.Sy_rot(odata.quintet.S.rot);
      quintet_rot_matrix_sq[MY] = M * M;
      M = odata.triplet_pair.Sx_rot(odata.quintet.S.rot);
      quintet_rot_matrix_sq[MX] = M * M;

     if ( PL.size() != PL_rate_names.size() ) { 
	 std::cerr << "# PL.size() != PL_rate_names.size()" << std::endl;
	 std::cerr << "# " << PL.size() << "   " << PL_rate_names.size() << std::endl;
      }
      if ( Gen.size() != Gen_rate_names.size() ) { 
	 std::cerr << "# Gen.size() != Gen_rate_names.size()" << std::endl;
	 std::cerr << "# " << Gen.size() << "   " << Gen_rate_names.size() << std::endl;
      }
   }
  
   //
   // Number of fit parameters is set here
   //
   int size(void) { return PL.size() + Gen.size() + 4; }

   //
   // Decoding from fit vector into the FitParameters class 
   //
   void set_from_vector(const std::vector<double> &vec)
   { 
      VectorPL pl_rate( vec.data() );
      PL = pl_rate;
      VectorGen generation_rate( vec.data() + pl_rate.size() );
      Gen  = generation_rate;
      theta = vec[ pl_rate.size() + generation_rate.size() ];
      itaurec = vec[ pl_rate.size() + generation_rate.size() + 1];
      rf = vec[ pl_rate.size() + generation_rate.size() + 2];
      alpha = vec[ pl_rate.size() + generation_rate.size() + 3];
   }


   bool with_thermal_generation(void) const {
      return Gen.size() == PL.size() + 2;
   }
  
   std::vector< double > upper_limit(void) {     
     std::vector<double> ub(size(), 2.0);
     //     std::vector<double> ub(size(), 6.0);     
     if (with_thermal_generation()) { 
        ub[ PL.size() + Gen.size() - 1. ] = 0.15; // temperature
     }
     ub[ PL.size() + Gen.size() ] = 2. * M_PI;
     return ub;
   }

   std::vector< double > lower_limit(void) {
      std::vector<double> lb(size(), 0.);
      //      lb[2] = -1.;
      //      lb[5] = -1.;
      if (with_thermal_generation()) { 
         lb[ PL.size() + Gen.size() - 1. ] = -0.15; // temperature
      }
      return lb;
   }

   double rate_for_evec(std::string rate, const TripletPair &tp, const TripletPair::SpinVector &v) const { 
      if (rate.find("Fz2")  != std::string::npos)  // total spin in molecular eigenbasis of molecule 1 
	 return real( static_cast<complexg> (v.adjoint() * tp.Sz_mol1() * tp.Sz_mol1() * v) );	  
      if (rate.find("Fy2")  != std::string::npos) 
	 return real( static_cast<complexg> (v.adjoint() * tp.Sy_mol1() * tp.Sy_mol1() * v) );	  
      if (rate.find("Fx2")  != std::string::npos) 
	 return real( static_cast<complexg> (v.adjoint() * tp.Sx_mol1() * tp.Sx_mol1() * v) );	  

      if (rate.find("Qz2")  != std::string::npos)  // total spin in molecular eigenbasis of molecule 1 
	 return real( static_cast<complexg> (v.adjoint() * quintet_rot_matrix_sq[MZ] * v) );	  
      if (rate.find("Qy2")  != std::string::npos) 
	 return real( static_cast<complexg> (v.adjoint() * quintet_rot_matrix_sq[MY] * v) );	  
      if (rate.find("Qx2")  != std::string::npos) 
	 return real( static_cast<complexg> (v.adjoint() * quintet_rot_matrix_sq[MX] * v) );	  

      
      if (rate.find("Fxz")  != std::string::npos) 
	 return real( static_cast<complexg> (v.adjoint() * (tp.Sx_mol1() * tp.Sz_mol1() + tp.Sz_mol1() * tp.Sx_mol1()) * v) );	  
      if (rate.find("Fyz")  != std::string::npos) 
	 return real( static_cast<complexg> (v.adjoint() * (tp.Sy_mol1() * tp.Sz_mol1() + tp.Sz_mol1() * tp.Sy_mol1()) * v) );	  
      if (rate.find("Fxy")  != std::string::npos) 
	 return real( static_cast<complexg> (v.adjoint() * (tp.Sx_mol1() * tp.Sy_mol1() + tp.Sy_mol1() * tp.Sx_mol1()) * v) );	  
      

      if (rate.find("Sz2")  != std::string::npos) 	 
	 return norm(static_cast<complexg>(v.adjoint() * (tp.Sz1_mol() * tp.Sz1_mol())*tp.singlet())) + norm(static_cast<complexg>(v.adjoint() * (tp.Sz2_mol() * tp.Sz2_mol())*tp.singlet()));
      if (rate.find("Sy2")  != std::string::npos) 
	 return norm(static_cast<complexg>(v.adjoint() * (tp.Sy1_mol() * tp.Sy1_mol())*tp.singlet())) + norm(static_cast<complexg>(v.adjoint() * (tp.Sy2_mol() * tp.Sy2_mol())*tp.singlet()));
      if (rate.find("Sx2")  != std::string::npos) 
	 return norm(static_cast<complexg>(v.adjoint() * (tp.Sx1_mol() * tp.Sx1_mol())*tp.singlet())) + norm(static_cast<complexg>(v.adjoint() * (tp.Sx2_mol() * tp.Sx2_mol())*tp.singlet()));


      if (rate.find("SPz2")  != std::string::npos) 	 
	 return norm(static_cast<complexg>(v.adjoint() * (tp.Sz1_mol() * tp.Sz1_mol() + tp.Sz2_mol() * tp.Sz2_mol() )*tp.singlet()));
      if (rate.find("SPy2")  != std::string::npos) 
	 return norm(static_cast<complexg>(v.adjoint() * (tp.Sy1_mol() * tp.Sy1_mol() + tp.Sy2_mol() * tp.Sy2_mol() )*tp.singlet()));
      if (rate.find("SPx2")  != std::string::npos) 
	 return norm(static_cast<complexg>(v.adjoint() * (tp.Sx1_mol() * tp.Sx1_mol() + tp.Sx2_mol() * tp.Sx2_mol() )*tp.singlet()));

      if (rate.find("PS")  != std::string::npos) 	 
	 return norm(static_cast<complexg>(v.adjoint() * tp.singlet()));

      
      if (rate.find("Sxz")  != std::string::npos) 	 
	 return norm(static_cast<complexg>(v.adjoint() * (
	     tp.Sx1_mol() * tp.Sz1_mol() + tp.Sx2_mol() * tp.Sz2_mol() + tp.Sz1_mol() * tp.Sx1_mol() + tp.Sz2_mol() * tp.Sx2_mol()
							  )*tp.singlet()));
      if (rate.find("Syz")  != std::string::npos) 
	 return norm(static_cast<complexg>(v.adjoint() * (
	     tp.Sy1_mol() * tp.Sz1_mol() + tp.Sy2_mol() * tp.Sz2_mol() + tp.Sz1_mol() * tp.Sy1_mol() + tp.Sz2_mol() * tp.Sy2_mol()
							  )*tp.singlet()));
      if (rate.find("Sxy")  != std::string::npos) 
	 return norm(static_cast<complexg>(v.adjoint() * (
	     tp.Sy1_mol() * tp.Sx1_mol() + tp.Sy2_mol() * tp.Sx2_mol() + tp.Sx1_mol() * tp.Sy1_mol() + tp.Sx2_mol() * tp.Sy2_mol()
							  )*tp.singlet()));  

      if (rate.find("ZFH1")  != std::string::npos) 	 
	 return norm(static_cast<complexg>(v.adjoint() * evecZFH.col(4) ));
      if (rate.find("ZFH2")  != std::string::npos) 	 
	 return norm(static_cast<complexg>(v.adjoint() * evecZFH.col(5) ));
      if (rate.find("ZFH3")  != std::string::npos) 	 
	 return norm(static_cast<complexg>(v.adjoint() * evecZFH.col(6) ));
      if (rate.find("ZFH4")  != std::string::npos) 	 
	 return norm(static_cast<complexg>(v.adjoint() * evecZFH.col(7) ));
      if (rate.find("ZFH5")  != std::string::npos) 	 
	 return norm(static_cast<complexg>(v.adjoint() * evecZFH.col(8) ));

      
      std::cerr << "unmatched rate type" << rate << std::endl;
      exit(0);
   }

   double PL_rate_for_evec(int n, const TripletPair &tp, const TripletPair::SpinVector &v) const { 
      return rate_for_evec( PL_rate_names[n], tp, v );
   }

   double Gen_rate_for_evec(int n, const TripletPair &tp, const TripletPair::SpinVector &v) const { 
      return rate_for_evec( Gen_rate_names[n], tp, v );
   }
  
   void print(void) { 
      std::cout << "# Bmin=" << Blimit.lower() << " Bmax=" << Blimit.upper() << " opt.exp=" << opt_exp << std::endl;
      std::cout << "# ";
      for (int n = 0; n < PL_rate_names.size(); n++)
	 std::cout << PL_rate_names[n] << " ";
      std::cout << std::endl;
      std::cout << PL << std::endl;
      std::cout << "# ";
      for (int n = 0; n < Gen_rate_names.size(); n++)
	 std::cout << Gen_rate_names[n] << " ";
      std::cout << std::endl;
      std::cout << Gen << std::endl;
      std::cout << theta << "   " << "theta" << std::endl;
      std::cout << itaurec << "   " << "itaurec" << std::endl;
      std::cout << itauspin << "   " << "itauspin" << std::endl;
      std::cout << rf << "   " <<  "rf" << std::endl;
      std::cout << alpha << "   " <<  "alpha" << std::endl;
   }
};


class TripletPairODMR {
   const TripletPair &tp;
   VectorXd quintet;    // quintet content 
   double gamma;        // non radiative recombination rate - assumed spin ind ependent 
public:
   VectorXd plrate;     // fluorescence recombination   rates 
   VectorXd alpha;      // photogeneration rates 
   double alpha0;
   TripletPairODMR(const TripletPair &tparg) : tp(tparg),
					       plrate( TripletPair::matrix_size ),
					       alpha( TripletPair::matrix_size ),
					       quintet( TripletPair::matrix_size )
   { 
   }

   void update_rates(const FitParameters &Fit)
   {
      gamma = Fit.itaurec;
      alpha0 = Fit.alpha;
      for (int n = 0; n < TripletPair::matrix_size; n++) {
	quintet(n) = tp.quintet_content(n);
	plrate(n) = 0;
	alpha(n) = 0;
	for (unsigned int q=0; q < Fit.PL.size(); q++) { 
	  plrate(n) += Fit.PL(q) * Fit.PL_rate_for_evec(q, tp, tp.evec.col(n) )  ;
	}
	for (unsigned int q=0; q < Fit.PL.size(); q++) { 
	  if (quintet(n) > 0.9) {
	     alpha(n) += Fit.Gen(q) * Fit.Gen_rate_for_evec(q, tp, tp.evec.col(n) );
	  }
	}
      }

      if ( Fit.with_thermal_generation() ) {
	 double eav = 0.0;
	 int neav = 0;
	 for (int n = 0; n < TripletPair::matrix_size; n++) {
	   if (quintet(n) > 0.9) {
	     eav += tp.eval(n);
	     neav++;
	   }
	 }
	 eav /= (double) neav;
	 
	 int ngen = Fit.Gen.size();
	 
	 double Z = 0.0;
	 for (int n = 0; n < TripletPair::matrix_size; n++) {
	   if (quintet(n) > 0.9) {
	     Z += exp(-Fit.Gen(ngen - 1) * (tp.eval(n) - eav) );
	   }
	 }
	 
	 
	 for (int n = 0; n < TripletPair::matrix_size; n++) {
	   if (quintet(n) > 0.9) {
	     alpha(n) += ( Fit.Gen(ngen - 2) * exp( -Fit.Gen(ngen - 1) * (tp.eval(n) - eav) ) / Z );
	   } 
	 }
      }

   }

  
   double Gamma(int n) {
      return plrate(n) + gamma;
   }

   double PL(void) { 
      double pl = 0;
      for (int n = 0; n < TripletPair::matrix_size; n++) {
	 if (quintet(n) > 0.9) {
	    pl += plrate(n) * (alpha(n) + alpha0) / ( plrate(n) + gamma );
	 } 
      }
      return pl;
   }

   double PL(const FitParameters &Fit) {
      update_rates(Fit);
      return PL();
   }
  
   double odmr(double omega, double mw_pow, double itau, double theta, double alpha0) { 
      double s = 0;
      for (int n = 0; n < tp.size(); n++) {
	for (int m = n+1; m < tp.size(); m++) { 
	  if (quintet(n) > 0.9 && quintet(m) > 0.9) {  // computes signal only from quintets 
	    double Sxy2 = norm( tp.Sx(n, m) * cos(theta) + tp.Sy(n, m) * sin(theta) );
	    double Amn = mw_pow * Sxy2 /
	      ( itau*itau +
		//		+ 0.5 * ( norm(Gamma(n)) + norm(Gamma(m)) )
		+ norm(omega - tp.eval(m) + tp.eval(n))
		//		+ mw_pow * Sxy2
		);

	    s += Amn * (
			alpha0 * norm(Gamma(n) - Gamma(m)) / (Gamma(n)*Gamma(m))
			+ alpha(n) * (Gamma(m) - Gamma(n)) / Gamma(n)
			+ alpha(m) * (Gamma(n) - Gamma(m)) / Gamma(m)
			);
	  }
	}
      }
      return s;      
   }


   std::complex<double> odmr_am(double omega, double omega_am, double mw_pow, double itau, double theta, double alpha0) { 
      std::complex<double> s = 0;
      for (int n = 0; n < tp.size(); n++) {
	for (int m = n+1; m < tp.size(); m++) { 
	  if (quintet(n) > 0.9 && quintet(m) > 0.9) {  // computes signal only from quintets 
	    double Sxy2 = norm( tp.Sx(n, m) * cos(theta) + tp.Sy(n, m) * sin(theta) );
	    double Amn = mw_pow * Sxy2 /
	      ( itau*itau +
		//		+ 0.5 * ( norm(Gamma(n)) + norm(Gamma(m)) )
		+ norm(omega - tp.eval(m) + tp.eval(n))
		//		+ mw_pow * Sxy2
		);

	    double Pn = (alpha0 + alpha(n))/Gamma(n);
	    double Pm = (alpha0 + alpha(m))/Gamma(m);
	    
	    s += Amn * ( Gamma(n)*Gamma(n) / (Gamma(n) + iii * omega_am ) -
			 Gamma(m)*Gamma(m) / (Gamma(m) + iii * omega_am ) 			 
			 ) * (Pn - Pm) / 2.;
	  }
	}
      }
      return s;      
   }

  
   double odmr(double omega, const FitParameters &Fit) { 
      update_rates(Fit);
      return odmr(omega, Fit.rf, Fit.itauspin, Fit.theta, Fit.alpha);
   }
  
   std::complex<double> odmr_am(double omega, double omega_am, const FitParameters &Fit) { 
      update_rates(Fit);
      return odmr_am(omega, omega_am, Fit.rf, Fit.itauspin, Fit.theta, Fit.alpha);
   }
  

   // very specific to quintets 
   Matrix<double, 5, 1> spin_pop(void) {
      Matrix<double, 5, 1> Spop;
      for (int n = 4; n < 9; n++) {
	 Spop(n-4) = (alpha0 + alpha(n))/Gamma(n);
      }
      return Spop;
   }

};



std::vector< std::vector<double> > Evaltheo(std::vector< std::vector<double> > &data, Vector3d &uB, TripletPair &triplet_pair, FitParameters &Fit){
  
   std::vector< std::vector<double> > theo( data.size() );
   for (int i = 0; i < theo.size(); i++) theo[i].resize(3);
  
   for (int ni = 0; ni < data.size(); ni++) {
      double b=mAtoGauss(data[ni][0])/MHztoGauss;
     triplet_pair.S1.B = b * uB;
     triplet_pair.S2.B = b * uB;
     triplet_pair.update_hamiltonian();
     triplet_pair.diag();
     TripletPairODMR tpo(triplet_pair);

     double delta_omega; 
     bool initializing = true;
     int Nn, Nm;
     for (int n = 4; n < TripletPair::matrix_size; n += 1) {        
       for (int m = n+1; m < TripletPair::matrix_size; m += 1) { 
	   double omega =  triplet_pair.eval(m) - triplet_pair.eval(n);	   
	   if (initializing || fabs(data[ni][2]-omega)<fabs(data[ni][2]-delta_omega)) {
	       delta_omega=omega;
	       Nn = n;
	       Nm = m;
	       initializing = false;
	   }
	}
     }

     theo[ni][2]=delta_omega;
     theo[ni][1]= tpo.odmr(delta_omega, Fit)*2.e4;
     theo[ni][0]= Fit.itauspin; // don't attempt to fit the transition linewidth 
   }

   return theo;  
}




Vector2d Evalerror2(std::vector< std::vector<double> > &data, Vector3d &uB, TripletPair &triplet_pair, FitParameters &Fit){
  std::vector< std::vector<double> > Theo = Evaltheo(data, uB, triplet_pair, Fit);
   double error=0.;
   double Werror=0.;
   
   for (int ni = 0; ni < data.size(); ni++) {  
     if ( Blimit.inside_range( mAtoGauss(data[ni][0]) ) ) { 
       error += pow( fabs(data[ni][1]-Theo[ni][1]), opt_exp );
       Werror += pow( fabs(data[ni][3]-Theo[ni][0]), opt_exp );
     }
   }
   Vector2d v;
   v << error, Werror;
   return v;
}


double minS(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
    static int count = 0;

    bool print = !(count % 100);
    
    DataForODMR *odmr_data = reinterpret_cast<DataForODMR *>(data);
    FitParameters Fit(*odmr_data);
    Fit.set_from_vector(x);

    double error = 0;
    if (print) { 
      std::cout << "##########################" << std::endl;
    }
    for (int n = 0; n < odmr_data->data.size(); n++) { 
      Vector2d v = Evalerror2( odmr_data->data[n], odmr_data->uB, odmr_data->triplet_pair, Fit);
      // v(0) amplitude error
      // v(1) width error 
      double err = v(0) * odmr_data->weights[n]/ (double) odmr_data->data[n].size();
      /***
      double werr;
      if (n == 4) {
	 werr =  v(1) * odmr_data->weights[n]/ (double) odmr_data->data[n].size();
	 error += werr;
      }
      ****/
      if (print) { 
	std::cout << "#err " << n << "   " << err << "   " << odmr_data->data[n].size() << std::endl;
	//	if (n == 4) 	std::cout << "#Werr " << n << "   " << werr << "   " << odmr_data->data[n].size() << std::endl;
      }
      error += err ;
    }
    if (print) { 
      Fit.print();
      std::cout << count << "    " << error << " ERR " << std::endl;
    }
    count++;
    return error;
}


std::vector< std::vector< std::vector<double> > > Evaltheo(const std::vector<double> &x, DataForODMR *odmr_data)
{
    FitParameters Fit(*odmr_data);
    Fit.set_from_vector(x);
    std::vector< std::vector< std::vector<double> > >  Theo;

    for (int n = 0; n < odmr_data->data.size(); n++) {     
      Theo.push_back( Evaltheo( odmr_data->data[n], odmr_data->uB, odmr_data->triplet_pair, Fit) );
    }
    return Theo;
}



int main(int argc, char **argv)
{   
  // 0.04 0.04 1 1450 0
   double ux = 0.04;
   double uy = 0.04;
   double uz = 1.0;

   DataForODMR odata;
   std::vector< std::string > files { "freqampwidthVsBpeak41.dat", "freqampwidthVsBpeak42.dat", "freqampwidthVsBpeak3.dat", "freqampwidthVsBpeak2.dat", "freqampwidthVsBpeak1.dat", "freqampwidthVsBpeak51.dat", "freqampwidthVsBpeak52.dat" , "freqampwidthVsBpeak53.dat",  "freqampwidthVsBpeak6.dat" };


   std::vector< double > weights { 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };   
   
   int Nfiles = files.size();
   odata.data.resize(Nfiles);
   odata.weights.resize(Nfiles);


   std::cout << opt_exp << std::endl;
   
   for (int n = 0; n < Nfiles; n++) {
     int rows, colmin, colmax;
     load_transition(files[n].c_str(), odata.data[n], rows, colmin, colmax);
     odata.weights[n] = weights[n];
   }
   
   
   for (int n = 0; n < Nfiles; n++) {
      std::cout << "# weight" << n << "   " << odata.weights[n] << std::endl;
   }
   
   
   odata.uB << ux, uy, uz;
   odata.uB.normalize();
    
    
   double D, E, J;
   D = 1450.0;
   E = 0;
   J = 1e7;
   
   
   double GammaD = 51.92;
   double Bz = 3000.0;

   int t1, t2;
   t1 = 1;
   t2 = 2;
	
   Vector3d r12_inside_cell = TIPS_Lattice::t(t1).r() - TIPS_Lattice::t(t2).r();
   Vector3d r12 = r12_inside_cell;
   for (int nx = -1; nx <= 1; nx++) {
     for (int ny = -1; ny <= 1; ny++) {
       for (int nz = -1; nz <= 1; nz++) {
	 Vector3d rxyz = r12_inside_cell + TIPS_Lattice::lattice_vector(nx, ny, nz);
	 if (rxyz.norm() < r12.norm()) { 
	   r12 = rxyz;
	 }
       }
     }
   }
	 
   Vector3d u12 = r12.normalized();
   double d12 = r12.norm(); 	 

   Vector3d Bhf1 = 0.0 * random_unit_vector();
   Vector3d Bhf2 = 0.0 * random_unit_vector();

   odata.triplet_pair.S1.D = D;
   odata.triplet_pair.S2.D = D;
   odata.triplet_pair.S1.E = E;
   odata.triplet_pair.S2.E = E;
   odata.triplet_pair.S1.rot = TIPS_Lattice::t(t1).rot().transpose();
   odata.triplet_pair.S2.rot = TIPS_Lattice::t(t2).rot().transpose();
   odata.triplet_pair.r12 << u12(0), u12(1), u12(2);
   odata.triplet_pair.J = J;
   odata.triplet_pair.Jdip = GammaD / (d12*d12*d12);
   
   //   std::cout << Evalerror(data, uB, odata.triplet_pair, pl_rate, generation_rate);
   
   odata.triplet_pair.S1.B = 0. * odata.uB;
   odata.triplet_pair.S2.B = 0. * odata.uB;
   odata.triplet_pair.update_hamiltonian();
   odata.triplet_pair.diag(); 
   odata.evecZFH = odata.triplet_pair.evec;

   std::cout << "# Filling F12 " << std::endl;
   Matrix3d F12 = ( odata.triplet_pair.S1.fine_structure_tensor() + odata.triplet_pair.S2.fine_structure_tensor() )/ 6.0; 
   Matrix3d I3 = Matrix3d::Identity();
   F12 -= ( GammaD / (d12*d12*d12) * ( u12 * u12.transpose() - I3 / 3.0 ) );
   std::cout << "# F12 done " << std::endl;
   
   SelfAdjointEigenSolver<Matrix3d> eigensolver(F12);
   MatrixXd evec = eigensolver.eigenvectors();
   Vector3d eval = eigensolver.eigenvalues();
   double DQ = eval(2) - (eval(0) + eval(1))/2.0;
   double EQ = (eval(1) - eval(0))/2.0;
   
   Vector3d swap = evec.col(0);
   evec.col(0) = evec.col(1);
   evec.col(1) = swap;
   double xyz = evec.col(2).transpose() * vector_cross( evec.col(0), evec.col(1) );
   if (xyz < 0) { 
     evec.col(0) = -evec.col(0);
   }
   
   odata.quintet.S.D = DQ;
   odata.quintet.S.E = EQ;
   odata.quintet.S.rot = evec.transpose();
   
   
   FitParameters Fit(odata);

   Blimit.set_limits( -50, 150 ) ; 

   /*** parameters for itauspin = 3
   std::vector<double> xinit {  1.92067, 0.313912,   0,
				1.93749, 0.536845,   0. * 0.0314619,
				4.73669, 0, 0.0512687, 0 
   };
   ****/
   // parameters for itauspin = 4 
   std::vector<double> xinit { 1.99954, 0.301644, 0 * 1.28508e-05,
			       1.99987, 0.396984, 0 * 0.01553,
			       4.78889, 0 * 2.99182e-05, 0.212169, 0 };

   Fit.set_from_vector(xinit);
   

   std::cout << "# evaluting starting error " << std::endl;
   std::cout << minS( xinit, xinit, reinterpret_cast<void *>(&odata)) << std::endl;


   TripletPairODMR tpo(odata.triplet_pair);
   std::cout << "# PL RATES " << std::endl;
   std::cout << tpo.plrate << std::endl;
   std::cout << std::endl;
   std::cout << "# Generation RATES " << std::endl;
   std::cout << tpo.alpha << std::endl;
   std::cout << std::endl;

   //   exit(0);
   
    int NOPT = xinit.size();
    nlopt::opt opt(nlopt::GN_CRS2_LM, NOPT);
    //    nlopt::opt opt(nlopt::LN_COBYLA, NOPT);    
    std::vector<double> lb;
    lb = Fit.lower_limit();
    std::vector<double> ub;
    ub = Fit.upper_limit();
    
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_min_objective(minS, reinterpret_cast<void *>(&odata));
    opt.set_xtol_rel(1e-5);
    //    opt.set_maxtime(1000.);
    opt.set_maxtime(1);    
    
    double minf;

    //    Blimit.set_limits( 50, 1000 ) ;    

    try {
      opt.optimize(xinit, minf);
    }    
    catch(std::exception &e) {
        std::cerr << "nlopt failed: " << e.what() << std::endl;
    }
    
    std::cout << "# optimization finished " << std::endl;
    std::cout << "# lowest error " << minf << std::endl;
    Fit.set_from_vector(xinit);
    Fit.print();
   
   std::vector < std::vector < std::vector<double> > > theo = Evaltheo( xinit, &odata );
   std::cout << "% " << odata.data.size() << std::endl;
   std::cout << "% " << odata.data[0].size() << std::endl;
   std::cout << "% " << odata.data[1].size() << std::endl;
   std::cout << "% " << theo[0].size() << std::endl;
   std::cout << "% " << theo[1].size() << std::endl;

   
   for (int f = 0; f < odata.data.size(); f++)  {
     for (int n = 0; n < odata.data[f].size(); n++) {
       if ( Blimit.inside_range ( mAtoGauss(odata.data[f][n][0]) ) ) {
	 std::cout << theo[f][n][2]  << "   "<< theo[f][n][1] << "    " << odata.data[f][n][2] << "   " << odata.data[f][n][1] << "   " << mAtoGauss(odata.data[f][n][0]) << " R" << f << std::endl;
       }
     }
   }



   
   Fit.set_from_vector(xinit);
   double b = 0.0;
   odata.triplet_pair.S1.B = b * odata.uB;
   odata.triplet_pair.S2.B = b * odata.uB;
   odata.triplet_pair.update_hamiltonian();
   odata.triplet_pair.diag(); 
   odata.quintet.S.B = b * odata.uB;
   odata.quintet.update_hamiltonian();
   odata.quintet.diag(); 

   std::cout << "# AMFREQ s12 : " << odata.triplet_pair.eval(5) - odata.triplet_pair.eval(4) << std::endl;
   std::cout << "# AMFREQ s13 : " << odata.triplet_pair.eval(6) - odata.triplet_pair.eval(4) << std::endl;
   std::cout << "# AMFREQ GAMMA(1) : " << tpo.plrate(4) << std::endl;
   std::cout << "# AMFREQ GAMMA(2) : " << tpo.plrate(5) << std::endl;
   std::cout << "# AMFREQ GAMMA(3) : " << tpo.plrate(6) << std::endl;
   for (double omega_am = 0.01; omega_am < 50.; omega_am += 0.02) {
     std::complex<double> s12 = tpo.odmr_am( odata.triplet_pair.eval(5) - odata.triplet_pair.eval(4) , omega_am, Fit);
     std::complex<double> s13 = tpo.odmr_am( odata.triplet_pair.eval(6) - odata.triplet_pair.eval(4) , omega_am, Fit);
     std::cout << omega_am << "    " << abs(s12) << "   " << abs(s13) << " AMFREQ " << std::endl;
   }


   
   for (b = -200.0; b < Bz; b += 2.0) { 
     odata.triplet_pair.S1.B = b * odata.uB;
     odata.triplet_pair.S2.B = b * odata.uB;
     odata.triplet_pair.update_hamiltonian();
     odata.triplet_pair.diag(); 

     odata.quintet.S.B = b * odata.uB;
     odata.quintet.update_hamiltonian();
     odata.quintet.diag(); 

     TripletPairODMR tpo(odata.triplet_pair);
     for (double omega = 0.01 * D; omega <= 2.5 * D; omega += 0.003 * D) {
       if ( Blimit.inside_range( b * MHztoGauss ) ) {
	double odmramp = tpo.odmr(omega, Fit);
	  std::cout << omega << "    " << b * MHztoGauss << "    " << odmramp << " M " << std::endl;
       }
     }

     if ( Blimit.inside_range( b * MHztoGauss ) ) {
        Matrix<double, 5, 1> pop = tpo.spin_pop();
	cout << b  * MHztoGauss << "  " << pop(0) << "  " << pop(1) << "  " << pop(2) << "   " << pop(3) << "   " << pop(4) << "  " << " POP " << std::endl;
	cout << b  * MHztoGauss << "  "
	     << odata.triplet_pair.eval(4)-J << "  "
	     << odata.triplet_pair.eval(5)-J << "  "
	     << odata.triplet_pair.eval(6)-J << "  "
	     << odata.triplet_pair.eval(7)-J << "  "
	     << odata.triplet_pair.eval(8)-J << "  " << " EVAL " << std::endl;
	cout << b  * MHztoGauss << "  "
	     << odata.quintet.eval(0) << "  "
	     << odata.quintet.eval(1) << "  "
	     << odata.quintet.eval(2) << "  "
	     << odata.quintet.eval(3) << "  "
	     << odata.quintet.eval(4) << "  " << " QUINTET " << std::endl;       
	cout << b  * MHztoGauss << "  "
	     << tpo.plrate(4) << "  "
	     << tpo.plrate(5) << "  "
	     << tpo.plrate(6) << "  "
	     << tpo.plrate(7) << "  "
	     << tpo.plrate(8) << "  " << " GAMMA " << std::endl;
	cout << b  * MHztoGauss << "  "
	     << tpo.alpha(4) << "  "
	     << tpo.alpha(5) << "  "
	     << tpo.alpha(6) << "  "
	     << tpo.alpha(7) << "  "
	     << tpo.alpha(8) << "  " << " ALPHA " << std::endl;
        std::cout << " M " << std::endl;
     }
   }

   double deltaB = 1.0;
   double prev_pl = -1;
   for (double b = -200.0; b < 5000; b += deltaB) { 
     odata.triplet_pair.S1.B = b * odata.uB;
     odata.triplet_pair.S2.B = b * odata.uB;
     odata.triplet_pair.update_hamiltonian();
     odata.triplet_pair.diag(); 

     odata.quintet.S.B = b * odata.uB;
     odata.quintet.update_hamiltonian();
     odata.quintet.diag(); 
     TripletPairODMR tpo(odata.triplet_pair);
     double pl = tpo.PL(Fit);
     if (prev_pl > 0) {
	cout << b * MHztoGauss << "   " << pl << "   " << (pl- prev_pl) / (MHztoGauss * deltaB) << " PLTOT " << std::endl;
     }
     prev_pl = pl;
   }

}


