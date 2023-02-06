#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>
#include <boost/scoped_ptr.hpp>

#ifndef ESPINDENSE
#define ESPINDENSE

#include "espingen.cc"


struct Spin0p5Matrix { 
    enum { matrix_size = 2 };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    typedef Matrix<double, matrix_size, matrix_size>  SpinMatrixReal;
    static const SpinMatrixReal rsx;
    static const SpinMatrixReal isy;
    static const SpinMatrixReal rsz;
    static const SpinMatrixReal rid;
    static const SpinMatrixReal rsp;
    static const SpinMatrixReal rsm;
    static const SpinMatrix sx;
    static const SpinMatrix sy;
    static const SpinMatrix sz;
    static const SpinMatrix id;

    const SpinMatrixReal& Sx(void) const { return rsx; }
    const SpinMatrix& Sy(void) const { return sy; }
    const SpinMatrixReal& Sz(void) const { return rsz; }
    const SpinMatrixReal& Sp(void) const { return rsp; }
    const SpinMatrixReal& Sm(void) const { return rsm; }
    const SpinMatrixReal& Id(void) const { return rid; }
};
const Spin0p5Matrix::SpinMatrixReal Spin0p5Matrix::rsx ( ( (SpinMatrixReal() <<  0, 1./2., 1./2., 0).finished() ) );
const Spin0p5Matrix::SpinMatrixReal Spin0p5Matrix::isy ( ( (SpinMatrixReal() <<  0, - 1./2., 1./2., 0).finished() ) );
const Spin0p5Matrix::SpinMatrixReal Spin0p5Matrix::rsz ( ( (SpinMatrixReal() <<  1./2., 0, 0, -1./2.).finished() ) );
const Spin0p5Matrix::SpinMatrixReal Spin0p5Matrix::rid (  (SpinMatrixReal() = SpinMatrixReal::Identity()) );
const Spin0p5Matrix::SpinMatrixReal Spin0p5Matrix::rsp (  (SpinMatrixReal() = Spin0p5Matrix::rsx + Spin0p5Matrix::isy) );
const Spin0p5Matrix::SpinMatrixReal Spin0p5Matrix::rsm (  (SpinMatrixReal() = Spin0p5Matrix::rsx - Spin0p5Matrix::isy) );
const Spin0p5Matrix::SpinMatrix Spin0p5Matrix::sx (  (SpinMatrix() = Spin0p5Matrix::rsx) );
const Spin0p5Matrix::SpinMatrix Spin0p5Matrix::sy (  (SpinMatrix() = -iii * Spin0p5Matrix::isy) );
const Spin0p5Matrix::SpinMatrix Spin0p5Matrix::sz (  (SpinMatrix() = Spin0p5Matrix::rsz) );
const Spin0p5Matrix::SpinMatrix Spin0p5Matrix::id (  (SpinMatrix() = Spin0p5Matrix::rid) );

struct Spin1Matrix { 
    enum { matrix_size = 3 };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    typedef Matrix<double, matrix_size, matrix_size>  SpinMatrixReal;
    static const SpinMatrixReal rsx;
    static const SpinMatrixReal isy;
    static const SpinMatrixReal rsz;
    static const SpinMatrixReal rid;
    static const SpinMatrixReal rsp;
    static const SpinMatrixReal rsm;
    static const SpinMatrix sx;
    static const SpinMatrix sy;
    static const SpinMatrix sz;
    static const SpinMatrix id;

    const SpinMatrixReal& Sx(void) const { return rsx; }
    const SpinMatrix& Sy(void) const { return sy; }
    const SpinMatrixReal& Sz(void) const { return rsz; }
    const SpinMatrixReal& Sp(void) const { return rsp; }
    const SpinMatrixReal& Sm(void) const { return rsm; }
    const SpinMatrixReal& Id(void) const { return rid; }
};
const Spin1Matrix::SpinMatrixReal Spin1Matrix::rsx ( ( (SpinMatrixReal() <<  0, sqrt(1./2.), 0, sqrt(1./2.), 0, sqrt(1./2.), 0, sqrt(1./2.), 0).finished() ) );
const Spin1Matrix::SpinMatrixReal Spin1Matrix::isy ( ( (SpinMatrixReal() <<  0, - sqrt(1./2.), 0, sqrt(1./2.), 0, - sqrt(1./2.), 0, sqrt(1./2.), 0).finished() ) );
const Spin1Matrix::SpinMatrixReal Spin1Matrix::rsz ( ( (SpinMatrixReal() <<  1, 0, 0, 0, 0, 0, 0, 0, -1).finished() ) );
const Spin1Matrix::SpinMatrixReal Spin1Matrix::rid (  (SpinMatrixReal() = SpinMatrixReal::Identity()) );
const Spin1Matrix::SpinMatrixReal Spin1Matrix::rsp (  (SpinMatrixReal() = Spin1Matrix::rsx + Spin1Matrix::isy) );
const Spin1Matrix::SpinMatrixReal Spin1Matrix::rsm (  (SpinMatrixReal() = Spin1Matrix::rsx - Spin1Matrix::isy) );
const Spin1Matrix::SpinMatrix Spin1Matrix::sx (  (SpinMatrix() = Spin1Matrix::rsx) );
const Spin1Matrix::SpinMatrix Spin1Matrix::sy (  (SpinMatrix() = -iii * Spin1Matrix::isy) );
const Spin1Matrix::SpinMatrix Spin1Matrix::sz (  (SpinMatrix() = Spin1Matrix::rsz) );
const Spin1Matrix::SpinMatrix Spin1Matrix::id (  (SpinMatrix() = Spin1Matrix::rid) );

struct Spin1p5Matrix { 
    enum { matrix_size = 4 };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    typedef Matrix<double, matrix_size, matrix_size>  SpinMatrixReal;
    static const SpinMatrixReal rsx;
    static const SpinMatrixReal isy;
    static const SpinMatrixReal rsz;
    static const SpinMatrixReal rid;
    static const SpinMatrixReal rsp;
    static const SpinMatrixReal rsm;
    static const SpinMatrix sx;
    static const SpinMatrix sy;
    static const SpinMatrix sz;
    static const SpinMatrix id;

    const SpinMatrixReal& Sx(void) const { return rsx; }
    const SpinMatrix& Sy(void) const { return sy; }
    const SpinMatrixReal& Sz(void) const { return rsz; }
    const SpinMatrixReal& Sp(void) const { return rsp; }
    const SpinMatrixReal& Sm(void) const { return rsm; }
    const SpinMatrixReal& Id(void) const { return rid; }
};
const Spin1p5Matrix::SpinMatrixReal Spin1p5Matrix::rsx ( ( (SpinMatrixReal() <<  0, sqrt(3.)/2., 0, 0, sqrt(3.)/2., 0, 1., 0, 0, 1., 0, sqrt(3.)/2., 0, 0, sqrt(3.)/2., 0).finished() ) );
const Spin1p5Matrix::SpinMatrixReal Spin1p5Matrix::isy ( ( (SpinMatrixReal() <<  0, - sqrt(3.)/2., 0, 0, sqrt(3.)/2., 0, - 1., 0, 0, 1., 0, - sqrt(3.)/2., 0, 0, sqrt(3.)/2., 0).finished() ) );
const Spin1p5Matrix::SpinMatrixReal Spin1p5Matrix::rsz ( ( (SpinMatrixReal() <<  3./2., 0, 0, 0, 0, 1./2., 0, 0, 0, 0, -1./2., 0, 0, 0, 0, -3./2.).finished() ) );
const Spin1p5Matrix::SpinMatrixReal Spin1p5Matrix::rid (  (SpinMatrixReal() = SpinMatrixReal::Identity()) );
const Spin1p5Matrix::SpinMatrixReal Spin1p5Matrix::rsp (  (SpinMatrixReal() = Spin1p5Matrix::rsx + Spin1p5Matrix::isy) );
const Spin1p5Matrix::SpinMatrixReal Spin1p5Matrix::rsm (  (SpinMatrixReal() = Spin1p5Matrix::rsx - Spin1p5Matrix::isy) );
const Spin1p5Matrix::SpinMatrix Spin1p5Matrix::sx (  (SpinMatrix() = Spin1p5Matrix::rsx) );
const Spin1p5Matrix::SpinMatrix Spin1p5Matrix::sy (  (SpinMatrix() = -iii * Spin1p5Matrix::isy) );
const Spin1p5Matrix::SpinMatrix Spin1p5Matrix::sz (  (SpinMatrix() = Spin1p5Matrix::rsz) );
const Spin1p5Matrix::SpinMatrix Spin1p5Matrix::id (  (SpinMatrix() = Spin1p5Matrix::rid) );

struct Spin2Matrix { 
    enum { matrix_size = 5 };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    typedef Matrix<double, matrix_size, matrix_size>  SpinMatrixReal;
    static const SpinMatrixReal rsx;
    static const SpinMatrixReal isy;
    static const SpinMatrixReal rsz;
    static const SpinMatrixReal rid;
    static const SpinMatrixReal rsp;
    static const SpinMatrixReal rsm;
    static const SpinMatrix sx;
    static const SpinMatrix sy;
    static const SpinMatrix sz;
    static const SpinMatrix id;

    const SpinMatrixReal& Sx(void) const { return rsx; }
    const SpinMatrix& Sy(void) const { return sy; }
    const SpinMatrixReal& Sz(void) const { return rsz; }
    const SpinMatrixReal& Sp(void) const { return rsp; }
    const SpinMatrixReal& Sm(void) const { return rsm; }
    const SpinMatrixReal& Id(void) const { return rid; }
};
const Spin2Matrix::SpinMatrixReal Spin2Matrix::rsx ( ( (SpinMatrixReal() <<  0, 1., 0, 0, 0, 1., 0, sqrt(3./2.), 0, 0, 0, sqrt(3./2.), 0, sqrt(3./2.), 0, 0, 0, sqrt(3./2.), 0, 1., 0, 0, 0, 1., 0).finished() ) );
const Spin2Matrix::SpinMatrixReal Spin2Matrix::isy ( ( (SpinMatrixReal() <<  0, - 1., 0, 0, 0, 1., 0, - sqrt(3./2.), 0, 0, 0, sqrt(3./2.), 0, - sqrt(3./2.), 0, 0, 0, sqrt(3./2.), 0, - 1., 0, 0, 0, 1., 0).finished() ) );
const Spin2Matrix::SpinMatrixReal Spin2Matrix::rsz ( ( (SpinMatrixReal() <<  2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -2).finished() ) );
const Spin2Matrix::SpinMatrixReal Spin2Matrix::rid (  (SpinMatrixReal() = SpinMatrixReal::Identity()) );
const Spin2Matrix::SpinMatrixReal Spin2Matrix::rsp (  (SpinMatrixReal() = Spin2Matrix::rsx + Spin2Matrix::isy) );
const Spin2Matrix::SpinMatrixReal Spin2Matrix::rsm (  (SpinMatrixReal() = Spin2Matrix::rsx - Spin2Matrix::isy) );
const Spin2Matrix::SpinMatrix Spin2Matrix::sx (  (SpinMatrix() = Spin2Matrix::rsx) );
const Spin2Matrix::SpinMatrix Spin2Matrix::sy (  (SpinMatrix() = -iii * Spin2Matrix::isy) );
const Spin2Matrix::SpinMatrix Spin2Matrix::sz (  (SpinMatrix() = Spin2Matrix::rsz) );
const Spin2Matrix::SpinMatrix Spin2Matrix::id (  (SpinMatrix() = Spin2Matrix::rid) );

struct Spin2p5Matrix { 
    enum { matrix_size = 6 };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    typedef Matrix<double, matrix_size, matrix_size>  SpinMatrixReal;
    static const SpinMatrixReal rsx;
    static const SpinMatrixReal isy;
    static const SpinMatrixReal rsz;
    static const SpinMatrixReal rid;
    static const SpinMatrixReal rsp;
    static const SpinMatrixReal rsm;
    static const SpinMatrix sx;
    static const SpinMatrix sy;
    static const SpinMatrix sz;
    static const SpinMatrix id;

    const SpinMatrixReal& Sx(void) const { return rsx; }
    const SpinMatrix& Sy(void) const { return sy; }
    const SpinMatrixReal& Sz(void) const { return rsz; }
    const SpinMatrixReal& Sp(void) const { return rsp; }
    const SpinMatrixReal& Sm(void) const { return rsm; }
    const SpinMatrixReal& Id(void) const { return rid; }
};
const Spin2p5Matrix::SpinMatrixReal Spin2p5Matrix::rsx ( ( (SpinMatrixReal() <<  0, sqrt(5.)/2., 0, 0, 0, 0, sqrt(5.)/2., 0, sqrt(8.)/2., 0, 0, 0, 0, sqrt(8.)/2., 0, 3./2., 0, 0, 0, 0, 3./2., 0, sqrt(8.)/2., 0, 0, 0, 0, sqrt(8.)/2., 0, sqrt(5.)/2., 0, 0, 0, 0, sqrt(5.)/2., 0).finished() ) );
const Spin2p5Matrix::SpinMatrixReal Spin2p5Matrix::isy ( ( (SpinMatrixReal() <<  0, - sqrt(5.)/2., 0, 0, 0, 0, sqrt(5.)/2., 0, - sqrt(8.)/2., 0, 0, 0, 0, sqrt(8.)/2., 0, - 3./2., 0, 0, 0, 0, 3./2., 0, - sqrt(8.)/2., 0, 0, 0, 0, sqrt(8.)/2., 0, - sqrt(5.)/2., 0, 0, 0, 0, sqrt(5.)/2., 0).finished() ) );
const Spin2p5Matrix::SpinMatrixReal Spin2p5Matrix::rsz ( ( (SpinMatrixReal() <<  5./2., 0, 0, 0, 0, 0, 0, 3./2., 0, 0, 0, 0, 0, 0, 1./2., 0, 0, 0, 0, 0, 0, -1./2., 0, 0, 0, 0, 0, 0, -3./2., 0, 0, 0, 0, 0, 0, -5./2.).finished() ) );
const Spin2p5Matrix::SpinMatrixReal Spin2p5Matrix::rid (  (SpinMatrixReal() = SpinMatrixReal::Identity()) );
const Spin2p5Matrix::SpinMatrixReal Spin2p5Matrix::rsp (  (SpinMatrixReal() = Spin2p5Matrix::rsx + Spin2p5Matrix::isy) );
const Spin2p5Matrix::SpinMatrixReal Spin2p5Matrix::rsm (  (SpinMatrixReal() = Spin2p5Matrix::rsx - Spin2p5Matrix::isy) );
const Spin2p5Matrix::SpinMatrix Spin2p5Matrix::sx (  (SpinMatrix() = Spin2p5Matrix::rsx) );
const Spin2p5Matrix::SpinMatrix Spin2p5Matrix::sy (  (SpinMatrix() = -iii * Spin2p5Matrix::isy) );
const Spin2p5Matrix::SpinMatrix Spin2p5Matrix::sz (  (SpinMatrix() = Spin2p5Matrix::rsz) );
const Spin2p5Matrix::SpinMatrix Spin2p5Matrix::id (  (SpinMatrix() = Spin2p5Matrix::rid) );

struct Spin3Matrix { 
    enum { matrix_size = 7 };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    typedef Matrix<double, matrix_size, matrix_size>  SpinMatrixReal;
    static const SpinMatrixReal rsx;
    static const SpinMatrixReal isy;
    static const SpinMatrixReal rsz;
    static const SpinMatrixReal rid;
    static const SpinMatrixReal rsp;
    static const SpinMatrixReal rsm;
    static const SpinMatrix sx;
    static const SpinMatrix sy;
    static const SpinMatrix sz;
    static const SpinMatrix id;

    const SpinMatrixReal& Sx(void) const { return rsx; }
    const SpinMatrix& Sy(void) const { return sy; }
    const SpinMatrixReal& Sz(void) const { return rsz; }
    const SpinMatrixReal& Sp(void) const { return rsp; }
    const SpinMatrixReal& Sm(void) const { return rsm; }
    const SpinMatrixReal& Id(void) const { return rid; }
};
const Spin3Matrix::SpinMatrixReal Spin3Matrix::rsx ( ( (SpinMatrixReal() <<  0, sqrt(3./2.), 0, 0, 0, 0, 0, sqrt(3./2.), 0, sqrt(5./2.), 0, 0, 0, 0, 0, sqrt(5./2.), 0, sqrt(6./2.), 0, 0, 0, 0, 0, sqrt(6./2.), 0, sqrt(6./2.), 0, 0, 0, 0, 0, sqrt(6./2.), 0, sqrt(5./2.), 0, 0, 0, 0, 0, sqrt(5./2.), 0, sqrt(3./2.), 0, 0, 0, 0, 0, sqrt(3./2.), 0).finished() ) );
const Spin3Matrix::SpinMatrixReal Spin3Matrix::isy ( ( (SpinMatrixReal() <<  0, - sqrt(3./2.), 0, 0, 0, 0, 0, sqrt(3./2.), 0, - sqrt(5./2.), 0, 0, 0, 0, 0, sqrt(5./2.), 0, - sqrt(6./2.), 0, 0, 0, 0, 0, sqrt(6./2.), 0, - sqrt(6./2.), 0, 0, 0, 0, 0, sqrt(6./2.), 0, - sqrt(5./2.), 0, 0, 0, 0, 0, sqrt(5./2.), 0, - sqrt(3./2.), 0, 0, 0, 0, 0, sqrt(3./2.), 0).finished() ) );
const Spin3Matrix::SpinMatrixReal Spin3Matrix::rsz ( ( (SpinMatrixReal() <<  3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, -3).finished() ) );
const Spin3Matrix::SpinMatrixReal Spin3Matrix::rid (  (SpinMatrixReal() = SpinMatrixReal::Identity()) );
const Spin3Matrix::SpinMatrixReal Spin3Matrix::rsp (  (SpinMatrixReal() = Spin3Matrix::rsx + Spin3Matrix::isy) );
const Spin3Matrix::SpinMatrixReal Spin3Matrix::rsm (  (SpinMatrixReal() = Spin3Matrix::rsx - Spin3Matrix::isy) );
const Spin3Matrix::SpinMatrix Spin3Matrix::sx (  (SpinMatrix() = Spin3Matrix::rsx) );
const Spin3Matrix::SpinMatrix Spin3Matrix::sy (  (SpinMatrix() = -iii * Spin3Matrix::isy) );
const Spin3Matrix::SpinMatrix Spin3Matrix::sz (  (SpinMatrix() = Spin3Matrix::rsz) );
const Spin3Matrix::SpinMatrix Spin3Matrix::id (  (SpinMatrix() = Spin3Matrix::rid) );

struct Spin3p5Matrix { 
    enum { matrix_size = 8 };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    typedef Matrix<double, matrix_size, matrix_size>  SpinMatrixReal;
    static const SpinMatrixReal rsx;
    static const SpinMatrixReal isy;
    static const SpinMatrixReal rsz;
    static const SpinMatrixReal rid;
    static const SpinMatrixReal rsp;
    static const SpinMatrixReal rsm;
    static const SpinMatrix sx;
    static const SpinMatrix sy;
    static const SpinMatrix sz;
    static const SpinMatrix id;

    const SpinMatrixReal& Sx(void) const { return rsx; }
    const SpinMatrix& Sy(void) const { return sy; }
    const SpinMatrixReal& Sz(void) const { return rsz; }
    const SpinMatrixReal& Sp(void) const { return rsp; }
    const SpinMatrixReal& Sm(void) const { return rsm; }
    const SpinMatrixReal& Id(void) const { return rid; }
};
const Spin3p5Matrix::SpinMatrixReal Spin3p5Matrix::rsx ( ( (SpinMatrixReal() <<  0, sqrt(7.)/2., 0, 0, 0, 0, 0, 0, sqrt(7.)/2., 0, sqrt(12.)/2., 0, 0, 0, 0, 0, 0, sqrt(12.)/2., 0, sqrt(15.)/2., 0, 0, 0, 0, 0, 0, sqrt(15.)/2., 0, 2., 0, 0, 0, 0, 0, 0, 2., 0, sqrt(15.)/2., 0, 0, 0, 0, 0, 0, sqrt(15.)/2., 0, sqrt(12.)/2., 0, 0, 0, 0, 0, 0, sqrt(12.)/2., 0, sqrt(7.)/2., 0, 0, 0, 0, 0, 0, sqrt(7.)/2., 0).finished() ) );
const Spin3p5Matrix::SpinMatrixReal Spin3p5Matrix::isy ( ( (SpinMatrixReal() <<  0, - sqrt(7.)/2., 0, 0, 0, 0, 0, 0, sqrt(7.)/2., 0, - sqrt(12.)/2., 0, 0, 0, 0, 0, 0, sqrt(12.)/2., 0, - sqrt(15.)/2., 0, 0, 0, 0, 0, 0, sqrt(15.)/2., 0, - 2., 0, 0, 0, 0, 0, 0, 2., 0, - sqrt(15.)/2., 0, 0, 0, 0, 0, 0, sqrt(15.)/2., 0, - sqrt(12.)/2., 0, 0, 0, 0, 0, 0, sqrt(12.)/2., 0, - sqrt(7.)/2., 0, 0, 0, 0, 0, 0, sqrt(7.)/2., 0).finished() ) );
const Spin3p5Matrix::SpinMatrixReal Spin3p5Matrix::rsz ( ( (SpinMatrixReal() <<  7./2., 0, 0, 0, 0, 0, 0, 0, 0, 5./2., 0, 0, 0, 0, 0, 0, 0, 0, 3./2., 0, 0, 0, 0, 0, 0, 0, 0, 1./2., 0, 0, 0, 0, 0, 0, 0, 0, -1./2., 0, 0, 0, 0, 0, 0, 0, 0, -3./2., 0, 0, 0, 0, 0, 0, 0, 0, -5./2., 0, 0, 0, 0, 0, 0, 0, 0, -7./2.).finished() ) );
const Spin3p5Matrix::SpinMatrixReal Spin3p5Matrix::rid (  (SpinMatrixReal() = SpinMatrixReal::Identity()) );
const Spin3p5Matrix::SpinMatrixReal Spin3p5Matrix::rsp (  (SpinMatrixReal() = Spin3p5Matrix::rsx + Spin3p5Matrix::isy) );
const Spin3p5Matrix::SpinMatrixReal Spin3p5Matrix::rsm (  (SpinMatrixReal() = Spin3p5Matrix::rsx - Spin3p5Matrix::isy) );
const Spin3p5Matrix::SpinMatrix Spin3p5Matrix::sx (  (SpinMatrix() = Spin3p5Matrix::rsx) );
const Spin3p5Matrix::SpinMatrix Spin3p5Matrix::sy (  (SpinMatrix() = -iii * Spin3p5Matrix::isy) );
const Spin3p5Matrix::SpinMatrix Spin3p5Matrix::sz (  (SpinMatrix() = Spin3p5Matrix::rsz) );
const Spin3p5Matrix::SpinMatrix Spin3p5Matrix::id (  (SpinMatrix() = Spin3p5Matrix::rid) );

struct Spin4Matrix { 
    enum { matrix_size = 9 };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    typedef Matrix<double, matrix_size, matrix_size>  SpinMatrixReal;
    static const SpinMatrixReal rsx;
    static const SpinMatrixReal isy;
    static const SpinMatrixReal rsz;
    static const SpinMatrixReal rid;
    static const SpinMatrixReal rsp;
    static const SpinMatrixReal rsm;
    static const SpinMatrix sx;
    static const SpinMatrix sy;
    static const SpinMatrix sz;
    static const SpinMatrix id;

    const SpinMatrixReal& Sx(void) const { return rsx; }
    const SpinMatrix& Sy(void) const { return sy; }
    const SpinMatrixReal& Sz(void) const { return rsz; }
    const SpinMatrixReal& Sp(void) const { return rsp; }
    const SpinMatrixReal& Sm(void) const { return rsm; }
    const SpinMatrixReal& Id(void) const { return rid; }
};
const Spin4Matrix::SpinMatrixReal Spin4Matrix::rsx ( ( (SpinMatrixReal() <<  0, sqrt(4./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(4./2.), 0, sqrt(7./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(7./2.), 0, sqrt(9./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(9./2.), 0, sqrt(10./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(10./2.), 0, sqrt(10./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(10./2.), 0, sqrt(9./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(9./2.), 0, sqrt(7./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(7./2.), 0, sqrt(4./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(4./2.), 0).finished() ) );
const Spin4Matrix::SpinMatrixReal Spin4Matrix::isy ( ( (SpinMatrixReal() <<  0, - sqrt(4./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(4./2.), 0, - sqrt(7./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(7./2.), 0, - sqrt(9./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(9./2.), 0, - sqrt(10./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(10./2.), 0, - sqrt(10./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(10./2.), 0, - sqrt(9./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(9./2.), 0, - sqrt(7./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(7./2.), 0, - sqrt(4./2.), 0, 0, 0, 0, 0, 0, 0, sqrt(4./2.), 0).finished() ) );
const Spin4Matrix::SpinMatrixReal Spin4Matrix::rsz ( ( (SpinMatrixReal() <<  4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4).finished() ) );
const Spin4Matrix::SpinMatrixReal Spin4Matrix::rid (  (SpinMatrixReal() = SpinMatrixReal::Identity()) );
const Spin4Matrix::SpinMatrixReal Spin4Matrix::rsp (  (SpinMatrixReal() = Spin4Matrix::rsx + Spin4Matrix::isy) );
const Spin4Matrix::SpinMatrixReal Spin4Matrix::rsm (  (SpinMatrixReal() = Spin4Matrix::rsx - Spin4Matrix::isy) );
const Spin4Matrix::SpinMatrix Spin4Matrix::sx (  (SpinMatrix() = Spin4Matrix::rsx) );
const Spin4Matrix::SpinMatrix Spin4Matrix::sy (  (SpinMatrix() = -iii * Spin4Matrix::isy) );
const Spin4Matrix::SpinMatrix Spin4Matrix::sz (  (SpinMatrix() = Spin4Matrix::rsz) );
const Spin4Matrix::SpinMatrix Spin4Matrix::id (  (SpinMatrix() = Spin4Matrix::rid) );

struct Spin4p5Matrix { 
    enum { matrix_size = 10 };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    typedef Matrix<double, matrix_size, matrix_size>  SpinMatrixReal;
    static const SpinMatrixReal rsx;
    static const SpinMatrixReal isy;
    static const SpinMatrixReal rsz;
    static const SpinMatrixReal rid;
    static const SpinMatrixReal rsp;
    static const SpinMatrixReal rsm;
    static const SpinMatrix sx;
    static const SpinMatrix sy;
    static const SpinMatrix sz;
    static const SpinMatrix id;

    const SpinMatrixReal& Sx(void) const { return rsx; }
    const SpinMatrix& Sy(void) const { return sy; }
    const SpinMatrixReal& Sz(void) const { return rsz; }
    const SpinMatrixReal& Sp(void) const { return rsp; }
    const SpinMatrixReal& Sm(void) const { return rsm; }
    const SpinMatrixReal& Id(void) const { return rid; }
};
const Spin4p5Matrix::SpinMatrixReal Spin4p5Matrix::rsx ( ( (SpinMatrixReal() <<  0, 3./2., 0, 0, 0, 0, 0, 0, 0, 0, 3./2., 0, 2., 0, 0, 0, 0, 0, 0, 0, 0, 2., 0, sqrt(21.)/2., 0, 0, 0, 0, 0, 0, 0, 0, sqrt(21.)/2., 0, sqrt(24.)/2., 0, 0, 0, 0, 0, 0, 0, 0, sqrt(24.)/2., 0, 5./2., 0, 0, 0, 0, 0, 0, 0, 0, 5./2., 0, sqrt(24.)/2., 0, 0, 0, 0, 0, 0, 0, 0, sqrt(24.)/2., 0, sqrt(21.)/2., 0, 0, 0, 0, 0, 0, 0, 0, sqrt(21.)/2., 0, 2., 0, 0, 0, 0, 0, 0, 0, 0, 2., 0, 3./2., 0, 0, 0, 0, 0, 0, 0, 0, 3./2., 0).finished() ) );
const Spin4p5Matrix::SpinMatrixReal Spin4p5Matrix::isy ( ( (SpinMatrixReal() <<  0, - 3./2., 0, 0, 0, 0, 0, 0, 0, 0, 3./2., 0, - 2., 0, 0, 0, 0, 0, 0, 0, 0, 2., 0, - sqrt(21.)/2., 0, 0, 0, 0, 0, 0, 0, 0, sqrt(21.)/2., 0, - sqrt(24.)/2., 0, 0, 0, 0, 0, 0, 0, 0, sqrt(24.)/2., 0, - 5./2., 0, 0, 0, 0, 0, 0, 0, 0, 5./2., 0, - sqrt(24.)/2., 0, 0, 0, 0, 0, 0, 0, 0, sqrt(24.)/2., 0, - sqrt(21.)/2., 0, 0, 0, 0, 0, 0, 0, 0, sqrt(21.)/2., 0, - 2., 0, 0, 0, 0, 0, 0, 0, 0, 2., 0, - 3./2., 0, 0, 0, 0, 0, 0, 0, 0, 3./2., 0).finished() ) );
const Spin4p5Matrix::SpinMatrixReal Spin4p5Matrix::rsz ( ( (SpinMatrixReal() <<  9./2., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7./2., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5./2., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3./2., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1./2., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1./2., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3./2., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5./2., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -7./2., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -9./2.).finished() ) );
const Spin4p5Matrix::SpinMatrixReal Spin4p5Matrix::rid (  (SpinMatrixReal() = SpinMatrixReal::Identity()) );
const Spin4p5Matrix::SpinMatrixReal Spin4p5Matrix::rsp (  (SpinMatrixReal() = Spin4p5Matrix::rsx + Spin4p5Matrix::isy) );
const Spin4p5Matrix::SpinMatrixReal Spin4p5Matrix::rsm (  (SpinMatrixReal() = Spin4p5Matrix::rsx - Spin4p5Matrix::isy) );
const Spin4p5Matrix::SpinMatrix Spin4p5Matrix::sx (  (SpinMatrix() = Spin4p5Matrix::rsx) );
const Spin4p5Matrix::SpinMatrix Spin4p5Matrix::sy (  (SpinMatrix() = -iii * Spin4p5Matrix::isy) );
const Spin4p5Matrix::SpinMatrix Spin4p5Matrix::sz (  (SpinMatrix() = Spin4p5Matrix::rsz) );
const Spin4p5Matrix::SpinMatrix Spin4p5Matrix::id (  (SpinMatrix() = Spin4p5Matrix::rid) );

struct Spin5Matrix { 
    enum { matrix_size = 11 };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    typedef Matrix<double, matrix_size, matrix_size>  SpinMatrixReal;
    static const SpinMatrixReal rsx;
    static const SpinMatrixReal isy;
    static const SpinMatrixReal rsz;
    static const SpinMatrixReal rid;
    static const SpinMatrixReal rsp;
    static const SpinMatrixReal rsm;
    static const SpinMatrix sx;
    static const SpinMatrix sy;
    static const SpinMatrix sz;
    static const SpinMatrix id;

    const SpinMatrixReal& Sx(void) const { return rsx; }
    const SpinMatrix& Sy(void) const { return sy; }
    const SpinMatrixReal& Sz(void) const { return rsz; }
    const SpinMatrixReal& Sp(void) const { return rsp; }
    const SpinMatrixReal& Sm(void) const { return rsm; }
    const SpinMatrixReal& Id(void) const { return rid; }
};
const Spin5Matrix::SpinMatrixReal Spin5Matrix::rsx ( ( (SpinMatrixReal() <<  0, sqrt(5./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(5./2.), 0, sqrt(9./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(9./2.), 0, sqrt(12./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(12./2.), 0, sqrt(14./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(14./2.), 0, sqrt(15./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(15./2.), 0, sqrt(15./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(15./2.), 0, sqrt(14./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(14./2.), 0, sqrt(12./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(12./2.), 0, sqrt(9./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(9./2.), 0, sqrt(5./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(5./2.), 0).finished() ) );
const Spin5Matrix::SpinMatrixReal Spin5Matrix::isy ( ( (SpinMatrixReal() <<  0, - sqrt(5./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(5./2.), 0, - sqrt(9./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(9./2.), 0, - sqrt(12./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(12./2.), 0, - sqrt(14./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(14./2.), 0, - sqrt(15./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(15./2.), 0, - sqrt(15./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(15./2.), 0, - sqrt(14./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(14./2.), 0, - sqrt(12./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(12./2.), 0, - sqrt(9./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(9./2.), 0, - sqrt(5./2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(5./2.), 0).finished() ) );
const Spin5Matrix::SpinMatrixReal Spin5Matrix::rsz ( ( (SpinMatrixReal() <<  5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5).finished() ) );
const Spin5Matrix::SpinMatrixReal Spin5Matrix::rid (  (SpinMatrixReal() = SpinMatrixReal::Identity()) );
const Spin5Matrix::SpinMatrixReal Spin5Matrix::rsp (  (SpinMatrixReal() = Spin5Matrix::rsx + Spin5Matrix::isy) );
const Spin5Matrix::SpinMatrixReal Spin5Matrix::rsm (  (SpinMatrixReal() = Spin5Matrix::rsx - Spin5Matrix::isy) );
const Spin5Matrix::SpinMatrix Spin5Matrix::sx (  (SpinMatrix() = Spin5Matrix::rsx) );
const Spin5Matrix::SpinMatrix Spin5Matrix::sy (  (SpinMatrix() = -iii * Spin5Matrix::isy) );
const Spin5Matrix::SpinMatrix Spin5Matrix::sz (  (SpinMatrix() = Spin5Matrix::rsz) );
const Spin5Matrix::SpinMatrix Spin5Matrix::id (  (SpinMatrix() = Spin5Matrix::rid) );


typedef Spin1Matrix PauliTripletMatrices;

struct PauliMatrices { 
    enum { matrix_size = 2 };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    static const SpinMatrix sx;
    static const SpinMatrix sy;
    static const SpinMatrix sz;
    static const SpinMatrix id;
};
const PauliMatrices::SpinMatrix PauliMatrices::sx ( ( (SpinMatrix() << 0.0, 1.0, 1.0, 0.0).finished() ) );
const PauliMatrices::SpinMatrix PauliMatrices::sy ( ( (SpinMatrix() << 0.0, -iii, iii, 0.0).finished() ) );
const PauliMatrices::SpinMatrix PauliMatrices::sz ( ( (SpinMatrix() << 1.0, 0.0, 0.0, -1.0).finished() ) );
const PauliMatrices::SpinMatrix PauliMatrices::id (  (SpinMatrix() = SpinMatrix::Identity()) );


typedef Spin0p5Matrix PauliDoubletMatrices;
typedef Spin1Matrix PauliQuintetMatrices;
typedef complexg *TransferSpinMatrix;

struct GenericSpinBase {
    double D; // D 
    double E; // E
    Rotation rot; // rotation from laboratory frame to molecular frame 
    Vector3d g3; // g factors in molecular frame 
    Vector3d B;  // magnetic field in laboratory frame 

    GenericSpinBase(void) :
      D(0),
      E(0),
      rot ( Matrix3d() = Matrix3d::Identity() ),
      g3 ( Vector3d() = Vector3d::Constant(1.0) ),
      B (  Vector3d() = Vector3d::Constant(0.0) )
    { 
    }

    virtual TransferSpinMatrix hamiltonian_gen(void) = 0;
    virtual const TransferSpinMatrix Sx_gen(void) const = 0;
    virtual const TransferSpinMatrix Sy_gen(void) const = 0;
    virtual const TransferSpinMatrix Sz_gen(void) const = 0;
    virtual const TransferSpinMatrix Id_gen(void) const = 0;


};


template <class Pauli> class GenericSpin : public Pauli, public GenericSpinBase { 
public :
    typedef typename Pauli::SpinMatrix SpinMatrix;
private :
    SpinMatrix Hfull;
public:
    enum { matrix_size = Pauli::matrix_size };


    const SpinMatrix& update_hamiltonian(void) { 
       SpinMatrix rotSx = rot.matrix(0, 0) * Pauli::rsx - (iii * rot.matrix(0, 1)) * Pauli::isy + rot.matrix(0, 2) * Pauli::rsz;
       SpinMatrix rotSy = rot.matrix(1, 0) * Pauli::rsx - (iii * rot.matrix(1, 1)) * Pauli::isy + rot.matrix(1, 2) * Pauli::rsz;
       SpinMatrix rotSz = rot.matrix(2, 0) * Pauli::rsx - (iii * rot.matrix(2, 1)) * Pauli::isy + rot.matrix(2, 2) * Pauli::rsz;
       Vector3d rBvec = rot.matrix() * B;
       Hfull = D * (rotSz * rotSz - 2.0*Pauli::id/3.0) + E * (rotSx * rotSx -  rotSy * rotSy) 
	   + g3[0] * rotSx * rBvec[0] + g3[1] * rotSy * rBvec[1] + g3[2] * rotSz * rBvec[2];
       return Hfull;
    }

    const SpinMatrix Sx_mol(void) const  { return rot.matrix(0, 0)*Pauli::rsx - (iii*rot.matrix(0, 1))*Pauli::isy + rot.matrix(0, 2)*Pauli::rsz; }
    const SpinMatrix Sy_mol(void) const { return rot.matrix(1, 0)*Pauli::rsx - (iii*rot.matrix(1, 1))*Pauli::isy + rot.matrix(1, 2)*Pauli::rsz; }
    const SpinMatrix Sz_mol(void) const  { return rot.matrix(2, 0)*Pauli::rsx - (iii*rot.matrix(2, 1))*Pauli::isy + rot.matrix(2, 2)*Pauli::rsz; }
    const SpinMatrix Sx2_mol(void) const  { return Sx_mol() * Sx_mol(); }
    const SpinMatrix Sy2_mol(void) const  { return Sy_mol() * Sy_mol(); }
    const SpinMatrix Sz2_mol(void) const  { return Sz_mol() * Sz_mol(); }
    const SpinMatrix Sx(void) const { return Pauli::rsx; }
    const SpinMatrix Sy(void) const  { return -iii*Pauli::isy; }
    const SpinMatrix Sz(void) const { return Pauli::rsz; }
    const SpinMatrix Id(void) const { return Pauli::id; }
 
    const SpinMatrix& hamiltonian(void) const { 
       return Hfull;
    }

    TransferSpinMatrix hamiltonian_gen(void) { 
       update_hamiltonian();
       return Hfull.data();
    }

    int size(void) const { return matrix_size; }

    const TransferSpinMatrix Sx_gen(void) const { return (TransferSpinMatrix) Pauli::sx.data(); }
    const TransferSpinMatrix Sy_gen(void) const { return (TransferSpinMatrix) Pauli::sy.data(); }
    const TransferSpinMatrix Sz_gen(void) const { return (TransferSpinMatrix) Pauli::sz.data(); }
    const TransferSpinMatrix Id_gen(void) const { return (TransferSpinMatrix) Pauli::id.data(); }

    const SpinMatrix& add_matrix(const SpinMatrix &M) { 
       Hfull += M;
       return Hfull;
    }

    Matrix3d fine_structure_tensor(void) { 
       return rot.matrix().transpose() *  (Vector3d() << E-D/3.0, -E-D/3.0, 2.0*D/3.0).finished().asDiagonal() * rot.matrix();    
    }
  //    template <class Spin2> GenericSpin<Pauli>& operator=(const GenericSpin<Spin2> & S) {
    template <class Spin2> GenericSpin<Pauli>& operator=(const Spin2 & S) {   
       this->D = S.D;
       this->E = S.E;
       this->rot = S.rot;
       this->g3 = S.g3;
       this->B = S.B;
       update_hamiltonian();
       return *this;
    }  


};



typedef GenericSpin<Spin0p5Matrix> Spin0p5;
typedef GenericSpin<Spin0p5Matrix> SpinHalf;
typedef GenericSpin<Spin1Matrix> Spin1;
typedef GenericSpin<Spin1Matrix> TripletSpin;
typedef GenericSpin<Spin1p5Matrix> Spin1p5;
typedef GenericSpin<Spin2Matrix> Spin2;
typedef GenericSpin<Spin2Matrix> QuintetSpin;
typedef GenericSpin<Spin2p5Matrix> Spin2p5;
typedef GenericSpin<Spin3Matrix> Spin3;
typedef GenericSpin<Spin3p5Matrix> Spin3p5;
typedef GenericSpin<Spin4Matrix> Spin4;
typedef GenericSpin<Spin4p5Matrix> Spin4p5;
typedef GenericSpin<Spin5Matrix> Spin5;



template <class Spin1> class SingleSpin { 
public:
    Spin1 S;
    typedef Matrix<complexg, Spin1::matrix_size, Spin1::matrix_size> SpinMatrix;
    typedef Matrix<double, Spin1::matrix_size, Spin1::matrix_size> SpinMatrixReal;
    typedef Matrix<complexg, Spin1::matrix_size, 1> SpinVector;
    typedef Matrix<double, Spin1::matrix_size, 1> SpinVectorReal;

    SpinVectorReal eval;   // eigenvalues
    SpinMatrix evec; // eigenvectors
    enum { matrix_size = Spin1::matrix_size };

    SingleSpin(void) : S()
    { 
    }

    int size(void) { return matrix_size; }
  
    const SpinMatrix& update_hamiltonian(void) { 
       return S.update_hamiltonian();
    }

    const SpinMatrix& hamiltonian(void) const { 
       return S.hamiltonian();
    }

    void diag(void) { 
       SelfAdjointEigenSolver<SpinMatrix> eigensolver( S.hamiltonian() );
       if (eigensolver.info() != Success) abort();
       eval = eigensolver.eigenvalues();
       evec = eigensolver.eigenvectors();
    }

    void diag_real(void) { 
       SelfAdjointEigenSolver<SpinMatrixReal> eigensolver( S.hamiltonian().real() );
       if (eigensolver.info() != Success) abort();
       eval = eigensolver.eigenvalues();
       evec = eigensolver.eigenvectors();
    }

    void diag_eval_only(void) { 
       SelfAdjointEigenSolver<SpinMatrix> eigensolver( S.hamiltonian(), EigenvaluesOnly );
       if (eigensolver.info() != Success) abort();
       eval = eigensolver.eigenvalues();
    }

    SpinMatrix Bac_field_basis_matrix(void) { 
       return S.Sx();
    }

    SpinMatrix Sx(void) { 
       return S.Sx();
    }
    SpinMatrix Sy(void) { 
       return S.Sy();
    }
    SpinMatrix Sz(void) { 
       return S.Sz();
    }
    complexg Sx(int n, int m) { 
       return evec.col(n).adjoint() * Sx() * evec.col(m);
    }

    complexg Sy(int n, int m) { 
       return evec.col(n).adjoint() * Sy() * evec.col(m);
    }

    complexg Sz(int n, int m) { 
       return evec.col(n).adjoint() * Sz() * evec.col(m);
    }

  

};



template <class Spin1, class Spin2> class SpinPair { 
public: 
    Spin1 S1;
    Spin2 S2;
    typedef Matrix<complexg, Spin1::matrix_size * Spin2::matrix_size, Spin1::matrix_size * Spin2::matrix_size> SpinMatrix;
    typedef Matrix<double, Spin1::matrix_size * Spin2::matrix_size, Spin1::matrix_size * Spin2::matrix_size> SpinMatrixReal;
    typedef Matrix<complexg, Spin1::matrix_size * Spin2::matrix_size, 1> SpinVector;
    typedef Matrix<double, Spin1::matrix_size * Spin2::matrix_size, 1> SpinVectorReal;
private : 
    SpinMatrix Hfull; // Hamiltonian 

public : 
    static SpinMatrixReal exchange_matrix(void) { 
      return kroneckerProduct(Spin1::rsx, Spin2::rsx).eval() - kroneckerProduct(Spin1::isy, Spin2::isy).eval() + kroneckerProduct(Spin1::rsz, Spin2::rsz).eval();
    }

    static SpinMatrix dipole_dipole_matrix(Vector3d uvec) {       
       double unorm = uvec.norm();
       if (abs(unorm) > 1e-15) { 
	  uvec /= unorm;
	  typename Spin1::SpinMatrix uS1 = uvec(0) * Spin1::rsx + uvec(1) * Spin1::sy + uvec(2) * Spin1::rsz;       
	  typename Spin2::SpinMatrix uS2 = uvec(0) * Spin2::rsx + uvec(1) * Spin2::sy + uvec(2) * Spin2::rsz;       
	  return (exchange_matrix() - 3.0 * kroneckerProduct(uS1, uS2).eval());
       } else {
	  return SpinMatrix() = SpinMatrix::Zero();
       }
    }

    enum { matrix_size = Spin1::matrix_size * Spin2::matrix_size };
    double J;
    double Jdip;
    Vector3d r12;

    SpinVectorReal eval;   // eigenvalues
    SpinMatrix evec; // eigenvectors

    SpinPair(void) : S1(), 
		     S2() 
    { 
       J = Jdip = 0;
       r12 << 0, 0, 1;
    }

    SpinMatrix update_hamiltonian(void) { 
       Hfull = kroneckerProduct( S1.update_hamiltonian(), S2.id ).eval() + kroneckerProduct( S1.id, S2.update_hamiltonian() ).eval() 
	 + J * exchange_matrix() + Jdip * dipole_dipole_matrix(r12);
       return Hfull;
    }

    SpinMatrix add_matrix(const SpinMatrix &M) { 
       Hfull += M;
       return Hfull;
    }

    SpinMatrix hamiltonian(void) const { 
       return Hfull;
    }

    void diag(void) { 
       SelfAdjointEigenSolver<SpinMatrix> eigensolver(Hfull);
       if (eigensolver.info() != Success) abort();
       eval = eigensolver.eigenvalues();
       evec = eigensolver.eigenvectors();
    }

    void diag_real(void) { 
       SelfAdjointEigenSolver<SpinMatrixReal> eigensolver( Hfull.real() );
       if (eigensolver.info() != Success) abort();
       eval = eigensolver.eigenvalues();
       evec = eigensolver.eigenvectors();
    }
  
    void diag_eval_only(void) { 
       SelfAdjointEigenSolver<SpinMatrix> eigensolver(Hfull);
       if (eigensolver.info() != Success) abort();
       eval = eigensolver.eigenvalues();
    }
 
    double sz_elem(int i) { 
       SpinMatrixReal Sz2 = kroneckerProduct(S1.rsz, S2.rid).eval() + kroneckerProduct(S1.rid, S2.rsz).eval();
       Matrix<complexg, matrix_size, 1> vi = evec.col(i);
       Matrix<complexg, 1, 1> Sz2ii = vi.adjoint() * Sz2 * vi;
       return real(Sz2ii(0));
    }


    void load_field_basis_Hamiltonian(const Rotation &t1_rot, const Rotation &t2_rot, const Vector3d &rdip, const Vector3d &Bvec) { 
       S1.rot = t1_rot;
       S2.rot = t2_rot;
       r12 = rdip;
       update_hamiltonian();
    }

    SpinMatrix Bac_field_basis_matrix(void) { 
       return kroneckerProduct(S1.sx, S2.id).eval() + kroneckerProduct(S1.id, S2.sx).eval();
    }

    const SpinMatrix Sx(void) const { 
       return kroneckerProduct(S1.sx, S2.id).eval() + kroneckerProduct(S1.id, S2.sx).eval();
    }
    const SpinMatrix Sy(void) const { 
       return kroneckerProduct(S1.sy, S2.id).eval() + kroneckerProduct(S1.id, S2.sy).eval();
    }
    const SpinMatrix Sz(void) const { 
       return kroneckerProduct(S1.sz, S2.id).eval() + kroneckerProduct(S1.id, S2.sz).eval();
    }

    SpinMatrix S2tot(void) { return Sx()*Sx() + Sy()*Sy() + Sz()*Sz(); }


    const SpinMatrix Sx(int s) const { 
       if (!s) return kroneckerProduct(S1.sx, S2.id);
       else return  kroneckerProduct(S1.id, S2.sx);
    }
    const SpinMatrix Sy(int s) const { 
       if (!s) return kroneckerProduct(S1.sy, S2.id);
       else return  kroneckerProduct(S1.id, S2.sy);
    }
    const SpinMatrix Sz(int s) const { 
       if (!s) return kroneckerProduct(S1.sz, S2.id);
       else return  kroneckerProduct(S1.id, S2.sz);
    }

    const SpinMatrix Sx_rot(const Rotation &rot) const  { return rot.matrix(0, 0)*Sx() + rot.matrix(0, 1) * Sy() + rot.matrix(0, 2)*Sz(); }
    const SpinMatrix Sy_rot(const Rotation &rot) const  { return rot.matrix(1, 0)*Sx() + rot.matrix(1, 1) * Sy() + rot.matrix(1, 2)*Sz(); }
    const SpinMatrix Sz_rot(const Rotation &rot) const  { return rot.matrix(2, 0)*Sx() + rot.matrix(2, 1) * Sy() + rot.matrix(2, 2)*Sz(); }
    const SpinMatrix Sx_rot(const Rotation &rot, int s) const  { return rot.matrix(0, 0)*Sx(s) + rot.matrix(0, 1) * Sy(s) + rot.matrix(0, 2)*Sz(s); }
    const SpinMatrix Sy_rot(const Rotation &rot, int s) const  { return rot.matrix(1, 0)*Sx(s) + rot.matrix(1, 1) * Sy(s) + rot.matrix(1, 2)*Sz(s); }
    const SpinMatrix Sz_rot(const Rotation &rot, int s) const  { return rot.matrix(2, 0)*Sx(s) + rot.matrix(2, 1) * Sy(s) + rot.matrix(2, 2)*Sz(s); }
  
    const SpinMatrix Sx_mol1(void) const { return Sx_rot(S1.rot); }
    const SpinMatrix Sy_mol1(void) const { return Sy_rot(S1.rot); }
    const SpinMatrix Sz_mol1(void) const { return Sz_rot(S1.rot); }
    const SpinMatrix Sx_mol2(void) const { return Sx_rot(S2.rot); }
    const SpinMatrix Sy_mol2(void) const { return Sy_rot(S2.rot); }
    const SpinMatrix Sz_mol2(void) const { return Sz_rot(S2.rot); }

    const SpinMatrix Sx1_mol(void) const { return Sx_rot(S1.rot, 0); }
    const SpinMatrix Sy1_mol(void) const { return Sy_rot(S1.rot, 0); }
    const SpinMatrix Sz1_mol(void) const { return Sz_rot(S1.rot, 0); }
    const SpinMatrix Sx2_mol(void) const { return Sx_rot(S2.rot, 1); }
    const SpinMatrix Sy2_mol(void) const { return Sy_rot(S2.rot, 1); }
    const SpinMatrix Sz2_mol(void) const { return Sz_rot(S2.rot, 1); }
 
      
    const complexg Sx(int n, int m) const { return evec.col(n).adjoint() * Sx() * evec.col(m); }
    const complexg Sy(int n, int m) const { return evec.col(n).adjoint() * Sy() * evec.col(m); }
    const complexg Sz(int n, int m) const { return evec.col(n).adjoint() * Sz() * evec.col(m); }
    complexg Sx2(int n, int m) { return evec.col(n).adjoint() * Sx() * Sx() * evec.col(m); }
    complexg Sy2(int n, int m) { return evec.col(n).adjoint() * Sy() * Sy() * evec.col(m); }
    complexg Sz2(int n, int m) { return evec.col(n).adjoint() * Sz() * Sz() * evec.col(m); }
    complexg S2tot(int n, int m) { return evec.col(n).adjoint() * S2tot() * evec.col(m); }
  
    complexg Perm(int n, int m) { 
       complexg Psum = 0;
       if (Spin1::matrix_size == Spin2::matrix_size) { 
	  int matrix_size = Spin1::matrix_size;
	  for (int i = 0; i < matrix_size; i++) { 
	     for (int j = 0; j < matrix_size; j++) { 
	        int k = i * matrix_size + j;
	        int q = j * matrix_size + i;
		Psum += conj( evec(k,n) ) * evec(q,m);
	     }
	  }
       }
       return Psum;
    }

    int size(void) const { return matrix_size; }

};

typedef SingleSpin<Spin0p5>  SingleSpin0p5;
typedef SingleSpin<Spin1>  SingleSpin1;
typedef SingleSpin<Spin1p5>  SingleSpin1p5;
typedef SingleSpin<Spin2>  SingleSpin2;
typedef SingleSpin<Spin2p5>  SingleSpin2p5;
typedef SingleSpin<Spin3>  SingleSpin3;
typedef SingleSpin<Spin3p5>  SingleSpin3p5;
typedef SingleSpin<Spin4>  SingleSpin4;
typedef SingleSpin<Spin4p5>  SingleSpin4p5;
typedef SingleSpin<Spin5>  SingleSpin5;

class SingleTriplet : public SingleSpin<TripletSpin> { 
public : 
    static const SpinMatrix singlet_projector(void) { 
      return (SpinMatrix() << 
         0.0, 0.0, 0.0, 
         0.0, 1.0, 0.0, 
         0.0, 0.0, 0.0
              ).finished();
    };
};

class SingleQuintet : public SingleSpin<QuintetSpin> { 
public : 
    static const SpinMatrix singlet_projector(void) { 
      return (SpinMatrix() << 
         0.0, 0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0
              ).finished();
    };
};


class TripletPair : public SpinPair<TripletSpin, TripletSpin> {
private: 
    static const double s2i3(void) { return sqrt(2.0/3.0); }
    static const double si2(void)  { return 1.0/sqrt(2.0); }
    static const double si3(void)  { return 1.0/sqrt(3.0); } 
    static const double si6(void)  { return 1.0/sqrt(6.0); }

public : 
    static const SpinMatrix Jproj;
    TripletPair(void) 
    {
    }


    const double quintet_content(int i) const {
       Matrix<complexg, 5, 1> iProj = Jproj.block(4, 0, 5, 9) * evec.block(0, i, 9, 1);
       Matrix<complexg, 1, 1> norm2 = iProj.adjoint() * iProj;
       return real(norm2(0));
    }

    const double triplet_content(int i) const {
       Matrix<complexg, 3, 1> iProj = Jproj.block(1, 0, 3, 9) * evec.block(0, i, 9, 1);
       Matrix<complexg, 1, 1> norm2 = iProj.adjoint() * iProj;
       return real(norm2(0));
    }

    const double singlet_content(int i) const {
       Matrix<complexg, 1, 1> iProj = Jproj.block(0, 0, 1, 9) * evec.block(0, i, 9, 1);
       Matrix<complexg, 1, 1> norm2 = iProj.adjoint() * iProj;
       return real(norm2(0));
    }

    static const SpinVector singlet(void) { 
       return Jproj.row(0).adjoint();
    }
    static const SpinMatrix singlet_projector(void) { 
       return Jproj.row(0).adjoint() * Jproj.row(0);
    }

    static const SpinVector singlet_wavefunction(void) { 
       return Jproj.col(0);
    }
};


const TripletPair::SpinMatrix TripletPair::Jproj ( 
						   (SpinMatrix() << 0, 0, si3(), 0, -si3(), 0, si3(), 0, 0,
						    0, 0, 0, 0, 0, -si2(), 0, si2(), 0,
						    0, 0, -si2(), 0, 0, 0, si2(), 0, 0,
						    0, -si2(), 0, si2(), 0, 0, 0, 0, 0,
						    0,    0, 0,   0, 0, 0, 0, 0, 1.0, 
						    0, 0, 0, 0, 0, si2(), 0, si2(), 0,
						    0, 0, si6(), 0, s2i3(), 0, si6(), 0, 0,
						    0, si2(), 0, si2(), 0, 0, 0, 0, 0,
						    1.0, 0, 0, 0, 0, 0, 0, 0, 0 ).finished()
						    );




class QuintetPair : public SpinPair<QuintetSpin, QuintetSpin> {
public : 
    static const SpinMatrix Jproj;

    static const SpinMatrix compute_singlet_projector(void) { 
       SpinPair<QuintetSpin, QuintetSpin> HJ;
       HJ.J = 1.0;
       HJ.update_hamiltonian();
       HJ.diag();
       return HJ.evec;
    }


    QuintetPair(void) 
    {
    }

    static const SpinMatrix singlet_projector(void) { 
       return Jproj.row(0).adjoint() * Jproj.row(0);
    }

};

const QuintetPair::SpinMatrix QuintetPair::Jproj ( QuintetPair::compute_singlet_projector() );




#endif
