// Explicit compatibility prescriptions for NuWro mec_kind=6, density_model=1.
// Reference implementation: NuWro ca4d1d9690a44e261c26f690b833c87ce6be4058,
// nucleus_data.cc, nucleus.{h,cc}, mecevent_2020Valencia.cc.
#ifndef GENIE_VALENCIA2020_NUWRO_H
#define GENIE_VALENCIA2020_NUWRO_H
#include <array>
#include <map>
#include <TVector3.h>
#include "Physics/Multinucleon/EventGen/Valencia2020Kinematics.h"

namespace genie { namespace utils { namespace valencia2020 {

inline double NuWroThreeBodyChargeWeight(int Z, int N, int finalZ, bool anti)
{
  // Combinatorics use the charge-exchanged target; remnant availability and
  // baryon/charge bookkeeping still refer to the original nucleus.
  const int initialZ = finalZ + (anti ? 1 : -1);
  if (initialZ < 0 || initialZ > 3 || initialZ > Z || 3-initialZ > N) return 0.;
  return Choose(Z+(anti ? -1:1),finalZ)*Choose(N+(anti ? 1:-1),3-finalZ);
}

inline int NuWroAngularChannel(int channel, bool anti)
{
  if (channel < 0 || channel > 2) throw std::invalid_argument("Not a two-body channel");
  // Native code classifies by out[0] and flag_pn, including nn as 'np'.
  const int conjugate[] = {1,2,1};
  return anti ? conjugate[channel] : channel;
}

inline TVector3 NuWroDirection(TRandom& rng)
{
  const double c = 2.*rng.Rndm()-1., phi = 2.*std::acos(-1.)*rng.Rndm();
  const double s = std::sqrt(std::max(0.,1.-c*c));
  return TVector3(s*std::cos(phi),s*std::sin(phi),c);
}

inline TVector3 NuWroFermiSphere(TRandom& rng, double kf)
{
  const double p = kf*std::cbrt(rng.Rndm());
  return p*NuWroDirection(rng);
}

// Point-nucleon Fourier-Bessel profiles. The four tagged NuWro profiles are
// C12, O16, Ar40 and Fe56. Nearest-profile selection and A^(1/3) dilation are
// the native density_model=1 prescription (Ca40 therefore uses Ar40).
class NuWroDensity {
public:
  NuWroDensity(int Z, int N) : fZ(Z), fA(Z+N)
  {
    if (Z < 0 || N < 0 || fA < 4) throw std::invalid_argument("Invalid NuWro density target");
    const int zs[] = {6,8,18,26}, as[] = {12,16,40,56};
    const double profiles[4][18] = {
      {8.00043364,.01524311,.04341582,.03984473,.01843899,-.00501202,-.01579038,
       -.01155837,-.00453332,-.00194829,-.00044933,.00099081,.00005054,
       -.00085590,.00047011,.00056445,-.00111315,.00057377},
      {8.00049515,.02019217,.05017159,.03650542,.00516551,-.01675677,-.01575877,
       -.00505903,-.00164696,-.00181031,.00003029,.00006469,-.00058935,
       .00073632,.00017952,-.00104117,.00099052,-.00034560},
      {9.00051447,.03049435,.05906711,.02004217,-.01741058,-.01814531,-.00039990,
       .00285990,-.00202572,.00025578,.00091830,-.00078574,.00035888,
       -.00030213,-.00021802,.00128401,-.00152466,.00062320},
      {9.00072200,.04231463,.06605488,-.00095655,-.03714238,-.01080343,.01555390,
       .00624856,-.00255266,-.00054963,-.00029042,.00098031,-.00038472,
       -.00071218,.00113815,-.00090063,.00064549,-.00028920}
    };
    int best=0, distance=1000000000;
    for (int i=0; i<4; ++i) {
      const int dz=zs[i]-Z, dn=as[i]-zs[i]-N, da=as[i]-fA;
      const int d=dz*dz+dn*dn+da*da;
      if (d<distance) { distance=d; best=i; }
    }
    std::copy(profiles[best],profiles[best]+18,fProfile.begin());
    fRadius=fProfile[0]*std::cbrt(double(fA)/as[best]);
    fNormalization=double(as[best])/zs[best];
    // Invert the radial CDF. 8192 intervals resolve the 17-term expansion;
    // the density itself and kF are always evaluated analytically.
    fCDF[0]=0.;
    double previous=0.;
    for (unsigned i=1; i<fCDF.size(); ++i) {
      const double r=fRadius*i/(fCDF.size()-1), value=r*r*Density(r);
      fCDF[i]=fCDF[i-1]+.5*(previous+value)*fRadius/(fCDF.size()-1);
      previous=value;
    }
    for (double& c:fCDF) c/=fCDF.back();
  }
  double Radius() const { return fRadius; }
  double Density(double r) const
  {
    if (r < 0. || r > fRadius) return 0.;
    const double x=std::acos(-1.)*r/fRadius;
    double sum=0.;
    for (int j=1; j<=17; ++j) {
      const double y=j*x;
      sum+=fProfile[j]*(std::abs(y)<1.e-12 ? 1. : std::sin(y)/y);
    }
    return fNormalization*std::max(0.,sum);
  }
  double FermiMomentum(int pdg, double r) const
  {
    const double fraction=double(pdg==2212 ? fZ:fA-fZ)/fA;
    return .1974*std::cbrt(3.*std::acos(-1.)*std::acos(-1.)*Density(r)*fraction);
  }
  TVector3 SampleVertex(TRandom& rng) const
  {
    const double u=rng.Rndm();
    auto upper=std::upper_bound(fCDF.begin(),fCDF.end(),u);
    const unsigned i=std::min(unsigned(upper-fCDF.begin()),unsigned(fCDF.size()-1));
    const double fraction=(u-fCDF[i-1])/(fCDF[i]-fCDF[i-1]);
    return (fRadius*(i-1+fraction)/(fCDF.size()-1))*NuWroDirection(rng);
  }
private:
  int fZ, fA;
  double fRadius, fNormalization;
  std::array<double,18> fProfile;
  std::array<double,8193> fCDF;
};

inline const NuWroDensity& NuWroDensityFor(int Z, int N)
{
  static thread_local std::map<std::pair<int,int>,NuWroDensity> cache;
  const auto key=std::make_pair(Z,N);
  auto entry=cache.find(key);
  if (entry==cache.end()) entry=cache.emplace(key,NuWroDensity(Z,N)).first;
  return entry->second;
}

}}}
#endif
