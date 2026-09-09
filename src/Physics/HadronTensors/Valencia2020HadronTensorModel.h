//____________________________________________________________________________
/*!

\class    genie::Valencia2020HadronTensorModel

\brief    Creates hadron tensor objects for calculations of exclusive MEC
          cross sections using the Valencia 2020 model (Sobczyk et al.)

\ref      J.E. Sobczyk, J. Nieves, F. Sánchez, Phys. Rev. C 102, 024601 (2020)

\cpright  Copyright (c) 2003-2026, The GENIE Collaboration
          For the full text of the license visit http://copyright.genie-mc.org
*/
//____________________________________________________________________________

#ifndef _VALENCIA2020_HADRON_TENSOR_MODEL_H_
#define _VALENCIA2020_HADRON_TENSOR_MODEL_H_

// GENIE includes
#include "Physics/HadronTensors/TabulatedHadronTensorModelI.h"

namespace genie {

class Valencia2020HadronTensorModel : public TabulatedHadronTensorModelI {

public:

  Valencia2020HadronTensorModel();
  Valencia2020HadronTensorModel(std::string config);

  virtual ~Valencia2020HadronTensorModel();

protected:

  // Implementation of TabulatedHadronTensorModelI interface
  virtual HadronTensorI* ParseTensorFile( const std::string& full_file_name ) const;

};

} // namespace genie

#endif // _VALENCIA2020_HADRON_TENSOR_MODEL_H_
