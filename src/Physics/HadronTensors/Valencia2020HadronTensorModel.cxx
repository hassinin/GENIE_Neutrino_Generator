//____________________________________________________________________________
/*
 Copyright (c) 2003-2026, The GENIE Collaboration
 For the full text of the license visit http://copyright.genie-mc.org
 or see $GENIE/LICENSE

 For the class documentation see the corresponding header file.
*/
//____________________________________________________________________________

// GENIE includes
#include "Physics/HadronTensors/Valencia2020HadronTensorModel.h"
#include "Physics/HadronTensors/TabulatedHadronTensorModelI.h"
#include "Physics/HadronTensors/TabulatedLabFrameHadronTensor.h"

//____________________________________________________________________________
genie::Valencia2020HadronTensorModel::Valencia2020HadronTensorModel()
  : genie::TabulatedHadronTensorModelI("genie::Valencia2020HadronTensorModel")
{

}

//____________________________________________________________________________
genie::Valencia2020HadronTensorModel::Valencia2020HadronTensorModel(std::string config)
  : genie::TabulatedHadronTensorModelI("genie::Valencia2020HadronTensorModel", config)
{

}

//____________________________________________________________________________
genie::Valencia2020HadronTensorModel::~Valencia2020HadronTensorModel()
{

}

//____________________________________________________________________________
genie::HadronTensorI* genie::Valencia2020HadronTensorModel::ParseTensorFile(
  const std::string& full_file_name) const
{
  return new TabulatedLabFrameHadronTensor( full_file_name );
}
