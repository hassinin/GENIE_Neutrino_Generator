#
# Makefile for GENIE Neutrino Generator
#
# Author: Costas Andreopoulos <c.andreopoulos \at cern.ch>
#

SHELL    = /bin/sh
NAME     = all
MAKEFILE = Makefile

# Include machine specific flags and locations (inc. files & libs)
#
include $(GENIE)/src/make/Make.include

PACKAGE           = Physics/HadronTensors
PACKAGE_ABBREV    = PhHadTens
DICTIONARY        = _ROOT_DICT_$(PACKAGE_ABBREV)
LIBNAME           = libG$(PACKAGE_ABBREV)
EXTRA_EXT_LIBS    = $(GPUSPLINE_LIBS)

all     : rootcint lib lib-link
install : install-inc install-lib

# GPU code is compiled in or out depending on ./configure --enable-gpu.
TabulatedLabFrameHadronTensor.o: $(GENIE)/src/make/Make.config
ifeq ($(strip $(GOPT_ENABLE_GPU)),YES)
TabulatedLabFrameHadronTensor.o: $(GENIE)/gpu_spline/include/gpu_spline/gpu_hadron_tensor.h
endif

# Include standard package makefile targets
#
include $(GENIE)/src/make/Make.std-package-targets

FORCE:
