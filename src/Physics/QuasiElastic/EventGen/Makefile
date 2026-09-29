#
# Makefile for GENIE Neutrino Generator
#
# Costas Andreopoulos <c.andreopoulos \at cern.ch>
#

SHELL    = /bin/sh
NAME     = all
MAKEFILE = Makefile

# Include machine specific flags and locations (inc. files & libs)
#
include $(GENIE)/src/make/Make.include

PACKAGE           = Physics/QuasiElastic/EventGen
PACKAGE_ABBREV    = PhQELEG
DICTIONARY        = _ROOT_DICT_$(PACKAGE_ABBREV)
LIBNAME           = libG$(PACKAGE_ABBREV)
EXTRA_EXT_LIBS    = $(XML_LIBRARIES) -L$(GENIE_LIB_PATH) -lGFwAlg -lGFwEG -lGFwGHEP -lGPhQELXS -lGPhNuclSt -lGPhHadTens -L$(GENIE)/gpu_spline/lib -lgpuspline -L/usr/local/cuda-12.9/lib64 -lcudart -Wl,-rpath,$(GENIE)/gpu_spline/lib -Wl,-rpath,/usr/local/cuda-12.9/lib64

all     : rootcint lib lib-link
install : install-inc install-lib

# Include standard package makefile targets
#
include $(GENIE)/src/make/Make.std-package-targets

FORCE:
