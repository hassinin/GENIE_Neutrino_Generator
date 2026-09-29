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

PACKAGE           = Physics/Multinucleon/XSection
PACKAGE_ABBREV    = PhMNucXS
DICTIONARY        = _ROOT_DICT_$(PACKAGE_ABBREV)
LIBNAME           = libG$(PACKAGE_ABBREV)
EXTRA_EXT_LIBS    = $(XML_LIBRARIES) -L$(GENIE_LIB_PATH) -lGPhXSIg -lGFwEG -lGFwGHEP -lGPhCmn -lGFwUtl -lGPhHadTens -lGPhQELXS -L$(GENIE)/gpu_spline/lib -lgpuspline -L/usr/local/cuda-12.9/lib64 -lcudart

all     : rootcint lib lib-link
install : install-inc install-lib

# Include standard package makefile targets
#
include $(GENIE)/src/make/Make.std-package-targets

FORCE:
