#
# Makefile for GENIE Neutrino Generator
#
# Costas Andreopoulos <c.andreopoulos \at cern.ch>
#

SHELL    = /bin/sh
NAME     = all
MAKEFILE = Makefile

# Include machine specific flags and locations (inc. files & libs)
include $(GENIE)/src/make/Make.include

PACKAGE           = Physics/QuasiElastic/XSection
PACKAGE_ABBREV    = PhQELXS
DICTIONARY        = _ROOT_DICT_$(PACKAGE_ABBREV)
LIBNAME           = libG$(PACKAGE_ABBREV)
EXTRA_EXT_LIBS    = $(GPUSPLINE_LIBS)

all     : rootcint lib lib-link
install : install-inc install-lib

# Rebuild when ./configure changes GPU support. QELGpuUtils is a .cpp file so
# rootcling never parses its header (which pulls in the GPU toolkit headers).
NewQELXSec.o QELGpuUtils.o: $(GENIE)/src/make/Make.config QELGpuUtils.h \
  $(wildcard $(GENIE)/gpu_spline/include/gpu_spline/*.h)
QELGpuUtils.o: QELGpuUtils.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@

# Include standard package makefile targets
#
include $(GENIE)/src/make/Make.std-package-targets

FORCE:
