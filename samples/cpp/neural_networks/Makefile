#===============================================================================
# Copyright 2017-2018 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#
# License:
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
# eement/
#===============================================================================

##  Content:
##     Intel(R) Data Analytics Acceleration Library samples creation and run
##******************************************************************************

help:
	@echo "Usage: make {libia32|soia32|libintel64|sointel64|help}"
	@echo "[compiler=compiler_name] [mode=mode_name] [sample=sample_name]"
	@echo
	@echo "compiler_name       - intel or gnu"
	@echo "                      Intel(R) C++ Compiler as default"
	@echo
	@echo "mode_name           - can be build or run. Default is run"
	@echo "sample_name         - sample name. Please see daal.lst file"

##------------------------------------------------------------------------------
## samples of using:
##
## make sointel64                - build by Intel(R) C++ Compiler
##                                 and run all samples for Intel(R)64 processor
##                                 family  applications, dynamic linking
##
## make sointel64 mode=build     - build only (not run) by Intel(R) C++ Compiler
##                                 all samples for Intel(R)64
##                                 processor family  applications, dynamic linking
##
## make help                     - show help
##
##------------------------------------------------------------------------------

include daal.lst

ifndef compiler
    compiler = intel
endif

ifndef DAALROOT
    DAALROOT = ./../..
endif

ifndef TBBROOT
    TBBROOT = ./../../../tbb
endif
TBB_PATH = "$(TBBROOT)/lib/$(_IA)_lin/gcc4.4"

ifndef DATA_PATH
    DATA_PATH = ./data
endif

ifndef sample
    sample = $(NEURAL_NETWORKS)
endif

DAAL_PATH = "$(DAALROOT)/lib/$(_IA)_lin"
DAAL_LIB := $(DAAL_PATH)/libdaal_core.$(RES_EXT) $(DAAL_PATH)/libdaal_thread.$(RES_EXT)
RES_DIR=_results/$(compiler)_$(_IA)_$(RES_EXT)

EXT_LIB := -L$(TBB_PATH) -ldl

ifneq ($(mode),build)
    override mode = run
endif

ifeq ($(threading),sequential)
    DAAL_LIB_T := $(DAAL_PATH)/libdaal_sequential.$(RES_EXT)
else
    override threading = parallel
    EXT_LIB += -L$(TBB_PATH) -ltbb -ltbbmalloc -lpthread
endif

ifeq ($(debug),on)
   COPTS := -Wall -w -ggdb -DDEBUG
else
   COPTS := -Wall -w -ggdb
endif
LOPTS := -Wl,--start-group $(DAAL_LIB) $(EXT_LIB) -Wl,--end-group

override proc = 4

RES = $(addprefix $(RES_DIR)/, $(if $(filter run, $(mode)), $(addsuffix .res ,$(sample)), $(addsuffix .exe,$(sample))))

ifeq ($(compiler),intel)
    CC = icc
endif

ifeq ($(compiler),gnu)
    CC = g++
    COPTS += $(if $(filter ia32, $(_IA)), -m32, -m64)
endif


.SECONDARY:
$(RES_DIR)/%.exe: ./sources/%.cpp | $(RES_DIR)/.
	$(CC) $(COPTS) $< -o $@ $(LOPTS)

$(RES_DIR)/%.res: $(RES_DIR)/%.exe
	$(CRUN) $(RUNOPTS) $< $(DATA_PATH) > $@

LOPTS := $(DAAL_LIB) $(EXT_LIB)

_make_ex: $(RES)

%/.:; mkdir -p $*

libia32:
	$(MAKE) _make_ex _IA=ia32 RES_EXT=a
soia32:
	$(MAKE) _make_ex _IA=ia32 RES_EXT=so
libintel64:
	$(MAKE) _make_ex _IA=intel64 RES_EXT=a
sointel64:
	$(MAKE) _make_ex _IA=intel64 RES_EXT=so
