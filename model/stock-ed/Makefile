#------------------------------------------------------------------------------#
#                     Makefile for ED and Miami-LU models                      #
#------------------------------------------------------------------------------#


CXX = gcc

ED = /path/to/EDv3_code
NETCDF = $(ED)/external_apps/installed_apps/netcdf/4.1.3
TBB = $(ED)/external_apps/installed_apps/IntelTBB/2017U3
BERDB = $(ED)/external_apps/installed_apps/BerkeleyDB/4.6.21
CONGLIB = $(ED)/external_apps/installed_apps/libconfig/1.7.3

INC = -I$(NETCDF)/include -I$(TBB)/include -I$(BERDB)/include -I$(CONGLIB)/include

LIB = -L$(NETCDF)/lib -L$(TBB)/lib -L$(BERDB)/lib -L$(CONGLIB)/lib


# add  -g for debug, -g3 for debug and optomize, -O2 for optimize
#      -Wall for full warnings
#      -pg for profiling

CXXFLAGS = $(INC) -c -Wall
LDFLAGS = $(LIB) -lm -lnetcdf -lnetcdf_c++ -ltbb -ldb_cxx -lconfig++ -lstdc++

CMN_SRCS = site.cc patch.cc miami.cc belowgrnd.cc \
           disturbance.cc fire.cc landuse.cc read_site_data.cc init_data.cc \
           outputter.cc print_output.cc restart.cc readconfiguration.cc photosynthesis.cc

EDM_SRCS = cohort.cc growth.cc allometry.cc phenology.cc mortality.cc \
           mechanism.cc odeint.cc


# Can't use target specific variables because they aren't available 
# when needed for dependency includes
ifeq ($(MAKECMDGOALS),mlu)
	TGT = mlu
   CXXFLAGS += -DMIAMI_LU -DMAIN
	SRCS = $(CMN_SRCS) main.cc
else ifeq ($(MAKECMDGOALS),ed_mpi)
   TGT = ed_mpi
   CXX = mpicxx
   CXXFLAGS += -DED -DUSEMPI -DMAIN
   SRCS = $(CMN_SRCS) $(EDM_SRCS) edmpi.cc main.cc
else ifeq ($(MAKECMDGOALS),libmlu.a)
	TGT = libmlu
   CXXFLAGS += -DMIAMI_LU -DCOUPLED
   SRCS = $(CMN_SRCS) edm_ied_interface.cc
else ifeq ($(MAKECMDGOALS),libed.a)
	TGT = libed
   CXXFLAGS += -DED -DCOUPLED
   SRCS = $(CMN_SRCS) $(EDM_SRCS) edm_ied_interface.cc
else
	TGT = edlu
   CXXFLAGS += -DED -DMAIN
   SRCS = $(CMN_SRCS) $(EDM_SRCS) main.cc
endif

OBJS = $(SRCS:.cc=.o)

all: edlu

edlu ed_mpi mlu: $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

libed.a libmlu.a: $(OBJS)
	ar rcs $@ $(OBJS)

%.tgt: 
	@rm -f *.tgt; \
	touch $@; 

# make sure that .o and .d files get rebuilt if we switch targets
$(OBJS) $(SRCS:.cc=.d): $(TGT).tgt

# automatically determine header dependencies 
%.d: %.cc
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

ifneq ($(MAKECMDGOALS),clean)
   -include $(SRCS:.cc=.d)
endif


.PHONY: clean
clean:
	- rm -f *.d *.o core.* edlu mlu

