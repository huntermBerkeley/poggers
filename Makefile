TARGETS=power_of_one_test team_test

ifdef D
	DEBUG=-g -G
	OPT=
else
	DEBUG=
	OPT=-O3
endif

ifdef NH
	ARCH=
else
	ARCH=-msse4.2 -D__SSE4_2_
endif

ifdef P
	PROFILE=-pg -no-pie # for bug in gprof.
endif

LOC_INCLUDE=include
LOC_SRC=src
LOC_TEST=test
OBJDIR=obj



CC = gcc -std=gnu11
CXX = g++ -std=c++11
CU = nvcc -dc -x cu
LD = nvcc

CXXFLAGS = -Wall $(DEBUG) $(PROFILE) $(OPT) $(ARCH) -m64 -I. -Iinclude

CUFLAGS = $(DEBUG) $(OPT) -arch=sm_70 -rdc=true -I. -Iinclude

CUDALINK = -L/usr/common/software/sles15_cgpu/cuda/11.1.1/lib64/compat -L/usr/common/software/sles15_cgpu/cuda/11.1.1/lib64 -L/usr/common/software/sles15_cgpu/cuda/11.1.1/extras/CUPTI/lib6 -lcurand --nvlink-options -suppress-stack-size-warning

LDFLAGS = $(DEBUG) $(PROFILE) $(OPT) $(CUDALINK) -arch=sm_70 -lpthread -lssl -lcrypto -lm -lcuda -lcudart


#
# declaration of dependencies
#

all: $(TARGETS)

# dependencies between programs and .o files

test:							$(OBJDIR)/test.o \
								$(OBJDIR)/vqf_block.o

team_test:							$(OBJDIR)/team_test.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o

vqf_tests:						$(OBJDIR)/vqf_tests.o \
								$(OBJDIR)/vqf.o \
								$(OBJDIR)/vqf_block.o

team_vqf_tests:						$(OBJDIR)/team_vqf_tests.o \
								$(OBJDIR)/team_vqf.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o


sort_vqf_tests:						$(OBJDIR)/sort_vqf_tests.o \
								$(OBJDIR)/team_vqf.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o

vqf_correctness_check:				$(OBJDIR)/vqf_correctness_check.o \
								$(OBJDIR)/team_vqf.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o

locking_vqf_test:						$(OBJDIR)/locking_vqf_test.o \
								$(OBJDIR)/vqf.o \
								$(OBJDIR)/vqf_block.o


small_vqf_tests:						$(OBJDIR)/small_vqf_tests.o \
								$(OBJDIR)/team_vqf.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o

mega_vqf_tests:						$(OBJDIR)/mega_vqf_tests.o \
								$(OBJDIR)/mega_vqf.o \
								$(OBJDIR)/megablock.o \
								$(OBJDIR)/warp_utils.o

power_of_one_test:					$(OBJDIR)/power_of_one_test.o \
								$(OBJDIR)/single_vqf.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o \
								$(OBJDIR)/hashutil.o


shared_test:					$(OBJDIR)/shared_test.o

# dependencies between .o files and .cc (or .c) files


#$(OBJDIR)/RSQF.o: $(LOC_SRC)/RSQF.cu $(LOC_INCLUDE)/RSQF.cuh

$(OBJDIR)/vqf_block.o: $(LOC_SRC)/vqf_block.cu $(LOC_INCLUDE)/vqf_block.cuh
$(OBJDIR)/vqf.o: $(LOC_SRC)/vqf.cu $(LOC_INCLUDE)/vqf.cuh $(LOC_SRC)/vqf_block.cu $(LOC_INCLUDE)/vqf_block.cuh
$(OBJDIR)/vqf_team_block.o: $(LOC_SRC)/vqf_team_block.cu $(LOC_INCLUDE)/vqf_team_block.cuh
$(OBJDIR)/warp_utils.o: $(LOC_SRC)/warp_utils.cu $(LOC_INCLUDE)/warp_utils.cuh
$(OBJDIR)/team_vqf.o: $(LOC_SRC)/team_vqf.cu $(LOC_INCLUDE)/team_vqf.cuh $(LOC_SRC)/vqf_team_block.cu $(LOC_INCLUDE)/vqf_team_block.cuh
$(OBJDIR)/megablock.o: $(LOC_SRC)/megablock.cu $(LOC_INCLUDE)/megablock.cuh
$(OBJDIR)/mega_vqf.o: $(LOC_SRC)/mega_vqf.cu $(LOC_INCLUDE)/mega_vqf.cuh $(LOC_SRC)/megablock.cu $(LOC_INCLUDE)/megablock.cuh
$(OBJDIR)/single_vqf.o: $(LOC_SRC)/single_vqf.cu $(LOC_INCLUDE)/single_vqf.cuh $(LOC_SRC)/vqf_team_block.cu $(LOC_INCLUDE)/vqf_team_block.cuh
$(OBJDIR)/hashutil.o: $(LOC_SRC)/hashutil.cu $(LOC_INCLUDE)/hashutil.cuh


#
# generic build rules
#

$(TARGETS):
	$(LD) $^ -o $@ $(LDFLAGS)


$(OBJDIR)/%.o: $(LOC_SRC)/%.cu | $(OBJDIR)
	$(CU) $(CUFLAGS) $(INCLUDE) -dc $< -o $@




$(OBJDIR)/%.o: $(LOC_SRC)/%.cc | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -c -o $@

$(OBJDIR)/%.o: $(LOC_SRC)/%.c | $(OBJDIR)
	$(CC) $(CXXFLAGS) $(INCLUDE) $< -c -o $@

$(OBJDIR)/%.o: $(LOC_TEST)/%.c | $(OBJDIR)
	$(CC) $(CXXFLAGS) $(INCLUDE) $< -c -o $@

$(OBJDIR):
	@mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGETS) core
