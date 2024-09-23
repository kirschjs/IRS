# Disable the default rules
MAKEFLAGS += --no-builtin-rules --no-builtin-variables

# Project name
NAME := juelmanoo

# Configuration settings
FC := ifx
FFLAGS:= -Ofast -mcmodel=medium -I../trig
LD := $(FC) 
RM := rm -f
LIBS:= -L${MKLROOT} -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

# List of all source files
SRCS := trig.f90 \
	juelmanoo_pc.f90 \
	juelmanoo_subs.f90 \
	juelmanoo.f90 

# Create lists of the build artefacts in this project
OBJS =  $(SRCS:.f90=.o)
#OBJS:=$(addsuffix .o, $(SRCS))
EXE := $(NAME).exe

# Declare all public targets
.PHONY: all clean
all: $(EXE)

# Create the static library from the object files
$(EXE): $(OBJS)
	$(LD) $(LIBS) -o $(EXE) $(OBJS) 

# Create object files from Fortran source
$(OBJS) : %.o:
	$(FC) $(FFLAGS) -c -o $*.o $*.f90

# Define all module interdependencies
#csv_kinds.mod := src/csv_kinds.f90.o

# Cleanup, filter to avoid removing source code by accident
clean:
	$(RM) *.o *.exe  *.mod
