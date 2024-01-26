CC          = mpicc
F77         = mpif77

FOPTS  =       -Ofast -ffpe-summary=none -mcmodel=medium -fallow-argument-mismatch

#COPTS       = -O2 -Wall -ggdb3
#COPTS       = -O2
# option -xW generates code exclusivly for P4 processors
#FOPTS       = -O2 -xW -mcmodel medium -shared-intel
#MAKRO       = -DPC_FORTRAN 
#-DDEBUG_INT -DDEBUG_DOUBLE
#RM          = /bin/rm -f
#LIBS    =       -L/home/hmh/lib -L/opt/intel_mkl_61/lib -leispack -lsu2 -lmkl -lmkl_lapack -lmkl_lapack32 -lguide -lblas
#    "$@" expands to the target; "$?" expands to the dependency list
CFLAGS      = -o $@ $? $(PRECFLAGS)

.c.o:
	$(CC) $(COPTS) $(MAKRO) -c -o $*.o $< 

.f.o:
	$(F77) $(FOPTS) -c -o $*.o $< 

.F.o:
	$(F77) $(FOPTS) $(MAKRO) -c -o $*.o $< 

mpi_quaf_v7.o: mpi_quaf_v7.F par.h Makefile	

mpi_quaf_v7: mpi_quaf_v7.o 
	$(F77) $(FOPTS) $(MAKRO) -o $@ mpi_quaf_v7.o   $(LIBS) $(MPI) 

sammel: sammel.o
	mpif77 -Ofast -ffpe-summary=none -mcmodel=medium -o $@ sammel.o $(LIBS)

clean:
	$(RM) *.o *.exe pvm_tniq *~

all: mpi_quaf_v7 sammel clean
