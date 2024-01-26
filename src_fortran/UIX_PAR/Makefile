CC          = mpicc
F77         = mpif77
#MAKRO       = -DSUN_FORTRAN
#MAKRO       = -DPC_FORTRAN
#LIBS        = -parallel -static -L/opt/intel/mkl52sp1/lib/32 -lmkl_lapack -lmkl_p4
FOPTS       = -Ofast -ffpe-summary=none -mcmodel=medium -fallow-argument-mismatch
# -mcmodel=large
#LIBS        = -L/home/kirscher/lib/intel/mkl/10.0.1.014/lib/em64t -lmkl_em64t -lguide

LIBS    = -L -lgfortran
CFLAGS      = -o $@ $? $(PRECFLAGS)

.c.o:
	$(CC) $(COPTS) $(MAKRO) -c -o $*.o $< 

.f.o:
	$(F77) $(FOPTS) -c -o $*.o $< 

.F.o:
	$(F77) $(FOPTS) $(MAKRO) -c -o $*.o $<

clean:
	rm -f *.o core

mpi_drqua_v7: mpi_drqua_v7.o
	$(F77) $(FFLAGS) $(MAKRO) -o mpi_drqua_v7 mpi_drqua_v7.o $(LIBS) $(MPI)

SAMMEL-uix: SAMMEL-uix.o  par.h
	$(F77) $(FOPTS) $(MAKRO) -o $@ SAMMEL-uix.o $(LIBS)

all: mpi_drqua_v7 SAMMEL-uix clean
