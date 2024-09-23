CC          = mpiicx
F77         = mpiifx
FOPTS       = -Ofast -mcmodel=medium #-fallow-argument-mismatch -ffpe-summary=none 

LIBS    = -L${MKLROOT} -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
CFLAGS      = -o $@ $? $(PRECFLAGS)

.c.o:
	$(CC) $(COPTS) -c -o $*.o $< 

.f.o:
	$(F77) $(FOPTS) -c -o $*.o $< 

.F.o:
	$(F77) $(FOPTS) -c -o $*.o $<

clean:
	rm -f *.o core


mpi_drqua_n_v18-uix.o:  mpi_drqua_n_v18-uix.F  par.h Makefile

mpi_drqua_n_v18-uix: mpi_drqua_n_v18-uix.o
	$(F77) $(FFLAGS)  -o mpi_drqua_n_v18-uix mpi_drqua_n_v18-uix.o $(LIBS) $(MPI)

mpi_drqua_v7: mpi_drqua_v7.o
	$(F77) $(FFLAGS) -o mpi_drqua_v7 mpi_drqua_v7.o $(LIBS) $(MPI)

SAMMEL-uix: SAMMEL-uix.o  par.h
	$(F77) $(FOPTS) -o $@ SAMMEL-uix.o $(LIBS)

all: mpi_drqua_v7 mpi_drqua_n_v18-uix SAMMEL-uix clean
