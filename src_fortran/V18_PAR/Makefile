CC          = mpiicx
F77         = mpiifx
FOPTS       = -Ofast -mcmodel=medium #-fallow-argument-mismatch -ffpe-summary=none 

LIBS    = -L${MKLROOT} -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

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
	 $(F77) $(FOPTS) $(MAKRO) -o $@ sammel.o $(LIBS)

clean:
	$(RM) *.o *.exe pvm_tniq *~

all: mpi_quaf_v7 sammel clean
