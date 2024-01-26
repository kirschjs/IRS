# hexpresso
F77     =       gfortran
FFLAGS  =       -Ofast -ffpe-summary='none' -mcmodel=medium #-fallow-argument-mismatch # 
LIBS    = -L -lgfortran -L/usr/lib/x86_64-linux-gnu/lapack -llapack -lblas
#LIBS    = -L${MKLROOT} -llapack -lblas  -lm -ldl

#theogate
#F77     =       ifort
#FFLAGS  =       -Ofast -ffpe-summary='none'  -mcmodel=large
#LIBS    = -L${MKLROOT} -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

# ohne -g und -traceback erzeugt spoleinv runtime error in smatest,rrandb subroutinen wenn auf die variable nbess zugegriffen wird;
# mit den optionen tritt der fehler nicht auf;

.f.o:
	$(F77) $(FFLAGS) -c -o $*.o $<


clean:
	rm -f *.o core

QUAFL_M: QUAFL_M.o par/QUAF
	$(F77) $(FFLAGS) -o QUAFL_M.exe QUAFL_M.o 

LUDW_CN: LUDW_CN.o par/LUCN
	$(F77) $(FFLAGS) -o LUDW_CN.exe LUDW_CN.o $(LIBS)

DRQUA_AK_M: DRQUA_AK_M.o par/DRQUA_AK
	$(F77) $(FFLAGS) -o DRQUA_AK_M.exe DRQUA_AK_M.o $(LIBS)

TDR2END_AK: TDR2END_AK.o par/DR2END_AK
	$(F77) $(FFLAGS) -o TDR2END_AK.exe TDR2END_AK.o $(LIBS)

TDR2END_NORMAL: TDR2END_NORMAL.o par/DR2END_AK
	$(F77) $(FFLAGS) -o TDR2END_NORMAL.exe TDR2END_NORMAL.o $(LIBS)

TDR2END_PYpoolnoo: TDR2END_PYpoolnoo.o par/DR2END_AK
	$(F77) $(FFLAGS) -o TDR2END_PYpoolnoo.exe TDR2END_PYpoolnoo.o $(LIBS)

DR2END_AK: DR2END_AK.o par/DR2END_AK
	$(F77) $(FFLAGS) -o DR2END_AK.exe DR2END_AK.o $(LIBS)

DR2END_NORMAL: DR2END_NORMAL.o par/DR2END_AK
	$(F77) $(FFLAGS) -o DR2END_NORMAL.exe DR2END_NORMAL.o $(LIBS)

DRLUD: DRLUD.o par/DRLUD
	$(F77) $(FFLAGS) -o DRLUD.exe DRLUD.o $(LIBS)

DROBER: DROBER.o par/DROBER
	$(F77) $(FFLAGS) -o DROBER.exe DROBER.o $(LIBS)

KOBER: KOBER.o par/KOBER
	$(F77) $(FFLAGS) -o KOBER.exe KOBER.o $(LIBS)

juelmanoo: juelmanoo.o par/juelma
	$(F77) $(FFLAGS) -o juelmanoo.exe juelmanoo.o $(LIBS)

jobelmanoo: jobelmanoo.o par/jobelma
	$(F77) $(FFLAGS) -o jobelmanoo.exe jobelmanoo.o $(LIBS)

jquelmanoo: jquelmanoo.o par/jquelma
	$(F77) $(FFLAGS) -o jquelmanoo.exe jquelmanoo.o

jenelmasnoo: jenelmasnoo.o par/jenelmas
	$(F77) $(FFLAGS) -o jenelmasnoo.exe jenelmasnoo.o  $(LIBS)

all: LUDW_CN KOBER DROBER DRLUD QUAFL_M DRQUA_AK_M DR2END_NORMAL TDR2END_NORMAL TDR2END_PYpoolnoo juelmanoo jobelmanoo jquelmanoo jenelmasnoo clean