      PROGRAM QUAF_S
C     FUER SMIN STATTDESSEN
CC    SUBROUTINE QUAF
      IMPLICIT REAL*8 (A-H,O-Z)
      include 'mpif.h'
C 
C     FUER ARGONNE-18 POTENTIAL
C     VERSION FUER 6 CLUSTER
C     VERLANGT KURZOBFILES!!!!!!!!!!!!!!!!
C     FUER SMIN VORBEREITET
C     JEWEILS 'SMIN' SUCHEN UND DIE ENTSPRECHENDEN AENDERUNGEN DURCHFUEHREN
C     DIE LAENGE DES COMMON BLOCKS 'COMOV' MUSS MIT DER IN SMINH PASSEN
C     DESHALB DURCHGEHEND 'par.QUAF' durch 'par.SMIN' ANDERN
C
C     QUAF BERECHNET DIE REDUZIERTEN QRSTRAUMMATRIXELEMENTE
C     UND MULTILPLIZIERT SIE MIT DEN REDUZIERTEN SPINMATRIX-
C     ELEMENTEN
C
C
C     29.7.97 IN BEGRI LOOP 166 MODIFIZIERT H.M.H.
C     16.12.97 NUM(3 ZAEHLT JETZT DIE BASISVEKTOREN INNERHALB EINER
C              ZERLEGUNG, D.H. AUF DEN QUAOUT.n FILES STEHT NUR EINE
C              RELATIVE NUMMER, DAMIT AUFSETZEN IMMER MOEGLICH IST H.M.H.
C     16.1.98 NKSU VON KOBER WIRD ZUR ZUWEISUNG DER DREHIMPULSE BEI
C             6 CLUSTERN BENUTZT H.M.H.
C     30.1.98 DIE MATRIXELEMENTE WERDEN JETZT AUF FILES TQUAOUT.MFL
C             GESCHRIEBEN H.M.H.
C     1.9.98  POTENTIAL UND ZUGEHOERIGE PARAMETER WERDEN JETZT VON EXTRA
C             FILE EINGELESEN, DAMIT NNN-WECHSELWIRKUNG EINFACHER ZU     
C             HANDHABEN H.M.H. 
C     3.2.99 FUER INDEPO UNGLEICH NULL WIRD NUR DER KOPF VON KOBOUT GELESEN,
C            DIE MATRIXELEMENTE WERDEN VON EINZELNEN FILES GELESEN, DEREN     
C     19.4.5 IN BEGRI GEMEINSAME AUSDRUECKE ZUSAMMENGEFASST H.M.H.
C     19.4.5 IN MVK INDICES VERTAUSCHT UND DIREKTE PRODUKTE H.M.H.
C
C
C
C     1. OPERATOR: NORM
C     2. OPERATOR: KIN. ENERGIE
C     3. OPERATOR: COULOMB
C     4. OPERATOR: ZENTRAL
C     5. OPERATOR: P-QUADRAT
C     6. OPERATOR: L-QUADRAT
C     7. OPERATOR: LS-QUADRAT SKALAR
C     8. OPERATOR: SPIN-BAHN
C     9. OPERATOR: LS-QUADRAT VEKTOR
C     10. OPERATOR: TENSOR
C     11. OPERATOR: LS-QUADRAT TENSOR
C     12. OPERATOR: ISO-TENSOR SKALAR
C     13. OPERATOR: ISO-TENSOR TENSOR
C     14. OPERATOR: CHARGE ASYMMETRIC
C
C  
C     INPUT DES PROGRAMMS
C      INTEGERS IM FORMAT 24I3, REALS IM FORMAT 6E12.4
C
C      BEZEICHNUNG DER BAENDER
C      FILENAME FUER POTENTIAL,UNIT=25                                       
C     von unit=25: DIE STEUERKARTE FUER DIE ZU RECHNENDEN OPERATOREN  
C     von unit=25: DIE STEUERKARTEN FUER DIE KORRELATIONSFUNKTIONEN       
C     von unit=25: DIE DEFINITION DER POTENTIALE
C
C  EINGABE FUER DIE FUNKTIONEN
C
C JEDER INDEX I) ENTSPRICHT EINER KARTE
C
C DER FOLGENDE DATENSATZ MUSS FUER JEDE FUNKTION (K=1,NZF) WIEDERHOLT
C WERDEN
C  A) ZAHL DER PARAMETERSAETZE                                     NZPAR
C     DIE INNEREN WEITEPARAMETER
C     DIE RADIALWEITEPARAMETER
C WENN NBAND5=0,WIRD ALLES NEU GERECHNET,WENN NBAND5#0,WIRD AUFGESETZT
C
C
      include "par.h"
      
C     NZOPER: ANZAHL DER OPERATOREN IN QUAF
C     NZOPOB:   "     "      "      "  OBER
C     NZOPLU:   "     "      "      "  LUDWIG
C     NZTMAX: MAXIMALE ANZAHL DER TEILCHEN
C     NZFMAX:     "      "     "  ZERLEGUNGEN
C     NZCMAX:     "      "     "  CLUSTER
C     MZGMAX:     "      "     "  SPINFUNKTIONEN
C     NZLWMA:     "      "     "  DREHIMPULSSTRUKTUREN
C     NZRHOM:     "      "     "  BASISVEKTOREN PRO ZERLEGUNG
C     NZPARM:     "      "     "  SAETZTE INNERER WEITEN
C     MZPARM:     "      "     "  RADIALPARAMETER
C     NZPOMA:     "      "     "  POLYNOMSTRUKTUREN
C     NZIQMA:     "      "     "  SIGMAFAKTOREN AUS LUDWIG
C     NZPOTM:     "      "     "  POTENTIALE
C     NZSIOP:     "      "     "  SPIN-ISOSPIN OPERATOREN
C     NPDC:       "      "     "  PDC'S AUS OBER
C
C
      integer i, info, nproc, nhost
      INTEGER iStatus( MPI_STATUS_SIZE )
      integer myid
      integer speed
      character*18 nodename, host
      character*8 arch
      character*180 name, fnumber, qname, oname, cquaf
      character*180 cquaout, coutput
      CHARACTER*1 dummy(80)
      character*1 cdummy
      integer nteil

      INTEGER DATE_TIME (8)
      CHARACTER (LEN = 10) BIG_BEN (3)
c      
      parameter(lbeg = 1 + nzoper + 1,
     * lstore = nzparm*nzrhom*nzfmax + nzvmax*nztmax + nzpotm*6*nzoper +
     *         nztmax*nzvmax + mzgmax*mzgmax*nzsiop*npdc +
     *         nztmax*(nzcmax-1) + nzvmax*nzvmax*(2*nzcmax-1) +
     *         mzparm*nzfmax + (nzavma-1)*nzparm*nzfmax,
     * lluri = ndim1 + ndim1 + ndim1 + ndim1*nziqma + 1 + 1 +
     *        2*(nzlrec+1) + 1 + nzlrec+1 + (nziqma+1)*(nzlrec+1),
     * llurr = ndim1 + ndim4*ndim4,
     * limp = 1 + 1 + 1 + 1 + 1,
     * lkomi = 5*nzrhom*nzfmax + nzfmax + nzfmax + nzfmax + nzfmax +
     *        nzfmax + npdc + npdc + nztmax*npdc + 2*nzcmax*nzfmax +
     *        nzoper + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
     * lsc = 4 + 4 + 4,
     * lbig = ndim*ndim*2)
      PARAMETER (LIND=(NZFMAX*(NZFMAX+1))/2*NZOPER)
      DIMENSION INDEX(0:LIND)
C
      COMMON /STORE/ COF(NZPARM,NZRHOM,NZFMAX),
     *   RVEC(NZVMAX,NZTMAX), WECPOT(NZPOTM,6,NZOPER),
     *          SVEC(NZTMAX,NZVMAX), U(MZGMAX,MZGMAX,NZSIOP,NPDC),
     *  VW(NZTMAX,NZCMAX-1), QFCL(NZVMAX,NZVMAX,2*NZCMAX-1),
     *        RPAR(MZPARM,NZFMAX), CPAR(NZAVMA-1,NZPARM,NZFMAX)
      dimension rstore(lstore)
      equivalence (rstore(1), cof)
C
      COMMON /BEG/  IZLURE, INQ(NZOPER),NAUS
      dimension ibeg(lbeg)
      equivalence (ibeg(1), izlure)
C
      COMMON /LURI/ KVK(NDIM1), JQ(NDIM1), INDPO(NDIM1),
     *    MVK(NZIQMA,NDIM1), IZ,JZ, KZHV(2,NZLREC+1), LUPAUS,
     *    IQM(NZLREC+1), KZHX(0:NZIQMA,NZLREC+1)
      dimension iluri(lluri)
      equivalence (iluri(1), kvk)
C
      COMMON /LURR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)
      dimension rlurr(llurr)
      equivalence (rlurr(1), epo)
C
      COMMON /IMP/  MKC, NZV, MFL,NREB,NZAV
      dimension iimp(limp)
      equivalence (iimp(1), mkc)
C
      COMMON /KOMIX/  NUM(5,NZRHOM,NZFMAX), NZC(NZFMAX),
     *          NZPAR(NZFMAX), MZPAR(NZFMAX), NZRHO(NZFMAX),
     *          K1VEC(NZFMAX),
     *          NT1(NPDC), NT2(NPDC), LC(NZTMAX,NPDC),
     *          NGRV(2,NZCMAX,NZFMAX), NZPOT(NZOPER),
     *          ITV2, KORR,MFR,MKCM,NL,NR, NTE, NZT
      dimension ikomi(lkomi)
      equivalence (ikomi(1), num)
C
      COMMON /SC/ ABC(4), ABD(4), ABE(4)
      dimension rsc(lsc)
      equivalence (rsc(1), abc)
C
      COMMON /LUR/ IENT(NZOPLU), NBAND3
C
      COMMON /BIG/ DM(NDIM,NDIM,2)
      dimension rbig(lbig)
      equivalence (rbig(1),dm)
C
      DIMENSION NREG(NZOPOB)
      DIMENSION MZG(NZFMAX), NOL(NZFMAX), NKCM(NZOPER)
      DIMENSION MS(MZGMAX,NZFMAX)
      DIMENSION NOP(NZOPER)
      DIMENSION LREG(NZOPER), KOM(NZOPER,NZFMAX,NZFMAX)
      DIMENSION kom2(NZFMAX,NZFMAX)
      DIMENSION MMASSE(2,MZGMAX,NZFMAX), MLAD(2,MZGMAX,NZFMAX),
     *          MSS(2,MZGMAX,NZFMAX)
      DIMENSION NZLW(NZFMAX), LW(2*NZCMAX-1,NZLWMA,NZFMAX),
     *          LZWERT(5,NZLWMA,NZFMAX),
     *          NZC1(NZFMAX), IVEK(NZFMAX),
     *          NZPO(NZFMAX), KP(NZCMAX-1,NZPOMA,NZFMAX)
      DIMENSION ITPO(NZFMAX),
     *          JREOB(NZOPER), JRELU(NZOPER),
     *          WERTNO(NDIM4,NDIM4), WERTTE(NDIM4,NDIM4)
      DIMENSION IMV(NZMAT), UM(NZMAT), LCALT(NZTMAX), NKSU(NZFMAX)
C
      CHARACTER*100 INFIL  ,INFILU
      CHARACTER*50 VARFOR(NZOPER)
      CHARACTER*100 CPOTFIL
C
      DATA VARFOR/'('' NORM'')','('' KINETISCHE ENERGIE'')',
     *'('' COULOMB'')','('' ZENTRAL '')',
     *'('' P-QUADRAT-POTENTIAL'')','('' L-QUADRAT-POTENTIAL'')',
     *'('' LS-QUADRAT SKALAR'')', '(''  SPIN-BAHN'')',
     *'('' LS-QUADRAT VEKTOR'')', '('' TENSOR'')',
     7'('' LS-QUADRAT TENSOR '')', '('' ISO-TENSOR ZENTRAL '')',
     8'('' ISO-TENSOR TENSOR '')', '('' CHARGE-ASYMMETRIC '')'/
C
      DATA NZPOT/NZOPER*1/
C
      DATA NKCM/1,1,1,4,4,4,4,2,2,2,2,2,1,1/
      DATA JRELU/1,2,3,0,0,4,0,5,0,6,7,0,0 ,0/
      DATA JREOB/1,0,2,3,0,0,0,4,0,5,0,6,7,8/
C
C     FUER SMIN SIND DIE MIT 'CC' VERSEHENEN KARTEN ZU AKTIVIEREN UND
C     DIE ANGEGEBENEN ZEILEN ZU DEAKTIVIEREN
C
      character*(MPI_MAX_PROCESSOR_NAME) my_name
      integer my_len , error

      integer jdummy(1 : 80)

C     MPI initialisieren
      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
      call MPI_GET_PROCESSOR_NAME( my_name , my_len ,error )
      
C     ######################################################################
C     Der gesamte Master wird nur fuer myid .eq. 0 durchlaufen,
C     sonst muss in den slave gesprungen werden
C     ######################################################################
      if (myid .eq. 0) then

C     Anzahl der Prozessoren
         call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)
         
         OPEN(UNIT=9 ,FILE='LUCOUT',
     $        STATUS='OLD',FORM='UNFORMATTED')
         OPEN(UNIT=10,FILE='QUAOUT',STATUS='UNKNOWN',FORM='UNFORMATTED')
         OPEN(UNIT=15 ,FILE='INQUA_N',STATUS='OLD')
         OPEN(UNIT=16 ,FILE='OUTPUT',
     $        STATUS='UNKNOWN', FORM='FORMATTED')
         OPEN(UNIT=8 ,FILE='KOBOUT',
     $        STATUS='OLD',FORM='UNFORMATTED')
      NOUT=16
         write(NOUT,*) 'Anzahl der Prozessoren: ',nproc

      WRITE(NOUT, 1111)
      
1111  FORMAT('1 QUAF_S VERSION VOM 2.9.98')
      INPUT=15
      INPOT=25
      PHI=3.1415926535897932384D0
      E2=0.51098D0*2.81785D0*2.0/PHI**0.5
      WRITE (NOUT,1000)
      DO 1   K=1,NZPOTM
      DO 1 L=1,NZOPER
      DO 2  M=1,3,2
    2 WECPOT(K,M,L)=1.
    1 WECPOT(K,2,L)=0.
      READ (INPUT,1002) NBAND1,NBAND2,NBAND3,IDUM,naufset,NAUS,MOBAUS
     1 ,LUPAUS, ISTOP, NLES
      if(naufset.eq.1) stop 'naufset=1'
      open(unit=18,file='FINDEX',status='unknown',form='unformatted')
      if(naufset.eq.0) then
         naufset=1
         nband5=0
         do i=1,lind
           index(i)=-1
         enddo
      else
         nband5=12
            if(nles.eq.0) then
      read(18) nger,(index(i),i=1,nger)
            else
                READ(INPUT,6666) NLES
6666       FORMAT(I6)
                read(18) nger,(index(i),i=1,nles)
           write(nout,*) ' Von ',nger,' gerechneten operatoren werden',
     *          nles,' eingelesen'
                nger=nles
            endif
      write(nout,*) 'nger',nger,index(nger)
c     write(6,*) 'nger',nger,(index(i),i=1,nger)
      do 59 i=1,nger
      if(index(i).lt.naufset) goto 59
      write(6,*) ' Fuer ',i,' ist index ',index(i),
     *   'nicht groesser naufset'
      stop 'naufset falsch'
59    continue
      do i=nger+1,lind
           index(i)=-1
      enddo
      endif
      READ(INPUT,993) CPOTFIL                                               
993   FORMAT(A100)                                                   
      OPEN(UNIT=25,FILE=CPOTFIL,STATUS='OLD',FORM='FORMATTED')
      READ(INPOT,1002)(LREG(K),K=1, NZOPER)
 1020 FORMAT(6E12.4)
C     KORRELATIONSFUNKTION
      A=0.
      B=1.
      READ (INPOT,1002) NKOR,(NOP(K),K=1,NZOPER)
      WRITE(NOUT,1022)
      IF(NKOR-1) 200,201,7201
201   CONTINUE
C     B (ABFALLPARAMETER) WIRD STETS POS.EINGELESEN
      READ(INPOT,1020) A,B
      WRITE (NOUT,1023)A,B
  200 CONTINUE
      WRITE (NOUT,1024)
      WRITE(NOUT,1025) (NOP(K),K=1,NZOPER)
 1022 FORMAT(///26H KORRELATIONSFUNKTION = 1./)
1023  FORMAT(24X,F9.5,7H *EXP(-,F8.4,7H *R**2),/)
 1024 FORMAT(2X,11H R IN FERMI,//)
 1025 FORMAT(/,' BEIM OPERATOR  1  2  3  4  5  6  7  8  9 10 11 12 ',
     1'13 14 WIRD DIE KORELATIONSFUNKTION AUS',/,14X,14I3,
     2'  GAUSSFUNKTIONEN GEBILDET',/)
      ABC(1)=1.
      ABE(1)=.0
      ABD(1)=0.
      ABC(2)=A
      ABD(2)=B
      ABE(2)=.0
      ABC(3)=A
      ABD(3)=.0
      ABE(3)=B
      ABD(4)=B
      ABC(4)=A   *A
      ABE(4)=B
C      ENDE KORRELATIONFUNKTION
C     POTENTIAL EINLESEN
      READ (INPOT,1002)  (NZPOT(K), K=4,NZOPER)
      DO 530 K=4,NZOPER
      IF(NZPOT(K).EQ.0) LREG(K) = 0
  530 CONTINUE
      WECPOT(1,1,3)=WECPOT(1,1,3)*E2
      DO 742 MKC=4,NZOPER
      IF(NZPOT(MKC).EQ.0) GOTO 742
      DO 741 K=1,NZPOT(MKC)
741   READ(INPOT,1020)(WECPOT(K,M,MKC),M=1,NKCM(MKC)+2)
742   CONTINUE
1042  FORMAT(' ANZAHL DER POTENTIALE',I4)
1044  FORMAT(//)
1043  FORMAT(' VORFAKTOR=',G15.6,' BETA',F11.5,
     1   ' W  =',F9.3,' M  =',F9.3,' B  =',F9.3,' H  =',F9.3)
C     POTENTIAL EINLESEN
      DO 757 MKC=4,NZOPER
      IF(NZPOT(MKC).EQ.0) GOTO 757
      WRITE(NOUT,1044)
      WRITE(NOUT,VARFOR(MKC))
      WRITE(NOUT,1042) NZPOT(MKC)
      WRITE(NOUT,1043)((WECPOT(I,M,MKC),M=1,6),I=1,NZPOT(MKC))
757   CONTINUE
      CLOSE(UNIT=25, STATUS='KEEP')
C     EINLESEN DER FUNKTIONSEIGENSCHAFTEN UND PARAMETER
      REWIND  NBAND2
      READ(NBAND2) NZF,NZT,(NREG(K),K=1,NZOPOB), INDEPO
      IF(INDEPO.NE.0) WRITE(NOUT,*)' OBERMATRIXELEMENTE VON EINZELFILES'
      NZV=NZT-1
      IF(MOBAUS.GT.0) WRITE(NOUT,*)' NZF,NZT,(NREG(K),K=1,NZOPOB)',
     *                NZF,NZT,NZV,(NREG(K),K=1,NZOPOB)
      
      do kl=1,nzfmax
         do kr=1,kl
            kom2(kl,kr)=1
         enddo
      enddo

      IF(NBAND5.LE.0) GOTO 900
      
      write(nout,*) 'KOM:'
      i=0
      DO 92 KL=1,NZF
      DO 92 KR=1,KL
         
         KOM2(KL,KR)=0
         DO 91 L = 1, NZOPER
            kom(l,kl,kr)=0
            if(lreg(l).eq.0) goto 91
            i=i+1
            if(index(i).ne.-1) kom(l,kl,kr)=1
            KOM2(KL,KR)=KOM2(KL,KR)+LREG(L)*IABS(1-KOM(L,KL,KR))
91       continue
         WRITE(NOUT,*) 'K2',KOM2(KL,KR),kl,kr
92       continue
         
      write(nout,*) 'war KOM.'
      GO TO 93
 900  CONTINUE
      I=0
      WRITE(NOUT,6100)
      DO 94 KL=1,NZF
      DO 94 KR=1,KL
      DO 94 L=1,NZOPER
      I=I+1
      INDEX(I)=-1
   94 KOM(L,KL,KR)=0
   93 CONTINUE
      NZV1 = NZV - 1
      WECPOT(1,1,1)=2./ DBLE(NZT*NZV)
      DO 20  K = 1,NZF
      READ (NBAND2) NZC(K),MZG(K),NOL(K)
      IF(MOBAUS.GT.0) WRITE(NOUT,*)' NZC(K),MZG(K),NOL(K)',
     *                NZC(K),MZG(K),NOL(K)
      M = MZG(K)
      IF(M.LE.MZGMAX) GOTO 6002
      WRITE(NOUT,*) 'ZERLEGUNG,MZG,MZGMAX',K,MZG(K),MZGMAX
      GO TO 250
6002  NOL(K)=NOL(K)+1
      READ (NBAND2) ((MMASSE(N,L,K),MLAD(N,L,K),MSS(N,L,K),N=1,2),
     1     MS(L,K),L=1, M)
      IF(MOBAUS.GT.0) WRITE(NOUT,*) 'MMASSE,MLAD,MSS,MS',
     * ((MMASSE(N,L,K),MLAD(N,L,K),MSS(N,L,K),N=1,2),MS(L,K),L=1, M)
      M = NZC(K)
      READ (NBAND2) ((NGRV(N,L,K),L=1,M),N=1,2),NKSU(K)
      IF(MOBAUS.GT.0) WRITE(NOUT,*)'NGRV',((NGRV(N,L,K),L=1,M),N=1,2),
     *              ' NKSU ',NKSU(K)
   20 CONTINUE
C
      REWIND NBAND3
      READ(NBAND3) NZF1,(NZLW(K),NZC1(K),K=1,NZF1),(IENT(K),K=1,NZOPLU),
     1  (NZPO(MH),MH=1,NZF1),INDELU
      IF(INDELU.NE.0) WRITE(NOUT,*)' LUDWMATRIXELEMENTE VON EINZELFILES'
      IF(NAUS.GT.0) WRITE(NOUT,248) NZF1,IENT,(NZLW(K),NZC1(K),NZPO(K),
     1  K=1,NZF1)
248    FORMAT(' VON LUPO,ZAHL DER ZERLEGUNGEN',I4,' OPERATOREN',7I4,/,
     1  (' ZAHL DER DREHIMPULSE',I4,' ZAHL DER CLUSTER',I4,
     2  'ZAHL DER POLYNOME',I4,/))
       IF(NZF.EQ.NZF1) GOTO 251
       write (nout,*) 'nzf, nzf1',nzf,nzf1
C      STOP 251
      call fehler(251)
  250 call fehler(250)
  251 DO 252 K=1,NZF
      ITPO(K)=0
      IF(NZPO(K).NE.0) ITPO(K)=2
      NZPO(K)=MAX0(1,NZPO(K))
      IF(NZC(K).NE.NZC1(K)) call fehler( 252)
  252 CONTINUE
C
C     UEBERPRUEFUNG, OB GERECHNETE OPERATOREN IN OBER, LUDWIG UND
C     QUAF ZUSAMMENPASSEN
C
      IF (IENT(1).EQ.0) call fehler( 253)
      IF (LREG(2).EQ.0) GOTO 254
      IF (IENT(2).EQ.0) call fehler(  254)
254   IF (LREG(3).EQ.0) GOTO 256
      IF (IENT(3).EQ.0 .OR. NREG(2).EQ.0) call fehler(  256)
256   IF (LREG(4).EQ.0) GOTO 258
      IF (NREG(3).EQ.0) call fehler(  258)
258   IF (LREG(5).EQ.0) GOTO 260
      IF (NREG(3).EQ.0) call fehler(  260)
260   IF (LREG(6).EQ.0) GOTO 1261
      IF (NREG(3).EQ.0 .OR. IENT(4).EQ.0) call fehler( 1261)
1261  IF (LREG(7).EQ.0) GOTO 1260
      IF (NREG(3).EQ.0 .OR. IENT(4).EQ.0) call fehler( 1261)
1260  IF (LREG(8).EQ.0) GOTO 1258
      IF (NREG(4).EQ.0 .OR. IENT(5).EQ.0) call fehler( 1260)
1258  IF (LREG(9).EQ.0) GOTO 1256
      IF (NREG(4).EQ.0 .OR. IENT(5).EQ.0) call fehler( 1260)
1256  IF (LREG(10).EQ.0) GOTO 1262
      IF (NREG(5).EQ.0 .OR. IENT(6).EQ.0) call fehler( 1256)
1262  IF (LREG(11).EQ.0) GOTO 1362
      IF (NREG(5).EQ.0 .OR. IENT(7).EQ.0) call fehler( 1256)
1362  IF (LREG(12).EQ.0) GOTO 1361
      IF (NREG(6).EQ.0) call fehler( 1362)
1361  IF (LREG(13).EQ.0) GOTO 1363
      IF (NREG(7).EQ.0 .OR. IENT(6).EQ.0) call fehler( 1361)
1363  IF (LREG(14).EQ.0) GOTO 262
      IF (NREG(8).EQ.0) call fehler( 1363)
262   CONTINUE
C     CHECK OBER/LUDWIG ZU ENDE
C
C     KONSTRUKTION DER BAHNDREHIMPULSE
      DO 264 K=1,NZF
      M1=2*NZC(K)-3
      M2=NZLW(K)
      IF(M2.LE.NZLWMA) GOTO 6000
      WRITE(NOUT,*) 'ZERLEGUNG,NZLW,NZLWMA',K,NZLW(K),NZLWMA
      GO TO 250
 6000 CONTINUE
      M3=NZC(K)-1
      MPO=NZPO(K)
      READ(NBAND3) ((LW(M,L,K),M=1,M1),L=1,M2),((KP(M,KH,K),M=1,M3),
     1 KH=1,MPO)
      IF(LUPAUS.LT.1) GOTO 263
      DO 1333 L=1,M2
1333  WRITE (NOUT,1334) (LW(M,L,K),M=1,M1),NOL(K)
1334  FORMAT(' LWERT, NOL',10I5)
263   CONTINUE
      DO 265 L=1,M2
      DO 266 M=1,3
  266 LZWERT(M,L,K)=0
      LZWERT(4,L,K)=LW(M3,L,K)
      LZWERT(5,L,K)=LW(M1,L,K)
      GOTO(261,265,267,268,276,282,261), NZC(K)
C     NZC FALSCH
261   call fehler(261)
C     3-CLUSTER
  267 LZWERT(3,L,K)=LW(1,L,K)
      IF(NOL(K).EQ.NZC(K)) LZWERT(1,L,K)=LW(1,L,K)
      IF(NOL(K).EQ.2) LZWERT(2,L,K)=LW(1,L,K)
      GO TO 265
C     4-CLUSTER
  268 LZWERT(3,L,K)=LW(4,L,K)
      IF(NOL(K).LT.NZC(K)) GOTO 272
      LZWERT(1,L,K)=LZWERT(3,L,K)
      GO TO 265
  272 IF(NOL(K).GT.2) GOTO 275
      LZWERT(2,L,K)=LZWERT(3,L,K)
      GO TO 265
  275 LZWERT(1,L,K)=LW(1,L,K)
      LZWERT(2,L,K)=LW(2,L,K)
      GOTO 265
C     5-CLUSTER
276   LZWERT(3,L,K)=LW(6,L,K)
      GOTO (277,278,279,280,281,277),NOL(K)
C     NOL
277   call fehler(277)
278   LZWERT(2,L,K)=LW(6,L,K)
      GOTO 265
279   LZWERT(1,L,K)=LW(1,L,K)
      LZWERT(2,L,K)=LW(5,L,K)
      GOTO 265
280   LZWERT(1,L,K)=LW(5,L,K)
      LZWERT(2,L,K)=LW(3,L,K)
      GOTO 265
281   LZWERT(1,L,K)=LW(6,L,K)
      GOTO 265
C     6-CLUSTER
282   LZWERT(3,L,K)=LW(8,L,K)
      GOTO (277,283,284,285,286,287,277),NOL(K)
283   LZWERT(2,L,K)=LW(8,L,K)
      GOTO 265
284   LZWERT(1,L,K)=LW(1,L,K)
      LZWERT(2,L,K)=LW(7,L,K)
      GOTO 265
285   LZWERT(1,L,K)=LW(6,L,K)
      LZWERT(2,L,K)=LW(7,L,K)
      GOTO 265
286   LZWERT(1,L,K)=LW(7,L,K)
      LZWERT(2,L,K)=LW(NZC(K)-NKSU(K),L,K)
      GOTO 265
287   LZWERT(1,L,K)=LW(8,L,K)
  265 CONTINUE
  264 CONTINUE
C      BAHNDREHIMPULSE ENDE
C      KONSTRUKTION DER BASISVEKTOREN
      I=0
      WRITE (NOUT,1000)
 1000 FORMAT(1H1)
      IVEK(1)=0
      DO 22  K = 1,NZF
      NZPAR(K) = 0
      MZPAR(K) = 0
       READ(INPUT,1002) NZRHO(K)
      KK=NZRHO(K)
      IF(KK.LE.NZRHOM) GOTO 388
      WRITE(NOUT,*) 'ZERLEGUNG,NZRHO,NZRHOM',K,NZRHO(K),NZRHOM
      GOTO 250
388   IVEK(K)=I
      IF(KK.EQ.0) GOTO 22
      M = 2*NZC(K) - 2
      WRITE (NOUT,1001)  K
 1001 FORMAT(//25H WEITEPARAMETER FUER DIE ,  I3,
     1       47H TE CLUSTERSTRUKTUR IN REZIPROKEN FERMIQUADRAT     )
      READ (INPUT,1002)  NZPAR(K) ,MZPAR(K)
CC    NZPAR(K)=NOVQ(K)
CC    MZPAR(K)=MOVQ(K)
C     DIESE KOMMENTARE SIND IN SMIN ZU AKTIVIEREN UND DAS VORANGEHENDE READ
C     UNWIRKSAM ZU MACHEN
 1002 FORMAT(20I3)
C     MZPARM ZU KLEIN
      IF(MZPAR(K).GT.MZPARM) call fehler(9000)
      LM = NZPAR(K)
      KM=MZPAR(K)
      IF(LM.LE.NZPARM) GOTO 390
      WRITE(NOUT,*) ' ZUVIELE INTERNE PARAMETERSAETZE',NZPAR(K),NZPARM
      GOTO 250
390   DO 24  L = 1,LM
      READ (INPUT,1003)  (CPAR(N,L,K),N=1,M)
      DO 124 N=NZC(K)+1,M
      IF(CPAR(N,L,K).GT.0.)  GOTO 124
      WRITE(NOUT,*)' WEITE ',N,' FALSCH'
      STOP 'INTERNE WEITE FALSCH'
  124 CONTINUE
 1003 FORMAT(6E12.4)
   24 WRITE (NOUT,1004)L,(  CPAR(N,L,K),N=1,M)
 1004 FORMAT(/I3,' TER SATZ INNERER WEITEN',/,(8F12.6))
      READ(INPUT,1003)  (RPAR(L,K),L=1,KM)
      DO 125 L=1,KM                                             
CC    RPAR(L,K)=ROV(L,K)                                                 
C     FUER SMIN AKTIVIEREN UND DAS READ DEAKTIVIEREN                     
      IF(RPAR(L,K).GT.1.E-8) GOTO 125                                   
      WRITE(NOUT,*) ' IN ZERLEGUNG ',K,' IST DIE RELATIVWEITE ',L,     
     *              ' FALSCH'                                    
125   CONTINUE
C
      WRITE(NOUT,1005)   (RPAR(L,K),L=1,KM)
 1005 FORMAT(/22H SATZ RADIALPARAMETER   /3(8F12.6/))
      DO 240 N=1,KK
      READ(INPUT,1002) NUM(1,N,K),NUM(2,N,K),NUM(4,N,K),NUM(5,N,K)
C     NUM(5,.,.) # 0: BASISVEKTOR WIRD NICHT NEU GERECHNET
C
      NUM(4,N,K)=MAX(1,NUM(4,N,K))
      NUM(3,N,K)=N
      READ(INPUT,1003) (COF(L,N,K),L=1,LM)
      WRITE(NOUT,1050) NUM(3,N,K),K,NUM(1,N,K),NUM(2,N,K),NUM(4,N,K),
     1  (COF(L,N,K),L=1,LM)
 1050 FORMAT(//15H DEFINITION DER,I5,' TEN BASISFUNKTION IN DER ',
     1 I3,' TEN ZERLEGUNG IST',/,I3,23H TE SPINISOSPINFUNKTION,I3,
     2 26H TE BAHNDREHIMPULSFUNKTION,I3,' TES POLYNOM',/,
     3 ' SUMMATION UEBER INNERE WEITEN MIT KOEFFIZIENTEN',/,(1P6E12.4))
      IF (NUM(5,N,K).GT.0) WRITE (NOUT,1051)
  240 CONTINUE
      I=I+KK
   22 CONTINUE
C     ENDE DEFINITION BASISVEKTOREN
2011  FORMAT(////18H BERECHNET WERDEN )
      WRITE (NOUT,2011)
      DO 611 IPV=1,NZOPER
      IF(LREG(IPV).GT.0) WRITE(NOUT,VARFOR(IPV))
611   CONTINUE
      REWIND NBAND1
      WRITE(NBAND1) NZF,(LREG(K),K=1,NZOPER),I,(NZRHO(K),K=1,NZF)
      DO 950  K = 1,NZF
      M=MZG(K)
      N3=MZPAR(K)
      MC1=NZC(K)-1
      K1VEC(K)=N3
      KK=NZRHO(K)
      IF(KK*N3.LE.NDIM)GOTO 930
      WRITE(NOUT,*) ' NDIM FUER ZERLEGUNG ',NZF,' ZU KLEIN, MINIMAL ',
     *             KK*N3
      call fehler(930)
930   IF(KK.LE.0) GOTO 809
      DO 4536 N=1,KK
      N1=NUM(1,N,K)
      N2=NUM(2,N,K)
       N4=NUM(4,N,K)
      WRITE(NBAND1) N3,MMASSE(1,N1,K),MMASSE(2,N1,K),MLAD(1,N1,K),
     1 MLAD(2,N1,K),MSS(1,N1,K),MSS(2,N1,K),MS(N1,K),
     2 (LZWERT(L,N2,K),L=1,5),(RPAR(L,K),L=1,N3),KP(MC1,N4,K)
 4536 CONTINUE
809   CONTINUE
810   CONTINUE
  950 CONTINUE
      WECPOT(1,1,2) = -(2.72099*0.529172*137.0373)**2/
     1                 ((938.211+939.505)* DBLE(NZT))
      ISTOPP=0
C
C
C     BESTIMMUNG DER ORTSMATRIXELEMENTE VOR EINSETZEN DER DREHIMPULSE

         cquaf='DMOUT.'
         write(fnumber,*) naufset
         do i=1, 255
            if(fnumber(i:i).ne.' ') goto 370
         end do
 370      do j=i, 255
            if(fnumber(j:j).eq.' ') goto 380
         end do  
 380      do k=1, 80
            if(cquaf(k:k).eq.' ') goto 490
         end do
 490      oname = cquaf(1:k-1) // fnumber(i:j)
          open(unit=27,file=oname,status='replace',form='unformatted')

      nteil = 1
      narbeit = 0
      nsend= 1
 1933 format(a72)
      
      IF(INDEPO.NE.0) THEN                             
         MEFI=23                                           
         CLOSE(UNIT=8,STATUS='KEEP')                               
         NBAND2=MEFI                                         
      ENDIF   
         IF(INDELU.NE.0) THEN
         MEFLU=25
         CLOSE(UNIT=9,STATUS='KEEP')
         NBAND3=MEFLU
         ENDIF

      CLOSE(UNIT=10,STATUS='KEEP')
      
        icount=0
      DO 40  MFL = 1,NZF
      IZLW=NZLW(MFL)
      IZPW=NZPO(MFL)
      IRHO=NZRHO(MFL)
      IK1=K1VEC(MFL)
      MC = NZC(MFL)
      MC1= MC - 1
      MC2 = MC + MC1
      IF(INDEPO.EQ.0) THEN            
         READ(NBAND2)   ((RVEC(M,N),M=1,NZV),N=1,NZT)
         READ(NBAND2)   ((SVEC(N,M),M=1,NZV),N=1,NZT)
         READ(NBAND2)   (((QFCL(M,N,K),M=1,NZV),N=1,NZV),K=1,MC2)   
      ENDIF
      DO 40 MFR=1,MFL
       rewind 18
      JZLW=NZLW(MFR)
       JZPW=NZPO(MFR)
      JRHO=NZRHO(MFR)
      JK1=K1VEC(MFR)
      WRITE(NOUT,3009) MFL,MFR
      NLUDZ=0
3009  FORMAT(//,19H ZWISCHEN ZERLEGUNG,I3,15H UND ZERLEGUNG ,I3,
     124H WIRD ZUR ZEIT GERECHNET )
      MD=NZC(MFR)
      MD1= MD - 1
      IF(INDEPO.NE.0) THEN                            
         READ(INPUT,378) INFIL                      
 378     FORMAT(A100)               
         write(NOUT,*) 'Oeffnen? MFL,MFR,KOM2(MFL,MFR)',
     $        MFL,MFR,KOM2(MFL,MFR),INFIL
         if(KOM2(MFL,MFR).EQ.0) goto 9010
         write(NOUT,*) 'Oeffne ',INFIL
c        CLOSE(UNIT=6,STATUS='KEEP')
#ifdef  PC_FORTRAN
c        OPEN(UNIT=6,FILE='OUTPUT', ACCESS='APPEND')
#endif
#ifdef  CRAY_FORTRAN
c        OPEN(UNIT=6,FILE='OUTPUT', POSITION='APPEND')
#endif
         OPEN(UNIT=MEFI,FILE=INFIL,STATUS='OLD',FORM='UNFORMATTED')    
         REWIND MEFI                                      
         READ(NBAND2)   ((RVEC(M,N),M=1,NZV),N=1,NZT)                   
         READ(NBAND2)   ((SVEC(N,M),M=1,NZV),N=1,NZT)                
         READ(NBAND2)   (((QFCL(M,N,K),M=1,NZV),N=1,NZV),K=1,MC2)  
      ENDIF  
9010  IF(INDELU.NE.0) THEN
         READ(INPUT,378) INFILU
         if(KOM2(MFL,MFR).EQ.0) goto 9011
         OPEN(UNIT=MEFLU,FILE=INFILU,STATUS='OLD',FORM='UNFORMATTED')
          REWIND MEFLU
      ENDIF

         if(KOM2(MFL,MFR).EQ.0) goto 9011
      READ(NBAND2)    ((VW(M,K),M=1,NZT),K=1,MD1)
 9011 CONTINUE
      IZL=MC1*(1+ITPO(MFL))+MD1*(1+ITPO(MFR))
      INQ(1)=1
      INQ(2)=((IZL-1)*IZL)/2 +1
      INQ(3)=INQ(2) +1
      INQ(4)=INQ(1)
      INQ(5)=INQ(2)
      INQ(6)=((INQ(2) +1)*INQ(2))/2
      INQ(7)=INQ(6)
      INQ(8)=INQ(2)
      INQ(9)=INQ(8)
      INQ(10)=2
      INQ(11)=((INQ(2) +1)*INQ(2))/2 + IZL +1
      INQ(12)=INQ(1)
      INQ(13)=INQ(10)
      INQ(14)=INQ(1)
      DO 42 MKC=1, NZOPER
       rewind 18
      FOP=1.
      IF(MKC.EQ.7) FOP= 1./3.
      IF(MKC.EQ.9) FOP= -1./2.
      IF(MKC.EQ.11) FOP= SQRT(15./(2.*PHI))/12.
      NREB = 0
      IZ=(IZPW-1)*NZLWMA+IZLW
      JZ=(JZPW-1)*NZLWMA+JZLW
      IF(MKC.EQ.12 .OR. MKC.EQ.14) THEN
       DO 1142, J=1, JZ
      DO 1142, I=1, IZ
1142       WERTL(I,J)=WERTNO(I,J)
       ENDIF
      IF(MKC.EQ.13 ) THEN
       DO 1143, J=1, JZ
      DO 1143, I=1, IZ
1143       WERTL(I,J)=WERTTE(I,J)
       ENDIF
      NZAV=MC1+MD1
      IZLURE=1
      IF(LREG(MKC).GT.0) WRITE(NOUT,VARFOR(MKC))
C     OPERATOR 1   2    3    4    5    6    7    8    9    10
      GO TO (3008,3002,3003,3008,3002,3002,3002,3002,3002,3004,
     *       3002,3008,3004,3008),MKC
3002      NREB = 1
      IZLURE=MC1*(1+ITPO(MFL))+MD1*(1+ITPO(MFR))
      GO TO 3008
3003  IZLURE=2
3004      NZAV=NZAV+1
3008  CONTINUE
      KORR = (NOP(MKC)+1)**2
      NL = MZG(MFL)
      NR = MZG(MFR)
      MKCM=NKCM(MKC)
      IF(JREOB(MKC).EQ.0) GOTO 440
      IF(NREG(JREOB(MKC)).EQ.0) GOTO 440
      IF(KOM2(MFL,MFR).EQ.0) GOTO 440
      READ(NBAND2)    IDUM,NTE,NTE
      IF(NTE.LE.NPDC) GOTO 6029
      WRITE(NOUT,*) ' ZUVIELE PDCS',NTE,' DIMENSION ',NPDC
C     NPDC
      call fehler(9001)
6029  CONTINUE
      IF(NTE.EQ.0) GOTO 440
      DO 922   MTE = 1,NTE
      READ(NBAND2) NT1(MTE),NT2(MTE),IZT,IMQ,(LCALT(I),I=1,IZT)
     *  ,           (IMV(I),UM(I),I=1,IMQ)
      DO 11 M=1,MKCM
      DO 11 J=1,NR
      DO 11 I=1,NL
      U(I,J,M,MTE)= 0.
11    CONTINUE
      DO 12 IK=1,IMQ
      I= IMV(IK)/(NR*MKCM)
      IM=MOD(IMV(IK),NR*MKCM)
      J=IM/MKCM
      M=MOD(IM,MKCM)
12    U(I+1,J+1,M+1,MTE)=UM(IK)
      IF(IZT.EQ.NZT) THEN
      DO 7 I=1,IZT
7     LC(I,MTE)=LCALT(I)
      GOTO 921
      ENDIF
      DO 177 I=1,NZT
177    LC(I,MTE)=LC(I,MTE-1)
921   CONTINUE
      IF(MOBAUS.EQ.0) GOTO 922
      IF(LREG(MKC).EQ. 0) GOTO 922
      WRITE (NOUT,838) NT1(MTE), NT2(MTE), (LC(K,MTE),K=1,NZT)
 838  FORMAT (10H VON OBER ,20I3)
      IF(MOBAUS.LT.2) GOTO 922
      WRITE (NOUT,839) (((U(NFL,NFR,NKC,MTE),NKC=1,MKCM),NFL=1,NL),
     *       NFR=1,NR)
 839  FORMAT (1P10E12.4)
922   CONTINUE
C
C       READ LUPO ELEMENTE
440   IF(JRELU(MKC).EQ.0) GOTO 441
      IF(IENT(JRELU(MKC)).EQ.0 ) GOTO 441
      IF(KOM2(MFL,MFR).EQ.0 .and. INDELU.NE.0) GOTO 441
      CALL LURE(MKC,ITV2)
      IF(MKC.EQ.1) THEN
       DO 1144, J=1, JZ
      DO 1144, I=1, IZ
1144       WERTNO(I,J)=WERTL(I,J)
       ENDIF
      IF(MKC.EQ.10) THEN
       DO 1145, J=1, JZ
      DO 1145, I=1, IZ
1145       WERTTE(I,J)=WERTL(I,J)
       ENDIF
      IF(LUPAUS.GE.0) WRITE(NOUT,*)' ',ITV2,
     *           ' LUC-MATRIXELEMENTE +OBER ',NTE
441   IF(LREG(MKC).EQ.0)  GOTO 6033
      IF (KOM(MKC,MFL,MFR)) 6030,6031,6032
 6030 WRITE(NOUT,6034)
      GO TO 6033
 6031 WRITE(NOUT,6035)
      GO TO 6033
 6032 WRITE(NOUT,6036)
 6033 CONTINUE
      IF(MKC.EQ.4) ITV2=KZHV(2,1)
      IF(MKC.EQ.5) ITV2=KZHV(2,1)
      IF(KOM(MKC,MFL,MFR).EQ.0) GOTO 95
      if(mfl .ne. mfr) then
         nnn = nzrho(mfl)*nzrho(mfr)
      else
         nnn = (nzrho(mfl)*(nzrho(mfl)+1))/2
      endif
      if(lreg(mkc)*nnn .le. 0) goto 42
      
      IF(KOM(MKC,MFL,MFR).EQ.1) GOTO 942
      WRITE(NOUT,6037)
6037  FORMAT(' FUER DIESEN OPERATOR WIRD ERGAENZT')
95    IF(IRHO*JRHO.LE.0) GOTO 42
      IF(LREG(MKC).LE.0) GOTO 42
C
C     Nichts tun, koennen wir auch selber
      if( nte*itv2.eq.0 ) then


         write( nout,* ) 'Nichts zu tun.',nteil
         index(nteil)=0
         icount=icount+1
               WRITE (27) nteil,icount,index(nteil)


         goto 942
      endif

c schon alle beschaeftigt?
      nfertig=0
      write(NOUT,*) 'narbeit, nproc, nteil',narbeit,nproc,nteil
      if(narbeit .ge. (nproc-1) ) then
c        alle beschaeftigt, erstmal auf Ergebnisse warten
         call MPI_RECV(ngerechnet,1,MPI_INTEGER,MPI_ANY_SOURCE,
     *                 MPI_ANY_TAG,MPI_COMM_WORLD,istatus,ierr)
         call MPI_RECV(nteilalt,1,MPI_INTEGER,ngerechnet,0,
     *        MPI_COMM_WORLD,istatus,ierr)

      call MPI_RECV(rbig,lbig,MPI_DOUBLE_PRECISION,ngerechnet,0,
     *     MPI_COMM_WORLD,istatus,ierr)


         narbeit = narbeit - 1
         write(NOUT,*)  'Rueckmeldung von ', ngerechnet
         index(nteilalt)=naufset
         icount=icount+1
               WRITE (27) nteilalt,icount,index(nteilalt)


         write(18) nteil,(index(i),i=1,nteil)
         write(27) (rbig(lb),lb=1,lbig)

         nfertig=nfertig+1
C     Der Sklave darf gleich noch einmal, wenn noch Arbeit da ist
         nstarte=ngerechnet
      endif
            
c     noch einen starten
      if (nsend.lt. nproc) then
         if (nfertig.eq.0) then
            nstarte=nsend
         endif 
      endif


c     CLOSE(UNIT=6,STATUS='KEEP')
#ifdef  PC_FORTRAN
c        OPEN(UNIT=6,FILE='OUTPUT', ACCESS='APPEND')
#endif
#ifdef  CRAY_FORTRAN
c     OPEN(UNIT=6,FILE='OUTPUT', POSITION='APPEND')
#endif

c und mit Arbeit versorgen
      idummy=0
      call MPI_SEND(idummy,1,MPI_INTEGER,nstarte,0,
     *     MPI_COMM_WORLD,ierr)
      
      call MPI_SEND(nteil,1,MPI_INTEGER,nstarte,0,
     *     MPI_COMM_WORLD,ierr)


      read (coutput,'(80A1)') dummy

      do 9002 n=1,80
 9002    jdummy(n)=ichar(dummy(n))
      call MPI_SEND(jdummy(1),80,MPI_INTEGER,nstarte,0,
     *        MPI_COMM_WORLD,ierr)

      call MPI_SEND(rstore,lstore,MPI_DOUBLE_PRECISION,nstarte,0,
     *     MPI_COMM_WORLD,ierr)

      call MPI_SEND(ibeg,lbeg,MPI_INTEGER,nstarte,0,
     *     MPI_COMM_WORLD,ierr)

      call MPI_SEND(iluri,lluri,MPI_INTEGER,nstarte,0,
     *     MPI_COMM_WORLD,ierr)

      call MPI_SEND(rlurr,llurr,MPI_DOUBLE_PRECISION,nstarte,0,
     *     MPI_COMM_WORLD,ierr)

      call MPI_SEND(iimp,limp,MPI_INTEGER,nstarte,0,
     *     MPI_COMM_WORLD,ierr)

      call MPI_SEND(ikomi,lkomi,MPI_INTEGER,nstarte,0,
     *     MPI_COMM_WORLD,ierr)

      call MPI_SEND(rsc,lsc,MPI_DOUBLE_PRECISION,nstarte,0,
     *     MPI_COMM_WORLD,ierr)
      
c     ... mit Arbeit versorgt
      
C     CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2), 
C    $        BIG_BEN (3), DATE_TIME)
C      write(NOUT,*) 'Rechner ',nstarte," um :: ",date_time,
C    $     ' abgefertigt'

      narbeit = narbeit + 1
      nsend= nsend +1
            
 942  nteil = nteil + 1

   42 CONTINUE
C        LOOP OPERATOREN
c     CLOSE(UNIT=6,STATUS='KEEP')
#ifdef  PC_FORTRAN
c        OPEN(UNIT=6,FILE='OUTPUT', ACCESS='APPEND')
#endif
#ifdef  CRAY_FORTRAN
c     OPEN(UNIT=6,FILE='OUTPUT', POSITION='APPEND')
#endif
   40 CONTINUE
      
c     hier muss die Schleife zum Aufsammeln der restlichen Ergebnisse hin
      do i=1, narbeit
         call MPI_RECV(ngerechnet,1,MPI_INTEGER,MPI_ANY_SOURCE,
     *        MPI_ANY_TAG,MPI_COMM_WORLD,istatus,ierr)
         call MPI_RECV(nteilalt,1,MPI_INTEGER,ngerechnet,0,
     *        MPI_COMM_WORLD,istatus,ierr)

      call MPI_RECV(rbig,lbig,MPI_DOUBLE_PRECISION,ngerechnet,0,
     *     MPI_COMM_WORLD,istatus,ierr)

         index(nteilalt)=naufset
         icount=icount+1
               WRITE (27) nteilalt,icount,index(nteilalt)

         write(27) (rbig(lb),lb=1,lbig)
         write(18) nteil,(index(iy),iy=1,nteil)

         istop=66
         call MPI_SEND(istop,1,MPI_INTEGER,istatus(MPI_SOURCE),
     *        0,MPI_COMM_WORLD,ierr)
         write(NOUT,*)  'Rueckmeldung VON ', ngerechnet
      enddo
c      
      istop=666
      
      write (NOUT,*) 'Aufsammeln'

      call MPI_BCAST(istop,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)

      write (NOUT,*) 'BROADCAST'
      write (NOUT,*) 'AUFSAMMELN MUSS NOCH MANUELL GEMACHT WERDEN!'
      rewind 18
         write(18) nteil,(index(i),i=1,nteil)
         write(NOUT,*) nteil,(index(i),i=1,nteil)
      close(unit=18,status='keep')


c     CLOSE(UNIT=6,STATUS='KEEP')
#ifdef  PC_FORTRAN
c        OPEN(UNIT=6,FILE='OUTPUT', ACCESS='APPEND')
#endif
#ifdef  CRAY_FORTRAN
c     OPEN(UNIT=6,FILE='OUTPUT', POSITION='APPEND')
#endif
      
      goto 666

C     Das Aufsammeln, wie bisher, wird nicht mehr durchlaufen, sondern
C     ist ein extra (serielles!) Programm.
C     Viel Spass beim Warten.


CC    RETURN
C     FUER SMIN AKTIVIEREN UND STOP DEAKTIVIEREN
C     Hier ist der Master fertig, laso das endif zum myid .eq. 0
      endif

C     ######################################################################
C     Jetzt kommt der quaf Sklave
C     ######################################################################

C      open(12, file=output )
C      write(12,'(a7,i1)') "OUTPUT.",myid
      if (myid .ne. 0) then

 1234    continue

         call MPI_RECV(idummy,1,MPI_INTEGER,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)
         if(idummy.eq.66) then
            call MPI_BCAST(istop,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
            if(istop.eq.666) goto 666
         endif
      
         call MPI_RECV(nteil,1,MPI_INTEGER,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)
      
         call MPI_RECV(jdummy(1),80,MPI_INTEGER,0,0,
     *           MPI_COMM_WORLD,istatus,ierr)
         do 9003 n=1,80
 9003       coutput(n:n)=char(jdummy(n))

C         CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2), 
C     $           BIG_BEN (3), DATE_TIME)
C         write(12,*) 'STARTE Empfang  um :: ',date_time
   
         call MPI_RECV(rstore,lstore,MPI_DOUBLE_PRECISION,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

C         CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2), 
C     $        BIG_BEN (3), DATE_TIME)
C         write(12,*) 'Empfangen lstore um :: ',date_time,lstore

         call MPI_RECV(ibeg,lbeg,MPI_INTEGER,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

C         CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2), 
C     $        BIG_BEN (3), DATE_TIME)
C         write(12,*) 'Empfangen lbeg um :: ',date_time,lbeg

         call MPI_RECV(iluri,lluri,MPI_INTEGER,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

         call MPI_RECV(rlurr,llurr,MPI_DOUBLE_PRECISION,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

         call MPI_RECV(iimp,limp,MPI_INTEGER,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

         call MPI_RECV(ikomi,lkomi,MPI_INTEGER,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

         call MPI_RECV(rsc,lsc,MPI_DOUBLE_PRECISION,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

c Namen fuer Slave bestimmen
C         CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2), 
C     $        BIG_BEN (3), DATE_TIME)
C         write(12,*) 'Empfangen Alles um :: ',date_time


         write(fnumber,*) nteil
         do i=1, 255
            if(fnumber(i:i).ne.' ') goto 70
         end do
 70      do j=i, 255
            if(fnumber(j:j).eq.' ') goto 80
         end do
 80      do k=1, 80
            if(coutput(k:k).eq.' ') goto 90
         end do
 90      oname = coutput(1:k-1) // fnumber(i:j)

      

C         TIME0=MCLOCK()/100.
C         RT0=timef()/1000.

c        write(15,*) 'Ich bin ',myid
c        write(15, *) 'Programm hat Bisher ',time0,'s gerechnet.'
C         CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2), 
C     $        BIG_BEN (3), DATE_TIME)
C         write(NOUT,*) date_time
C         write(NOUT,*) date_time,my_name

C     Ein Aufruf
         CALL CALCU
C     ... und fertig
         
C         RT1=timef()/1000.

C         TIME1=MCLOCK()/100.
C         TIMED=MCLOCK()/100.-TIME0
c        write(15, *) 'Ich bin Nummer ',myid
c        
c        write(15, *) 'Programm hat bisher ',time0,'s gerechnet.'
c        write(15, *) 'Gerechnet: ',timed,' gebraucht: ',RT1,'s'
c        write(15, *) 'insgesamt jetzt ',time1,'s.'

C         CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2), 
C     $        BIG_BEN (3), DATE_TIME)
c        write(15,*) 'Beendet: ', date_time
         
C     Zurueckmelden
         
c        close(unit=15)
         call MPI_SEND(myid,1,MPI_INTEGER,0,0,MPI_COMM_WORLD,ierr)

         call MPI_SEND(nteil,1,MPI_INTEGER,0,0,
     *        MPI_COMM_WORLD,ierr)

      call MPI_SEND(rbig,lbig,MPI_DOUBLE_PRECISION,0,0,
     *     MPI_COMM_WORLD,ierr)


C     Sklave geht zurueck zum Anfang. Vielleicht kommt ja noch was nach
      
      goto 1234

      endif

C     Ende Hauptprogramm
      close(unit=6)
 666  continue
      close(unit=27,status='keep')
      ifehler = 42
      call MPI_FINALIZE(ierror)
       STOP

 6100 FORMAT(/31H ES WIRD VOELLIG NEU GERECHNET /)
 6034 FORMAT(34H FUER DIESEN OPERATOR WIRD ERSETZT /)
 6035 FORMAT(40H FUER DIESEN OPERATOR WIRD NEU GERECHNET /)
 6036 FORMAT(34H FUER DIESEN OPERATOR WIRD KOPIERT /)
 7201 WRITE(NOUT,7202) NKOR
 7202 FORMAT(' ZUVIELE KORRELATIONSFUNKTIONEN',I5)
 1051 FORMAT(1X,'DIESER BASISVEKTOR WIRD NICHT NEU GERECHNET')
      END

      SUBROUTINE CALCU
      IMPLICIT REAL*8 (A-H,O-Z)
C
      include "par.h"
C
      PARAMETER (NOUT=16, NBAND1=10)
C
      COMMON /LURI/ KVK(NDIM1), JQ(NDIM1), INDPO(NDIM1),
     *    MVK(NZIQMA,NDIM1), IZ,JZ, KZHV(2,NZLREC+1), LUPAUS,
     *    IQM(NZLREC+1), KZHX(0:NZIQMA,NZLREC+1)
C
      COMMON /LURR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)
C
C
      COMMON /BEG/  IZLURE, INQ(NZOPER),NAUS
C
      COMMON /STORE/ COF(NZPARM,NZRHOM,NZFMAX),
     *   RVEC(NZVMAX,NZTMAX), WECPOT(NZPOTM,6,NZOPER),
     *          SVEC(NZTMAX,NZVMAX), U(MZGMAX,MZGMAX,NZSIOP,NPDC),
     *  VW(NZTMAX,NZCMAX-1), QFCL(NZVMAX,NZVMAX,2*NZCMAX-1),
     *        RPAR(MZPARM,NZFMAX), CPAR(NZAVMA-1,NZPARM,NZFMAX)
C
      COMMON /IMP/  MKC, NZV, MFL,NREB,NZAV
C
      COMMON /CALBE/ QO(NZVMAX,NZVMAX), QOR(NZVMAX,NZVMAX),
     *         VR(NZTMAX,NZCMAX-1), UR(NZVMAX), WR(NZTMAX),
     *        VZ(NZVMAX,NZVMAX), RPAL(MZPARM)
C
      COMMON /CALCEI/  LPARR, MC1, MD1, NCC
C
      COMMON /KOMIX/  NUM(5,NZRHOM,NZFMAX), NZC(NZFMAX),
     *          NZPAR(NZFMAX), MZPAR(NZFMAX), NZRHO(NZFMAX),
     *          K1VEC(NZFMAX),
     *          NT1(NPDC), NT2(NPDC), LC(NZTMAX,NPDC),
     *          NGRV(2,NZCMAX,NZFMAX), NZPOT(NZOPER),
     *          ITV2, KORR,MFR,MKCM,NL,NR, NTE, NZT
C
      COMMON /BIG/ DM(NDIM,NDIM,2)
C
      COMMON /CSH/ NSH1(NDIM5), NSH2(NDIM5), SH(NDIM5), NSH
C
      DIMENSION 
     *          QFCR(NZVMAX,NZVMAX,2*NZCMAX-1), LC1(NZTMAX),
     *          QFO(NZVMAX,NZVMAX), POR(NZVMAX,NZVMAX),
     *          POL(NZVMAX,NZVMAX)
C
      DIMENSION  UU(MZGMAX,MZGMAX),
     *          POLL(NZVMAX,NZVMAX), PORR(NZVMAX,NZVMAX)
C
      PHI=3.1415926535897932384D0
      FOP=1.                                                    
      IF(MKC.EQ.7) FOP= 1./3.                          
      IF(MKC.EQ.9) FOP= -1./2.                             
      IF(MKC.EQ.11) FOP= SQRT(15./(2.*PHI))/12.
      DO 460 N=1,NDIM
      DO 460 M=1,NDIM
      DM(M,N,1)=.0
  460 DM(M,N,2)=.0
      DO 490 M=1,NZT
  490 LC1(M)=0
      IRHO=NZRHO(MFL)
      IK1=K1VEC(MFL)
      MC = NZC(MFL)
      MC1= MC - 1
      MC2 = MC + MC1
      MC3 = MC2 - 1
      NC= NZPAR(MFL)
      NCC = MZPAR(MFL)
      DO 961 IH=1,NCC
961   RPAL(IH)=RPAR(IH,MFL)
      JRHO=NZRHO(MFR)
      JK1=K1VEC(MFR)
      MD=NZC(MFR)
      MD1= MD - 1
      MD2 = MD + MD1
      MD3 = MD2 - 1
      ND = NZPAR(MFR)
      NDD = MZPAR(MFR)
464   IF (NTE*ITV2.LE.0) GOTO 900
      DO 44 MTE=1,NTE
c     IF(NAUS.LT.1) GOTO 1491
c     WRITE(NOUT,*) MTE,' TES OBERMATRIXELEMENT'
C
C     FUER SMIN MUESSEN DIESE STATEMENTS BESEITIGT WERDEN
C
1491  MM=0
      DO 491 M=1,NZT
  491 MM=MM+IABS(LC(M,MTE)-LC1(M))
      IF(MM.EQ.0) GOTO 492
      DO 494 M=1,NZT
  494 LC1(M)=LC(M,MTE)
      DO 470 M=1,NZT
      MM=LC(M,MTE)
  470 VR(MM,1)=VW(M,MD1)
      TS=.0
      TD=.0
      NA= 1
      DO 471 M=1,NZT
      TS=TS+ ABS(VR(M,1)+SVEC(M,NZV))
  471 TD=TD+ ABS(VR(M,1)-SVEC(M,NZV))
      IF(TS*TD.GT.0.00001) NA=0
      DO 51   K= 1,MD2
      DO 51   M = 1,NZV
      DO 51   N = 1,M
   51 QFCR(M,N,K) = .0
      DO 50 K=1,MD1
      DO 50   M = 1,NZV
   50 VR(M,K) = .0
      DO 290  K = 1,MD
      I1 = NGRV(1,K,MFR)
      I2 = NGRV(2,K,MFR) + I1 - 1
      IF(NGRV(2,K,MFR).LE.1)GOTO 290
      FZZ = 1./ DBLE(NGRV(2,K,MFR))
      I3=I2-1
      DO 52  L1 = I1,I3
      I4=L1+1
      DO 52  L2 =I4,I2
      L3 = LC(L1,MTE)
      L4 = LC(L2,MTE)
      DO 52  M = 1,NZV
      DO 52  N = 1,NZV
      MM=MIN0(M,N)
      NN=MAX0(N,M)
   52 QFCR(NN,MM,K) = QFCR(NN,MM,K) + (RVEC(M,L3)-RVEC(M,L4))*
     1(RVEC(N,L3)-RVEC(N,L4)) *FZZ
  290 CONTINUE
      DO 53  K = 2,MD
      KI = MD1 + K
      DO 100   M = 1,NZT
      MM = LC(M,MTE)
      DO 100  N = 1,NZV
  100 VR(N,K-1) = VR(N,K-1) + VW(M,K-1) *RVEC(N,MM)
      DO 53   M = 1,NZV
      DO 53   N = 1,NZV
      MM=MIN0(M,N)
      NN=MAX0(N,M)
      QFCR(NN,MM,KI ) = QFCR(NN,MM,KI ) + VR(M,K-1)*VR(N,K-1)
   53 CONTINUE
  492 CONTINUE
      DO 54  M = 1,NZV
      UR(M) = RVEC(M,NT1(MTE)) - RVEC(M,NT2(MTE))
      WR(M) = SVEC(NT1(MTE),M) - SVEC(NT2(MTE),M)
      DO 54  N = 1,NZV
   54 QFO(N,M)=.0
      DO 55   M = 1,NZV
      DO 55   N = 1,NZV
      MM=MIN0(M,N)
      NN=MAX0(N,M)
      QFO(NN,MM) = QFO(NN,MM) + UR(M)*UR(N)
   55 CONTINUE
      DO 80 IPOT=1, NZPOT(MKC)
      FAK=WECPOT(IPOT,1,MKC)
       POWE= WECPOT(IPOT,2,MKC)
      DO 82 N=1,NR
      DO 82 M=1,NL
      UU(M,N)=.0
      DO 81 NKC=1,MKCM
      UU(M,N)=UU(M,N)+WECPOT(IPOT,NKC+2,MKC)*U(M,N,NKC,MTE)*FOP
   81 CONTINUE
      if(abs(uu(m,n)).lt. 1.e-10) uu(m,n)=0.   
   82 CONTINUE
      DO 60   KPAR = 1,NC
      DO 62   M = 1,NZV
      DO 62   N = 1,M
   62 POL (M,N) = .0
      DO 64   M = 1,NZV
      IF(MC3.LE.0) GOTO 64
      DO 63   K = 1,MC3
   63 POL (M,M)=POL (M,M) + CPAR(K,KPAR,MFL)*QFCL(M,M,K)
   64 CONTINUE
       DO 70 LPAR=1,ND
      I=0
      DO 101 M=1,IRHO
      KSL= NUM(1,M,MFL)
      KLL= NUM(2,M,MFL)
      KPL=NUM(4,M,MFL)
      JPL=(KPL-1)*NZLWMA+KLL
      NUML= NUM(3,M,MFL)
      TS = COF(KPAR,M,MFL)
      IF (ABS(TS).LT.1.E-10) GOTO 101
      MADL = (M-1) * IK1
      DO 105 N=1,JRHO
      KSR= NUM(1,N,MFR)
      KLR= NUM(2,N,MFR)
      KPR=NUM(4,N,MFR)
      JPR=(KPR-1)*NZLWMA+KLR
      NUMR= NUM(3,N,MFR)
      IF (MFL.EQ.MFR .AND. NUML.LT.NUMR) GOTO 105
      IF(WERTL(JPL,JPR).NE.1.) GOTO 105
       A = TS * UU(KSL,KSR)*COF(LPAR,N,MFR)
      IF(ABS(A).LT.1.E-20) GOTO 105
      MADR = (N-1) * JK1
      IF (I.LT.NDIM5) GOTO 110
c     WRITE (NOUT,111) I
 111  FORMAT ('0NSH ZU KLEIN', I4)
      STOP
 110  I = I + 1
      NSH1(I) = NDIM * (MADR-1) + MADL
      NSH2(I) = NDIM4 * (JPR-1) + JPL
      SH(I) = A
 105  CONTINUE
 101  CONTINUE
      NSH = I
      IF (NSH.EQ.0) GOTO 70
      DO 72   N = 1,NZV
      DO 72   M = 1,NZV
   72 POR(M,N) = .0
      DO 74  M = 1,NZV
      DO 74  N = M,NZV
      IF(MD3.LE.0) GOTO 74
      DO 73  K = 1,MD3
   73 POR(N,M)=POR(N,M)+CPAR(K,LPAR,MFR)*QFCR(N,M,K)
   74 CONTINUE
      DO 67 M=1,NZV
      DO 67  N = M,NZV
   67 POLL(N,M)=POL(N,M)+
     1  WECPOT(IPOT,2,MKC)*QFO(N,M)
      DO 76 LPARR=1,NDD
      DO 77 M=1,NZV
      DO 77 N=M,NZV
   77 PORR(N,M)=POR(N,M)+RPAR(LPARR,MFR)*QFCR(N,M,MD2)
      CALL SCORR(POLL,PORR,QFO,   NA,FAK,KORR,NZT,POWE)
   76 CONTINUE
C      LOOP RADIALWEITEN RECHTS
   70 CONTINUE
C     LOOP INNNERE WEITEN RECHTS
   60 CONTINUE
C      LOOP INNERE WEITEN LINKS
   80 CONTINUE
C       LOOP POTENTIAL
   44 CONTINUE
C       LOOP OBER
  900 CONTINUE
1021  FORMAT(1X,10E12.5)
 1002 FORMAT(20I3)
C      LOOP BASISVECTOR
C
      RETURN
      END
      SUBROUTINE SCORR(POL,POR,Q,NA,F,KORR,NZT,POWE)
      IMPLICIT REAL*8 (A-H,O-Z)
C     SCORR BERECHNET DIE KORRELATIONSFUNKTION
C
      include "par.h"
C
      COMMON /SC/ ABC(4), ABD(4), ABE(4)
C
      COMMON /IMP/  MKC, NZV, MFL,NREB,NZAV
C
      COMMON /CALBE/ QO(NZVMAX,NZVMAX), QOR(NZVMAX,NZVMAX),
     *         VR(NZTMAX,NZCMAX-1), UR(NZVMAX), WR(NZTMAX),
     *        VZ(NZVMAX,NZVMAX), RPAL(MZPARM)
C
      COMMON /CALCEI/  LPARR, MC1, MD1, NCC
C
      DIMENSION Q(NZVMAX,NZVMAX),POL(NZVMAX,NZVMAX),POR(NZVMAX,NZVMAX)
C
      DO 7 M=1,NZV
      DO 7 N=1,NZV
      QO(N,M)=0.
7     QOR(N,M)=0.
      DO 100K=1,KORR
      FF=F*ABC(K)
       GOTO (30,5,2,2,5,2,2,2,2,2,2,2,2,2,2),MKC
   30 IF(K.EQ.1) GOTO 8
      FF=FF* DBLE((NZT*NZV)/2)
      GO TO 4
    5 IF(K.EQ.1) GOTO 8
      FF = FF* DBLE(NZT)/2.
      IF(K-4)      1,8,8
    2 IF(MKC.EQ.6 .OR. MKC.EQ.7 .OR. MKC.EQ.8 .OR. MKC.EQ.9
     *            .OR. MKC.EQ.11) GOTO 8
    4 GO TO (8,1,11,8) ,K
   11 FF=2.*FF
    8 DO 10  M = 1,NZV
      DO 10  N = M,NZV
      QOR(N,M)=POR(N,M)+ABE(K)*Q(N,M)
   10 QO(N,M)=POL(N,M)+ABD(K)*Q(N,M)  + QOR(N,M)
      IF(NA.EQ.0) GOTO 26
      IF( ABS(UR(NZV)).GE.0.0000001)GOTO 24
      NB = 1
      GO TO 22
   24 IF(MKC.GT.2) GOTO 26
       IF(K.NE.1) GOTO 26
        NB=1
      GO TO 22
   26 NB=0
   22 CONTINUE
      CALL BEGRI(NB,FF,POWE)
    1 CONTINUE
100   CONTINUE
      RETURN
      END
      SUBROUTINE HAUPT
      IMPLICIT REAL*8 (A-H,O-Z)
C     HAUPT FUEHRT HAUPTACHSENTRANSFORMATION DURCH
C     TRANSFORMATION IN VZ, HAUPTACHSENWERTE IN QO
C
      include "par.h"
C
      COMMON /IMP/  MKC, NZV, MFL,NREB,NZAV
C
      COMMON /CALBE/ QO(NZVMAX,NZVMAX), QOR(NZVMAX,NZVMAX),
     *         VR(NZTMAX,NZCMAX-1), UR(NZVMAX), WR(NZTMAX),
     *        VZ(NZVMAX,NZVMAX), RPAL(MZPARM)
C
      DO 1   K = 1,NZV
      VZ(K,K) = 1.
      if(k.eq.nzv) goto 1
    2 A = .5/QO(K,K)
      DO 7   M = K+1,NZV
      VZ(M,K) = .0
      B = A*QO(M,K)
      DO 5   L = 1,K
    5 VZ(M,L) = VZ(M,L) - B*VZ(K,L)
      QO(M,M) = QO(M,M) - B*B*QO(K,K)
      if(m.eq.nzv) goto 7
      DO 4   N = M+1,NZV
    4 QO(N,M) = QO(N,M) - A*QO(M,K)*QO(N,K)
    7 CONTINUE
    1 CONTINUE
      RETURN
      END
      SUBROUTINE BEGRI(NB,FF1,POWE)
      IMPLICIT REAL*8 (A-H,O-Z)
C
C
      include "par.h"
C
      PARAMETER (PHI2=1.77245385090552D0)
      parameter (NOUT=26)
C
C
      COMMON /BIG/ DM(NDIM,NDIM,2)
C
      COMMON /BEG/  IZLURE, INQ(NZOPER),NAUS
C
      COMMON /IMP/  MKC, NZV, MFL,NREB,NZAV
C
      COMMON /BEMAT/ WERT(NDIM4,NDIM4)
C
      COMMON /CSH/ NSH1(NDIM5), NSH2(NDIM5), SH(NDIM5), NSH
C
      COMMON /CALBE/ QO(NZVMAX,NZVMAX), QOR(NZVMAX,NZVMAX),
     *         VR(NZTMAX,NZCMAX-1), UR(NZVMAX), WR(NZTMAX),
     *        VZ(NZVMAX,NZVMAX), RPAL(MZPARM)
C
      COMMON /CALCEI/  LPARR, MC1, MD1, NCC
C
      COMMON /ww/ws
C
      DIMENSION P(NZAVMA,NZVMAX), BETA(NZVMAX), RHO(NZVMAX),
     *          S((NZAVMA*(NZAVMA+1))/2+1), YGAM(NZAVMA),
     *          GAM(NZAVMA), ZZ(NZAVMA),
     *          ZH(NZAVMA), ZW(NZVMAX), 
     *          SS(NZAVMA*(NZAVMA+1)+2), HX(NZAVMA-1,NZAVMA-1)
      DIMENSION H(NZLREC+1), RT(NZVMAX), RTH(NZVMAX),
     *          YH(NZAVMA), ZZH(NZAVMA), C(NZAVMA,NZAVMA,NZAVMA,NZAVMA)
C
      DIMENSION WERTT(NDIM4*NDIM4), DMM(NDIM*NDIM,2)
C
      EQUIVALENCE (WERT,WERTT), (DM,DMM)
C
C
C       WR = KOORDINATEN DER WW-TEILCHEN
      NZV1=NZV-1
      G= .0
      G0=0.
      DO 79   M = 1,NZV
      RT(M)=0.
      RTH(M)=0.
79    RHO(M) = .0
      DO 80   M = 1,NZV
      DO 80   N = 1,NZAV
   80 P(N,M) = 0.
      INZ1=(NZAV*(NZAV+1))/2+1
      DO 81 M=1,INZ1
81    S(M)=0.
      IF(NREB.EQ.0) GOTO 82
      DO 179  N = 1,NZAV
      YH(N)=0.
      ZZH(N)=0.
179   GAM(N) = .0
      DO 86   N = MC1+1,NZAV
      DO 86   K = 1,NZV
   86 GAM(N) = GAM(N) + VR(K,N-MC1)*WR(K)
C       TRANSFORMATION DER RELATIV-VECTOREN RECHTS AUF JACOBI
C      UND SKALARE MULTIPLIKATION MIT TRANSFORMATION DER
C      WECHSELWIRKENDEN TEILCHEN AUF JACOBI
   84 CONTINUE
      DO 87   M = 1,NZV
      DO 87   K = 1,NZV
   87 RHO(M) = RHO(M) + WR(K)*(QOR(K,M)+QOR(M,K))
C      RHO = TRANSFORMATION DER WW-TEILCHEN AUF JACOBI* RHO-MATRIX RECHT
      GRHO= 0.
      WR2=0.
      DO 85 K=1,NZV
      WR2=WR2 + WR(K)**2
85    GRHO=GRHO + WR(K)* RHO(K)
C     GRHO IST RHO * WW-TEILCHEN
      IF(NAUS.LT.3) GOTO 82
c     WRITE(NOUT,*) ' QOR',((QOR(N,M),N=1,NZV),M=1,NZV)
c     WRITE(NOUT,*) ' QO',((QO(N,M),N=1,NZV),M=1,NZV)
c     WRITE(NOUT,*) ' GAM ',(GAM(N),N=1,NZAV)
c     WRITE(NOUT,*) ' RHO ',(RHO(N),N=1,NZV)
c     WRITE(NOUT,*) ' WR ',(WR(N),N=1,NZV)
   82 CONTINUE
c      WRITE(NOUT,*) ' QO',((QO(N,M),N=1,NZV),M=1,NZV)
      CALL HAUPT
c      WRITE(NOUT,*) ' QOm',(QO(N,n),N=1,NZV)
      FFF=1.
      IF(NZV1.EQ.0) GOTO 71
       DO 1   M = 1,NZV1
      BETA(M)=1.D0/ SQRT(QO(M,M))
      FFF=PHI2*BETA(M)*FFF
c     write(nout,*) m,beta(m),fff
    1 CONTINUE
   71 CONTINUE
      BETA(NZV) = 1.
      DO 2 N=1,MC1
      NN=NZV-MC1+N
      DO 2 M=NN,NZV
    2 P(N,M)=VZ(M,NN)*BETA(M)
C      VZ(M,N) TRANSFORMATIONSMATRIX VON S(N) AUF T(M)
C     VZ(M,N)=T(N,M) EQ 7.9
C      P(M,N)=P(M,N) EQ 7.11
      DO 4 N=1,MD1
      NN=MC1+N
      DO 4 M=1,NZV
      DO 6 K=1,M
    6 P(NN,M)=VR(K,N)*VZ(M,K)+P(NN,M)
    4 P(NN,M)=P(NN,M)*BETA(M)
C      P= TRANSFORMATIONS-MATRIX VON RELATIV AUF DIAGONALKOORDINATEN T
C      BEREITS DIVIDIERT DURCH WURZEL BETA
      GOTO (10,10,7,10,10,10,10,10,10,7,  10,10,7,10), MKC
    7 DO 8 M=1,NZV
      DO 9 K=1,M
    9 P(NZAV,M)=P(NZAV,M)+UR(K)*VZ(M,K)
    8 P(NZAV,M)=P(NZAV,M)*BETA(M)
   10 CONTINUE
      DO 3  M = 1,NZAV
    3 ZH(M) = P(M,NZV)
      DO 5   M = 1,NZV
    5 ZW(M) = VZ(NZV,M)
      IF (NZV1.EQ.0) GOTO 14
      DO 11 N=1,NZV1
      DO 11 m=n,nzv1
   11 VZ(M,N)=VZ(M,N)*BETA(M)
C     VZ DIVIDIERT DURCH WURZEL BETA
      I=0
      DO 12 M=1,NZAV
      DO 12 N=M,NZAV
      I=I+1
      DO 12   K = 1,NZV1
   12 S(I)=S(I)+P(N,K)*P(M,K)
C      S= SIGMA-FAKTOREN
C     SIGMA(N,N') EQ 7.14
      IF (MKC.NE.3)  GOTO 14
      I=I+1
      DO 16 K=1,NZV1
   16 S(I)=S(I)+P(NZAV,K)**2
   14 CONTINUE
      IF(NREB.EQ.0) GOTO 88
      IF (NZV1.EQ.0) GOTO 46
      DO 41 K=1,NZV1
      DO 41 M=1,K
      RT(K)=RT(K)+RHO(M)*VZ(K,M)
   41 RTH(K)=RTH(K)+WR(M)*VZ(K,M)
      DO 40 K=1,NZV1
   40 G=G+RT(K)*RT(K)
      G0=G
      DO 62 N=1,NZAV
      DO 62 K=1,NZV1
      ZZH(N)=ZZH(N) + RTH(K)*P(N,K)
62    YH(N)=YH(N) + RT(K)*P(N,K)
  46  GG = G - 2.*GRHO
      GOTO (88,88,88,88,47,47,47,88,88,88,  47,88,88,88), MKC
47    G1=0.
       G2=0.
       IF(NZV1.EQ.0) GOTO 88
      DO 140 K=1,NZV1
      G1=G1+RT(K)*RTH(K)
140    G2=G2+RTH(K)*RTH(K)
   88 CONTINUE
      DO 170  KPARR = 1,NCC
      BETA(NZV) = 1./ SQRT(QO(NZV,NZV)+RPAL(KPARR))
      FFW = FFF *PHI2*BETA(NZV)
      FF=FFW**3*FF1
c      IF(NAUS.GT.2) WRITE(NOUT,*) 'BETA',(BETA(K),K=1,NZV),' FF1,FF ',
c     *   FF1,FF
      DO 91   M = 1,NZAV
   91 P(M,NZV) = ZH(M)*BETA(NZV)
      DO 92    M = 1,NZV
   92 VZ(NZV,M) = ZW(M)*BETA(NZV)
      IF(NAUS.LT.3) GOTO 136
c      DO 132 M=1,NZV
c132   WRITE(NOUT,*) ' P ',(P(N,M),N=1,NZAV)
c      DO 133 M=1,NZV
c133   WRITE(NOUT,*) ' T ',(VZ(N,M),N=1,NZV)
136   I = 0
      DO 93   M = 1,NZAV
      DO 93   N = M,NZAV
      I = I + 1
   93 SS(I) = 2.*(S(I) + P(N,NZV)*P(M,NZV))
      SS(I+1) = 2.*(S(I+1) + P(NZAV,NZV)**2)
      I = (NZAV*(NZAV-1))/2
      IF(NREB.GT.0) GOTO 19
      G = FF
      H(1)=G
      IF(MKC.EQ.10 .OR. MKC.EQ.13) H(2)=G
      IF (MKC.NE.3) GOTO 33
      I=(NZAV*(NZAV+1))/2
      IC = I+1
      CC = 1. / SS(IC)
      G = G * SQRT(2.*CC)
      H(INQ(3)) = G
      DO 34  M=1,NZAV
      DO 34  N=M,NZAV
      IC = IC+1
      IC1 = M*NZAV - (M*(M-1))/2
      IC2 = N*NZAV - (N*(N-1))/2
34    SS(IC) = SS(IC1) * SS(IC2) * CC
      SS(IC+1) = SS(I+1)
33    CONTINUE
      CALL MAT(H,SS,INQ(MKC))
      GO TO 70
19    RT(NZV)=0.
      RTH(NZV)=0.
      DO 96    M = 1,NZV
      RT(NZV) = RT(NZV) + RHO(M)*VZ(NZV,M)
   96 RTH(NZV) = RTH(NZV) + WR(M)*VZ(NZV,M)
c      IF(NAUS.GT.2) WRITE(NOUT,*) ' RT' ,(RT(M),M=1,NZV)
c      IF(NAUS.GT.2) WRITE(NOUT,*) ' RTH' ,(RTH(M),M=1,NZV)
      DO 24 N=1,NZAV
      YGAM(N) = YH(N) + RT(NZV)*P(N,NZV) -2.*GAM(N)
24    ZZ(N) = ZZH(N) + RTH(NZV)*P(N,NZV)
      IF(MKC.EQ.2) GOTO 196
      IF(MKC.EQ.8 .OR. MKC .EQ. 9) GOTO 32
      G0P=G0+RT(NZV)*RT(NZV)
      G1P=G1+RT(NZV)*RTH(NZV)
      G2P=G2+RTH(NZV)*RTH(NZV)
      IF(MKC.EQ.5) G=.5*(GG+RT(NZV)*RT(NZV))+2.*POWE*(G1P
     1               -2.+POWE*2.*G2P)
      IF(MKC.GT.5) G=-(G0P*G2P - G1P*G1P + 2.*WR2*G1P - 2.*GRHO*G2P)/4.
c    1                (8.*SQRT(3.))
      GOTO 97
196   G = GG + RT(NZV)*RT(NZV)
97    II= 0
      IF(MKC.GT.10) GOTO 98
      H(1)=1.5*FF*G
      II= 1
98    DO 44 N=1,NZAV
      DO 44 M=1,NZAV
   44 HX(M,N)=2.*FF*YGAM(M)*YGAM(N)
      GOTO(30,52,30,30,150,160,160,30,30,30,  160,30,30,30), MKC
52    DO 144 M=2,IZLURE
      MXH=(M-2)/NZAV
      MH=M-1-MXH*NZAV
      DO 144 N=M,IZLURE
      NXH=(N-1)/NZAV
      NH=N-NXH*NZAV
      II=II+1
144   H(II)=HX(MH,NH)
145   CONTINUE
      IF(MKC.LT.6 .OR. MKC.EQ.8 .OR. MKC.EQ.9) GOTO 148
      DO 146 N=1,NZAV - 1
      DO 146 M=N+1,NZAV
      DO 146 NP=N,NZAV - 1
      MPMIN=NP+1
      IF(NP.EQ.N) MPMIN=MAX(M,NP+1)
      DO 146 MP=MPMIN,NZAV
      II=II+1
146   H(II)=(C(N,M,NP,MP) + C(M,N,MP,NP)-C(M,N,NP,MP)-C(N,M,MP,NP))
148   IF(NAUS.LT.3) GOTO 149
c      WRITE(NOUT,*) 'WR2 ',WR2,' GRHO ',GRHO,' FF ',FF
c      WRITE(NOUT,*) ' G0P ',G0P,' G1P ',G1P,' G2P ',G2P
c      WRITE(NOUT,*) ' ZZ ',(ZZ(N),N=1,NZAV)
c      WRITE(NOUT,*) ' H ',(H(N),N=1,II)
c      WRITE(NOUT,*) ' S ',(SS(N),N=1,(NZAV*(NZAV+1))/2)
149   CALL MAT(H,SS,INQ(MKC))
      GO TO 70
32    DO 51 N=1,NZAV
      DO 51 M=1,NZAV
   51 HX(M,N)=-.5*FF*(ZZ(M)*YGAM(N)-ZZ(N)*YGAM(M))
      II= 1
C     LS  
      GO TO 52
150   DO 154 N=1,NZAV
      DO 154 M=1,NZAV
154   HX(M,N)=FF*2.*POWE*(ZZ(M)*YGAM(N)+ZZ(N)*YGAM(M)
     1  +4.*POWE*ZZ(M)*ZZ(N))+.5*HX(M,N)
C     P**2
      GOTO 52
160   FR=1.
      IF(MKC.EQ.11) FR=-0.5
      DO 164 N=1,NZAV
      DO 164 M=N,NZAV
164   HX(N,M)=FR*(FF*.5*(
     1  ZZ(N)*ZZ(M)*(2.*GRHO - 1.*G0P)
     2  +(G1P-WR2)*(ZZ(N)*YGAM(M) + ZZ(M)*YGAM(N))
     3  -G2P*YGAM(M)*YGAM(N) ) )
      DO 166 N=1,NZAV
      DO 166 M=1,NZAV
      DO 166 NP=1,NZAV
      DO 166 MP=1,NZAV
      C(N,M,NP,MP)=-.125*FF*ZZ(N)*ZZ(NP)*YGAM(M)*YGAM(MP)
166   CONTINUE
C     L**2,LS**2 SKALAR
      IF(MKC.LT.11) GOTO 52
C     LS**2 TENSOR
      II=2
      DO 168 N=1,NZAV
      DO 168 M=N,NZAV
      II=II+1
168   H(II)= HX(N,M)
      GOTO 145
70    CONTINUE
      I0 = NDIM * LPARR + KPARR
      DO 100 M=1,NSH
c      I2 = NSH2(M)
c      CC= SH(M) * WERTT(I2)
      CC= SH(M) * WS
      IF (CC.EQ.0.) GOTO 100
      I1 = NSH1(M) + I0
      IF (NB) 110,110,111
C       DMM(.,.,2) ENTHAELT ALLE BEITRAEGE
C       DMM(.,.,1) ENTHAELT NUR BEITRAEGE FUER NB=0
 110  DMM(I1,1) = DMM(I1,1) + CC
 111  DMM(I1,2) = DMM(I1,2) + CC
c      WRITE(NOUT,120) M,I1,I2,SH(M),WS,DMM(I1,1),DMM(I1,2)
120   FORMAT(' M,I1,I2,SH,WS,DMM1,2 ',3I5,E14.6,3E20.14)
  100 CONTINUE
  170 CONTINUE
      RETURN
   30 CONTINUE
      STOP '30'
      END
      SUBROUTINE MAT(H,S,NQ)
      IMPLICIT REAL*8 (A-H,O-Z)
C
C
      include "par.h"
      parameter (NOUT=16)
C
      COMMON /LURI/ KVK(NDIM1), JQ(NDIM1), INDPO(NDIM1),
     *    MVK(NZIQMA,NDIM1), IZ,JZ, KZHV(2,NZLREC+1), LUPAUS,
     *    IQM(NZLREC+1), KZHX(0:NZIQMA,NZLREC+1)
C
      COMMON /LURR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)
C
      COMMON /IMP/  MKC, NZV, MFL,NREB,NZAV
C
      COMMON /BEMAT/ WERT(NDIM4,NDIM4)
C
      COMMON /ww/ws
C
      DIMENSION S(NZAVMA*(NZAVMA+1)+2), H(NZLREC+1)
      DIMENSION WERTT(NDIM4*NDIM4),F(NDIM1)
      EQUIVALENCE (WERT,WERTT)
c      DO 4   LR = 1,JZ
c      DO 4   LL = 1,IZ
c    4 WERT(LL,LR) = .0
      WS=0.
      NU=1
      NN=NQ
      GO TO(13,13,10,13,13,13,13,11,11,11,12,13,11,13),MKC
   10 NU = NQ
      GO TO 13
   11 NU = 2
12    IF (MKC.EQ.11) NU = 3
13    IF(LUPAUS.GT.3) THEN
c      WRITE(NOUT,*) ' H ',(H(N),N=NU,NN)
c      WRITE(NOUT,*) ' S ',(S(N),N=1,12)
      ENDIF
      DO 1 N=NU,NN
      K3ZAHL = KZHV(2,N)
      IF(K3ZAHL.EQ.0) GOTO 1
      IF(ABS(H(N)).LT.1.E-15)  GOTO 1
      K1ZAHL = KZHV(1,N) + 1
      K2ZAHL = K1ZAHL + K3ZAHL - 1
      if(jq(k1zahl).ne.jq(k2zahl)) stop 'mehrere jq'
      jqq=jq(k1zahl)
      w=0.
      IF(KZHX(0,N).GT.KZHX(1,N)) GOTO 41
      goto (21,22,23,24,25,26,27,28,29,30),jqq
21    DO 121 K=K1ZAHL,K2ZAHL
121    W=W+H(N)*EPO(K)*S(MVK(1,K))
      goto 40
22    DO 122 K=K1ZAHL,K2ZAHL
122    W=W+H(N)*EPO(K)*S(MVK(1,K))*S(MVK(2,K))
      goto 40
23    DO 123 K=K1ZAHL,K2ZAHL
123    W=W+H(N)*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))
      goto 40
24    DO 124 K=K1ZAHL,K2ZAHL
124    W=W+H(N)*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
      goto 40
25    DO 125 K=K1ZAHL,K2ZAHL
125    W=W+H(N)*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
     + *S(MVK(5,K))
      goto 40
26    DO 126 K=K1ZAHL,K2ZAHL
126    W=W+H(N)*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
     + *S(MVK(5,K))*S(MVK(6,K))
      goto 40
27    DO 127 K=K1ZAHL,K2ZAHL
127    W=W+H(N)*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
     + *S(MVK(5,K))*S(MVK(6,K))*S(MVK(7,K))
      goto 40
28    DO 128 K=K1ZAHL,K2ZAHL
128    W=W+H(N)*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
     + *S(MVK(5,K))*S(MVK(6,K))*S(MVK(7,K))*S(MVK(8,K))
      goto 40
29    DO 129 K=K1ZAHL,K2ZAHL
129    W=W+H(N)*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
     + *S(MVK(5,K))*S(MVK(6,K))*S(MVK(7,K))*S(MVK(8,K))*S(MVK(9,K))
      goto 40
30    stop 'jq zu gross'
41    DO 141 K=K1ZAHL,K2ZAHL
141    W=W+H(N)*EPO(K)
40    CONTINUE
c      wertt(1)=wertt(1)+W
      WS=WS+W
    1 CONTINUE
      RETURN
      END
      SUBROUTINE LURE(MKC,ITV2)
      IMPLICIT REAL*8 (A-H,O-Z)
C     LURE LIEST MATRIXELEMENTE AUS LUDWIG EIN
C
      include "par.h"
      PARAMETER (NOUT=16)
C
      COMMON /LURI/ KVK(NDIM1), JQ(NDIM1), INDPO(NDIM1),
     *    MVK(NZIQMA,NDIM1), IZ,JZ, KZHV(2,NZLREC+1), LUPAUS,
     *    IQM(NZLREC+1), KZHX(0:NZIQMA,NZLREC+1)
C
      COMMON /LURR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)
C
      COMMON /LUR/ IENT(NZOPLU), NBAND3
C
      COMMON /BEG/  IZLURE, INQ(NZOPER),NAUS
C
      ITV2=0
      IU=2
C    OPERATOR 1  2   3   4   5   6   7   8    9   10
      GOTO (10, 60, 40,400, 400, 60,400, 20, 400, 20,
     *      60, 400, 20,  400 ), MKC
C
10    IU=1
20    DO 30, I=1, IZ
       DO 30, J=1, JZ
        WERTL(I,J)=0.
30    CONTINUE
60    CONTINUE
C    OPERATOR 1  2   3   4   5   6   7   8    9   10
      GOTO(  61, 61,446,400,400,40, 400, 61, 400, 61,
     *       52,400, 61,400), MKC
40    IU=INQ(2) +1
      GOTO 61
C
52    IU=3
61    CONTINUE
      DO 200, IIZ=IU, INQ(MKC)
       READ (NBAND3) K1ZAHL, IQMH
       IF (IQMH.GT.NZIQMA) GOTO 445
c      IF (LUPAUS.GT.0) WRITE (NOUT,1000) IIZ, K1ZAHL, IQMH
       KZHV(2,IIZ)=K1ZAHL
       IF (IIZ.EQ.1) KZHV(1,1)=0
       IF (IIZ.NE.1) KZHV(1,IIZ)=KZHV(1,IIZ-1)+KZHV(2,IIZ-1)
       KZAHL=KZHV(1,IIZ)+K1ZAHL
       IF (K1ZAHL.EQ.0) GOTO 200
       IQMH=MAX0(IQMH,1)
       IQM(IIZ)=IQMH
       IF (KZAHL.GT.NDIM1) GOTO 444
       K2ZAHL=KZAHL-K1ZAHL+1
       READ (NBAND3) (JQ(KW),INDPO(KW),EPO(KW),(MVK(IW,KW),
     *                IW=1,IQMH), KW=K2ZAHL,KZAHL)
C    SORTIEREN NACH ANZAHL DER SIGMA FAKTOREN
        CALL ORDSRT(IIZ,K2ZAHL,KZAHL)
C
       CALL FAPOR(K2ZAHL,KZAHL)
200   CONTINUE
C
      ITV2=KZAHL
      IF (MKC.EQ.8.OR.MKC.EQ.10) ITV2=ITV2-KZHV(2,1)
      IF (MKC.EQ.11) ITV2=ITV2-KZHV(2,2)
C
400   RETURN
C
444   WRITE (NOUT,1001) IIZ, KZAHL, NDIM1
      STOP 444
445   WRITE (NOUT,1002) IIZ, IQMH, NZIQMA
      STOP 445
446   STOP 'MKC SPRINGT FALSCH'
C
1000  FORMAT(/,I3,'. RECORD AUS LUDWIG:',
     *       ' RECORDLAENGE',I7,'   IQM',I5)
1001  FORMAT(/,1X,'RECORD AUS LUDWIG ZU LANG',/,1X,
     *       'IIZ, KZAHL, NDIM1:',3I8)
1002  FORMAT(/,1X,'ZU VIELE SIGMAFAKTOREN',/,1X,
     *       'IIZ, IQM, NZIQMA:',3I4)
      END
      SUBROUTINE FAPOR(IA,IE)
      IMPLICIT REAL*8 (A-H,O-Z)
C     FAPOR REPRODUZIERT DIE DREIMPULS- UND POLYNOMSTRUKTUR
C
      include "par.h"
      PARAMETER (NOUT=16)
C
      COMMON /LURI/ KVK(NDIM1), JQ(NDIM1), INDPO(NDIM1),
     *    MVK(NZIQMA,NDIM1), IZ,JZ, KZHV(2,NZLREC+1), LUPAUS,
     *    IQM(NZLREC+1), KZHX(0:NZIQMA,NZLREC+1)
C
      COMMON /LURR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)
C
      IF(LUPAUS.EQ.0) GOTO 100
      write(NOUT,10) IA,IE
10    FORMAT(' VON LUDWIG ELEMENTE VON',I8,' BIS ',I10)
      IF(LUPAUS.LT.2) GOTO 100
      DO 15 I=IA,IE
15    write(NOUT,20) JQ(I),INDPO(I),EPO(I),(MVK(J,I),J=1,JQ(I))
20    FORMAT(I5,' INDEX',I10,' EPO ',E15.8,' MVK',19I3)
100   DO 1 I=IA,IE
      KPL=INDPO(I)/100000+1
      KPR=MOD(INDPO(I)/10000,10)+1
      LL=MOD(INDPO(I)/100,100)+1
      LR=MOD(INDPO(I),100)+1
      LPL=(KPL-1)*NZLWMA+LL
      LPR=(KPR-1)*NZLWMA+LR
      WERTL(LPL,LPR) = 1.
       L = NDIM4 * (LPR-1) + LPL
      KVK(I) = L
    1 CONTINUE
      RETURN
      END
      SUBROUTINE ORDSRT(IIZ,KUNTEH,KOBEH)
      IMPLICIT REAL*8 (A-H,O-Z)
C     ORDSRT SORTIERT DEN GESAMTEN COMMONBLOCK LUP NACH AUFSTEIGENDER
C     ANZAHL VON SIGMAFAKTOREN
      
      include "par.h"
      PARAMETER (NOUT=16)
C
      COMMON /LURI/ KVK(NDIM1), JQ(NDIM1), INDPO(NDIM1),
     *    MVK(NZIQMA,NDIM1), IZ,JZ, KZHV(2,NZLREC+1), LUPAUS,
     *    IQM(NZLREC+1), KZHX(0:NZIQMA,NZLREC+1)
C
      COMMON /LURR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)
C
C    KZHX(IQ,IIZ) GIBT DIE NIEDRIGSTE ELEMENT NUMMER AN MIT MINDESTENS
C                 IQ FAKTOREN IM RECORD IIZ
C
      KUNTH=KUNTEH
      KZAHL=KOBEH
      IQMH=IQM(IIZ)
      DO 10 IQ=0,IQMH
10    KZHX(IQ,IIZ)=0
C
      DO 100, IQ=0, IQMH
C        LOOP ANZAHL SIGMAFAKTOREN
C
         KZHX(IQ,IIZ)=KUNTH
         KUNTEN=KUNTH
         KOBH  =KZAHL
C
         DO 80, KH=KUNTEN, KZAHL
C
            IF (JQ(KH).EQ.IQ) GOTO 80
C           SIGMAFAKTOR FALSCH PLAZIERT
            KOBEN=KOBH
            DO 60, IH=KOBEN, KH+1, -1
               IF (JQ(IH).NE.IQ) GOTO 60
C              KANDIDAT ZUM AUSTAUSCH GEFUNDEN
               INT=JQ(IH)
               JQ(IH)=JQ(KH)
               JQ(KH)=INT
               INT=INDPO(IH)
               INDPO(IH)=INDPO(KH)
               INDPO(KH)=INT
               RE=EPO(IH)
               EPO(IH)=EPO(KH)
               EPO(KH)=RE
               DO 20, KQ=1, IQMH
                  IND=MVK(KQ,IH)
                  MVK(KQ,IH)=MVK(KQ,KH)
                  MVK(KQ,KH)=IND
20             CONTINUE
               KOBH=IH-1
               GOTO 70
C
60          CONTINUE
C           KEINEN TAUSCHKANDIDATEN MEHR GEFUNDEN, NAECHSTES IQ NEHMEN
            KUNTH=KH
            GOTO 90
C
70          CONTINUE
C           TAUSCHPARTNER GEFUNDEN UND GETAUSCHT, WEITERSUCHEN
C
80       CONTINUE
C
C        ENDE LOOP ANZAHL SIGMAFAKTOREN
90      IF(JQ(KZAHL).EQ.0) GOTO 110
100   CONTINUE
110   CONTINUE
c     IF(LUPAUS.GT.1) WRITE(NOUT,999) (KZHX(IH,IIZ),IH=0,IQMH)
999   FORMAT(' IIZ,KZHX ',20I4)
      RETURN
      END
      
      subroutine fehler(Nummer)
C     Statt einen Prozess mit STOP abzuschiessen, sollen alle
C     Prozesse von dem Unheil unterrichtet werden und daran 
C     eingehen. Scheint nicht richtig zu funktionieren.
      include "mpif.h"
      PARAMETER (NOUT=16)
      INTEGER iStatus( MPI_STATUS_SIZE )      
      call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)

      do i=1,nproc-1 
         istop=66
         call MPI_SEND(istop,1,MPI_INTEGER,i,0,MPI_COMM_WORLD,ierr)
      enddo

      
      write (NOUT,*) 'FEHLER! Gestoppt.'
      write (NOUT,*) 'STOP ',nummer
      if (nummer.eq.261) write (NOUT,*) 'NZC?'
      if (nummer.eq.277) write (NOUT,*) 'NOL?'
      if (nummer.eq.9000) write (NOUT,*) 'MZPARM ZU KLEIN?'
      if (nummer.eq.9000) write (NOUT,*) 'NPDC?'

      istop=666
      
      call MPI_BCAST(istop,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)

      CLOSE(UNIT=16,STATUS='KEEP')

      call MPI_ABORT(MPI_COMM_WORLD,ifehler,ierror)
      call MPI_FINALIZE(ierr)
      STOP 666
      end
