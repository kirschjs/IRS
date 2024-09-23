      PROGRAM DRQUA_ANTI_KOM_P
C ----- MPI VERSION VON DRQUA_AK_N -----------------------------------    
C     FUER SMIN STATTDESSEN
CC    SUBROUTINE QUAF
      IMPLICIT REAL*8 (A-H,O-Z)
      include 'mpif.h'
C
C     FUER URBANA 3N-POTENTIAL
C     VERSION FUER 6 CLUSTER
C     FUER SMIN VORBEREITET
C     JEWEILS 'SMIN' SUCHEN UND DIE ENTSPRECHENDEN AENDERUNGEN DURCHFUEHREN
C     DIE LAENGE DES COMMON BLOCKS 'COMOV' MUSS MIT DER IN SMINH PASSEN
C     DESHALB DURCHGEHEND 'par.h' durch 'par.SMIN' ANDERN
C
C     QUAF BERECHNET DIE REDUZIERTEN ORSTRAUMMATRIXELEMENTE
C     UND MULTIPLIZIERT SIE MIT DEN REDUZIERTEN SPINMATRIX-
C     ELEMENTEN
C
C     27.8.98 ZUORDNUNG FUER NB KORRIGIERT H.M.H.
C     1.9.98  POTENTIAL UND ZUGEHOERIGE PARAMETER WERDEN JETZT VON EXTRA
C             FILE EINGELESEN, DAMIT NNN-WECHSELWIRKUNG EINFACHER ZU
C             HANDHABEN H.M.H.
C     25.8.99 FUER TUCSON-MELBOURNE 'D1'-TERM WIRD JETZT 'LREG'=2 VERWENDET
C             MIT 'RJK' UND 'QJK' IM POTENTIAL FUER OPERATOR 3 UND 6 H.M.H.
C     29.11.99 ES WERDEN UNTEROPERATOREN MIT LPOT=2 ODER 1 ZUR BEHANDLUNG DER
C             KOORDINATEN BENUTZT, DAZU KOORDINATEN DER WW-TEILCHEN IM 
C             AUFRUF VON BEGRI H.M.H.
C     26.4.00 FUER TEXAS-LOS ALAMOS WIRD DAS PION AUF WECPOT(3) UND DAS OMEGA
C             AUF WECPOT(2) GELEGT IN DER FORM T_PI(R_JK)*(Y_O(R_IK)+Y_O(R_IJ))
C     16.5.00 FUER VPVS WERDEN ZUSAETZLICHE OPERATOREN SPIN-SPIN UND TENSOR
C             AUF DIE OPERATOREN 16 UNE 17 GELEGT H.M.H.
C
C     1. OPERATOR: NORM
C     2. OPERATOR: ZENTRAL
C     3. OPERATOR: SPIN-SPIN ANTI
C     4. OPERATOR: SPIN-SPIN KOM
C     5. OPERATOR: SPIN-TENSOR R_IK ANTI
C     6. OPERATOR: SPIN-TENSOR R_IJ ANTI
C     7. OPERATOR: SPIN-TENSOR R_IK KOM
C     8. OPERATOR: SPIN-TENSOR R_IJ KOM
C     9. OPERATOR: TENSOR-TENSOR RANG=0 ANTI
C    10. OPERATOR: TENSOR-TENSOR RANG=0 KOM
C    11. OPERATOR: TENSOR-TENSOR RANG=1 ANTI
C    12. OPERATOR: TENSOR-TENSOR RANG=1 KOM
C    13. OPERATOR: TENSOR-TENSOR RANG=2 ANTI
C    14. OPERATOR: TENSOR-TENSOR RANG=2 KOM
C    15. OPERATOR: TENSOR-TENSOR RANG=3 KOM
C    16 .OPERATOR: SPIN-SPIN
C    16 .OPERATOR: TENSOR
C
C
C     INPUT DES PROGRAMMS
C      INTEGERS IM FORMAT 24I3, REALS IM FORMAT 6E12.4
C 
C      BEZEICHNUNG DER BAENDER, AUSDRUCK VARIABLEN
C      FILENAME FUER POTENTIAL,UNIT=25
C     von INPOT=25: DIE STEUERKARTE FUER DIE ZU RECHNENDEN OPERATOREN
C     von INPOT=25: DIE STEUERKARTEN FUER DIE KORRELATIONSFUNKTIONEN
C     von INPOT=25: DIE DEFINITION DER POTENTIALE
C
C EINGABE FUER DIE FUNKTIONEN
C
C JEDER INDEX I) ENTSPRICHT EINER KARTE
C
C DER FOLGENDE DATENSATZ MUSS FUER JEDE FUNKTION (K=1,NZF) WIEDERHOLT
C WERDEN 
C  A) ZAHL DER PARAMETERSAETZE                                     NZPAR
C     DIE INNEREN WEITEPARAMETER
C     DIE RADIALWEITEPARAMETER
C WENN NBAND5=0,WIRD ALLES NEU GERECHNET,WENN NBAND5#0,WIRD VON DIESEM B
C  TEIL UBERNOMMEN
C
C
      INCLUDE 'par.h'
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
C     NPDC:       "      "     "  PDC'S AUS OBER
C--------------------------------------------------------------------

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

C     Ende Variablen zur Verwaltung der parallelen Rechnung

c    dimensionierung nzavma mag zu klein sein, besser 2*nzcmax hmh ??????
      PARAMETER (NZAVMA=2*NZCMAX-1, NZVMAX=NZTMAX-1,
     *           NZMAT=MZGMAX*MZGMAX,
     *           NDIM4=NZPOMA*NZLWMA, NZLRH=(2*NZCMAX-3)*(NZCMAX-1),
     *           lbig = ndim*ndim*2)
      PARAMETER (LIND=(NZFMAX*(NZFMAX+1))/2*NZOPER)
      DIMENSION INDEX(LIND+1)

CCCCC Daten, die mit dem Slave geteilt werden

      COMMON /INT/
     $     NZT , MKC  ,  NL , MC3   ,
     $     NR  , NC   , NTE , IRHO  , MFL,
     $     JRHO,  MFR , NDD , MD1   ,
     $     MD2 , MD   , ND  , MD3   , NZV,
     $     IK1 , JK1  , LPOT, MC1   , NCC,
     $     NZAV, NZV1 , NAUS
      PARAMETER( lint=4+5+4+5+5+3 )
      DIMENSION iint(lint)
      EQUIVALENCE( iint(1), nzt )

C     Integer Arrays
      COMMON /INTA/
     $     LC(NZTMAX,NPDC),NGRV(2,NZCMAX,NZFMAX),
     $     NUM(5,NZRHOM,NZFMAX),NT1(NPDC), NT2(NPDC),
     $     NT3(NPDC), NZPOT(NZOPER), LREG(NZOPER),
     $     NZRHO(NZFMAX)
      PARAMETER( linta = nztmax*npdc  + 2*nzcmax*nzfmax +
     $     5*nzrhom*nzfmax + npdc + npdc + npdc + nzoper + nzoper +
     $     nzfmax )
      DIMENSION iinta(linta)
      EQUIVALENCE( iinta(1), lc )

      COMMON /LUPI/ KVK(NDIM1), jqq,
     *    MVK(NZIQMA,NDIM1), IZ, JZ,
     *    KZAHL, LUPAUS, IQM, KZHX(0:NZIQMA)
      PARAMETER( llupi = ndim1 + 1 + ndim1*nziqma + 1 + 1 +
     $     1 + 1 + 1 + NZIQMA + 1 )
      DIMENSION ilupi(llupi)
      EQUIVALENCE( ilupi(1), kvk )

C     Double Arrays
      COMMON /DOPPELA/
     $     VW(NZTMAX,NZCMAX-1),       SVEC(NZTMAX,NZVMAX),
     $     RVEC(NZVMAX,NZTMAX),       WECPOT(NZPOTM,3,NZOPER),
     $     U(MZGMAX,MZGMAX,NPDC),     CPAR(NZAVMA-1,NZPARM,NZFMAX),
     $     QFCL(NZVMAX,NZVMAX,2*NZCMAX-1),
     $     COF(NZPARM,NZRHOM,NZFMAX), RPAR(MZPARM,NZFMAX)
      PARAMETER( ldoppela = nztmax*(nzcmax-1) + nztmax*nzvmax +
     $     nzvmax*nztmax + nzpotm*3*nzoper + mzgmax*mzgmax*npdc +
     $     (nzavma-1)*nzparm*nzfmax + nzvmax*nzvmax*(2*nzcmax-1) +
     $     nzparm*nzrhom*nzfmax + mzparm*nzfmax )
      DIMENSION rdoppela(ldoppela)
      EQUIVALENCE( rdoppela(1), vw )
      
      COMMON /LUPR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)
      PARAMETER( llupr = ndim1 + ndim4*ndim4 )
      DIMENSION rlupr(llupr)
      EQUIVALENCE( rlupr(1), epo )

CCCCC Ende der mit dem Sklaven gemeinsamen Daten


      COMMON /LUR/ NBAND3
C
      COMMON /BIG/ DM(NDIM,NDIM,2)
      dimension rbig(lbig)
      equivalence (rbig(1),dm)
C
      DIMENSION NZC(NZFMAX), MZG(NZFMAX), NOL(NZFMAX),
     $     IENT(NZOPLU), NZPAR(NZFMAX), NREG(NZOPOB),
     $     MS(MZGMAX,NZFMAX), MZPAR(NZFMAX)

      DIMENSION KOM(NZOPER,NZFMAX,NZFMAX),
     $     MMASSE(2,MZGMAX,NZFMAX), MLAD(2,MZGMAX,NZFMAX),
     $     MSS(2,MZGMAX,NZFMAX)
      DIMENSION NZLW(NZFMAX), LW(2*NZCMAX-1,NZLWMA,NZFMAX),
     *     LZWERT(5,NZLWMA,NZFMAX),
     *     NZC1(NZFMAX),  K1VEC(NZFMAX),
     *     NZPO(NZFMAX), KP(NZCMAX-1,NZPOMA,NZFMAX)
      DIMENSION  ITPO(NZFMAX),    JREOB(NZOPER), JRELU(NZOPER)
      DIMENSION kom2(NZFMAX,NZFMAX)
C
      DIMENSION IMV(NZMAT), UM(NZMAT)
C
      CHARACTER*100 INFILO, INFILU
      CHARACTER*50 VARFOR(NZOPER)
      CHARACTER*100 CPOTFIL

C--------------------------------------------------------------------
      DATA VARFOR/'('' NORM'')','('' ZENTRAL '')',
     *'('' SPIN-SPIN-POTENTIAL ANTI'')',
     *'('' SPIN-SPIN-POTENTIAL KOM'')',
     *'('' SPIN-TENSOR-POTENTIAL ANTI R-IK'')',
     *'('' SPIN-TENSOR-POTENTIAL ANTI R-IJ'')',
     *'('' SPIN-TENSOR-POTENTIAL KOM R-IK'')',
     *'('' SPIN-TENSOR-POTENTIAL KOM R-IJ'')',
     * '('' TENSOR-TENSOR RANG=0 ANTI '')',
     * '('' TENSOR-TENSOR RANG=0 KOM '')',
     * '('' TENSOR-TENSOR RANG=1 ANTI '')',
     * '('' TENSOR-TENSOR RANG=1 KOM '')',
     * '('' TENSOR-TENSOR RANG=2 ANTI '')',
     * '('' TENSOR-TENSOR RANG=2 KOM '')',
     * '('' TENSOR-TENSOR RANG=3 KOM '')',
     *'('' SPIN-SPIN-POTENTIAL VPVS'')',
     *'('' SPIN-TENSOR-POTENTIAL VPVS R-IK'')'/
C
      DATA NZPOT/NZOPER*1/
C
      DATA JRELU/1,0,0,0,2,0,0,0,3,0,4,0,5,0,6,7,8/
      DATA JREOB/1,0,2,3,4,0,5,6,7,8,9,10,11,12,13,14,15/
C
C     FUER SMIN SIND DIE MIT 'CC' VERSEHENEN KARTEN ZU AKTIVIEREN UND
C     DIE ANGEGEBENEN ZEILEN ZU DEAKTIVIEREN
C--------------------------------------------------------------------
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
C--------------------------------------------------------------------
      OPEN(UNIT=9 ,FILE='DRLUOUT',STATUS='OLD',FORM='UNFORMATTED')
      OPEN(UNIT=10,FILE='DRQUAOUT',STATUS='UNKNOWN',FORM='UNFORMATTED')
C
      OPEN(UNIT=15 ,FILE='INQUA_N'  ,STATUS='OLD')
      OPEN(UNIT=16 ,FILE='OUTPUT')
      OPEN(UNIT=8 ,FILE='DROBOUT', STATUS='OLD',FORM='UNFORMATTED')
      NOUT=16
       write(NOUT,*) 'Anzahl der Prozessoren: ',nproc
C     IN SMIN MUESSEN DIESE OPEN STATEMENTS UNWIRKSAM GEMACHT WERDEN
      WRITE(NOUT, 1111)
1111  FORMAT('1 DRQUA_AK_N_P VERSION VOM 16.02.2006')
      INPUT=15
      INPOT=25
      DO 1   K=1,NZPOTM
      DO 1 L=1,NZOPER
      WECPOT(K,1,L)=1.
      WECPOT(K,2,L)=0.
    1 WECPOT(K,3,L)=0.
      READ (INPUT,1002) NBAND1,NBAND2,NBAND3,IDUM,naufset,NAUS,MOBAUS
     1 ,LUPAUS, ISTOP, NLES
      WRITE (NOUT,1002) NBAND1,NBAND2,NBAND3,IDUM,naufset,NAUS,MOBAUS
     1 ,LUPAUS, ISTOP, NLES
C--------------------------------------------------------------------
      if(naufset.eq.1) stop 'naufset=1'
      open(unit=18,file='DRFINDEX',status='unknown',form='unformatted')
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
c     write(nout,*) 'nger',nger,' index ',index(nger)
      do 59 i=1,nger
      if(index(i).lt.naufset) goto 59
      write(nout,*) ' Fuer ',i,' ist index ',index(i),'groesser naufset'
      stop 'naufset falsch'
59    continue
      do i=nger+1,lind
           index(i)=-1
      enddo
      endif
C--------------------------------------------------------------------
      READ(INPUT,993) CPOTFIL                                                   
      WRITE(NOUT,993) CPOTFIL                                                   
993   FORMAT(A100)                                                               
      OPEN(UNIT=25,FILE=CPOTFIL,STATUS='OLD',FORM='FORMATTED')
      READ(INPOT,1002)(LREG(K),K=1, NZOPER)
 1020 FORMAT(6E12.4)
C     POTENTIAL EINLESEN
      READ (INPOT,1002)  (NZPOT(K), K=2,NZOPER)
      DO 530 K=2,NZOPER
      IF(NZPOT(K).EQ.0) LREG(K) = 0
  530 CONTINUE
      DO 742 MKC=2,NZOPER
      IF(NZPOT(MKC).EQ.0) GOTO 742
      DO 741 K=1,NZPOT(MKC)
741   READ(INPOT,1020)(WECPOT(K,M,MKC),M=1,3)
742   CONTINUE
1042  FORMAT(' ANZAHL DER POTENTIALE',I4)
1044  FORMAT(//)
1043  FORMAT(' VORFAKTOR=',G15.6,' BETA IJ ',F11.5,
     1   ' BETA IK  =',F11.5)
C     POTENTIAL EINLESEN
      DO 757 MKC=2,NZOPER
      IF(NZPOT(MKC).EQ.0) GOTO 757
      WRITE(NOUT,1044)
      WRITE(NOUT,VARFOR(MKC))
      WRITE(NOUT,1042) NZPOT(MKC)
      WRITE(NOUT,1043)((WECPOT(I,M,MKC),M=1,3),I=1,NZPOT(MKC))
757   CONTINUE
C     EINLESEN DER FUNKTIONSEIGENSCHAFTEN UND PARAMETER
      REWIND  NBAND2
      READ(NBAND2) NZF,NZT,(NREG(K),K=1,NZOPOB),INDEKO
      IF(INDEKO.NE.0) WRITE(NOUT,*)' OBERMATRIXELEMENTE VON EINZELFILES'
      NZV=NZT-1
      IF(MOBAUS.GT.0) WRITE(NOUT,*)' NZF,NZT,(NREG(K),K=1,NZOPOB)',
     *                NZF,NZT,NZV,(NREG(K),K=1,NZOPOB)
C
      do kr=1,nzfmax
      do kl=1,nzfmax
            kom2(kl,kr)=1
      enddo
      enddo
C
      IF(NBAND5.LE.0) GOTO 900
C
      write(nout,*) 'KOM:'
      i=0
C
      DO 92 KL=1,NZF
      DO 92 KR=1,KL
C
      KOM2(KL,KR)=0
      DO 91 L = 1, NZOPER
          kom(l,kl,kr)=0
          if(lreg(l).eq.0) goto 91
          i=i+1
          if(index(i).ne.-1) kom(l,kl,kr)=1
          KOM2(KL,KR)=KOM2(KL,KR)+LREG(L)*IABS(1-KOM(L,KL,KR))
91    continue
92    WRITE(NOUT,*) (KOM(L,KL,KR),L=1,NZOPER),'K2',KOM2(KL,KR)
C
      write(nout,*) 'war KOM.'
C
      GO TO 93
  900 CONTINUE
C
      I=0
C
      WRITE(NOUT,6100)
      DO 94 KL=1,NZF
      DO 94 KR=1,KL
      DO 94 L=1,NZOPER
C
      I=I+1
      INDEX(I)=-1
C
   94 KOM(L,KL,KR)=0
   93 CONTINUE
      NZV1 = NZV - 1
      WECPOT(1,1,1)=2./ DBLE(NZT*(NZT-1)*(NZT-2))
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
      READ (NBAND2) ((NGRV(N,L,K),L=1,M),N=1,2)
      IF(MOBAUS.GT.0) WRITE(NOUT,*)'NGRV',((NGRV(N,L,K),L=1,M),N=1,2)
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
      write(nout,*)'NZF,NZF1 = ',NZF,NZF1
      STOP 251
  250 STOP 250
  251 DO 252 K=1,NZF
      ITPO(K)=0
      IF(NZPO(K).NE.0) ITPO(K)=2
      NZPO(K)=MAX0(1,NZPO(K))
      IF(NZC(K).NE.NZC1(K)) STOP 252
  252 CONTINUE
C
C     UEBERPRUEFUNG, OB GERECHNETE OPERATOREN IN OBER, LUDWIG UND
C     QUAF ZUSAMMENPASSEN
C
      IF (IENT(1).EQ.0 .OR. NREG(1).EQ.0) STOP 253
      IF (LREG(3).EQ.0) GOTO 255
      IF (NREG(2).EQ.0) STOP 255
255   IF (LREG(4).EQ.0) GOTO 256
      IF (NREG(3).EQ.0) STOP  256
256   IF (LREG(5).EQ.0) GOTO 257
      IF (NREG(4)*IENT(2).EQ.0) STOP  257
257   IF (LREG(6).EQ.0) GOTO 258
      IF (NREG(4)*IENT(2).EQ.0) STOP  258
258   IF (LREG(7).EQ.0) GOTO 259
      IF (NREG(5)*IENT(2).EQ.0) STOP  259
259   IF (LREG(8).EQ.0) GOTO 1253
      IF (NREG(6)*IENT(2).EQ.0) STOP 1253
1253  IF (LREG(9).EQ.0) GOTO 1254
      IF (NREG(7)*IENT(3).EQ.0) STOP 1254
1254  IF (LREG(10).EQ.0) GOTO 1255
      IF (NREG(8)*IENT(3).EQ.0) STOP 1255
1255  IF (LREG(11).EQ.0) GOTO 1256
      IF (NREG(9)*IENT(4).EQ.0) STOP 1256
1256  IF (LREG(12).EQ.0) GOTO 1257
      IF (NREG(10)*IENT(4).EQ.0) STOP 1257
1257  IF (LREG(13).EQ.0) GOTO 1258
      IF (NREG(11)*IENT(5).EQ.0) STOP 1258
1258  IF (LREG(14).EQ.0) GOTO 1259
      IF (NREG(12)*IENT(5).EQ.0) STOP 1259
1259  IF (LREG(15).EQ.0) GOTO 1260
      IF (NREG(13)*IENT(6).EQ.0) STOP 1260
1260  IF (LREG(16).EQ.0) GOTO 1261
      IF (NREG(14)*IENT(7).EQ.0) STOP 1261
1261  IF (LREG(17).EQ.0) GOTO 262
      IF (NREG(15)*IENT(8).EQ.0) STOP 262
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
261   STOP 'NZC'
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
277   STOP 'NOL'
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
      LZWERT(2,L,K)=LW(4,L,K)
      GOTO 265
287   LZWERT(1,L,K)=LW(8,L,K)
  265 CONTINUE
  264 CONTINUE
C      BAHNDREHIMPULSE ENDE
C      KONSTRUKTION DER BASISVEKTOREN
      I=0
      WRITE (NOUT,1000)
 1000 FORMAT(1H1)
      DO 22  K = 1,NZF
      NZPAR(K) = 0
      MZPAR(K) = 0
       READ(INPUT,1002) NZRHO(K)
      KK=NZRHO(K)
      IF(KK.LE.NZRHOM) GOTO 388
      WRITE(NOUT,*) 'ZERLEGUNG,NZRHO,NZRHOM',K,NZRHO(K),NZRHOM
      GOTO 250
c388   IVEK(K)=I
388   IF(KK.EQ.0) GOTO 22
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
      IF(MZPAR(K).GT.MZPARM) STOP 'MZPARM ZU KLEIN'
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
c      NUM(3,N,K)=IVEK(K)+N
      NUM(3,N,K)=N
      READ(INPUT,1003) (COF(L,N,K),L=1,LM)
      WRITE(NOUT,1050) NUM(3,N,K),K,NUM(1,N,K),NUM(2,N,K),NUM(4,N,K),
     1  (COF(L,N,K),L=1,LM)
 1050 FORMAT(//15H DEFINITION DER,I3,22H TEN BASISFUNKTION IST/
     1 I3,13H TE ZERLEGUNG,I3,23H TE SPINISOSPINFUNKTION/I3,
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
      STOP 930
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
  950 CONTINUE
C--------------------------------------------------------------------
      CLOSE( unit=10 )
      if( naus.eq.666 ) stop 'DRQUAOUT geschrieben'

         write(fnumber,*) naufset
         do i=1, 255
            if(fnumber(i:i).ne.' ') goto 370
         end do
 370      do j=i, 255
            if(fnumber(j:j).eq.' ') goto 380
         end do
 380      oname = 'DRDMOUT.' // fnumber(i:j)
          open(unit=27,file=oname,status='replace',form='unformatted')
C
      nteil = 1
      narbeit = 0
      nsend = 1
C
 1933 format(a72)
C--------------------------------------------------------------------
      IF(INDEKO.NE.0) THEN
       MEFOB=23
       CLOSE(UNIT=8,STATUS='KEEP')
       NBAND2=MEFOB
      ENDIF
      IF(INDELU.NE.0) THEN
        MEFLU=25   
        CLOSE(UNIT=9,STATUS='KEEP')
        NBAND3=MEFLU
       ENDIF
c
      icount = 0
C     BESTIMMUNG DER ORTSMATRIXELEMENTE VOR EINSETZEN DER DREHIMPULSE
      DO 40  MFL = 1,NZF
      IZLW=NZLW(MFL)
      IZPW=NZPO(MFL)
      IRHO=NZRHO(MFL)
      IK1=K1VEC(MFL)
      MC = NZC(MFL)
      MC1= MC - 1
      MC2 = MC + MC1
      MC3 = MC2 - 1
      NC= NZPAR(MFL)
      NCC = MZPAR(MFL)
      IF(INDEKO.EQ.0) THEN
         READ(NBAND2)   ((RVEC(M,N),M=1,NZV),N=1,NZT)
         READ(NBAND2)   ((SVEC(N,M),M=1,NZV),N=1,NZT)
         READ(NBAND2)   (((QFCL(M,N,K),M=1,NZV),N=1,NZV),K=1,MC2)
      ENDIF
      DO 40 MFR=1,MFL
c--------------------------------------------------------------------
      CLOSE(UNIT=18,STATUS='KEEP')
      open(unit=18,file='DRFINDEX',status='unknown',form='unformatted')
      REWIND 18
c--------------------------------------------------------------------
      JZLW=NZLW(MFR)
      JZPW=NZPO(MFR)
      JRHO=NZRHO(MFR)
      JK1=K1VEC(MFR)
      WRITE(NOUT,3009) MFL,MFR
3009  FORMAT(//,19H ZWISCHEN ZERLEGUNG,I3,15H UND ZERLEGUNG ,I3,
     124H WIRD ZUR ZEIT GERECHNET )
      MD=NZC(MFR)
      MD1= MD - 1
      IF(INDEKO.NE.0) THEN
         READ(INPUT,378) INFILO
378      FORMAT(A100)
C
         write(NOUT,*) 'Oeffnen? MFL,MFR,KOM2(MFL,MFR)',
     $        MFL,MFR,KOM2(MFL,MFR),INFILO
         if(KOM2(MFL,MFR).EQ.0) goto 9010
         write(NOUT,*) 'Oeffne ',INFILO
C
         OPEN(UNIT=MEFOB,FILE=INFILO,STATUS='OLD',FORM='UNFORMATTED')
          REWIND MEFOB
         READ(NBAND2)   ((RVEC(M,N),M=1,NZV),N=1,NZT)
         READ(NBAND2)   ((SVEC(N,M),M=1,NZV),N=1,NZT)
         READ(NBAND2)   (((QFCL(M,N,K),M=1,NZV),N=1,NZV),K=1,MC2)
      ENDIF
9010  IF(INDELU.NE.0) THEN
         READ(INPUT,378) INFILU
C
         if(KOM2(MFL,MFR).EQ.0) goto 9011
C
         OPEN(UNIT=MEFLU,FILE=INFILU,STATUS='OLD',FORM='UNFORMATTED')
          REWIND MEFLU
      ENDIF
C
      if(KOM2(MFL,MFR).EQ.0) goto 9011
C
      READ(NBAND2)    ((VW(M,K),M=1,NZT),K=1,MD1)
C
9011  CONTINUE
C
      MD2 = MD + MD1
      MD3 = MD2 - 1
      ND = NZPAR(MFR)
      NDD = MZPAR(MFR)
      DO 42 MKC=1, NZOPER
      REWIND 18
      IZ=(IZPW-1)*NZLWMA+IZLW
      JZ=(JZPW-1)*NZLWMA+JZLW
      NZAV=MC1+MD1
      IF(LREG(MKC).GT.0) WRITE(NOUT,VARFOR(MKC))
      ELREG= LREG(MKC) - 1
C     OPERATOR 1   2    3    4    5    6    7    8    9    10
      GO TO (3008,3008,3008,3008,3004,3004,3004,3004,3002,3002,
     *       3002,3002,3002,3002,3002,3008,3004),MKC 
3002  NZAV=NZAV+2
      GO TO 3008
3004      NZAV=NZAV+1
3008  CONTINUE
      if(NZAV.GT.NZAVMA) STOP'NZAVMA'
      NL = MZG(MFL)
      NR = MZG(MFR)
      IF(JREOB(MKC).EQ.0) GOTO 440
      IF(NREG(JREOB(MKC)).EQ.0) GOTO 440
C
      IF(KOM2(MFL,MFR).EQ.0) goto 440
C
      READ(NBAND2)    IDUM,NTE,NTE
      IF(NTE.LE.NPDC) GOTO 6029
      WRITE(NOUT,*) ' ZUVIELE PDCS',NTE,' DIMENSION ',NPDC
      STOP 'NPDC'
6029  CONTINUE
      IF(NTE.EQ.0) GOTO 440
      DO 922   MTE = 1,NTE
      READ(NBAND2) LH,IMQ, (IMV(I),UM(I),I=1,IMQ)
      DO 11 I=1,NL
      DO 11 J=1,NR
      U(I,J,MTE)= 0.
11    CONTINUE
      DO 12 IK=1,IMQ
      I= IMV(IK)/NR
      J=MOD(IMV(IK),NR)
12    U(I+1,J+1,MTE)=UM(IK)
      DO 7 I=1,NZT
      LC(NZT+1-I,MTE)=MOD(LH,10)
7     LH=LH/10 
      NT3(MTE)=MOD(LH,10)
      LH=LH/10 
      NT2(MTE)=MOD(LH,10)
      NT1(MTE)= LH/10 
      IF(MOBAUS.EQ.0) GOTO 922
      IF(LREG(MKC).EQ. 0) GOTO 922
      WRITE (NOUT,838) NT1(MTE), NT2(MTE), NT3(MTE),(LC(K,MTE),K=1,NZT)
 838  FORMAT (10H VON OBER ,20I3)
      IF(MOBAUS.LT.2) GOTO 922
      WRITE (NOUT,839) ((U(NFL,NFR,MTE),NFL=1,NL), NFR=1,NR)
 839  FORMAT (1P10E12.4)
922   CONTINUE
C
C       READ LUPO ELEMENTE
440   IF(JRELU(MKC).EQ.0) GOTO 441
      IF(IENT(JRELU(MKC)).EQ.0 ) GOTO 441
C
      IF(KOM2(MFL,MFR).EQ.0 .and. INDELU.NE.0) goto 441
C
      CALL LURE(MKC,ITV2)
      IF(LUPAUS.GE.0) WRITE(NOUT,*)' ',ITV2,' LUC-MATRIXELEMENTE'
441   IF(LREG(MKC).EQ.0)  GOTO 6033
      IF (KOM(MKC,MFL,MFR)) 6030,6031,6032
 6030 WRITE(NOUT,6034)
      GO TO 6033
 6031 WRITE(NOUT,6035)
      GO TO 6033
 6032 WRITE(NOUT,6036)
 6033 CONTINUE
C
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
C     keine BV
95    IF(IRHO*JRHO.LE.0) GOTO 42
      IF(LREG(MKC).LE.0) GOTO 42

C     Nichts tun, koennen wir auch selber
      if( nte*itv2.eq.0 ) then
         write( nout,* ) 'Nichts zu tun.',nteil
         index(nteil)=0
         icount=icount+1
               WRITE (27) nteil,icount,index(nteil)

         goto 942
      endif
C--------------------------------------------------------------------
C          MPI master/slave austausch aus mpi_quaf_v6.f

c schon alle beschaeftigt?
      nfertig=0
      write(NOUT,*) 'narbeit, nproc, nteil',narbeit,nproc,nteil

      if(narbeit .ge. (nproc-1) ) then
c        alle beschaeftigt, erstmal auf Ergebnisse warten
         call MPI_RECV(ngerechnet,1,MPI_INTEGER,MPI_ANY_SOURCE,
     *                 MPI_ANY_TAG,MPI_COMM_WORLD,istatus,ierr)
         call MPI_RECV(nteilalt,1,MPI_INTEGER,ngerechnet,0,
     *                 MPI_COMM_WORLD,istatus,ierr)

         call MPI_RECV(rbig,lbig,MPI_DOUBLE_PRECISION,ngerechnet,0,
     *                 MPI_COMM_WORLD,istatus,ierr)


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
c         write(NOUT,*) 'STARTE Verteilung von Op.Nr. ', MKC
      call MPI_SEND(idummy,1,MPI_INTEGER,nstarte,0,
     *     MPI_COMM_WORLD,ierr)

      call MPI_SEND(nteil,1,MPI_INTEGER,nstarte,0,
     *     MPI_COMM_WORLD,ierr)

c------------------------------------------------
      read (coutput,'(80A1)') dummy

      do 9002 n=1,80
 9002    jdummy(n)=ichar(dummy(n))
      call MPI_SEND(jdummy(1),80,MPI_INTEGER,nstarte,0,
     *        MPI_COMM_WORLD,ierr)
c------------------------------------------------

      call MPI_SEND(rdoppela,ldoppela,MPI_DOUBLE_PRECISION,nstarte,0,
     *     MPI_COMM_WORLD,ierr)
c      call MPI_SEND(rstore,lstore,MPI_DOUBLE_PRECISION,nstarte,0,
c     *     MPI_COMM_WORLD,ierr)

      call MPI_SEND(iint,lint,MPI_INTEGER,nstarte,0,
     *     MPI_COMM_WORLD,ierr)
c      call MPI_SEND(ibeg,lbeg,MPI_INTEGER,nstarte,0,
c     *     MPI_COMM_WORLD,ierr)

      call MPI_SEND(ilupi,llupi,MPI_INTEGER,nstarte,0,
     *     MPI_COMM_WORLD,ierr)
c      call MPI_SEND(iluri,lluri,MPI_INTEGER,nstarte,0,
c     *     MPI_COMM_WORLD,ierr)

      call MPI_SEND(rlupr,llupr,MPI_DOUBLE_PRECISION,nstarte,0,
     *     MPI_COMM_WORLD,ierr)
c      call MPI_SEND(rlurr,llurr,MPI_DOUBLE_PRECISION,nstarte,0,
c     *     MPI_COMM_WORLD,ierr)

      call MPI_SEND(iinta,linta,MPI_INTEGER,nstarte,0,
     *     MPI_COMM_WORLD,ierr)
c      call MPI_SEND(iimp,limp,MPI_INTEGER,nstarte,0,
c     *     MPI_COMM_WORLD,ierr)

c      call MPI_SEND(ikomi,lkomi,MPI_INTEGER,nstarte,0,
c     *     MPI_COMM_WORLD,ierr)

c      call MPI_SEND(rsc,lsc,MPI_DOUBLE_PRECISION,nstarte,0,
c     *     MPI_COMM_WORLD,ierr)

c     ... mit Arbeit versorgt ---------------------------------------

C     CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2),
C    $        BIG_BEN (3), DATE_TIME)
C      write(NOUT,*) 'Rechner ',nstarte," um :: ",date_time,
C    $     ' abgefertigt'
c         write(NOUT,*) 'Alles verteilt von Op.: ', MKC

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
         write(NOUT,*)  'Queu wird geleert, warte auf jobs = ', narbeit
      do i=1, narbeit
         call MPI_RECV(ngerechnet,1,MPI_INTEGER,MPI_ANY_SOURCE,
     *        MPI_ANY_TAG,MPI_COMM_WORLD,istatus,ierr)
         call MPI_RECV(nteilalt,1,MPI_INTEGER,ngerechnet,0,
     *        MPI_COMM_WORLD,istatus,ierr)

         call MPI_RECV(rbig,lbig,MPI_DOUBLE_PRECISION,ngerechnet,0,
     *        MPI_COMM_WORLD,istatus,ierr)

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

c      open(12, file=output )
c      write(NOUT,'(a7,i1)') "OUTPUT.",myid
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

c------------------------------------------------
         call MPI_RECV(jdummy(1),80,MPI_INTEGER,0,0,
     *           MPI_COMM_WORLD,istatus,ierr)
         do 9003 n=1,80
 9003       coutput(n:n)=char(jdummy(n))
c------------------------------------------------

C         CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2),
C     $           BIG_BEN (3), DATE_TIME)
c         write(NOUT,*) 'STARTE Empfang  um :: ',date_time
c         write(NOUT,*) 'STARTE Empfang von Op.Nr./myid  ', MKC,'/',myid

         call MPI_RECV(rdoppela,ldoppela,MPI_DOUBLE_PRECISION,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

C         CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2),
C     $        BIG_BEN (3), DATE_TIME)
c         write(12,*) 'Empfangen lstore um :: ',date_time,lstore

         call MPI_RECV(iint,lint,MPI_INTEGER,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

C         CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2),
C     $        BIG_BEN (3), DATE_TIME)
c         write(12,*) 'Empfangen lbeg um :: ',date_time,lbeg

         call MPI_RECV(ilupi,llupi,MPI_INTEGER,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

         call MPI_RECV(rlupr,llupr,MPI_DOUBLE_PRECISION,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

         call MPI_RECV(iinta,linta,MPI_INTEGER,0,0,
     *        MPI_COMM_WORLD,istatus,ierr)

c         call MPI_RECV(ikomi,lkomi,MPI_INTEGER,0,0,
c     *        MPI_COMM_WORLD,istatus,ierr)

c         call MPI_RECV(rsc,lsc,MPI_DOUBLE_PRECISION,0,0,
c     *        MPI_COMM_WORLD,istatus,ierr)

c Namen fuer Slave bestimmen
C         CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2),
C     $        BIG_BEN (3), DATE_TIME)
c         write(NOUT,*) 'Empfangen Alles um :: ',date_time
c         write(NOUT,*) 'Empfangen Alles von Op./myid ',MKC,'/',myid


         write(fnumber,*) nteil
         do i=1, 255
            if(fnumber(i:i).ne.' ') goto 70
         end do
 70      do j=i, 255
            if(fnumber(j:j).eq.' ') goto 80
         end do
 80      do k=1, 80
            if(coutput(k:k).eq.' ') goto 83
         end do
 83      oname = coutput(1:k-1) // fnumber(i:j)



C         TIME0=MCLOCK()/100.
C         RT0=timef()/1000.

c        write(15,*) 'Ich bin ',myid
c        write(15, *) 'Programm hat Bisher ',time0,'s gerechnet.'
C         CALL DATE_AND_TIME (BIG_BEN (1), BIG_BEN (2),
C     $        BIG_BEN (3), DATE_TIME)
C         write(NOUT,*) date_time
C         write(NOUT,*) date_time,my_name

C     Ein Aufruf
c         write(NOUT,*) 'Betrete CALCU, Op/id:',MKC,'/',myid
         CALL CALCU
c         write(NOUT,*) 'Verlasse CALCU, Op/id:',MKC,'/',myid
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
C      STOP
 6100 FORMAT(/31H ES WIRD VOELLIG NEU GERECHNET /)
 6034 FORMAT(34H FUER DIESEN OPERATOR WIRD ERSETZT /)
 6035 FORMAT(40H FUER DIESEN OPERATOR WIRD NEU GERECHNET /)
 6036 FORMAT(34H FUER DIESEN OPERATOR WIRD KOPIERT /)
 1051 FORMAT(1X,'DIESER BASISVEKTOR WIRD NICHT NEU GERECHNET')
      END
C--------------------------------------------------------------------
      SUBROUTINE CALCU
      IMPLICIT REAL*8 (A-H,O-Z)

      INCLUDE 'par.h'

      PARAMETER (
     $     NZAVMA=2*NZCMAX-1,
     $     NZVMAX=NZTMAX-1,
     $     NZMAT=MZGMAX*MZGMAX,
     $     NDIM4=NZPOMA*NZLWMA,
     $     NZLRH=(2*NZCMAX-3)*(NZCMAX-1))
C      integer*4 time

CCCCC DIESE DATEN MUESSEN VOM MEISTER UEBERTRAGEN WERDEN
      COMMON /INT/
     $     NZT , MKC  ,  NL , MC3   ,
     $     NR  , NC   , NTE , IRHO  , MFL,
     $     JRHO,  MFR , NDD , MD1   ,
     $     MD2 , MD   , ND  , MD3   , NZV,
     $     IK1 , JK1  , LPOT, MC1   , NCC,
     $     NZAV, NZV1 , NAUS
      PARAMETER( lint=4+5+4+5+5+3 )
      DIMENSION iint(lint)
      EQUIVALENCE( iint(1), nzt )

C     Integer Arrays
      COMMON /INTA/
     $     LC(NZTMAX,NPDC),NGRV(2,NZCMAX,NZFMAX),
     $     NUM(5,NZRHOM,NZFMAX),NT1(NPDC), NT2(NPDC),
     $     NT3(NPDC), NZPOT(NZOPER), LREG(NZOPER),
     $     NZRHO(NZFMAX)
      PARAMETER( linta = nztmax*npdc  + 2*nzcmax*nzfmax +
     $     5*nzrhom*nzfmax + npdc + npdc + npdc + nzoper + nzoper +
     $     nzfmax )
      DIMENSION iinta(linta)
      EQUIVALENCE( iinta(1), lc )

      COMMON /LUPI/ KVK(NDIM1), jqq,
     *    MVK(NZIQMA,NDIM1), IZ, JZ,
     *    KZAHL, LUPAUS, IQM, KZHX(0:NZIQMA)
      PARAMETER( llupi = ndim1 + 1 + ndim1*nziqma + 1 + 1 +
     $     1 + 1 + 1 + NZIQMA + 1 )
      DIMENSION ilupi(llupi)
      EQUIVALENCE( ilupi(1), kvk )

C      Double Arrays
      COMMON /DOPPELA/
     $     VW(NZTMAX,NZCMAX-1),       SVEC(NZTMAX,NZVMAX),
     $     RVEC(NZVMAX,NZTMAX),       WECPOT(NZPOTM,3,NZOPER),
     $     U(MZGMAX,MZGMAX,NPDC),     CPAR(NZAVMA-1,NZPARM,NZFMAX),
     $     QFCL(NZVMAX,NZVMAX,2*NZCMAX-1),
     $     COF(NZPARM,NZRHOM,NZFMAX), RPAR(MZPARM,NZFMAX)
      PARAMETER( ldoppela = nztmax*(nzcmax-1) + nztmax*nzvmax +
     $     nzvmax*nztmax + nzpotm*3*nzoper + mzgmax*mzgmax*npdc +
     $     (nzavma-1)*nzparm*nzfmax + nzvmax*nzvmax*(2*nzcmax-1) +
     $     nzparm*nzrhom*nzfmax + mzparm*nzfmax )
      DIMENSION rdoppela(ldoppela)
      EQUIVALENCE( rdoppela(1), vw )

      COMMON /LUPR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)
      PARAMETER( llupr = ndim1 + ndim4*ndim4 )
      DIMENSION rlupr(llupr)
      EQUIVALENCE( rlupr(1), epo )

CCCCC ENDE DER ZU UEBERTRAGENDEN DATEN

C     Im Sklaven lokale Variablen, evtl. muessen
C     einige an begri und haupt uebergeben werden

      DIMENSION LC1(NZTMAX),
     $     QFCR(NZVMAX,NZVMAX,2*NZCMAX-1),
     $     QJK(NZVMAX,NZVMAX), RJK(NZVMAX),UR(NZVMAX),
     $     RIK(NZVMAX),
     $     QFO(NZVMAX,NZVMAX), POR(NZVMAX,NZVMAX),
     $     QIK(NZVMAX,NZVMAX), POL(NZVMAX,NZVMAX),
     $     UU(MZGMAX,MZGMAX),  POLL(NZVMAX,NZVMAX)


      PARAMETER (LBIG=NDIM*NDIM*2)

C     An Unterprogramme uebergeben
      COMMON /CSH/ NSH1(NDIM5), NSH2(NDIM5), SH(NDIM5), NSH
      COMMON /BIG/ DM(NDIM,NDIM,2)
      dimension rbig(lbig)
      equivalence (rbig(1),dm)
C
      COMMON /SKLAVE/ QO(NZVMAX,NZVMAX), VR(NZTMAX,NZCMAX-1)
      COMMON VZ(NZVMAX,NZVMAX), WERT(NDIM4,NDIM4), LPARR

CCCCC Buchhaltung fuer paralleles Programm
      CHARACTER*255 oname,  fnumber

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DO 460 N=1,NDIM
      DO 460 M=1,NDIM
      DM(M,N,1)=.0
  460 DM(M,N,2)=.0
      DO 490 M=1,NZT
  490 LC1(M)=0
c      IF(MFR.NE.MFL) GOTO 464
c464   IF (NTE*ITV2.LE.0) GOTO 900
      DO 44 MTE=1,NTE
      IF(NAUS.LT.1) GOTO 1491
      WRITE(NOUT,*) MTE,' TES OBERMATRIXELEMENT'
C
#ifdef PC_FORTRAN
#endif
#ifdef SUN_FORTRAN
#endif
#ifdef CRAY_FORTRAN
#endif
C
C     FUER SMIN MUESSEN DIESE STATEMENTS BESEITIGT WERDEN
c      CLOSE(UNIT=16,STATUS='KEEP')
c      OPEN(UNIT=16,FILE='OUTPUT',ACCESS='APPEND')
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
      IF(TS*TD.GT.0.000001) NA=0
C     PERMUTATION UEBER FRAGMENTGRENZEN
      DO 50   M = 1,NZV
      DO 50   N = 1,M
      DO 51   K= 1,MD2
   51 QFCR(M,N,K) = .0
      DO 50 K=1,MD1
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
      RIK(M) = RVEC(M,NT1(MTE)) - RVEC(M,NT3(MTE))
      RJK(M) = RVEC(M,NT2(MTE)) - RVEC(M,NT3(MTE))
      IF(MKC.GT.15) THEN
C     hier ist rik die jacobikoordinate des dritten teilchens
         RIK(M)= RVEC(M,NT1(MTE))+RVEC(M,NT2(MTE))-2.*RVEC(M,NT3(MTE))
         RIK(M)= RIK(M)/SQRT(6.)
      ENDIF
      DO 54  N = 1,NZV
      QIK(M,N)=.0
      QJK(M,N)=.0
   54 QFO(M,N)=.0
c      IF(NAUS.GT.2) WRITE(NOUT,*) 'UR ',(UR(M),M=1,NZV)
c      IF(NAUS.GT.2) WRITE(NOUT,*) 'RIK ',(RIK(M),M=1,NZV)
c      IF(NAUS.GT.2) WRITE(NOUT,*) 'RJK ',(RJK(M),M=1,NZV)
      DO 55   M = 1,NZV
      DO 55   N = 1,NZV
      MM=MIN0(M,N)
      NN=MAX0(N,M)
      QFO(NN,MM) = QFO(NN,MM) + UR(M)*UR(N)
      QIK(NN,MM) = QIK(NN,MM) + RIK(M)*RIK(N)
      QJK(NN,MM) = QJK(NN,MM) + RJK(M)*RJK(N)
   55 CONTINUE
      DO 80 IPOT=1, NZPOT(MKC)
      FAK=WECPOT(IPOT,1,MKC)
      DO 81 M=1,NL
      DO 81 N=1,NR
      UU(M,N)=U(M,N,MTE)
   81 CONTINUE
      DO 60   KPAR = 1,NC
      DO 62   M = 1,NZV
      DO 62   N = 1,NZV
   62 POL(M,N) = .0
      DO 64   M = 1,NZV
      IF(MC3.LE.0) GOTO 64
      DO 63   K = 1,MC3
   63 POL(M,M)=POL(M,M) + CPAR(K,KPAR,MFL)*QFCL(M,M,K)
   64 CONTINUE
       DO 70 LPAR=1,ND
C       FUER UNTEROPERATOREN
        LPOT=0
        IF(LREG(MKC).EQ.2)  LPOT=2
C
      I=0
      DO 101 M=1,IRHO
      KSL= NUM(1,M,MFL)
      KLL= NUM(2,M,MFL)
      KPL=NUM(4,M,MFL)
      JPL=(KPL-1)*NZLWMA+KLL
      NUML= NUM(3,M,MFL)
      TS = COF(KPAR,M,MFL)
      IF (ABS(TS).LT.1.D-10) GOTO 101
      MADL = (M-1) * IK1
      DO 105 N=1,JRHO
      KSR= NUM(1,N,MFR)
      KLR= NUM(2,N,MFR)
      KPR=NUM(4,N,MFR)
      JPR=(KPR-1)*NZLWMA+KLR
      NUMR= NUM(3,N,MFR)
C--------------------------------------------------------------------
      IF (MFL.EQ.MFR .AND. NUML.LT.NUMR) GOTO 105
C--------------------------------------------------------------------
      IF(WERTL(JPL,JPR).NE.1.) GOTO 105
       A = TS * UU(KSL,KSR)*COF(LPAR,N,MFR)
      IF(ABS(A).LT.1.D-20) GOTO 105
      MADR = (N-1) * JK1
      IF (I.LT.NDIM5) GOTO 110
c      WRITE (NOUT,111) I
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
      DO 72   M = 1,NZV
      DO 72   N = 1,NZV
   72 POR(M,N) = .0
      DO 74  M = 1,NZV
      DO 74  N = M,NZV
      IF(MD3.LE.0) GOTO 74
      DO 73  K = 1,MD3
   73 POR(N,M)=POR(N,M)+CPAR(K,LPAR,MFR)*QFCR(N,M,K)
   74 CONTINUE
  774 CONTINUE
C     EINSPRUNG FUER UNTEROPERATOREN
      DO 67 M=1,NZV
      DO 67  N = M,NZV
      IF(LPOT.EQ.0) POLL(N,M)=POL(N,M)+
     1  WECPOT(IPOT,2,MKC)*QFO(N,M)
     2  +WECPOT(IPOT,3,MKC)*QIK(N,M)
      IF(LPOT.EQ.2) POLL(N,M)=POL(N,M)+
     1  WECPOT(IPOT,2,MKC)*QIK(N,M)
     2  +WECPOT(IPOT,3,MKC)*QJK(N,M)
      IF(LPOT.EQ.1) POLL(N,M)=POL(N,M)+
     1  WECPOT(IPOT,2,MKC)*QFO(N,M)
     2  +WECPOT(IPOT,3,MKC)*QJK(N,M)
67    CONTINUE
      DO 76 LPARR=1,NDD
      DO 77 M=1,NZV
      DO 77 N=M,NZV
   77 QO(N,M)=POLL(N,M)+POR(N,M)+RPAR(LPARR,MFR)*QFCR(N,M,MD2)
      NB=NA
      IF(NA.EQ.0) GOTO 170
      IF((LPOT.EQ.0 .OR. LPOT.EQ.1).AND.ABS(UR(NZV)).GT.1.D-6) GOTO 160
      IF((LPOT.EQ.0 .OR. LPOT.EQ.2).AND.ABS(RIK(NZV)).GT.1.D-6) GOTO 160
      IF((LPOT.EQ.1 .OR. LPOT.EQ.2).AND.ABS(RJK(NZV)).GT.1.D-6) GOTO 160
      GOTO 170
160   IF(MKC.EQ.1) GOTO 170
      NB=0
170   CONTINUE
c     write(nout,*)'pol',((pol(n,m),n=1,nzv),m=1,nzv)
c     write(nout,*)'poll',((poll(n,m),n=1,nzv),m=1,nzv)
c     write(nout,*)'por',((por(n,m),n=1,nzv),m=1,nzv)
c     write(nout,*)'qo',((qo(n,m),n=1,nzv),m=1,nzv)
      IF(LPOT.EQ.0) CALL BEGRI(NB,FAK,UR,RIK)
      IF(LPOT.EQ.1) CALL BEGRI(NB,FAK,RJK,RIK)
      IF(LPOT.EQ.2) CALL BEGRI(NB,FAK,RJK,RIK)
   76 CONTINUE
C      LOOP RADIALWEITEN RECHTS
       IF(LPOT.NE.2) GOTO 70
         LPOT=LPOT-1
         GOTO 774
   70 CONTINUE
C     LOOP INNNERE WEITEN RECHTS
   60 CONTINUE
C      LOOP INNERE WEITEN LINKS
   80 CONTINUE
C       LOOP POTENTIAL
   44 CONTINUE
C       LOOP OBER
  900 CONTINUE
C
 1002 FORMAT(20I3)
      DO 480 M=1,IRHO
      NUML=NUM(3,M,MFL)
      DO 481 N=1,JRHO
      NUMR=NUM(3,N,MFR)
c      IF(NUML.LT.NUMR) GOTO 481
      IF (NUM(5,M,MFL)*NUM(5,N,MFR).GT.0) GOTO 482
C     HIER WERDEN DIE BASISVEKTOREN UEBERSPRUNGEN
      M1=(M-1)*IK1+1
      M2=M*IK1
      N1=(N-1)*JK1+1
      N2=N*JK1
      II1 = 1
c      A = 0.
c      DO 510 I=1,2
c      DO 510 L=N1,N2
c      DO 510 K=M1,M2
c 510  A = A + ABS(DM(K,L,I))
c      IF(A.GT.0.) GOTO512
c      WRITE (NBAND1) NUML,NUMR,II1,II1,A,A
c      GOTO 481
c512   WRITE(NBAND1) NUML,NUMR,IK1,JK1,(((DM(K,L,I),K=M1,M2),L=N1,N2),
c     1 I=1,2)
      IF (NAUS.EQ.0) GOTO 481
      WRITE (NOUT,1002) NUML, NUMR, IK1, JK1
      IF(NAUS.LT.1) GOTO 481
      WRITE (NOUT,1021) (((DM(K,L,I), K=M1,M2),L=N1,N2),I=1,2)
1021  FORMAT(1X,10E12.5)
482   IF(NUM(5,M,MFL)*NUM(5,N,MFR).GT.0) WRITE(NOUT,*) 'M,N',M,N,
     *  ' NUM5', NUM(5,M,MFL),NUM(5,N,MFR)
  481 CONTINUE
  480 CONTINUE
C      LOOP BASISVECTOR
C
      RETURN
      END
C
c      ISTOPP=ISTOPP+1
c      IF (ISTOP.GT.0.AND.ISTOPP.GE.ISTOP) STOP 777
C     HIER KANN PROGRAMM NACH BERECHNUNG VON ISTOP OPERATOREN GESTOPPT
C     WERDEN
C
C     FUER SMIN MUESSEN DIESE STATEMENTS BESEITIGT WERDEN
c      CLOSE(UNIT=16,STATUS='KEEP')
c      CLOSE(UNIT=10,STATUS='KEEP')
c      OPEN(UNIT=10,FILE='DRQUAOUT',STATUS='UNKNOWN',ACCESS='APPEND',
c     *      FORM='UNFORMATTED')
c      OPEN(UNIT=16,FILE='OUTPUT',ACCESS='APPEND')
c   42 CONTINUE
C        LOOP OPERATOREN
c   40 CONTINUE
c      WRITE(NOUT,3011)
c3011  FORMAT(//,' ENDE DER RECHNUNG VON QUAF')
c      CLOSE(UNIT=9,STATUS='KEEP')
c      CLOSE(UNIT=12,STATUS='KEEP')
CC    RETURN
C     FUER SMIN AKTIVIEREN UND STOP DEAKTIVIEREN
c      STOP
c 6100 FORMAT(/31H ES WIRD VOELLIG NEU GERECHNET /)
c 6034 FORMAT(34H FUER DIESEN OPERATOR WIRD ERSETZT /)
c 6035 FORMAT(40H FUER DIESEN OPERATOR WIRD NEU GERECHNET /)
c 6036 FORMAT(34H FUER DIESEN OPERATOR WIRD KOPIERT /)
c1051  FORMAT(1X,'DIESER BASISVEKTOR WIRD NICHT NEU GERECHNET')
c      END
c--------------------------------------------------------------------
      SUBROUTINE LURE(MKC,ITV2)
      IMPLICIT REAL*8 (A-H,O-Z)
C     LURE LIEST MATRIXELEMENTE AUS LUDWIG EIN
C
      INCLUDE 'par.h'
C
      PARAMETER (NZAVMA=2*NZCMAX-1, NDIM4=NZPOMA*NZLWMA,
     *           NZLRH=(2*NZCMAX-3)*(NZCMAX-1))
C
      COMMON /LUPI/ KVK(NDIM1), jqq,
     *    MVK(NZIQMA,NDIM1), IZ, JZ,
     *    KZAHL, LUPAUS, IQM, KZHX(0:NZIQMA)

      COMMON /LUPR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)

      COMMON /LUR/  NBAND3

C     Die beiden werden nur von lure, fapor und ordrsrt benoetigt
      COMMON /UPROG/ JQ(NDIM1), INDPO(NDIM1)
C
      ITV2=0
      IU=2
C    OPERATOR 1  2   3   4   5   6   7   8    9   10
      GOTO (10, 400, 400,400, 10,400,400,400, 10, 400,
     *      10, 400, 10 ,400,10, 10, 10), MKC
C
10    DO 30, I=1, IZ
       DO 30, J=1, JZ
        WERTL(I,J)=0.
30    CONTINUE
       READ (NBAND3) KZAHL, IQMH
       IF (IQMH.GT.NZIQMA) GOTO 445
       IF (LUPAUS.GT.0) WRITE (NOUT,1000)  KZAHL, IQMH
       IF (KZAHL.EQ.0) GOTO 200
       IQM=MAX0(IQMH,1)
       IF (KZAHL.GT.NDIM1) GOTO 444
       READ (NBAND3) (JQ(KW),INDPO(KW),EPO(KW),(MVK(IW,KW),
     *                IW=1,IQM), KW=1,KZAHL)
C    SORTIEREN NACH ANZAHL DER SIGMA FAKTOREN
        CALL ORDSRT(KZAHL)
C
      if(jq(1).ne.jq(kzahl)) stop 'mehrere jq'
      jqq=jq(1)
C
       CALL FAPOR(1,KZAHL)
200   CONTINUE
C
      ITV2=KZAHL
C
400   RETURN
C
444   WRITE (NOUT,1001)  KZAHL, NDIM1
      STOP 444
445   WRITE (NOUT,1002)  IQMH, NZIQMA
      STOP 445
1000  FORMAT(/,' RECORD AUS LUDWIG:',
     *       ' RECORDLAENGE',I7,'   IQM',I5)
1001  FORMAT(/,1X,'RECORD AUS LUDWIG ZU LANG',/,
     *       ' KZAHL, NDIM1:',3I8)
1002  FORMAT(/,1X,'ZU VIELE SIGMAFAKTOREN',/,
     *       ' IQM, NZIQMA:',3I4)
      END
c--------------------------------------------------------------------
      SUBROUTINE FAPOR(IA,IE)
      IMPLICIT REAL*8 (A-H,O-Z)
C     FAPOR REPRODUZIERT DIE DREIMPULS- UND POLYNOMSTRUKTUR
C
      INCLUDE 'par.h'
C
      PARAMETER (NZAVMA=2*NZCMAX-1, NDIM4=NZPOMA*NZLWMA,
     *           NZLRH=(2*NZCMAX-3)*(NZCMAX-1))
C
      COMMON /LUPI/ KVK(NDIM1), jqq,
     *    MVK(NZIQMA,NDIM1), IZ, JZ,
     *    KZAHL, LUPAUS, IQM, KZHX(0:NZIQMA)

      COMMON /LUPR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)

      COMMON /UPROG/ JQ(NDIM1), INDPO(NDIM1)
C
      IF(LUPAUS.EQ.0) GOTO 100
      WRITE(NOUT,10) IA,IE
10    FORMAT(' VON LUDWIG ELEMENTE VON',I8,' BIS ',I10)
      IF(LUPAUS.LT.2) GOTO 100
      DO 15 I=IA,IE
15    WRITE(NOUT,20) JQ(I),INDPO(I),EPO(I),(MVK(J,I),J=1,JQ(I))
20    FORMAT(I5,' INDEX',I10,' EPO ',E12.5,' MVK',19I3)
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
c--------------------------------------------------------------------
      SUBROUTINE HAUPT
      IMPLICIT REAL*8 (A-H,O-Z)
C     HAUPT FUEHRT HAUPTACHSENTRANSFORMATION DURCH
C     TRANSFORMATION IN VZ, HAUPTACHSENWERTE IN QO
C
      INCLUDE 'par.h'
      PARAMETER (NZVMAX=NZTMAX-1)
C
      COMMON /INT/
     $     NZT , MKC  ,  NL , MC3   ,
     $     NR  , NC   , NTE , IRHO  , MFL,
     $     JRHO,  MFR , NDD , MD1   ,
     $     MD2 , MD   , ND  , MD3   , NZV,
     $     IK1 , JK1  , LPOT, MC1   , NCC,
     $     NZAV, NZV1 , NAUS

      COMMON VZ(NZVMAX,NZVMAX)

      COMMON /SKLAVE/ QO(NZVMAX,NZVMAX), VR(NZTMAX,NZCMAX-1)
C
      DO 1   K = 1,NZV
      VZ(K,K) = 1.
      IF(NZV-K)   1,1,2
    2 A = .5/QO(K,K)
      K1 = K + 1
      DO 7   M = K1,NZV
      VZ(M,K) = .0
      B = A*QO(M,K)
      DO 5   L = 1,K
    5 VZ(M,L) = VZ(M,L) - B*VZ(K,L)
      QO(M,M) = QO(M,M) - B*B*QO(K,K)
      IF(NZV-M)   7,7,6
    6 M1 = M + 1
      DO 4   N = M1,NZV
    4 QO(N,M) = QO(N,M) - A*QO(M,K)*QO(N,K)
    7 CONTINUE
    1 CONTINUE
      RETURN
      END
c--------------------------------------------------------------------
      SUBROUTINE BEGRI(NB,FF1,UR,RIK)
      IMPLICIT REAL*8 (A-H,O-Z)
C
C
      INCLUDE 'par.h'
C
C
      PARAMETER (NZAVMA=2*NZCMAX-1, NZVMAX=NZTMAX-1,
     *           NDIM4=NZPOMA*NZLWMA, NZLRH=(2*NZCMAX-3)*(NZCMAX-1))
C
      COMMON /BIG/ DM(NDIM,NDIM,2)
C
      COMMON /CSH/ NSH1(NDIM5), NSH2(NDIM5), SH(NDIM5), NSH
C
      COMMON /SKLAVE/ QO(NZVMAX,NZVMAX), VR(NZTMAX,NZCMAX-1)
c
      COMMON /INT/
     $     NZT , MKC  ,  NL , MC3   ,
     $     NR  , NC   , NTE , IRHO  , MFL,
     $     JRHO,  MFR , NDD , MD1   ,
     $     MD2 , MD   , ND  , MD3   , NZV,
     $     IK1 , JK1  , LPOT, MC1   , NCC,
     $     NZAV, NZV1 , NAUS
c
C     Nur rpar wird verwendet!
      COMMON /DOPPELA/
     $     VW(NZTMAX,NZCMAX-1),       SVEC(NZTMAX,NZVMAX),
     $     RVEC(NZVMAX,NZTMAX),       WECPOT(NZPOTM,3,NZOPER),
     $     U(MZGMAX,MZGMAX,NPDC),     CPAR(NZAVMA-1,NZPARM,NZFMAX),
     $     QFCL(NZVMAX,NZVMAX,2*NZCMAX-1),
     $     COF(NZPARM,NZRHOM,NZFMAX), RPAR(MZPARM,NZFMAX)
c
      COMMON VZ(NZVMAX,NZVMAX), WERT(NDIM4,NDIM4), LPARR
c
      COMMON /ww/w
C
      DIMENSION RIK(NZVMAX), UR(NZVMAX)
C
      DIMENSION P(NZAVMA,NZVMAX), BETA(NZVMAX),
     *          S((NZAVMA*(NZAVMA+1))/2+1),
     *          ZH(NZAVMA),  SS(NZAVMA*(NZAVMA+1)+2)
C
      DIMENSION WERTT(NDIM4*NDIM4), DMM(NDIM*NDIM,2)
C
      EQUIVALENCE (WERT,WERTT), (DM,DMM)
C
      DATA PHI2 /1.77245385090552D0/
C
      DO 80   M = 1,NZV
      DO 80   N = 1,NZAV
   80 P(N,M) = 0.
      INZ1=(NZAV*(NZAV+1))/2
      DO 81 M=1,INZ1
81    S(M)=0.
      CALL HAUPT
      FFF=1.
      IF(NZV1.EQ.0) GOTO 71
       DO 1   M = 1,NZV1
      BETA(M)=1./ SQRT(QO(M,M))
    1 FFF=PHI2*BETA(M)*FFF
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
C   OPERATOR 1  2  3  4  5  6  7  8  9  10
      GOTO ( 10,10,10,10,72,7 ,72,7 ,70,70,
     *       70,70,70,70,70,10,72), MKC
   70 DO 88 M=1,NZV
      DO 90 K=1,M
      P(NZAV-1,M)=P(NZAV-1,M)+UR(K)*VZ(M,K)
   90 P(NZAV,M)=P(NZAV,M)+RIK(K)*VZ(M,K)
      P(NZAV-1,M)=P(NZAV-1,M)*BETA(M)
   88 P(NZAV,M)=P(NZAV,M)*BETA(M)
      GOTO 10
    7 DO 8 M=1,NZV
      DO 9 K=1,M
    9 P(NZAV,M)=P(NZAV,M)+UR(K)*VZ(M,K)
    8 P(NZAV,M)=P(NZAV,M)*BETA(M)
      GOTO 10
   72 DO 82 M=1,NZV
      DO 94 K=1,M
   94 P(NZAV,M)=P(NZAV,M)+RIK(K)*VZ(M,K)
   82 P(NZAV,M)=P(NZAV,M)*BETA(M)
   10 CONTINUE
      DO 3  M = 1,NZAV
    3 ZH(M) = P(M,NZV)
      I=0
      DO 12 M=1,NZAV
      DO 12 N=M,NZAV
      I=I+1
      DO 12   K = 1,NZV1
   12 S(I)=S(I)+P(N,K)*P(M,K)
C      S= SIGMA-FAKTOREN
C     SIGMA(N,N') EQ 7.14
 14   DO 170  KPARR = 1,NCC
      BETA(NZV) = 1./ SQRT(QO(NZV,NZV)+RPAR(KPARR,MFL))
      FFW = FFF *PHI2*BETA(NZV)
      FF=FFW**3*FF1
      DO 91   M = 1,NZAV
   91 P(M,NZV) = ZH(M)*BETA(NZV)
      IF(NAUS.LT.3) GOTO 136
      DO 132 M=1,NZV
132   WRITE(NOUT,*) ' P ',(P(N,M),N=1,NZAV)
136   I = 0
      DO 93   M = 1,NZAV
      DO 93   N = M,NZAV
      I = I + 1
   93 SS(I) = 2.*(S(I) + P(N,NZV)*P(M,NZV))
      IF(NAUS.GT.2) WRITE(NOUT,*)' SS',(SS(IH),IH=1,(NZAV*(NZAV+1))/2)
     *              ,' FF',FF
      CALL MAT(FF,SS)
      I0 = NDIM * LPARR + KPARR
      DO 100 M=1,NSH
c      I2 = NSH2(M)
c      CC= SH(M) * WERTT(I2)
      CC = SH(M) * W
      IF (CC.EQ.0.) GOTO 100
      I1 = NSH1(M) + I0
      IF (NB) 110,110,111
C       DMM(.,.,2) ENTHAELT ALLE BEITRAEGE
C       DMM(.,.,1) ENTHAELT NUR BEITRAEGE FUER NB=0
 110  DMM(I1,1) = DMM(I1,1) + CC
 111  DMM(I1,2) = DMM(I1,2) + CC
      IF(NAUS.GT.1) WRITE(NOUT,120) M,I1,I2,SH(M),CC,DMM(I1,1),DMM(I1,2)
  120   FORMAT(' M,I1,I2,SH,CC,DMM1,2 ',3I5,E14.6,3E20.14)
  100 CONTINUE
  170 CONTINUE
      RETURN
      END
c--------------------------------------------------------------------
      SUBROUTINE MAT(H,S)
      IMPLICIT REAL*8 (A-H,O-Z)
C
C
      INCLUDE 'par.h'
C
      PARAMETER (NZAVMA=2*NZCMAX-1, NZVMAX=NZTMAX-1,
     *           NDIM4=NZPOMA*NZLWMA, NZLRH=(2*NZCMAX-3)*(NZCMAX-1))
C
      COMMON /LUPI/ KVK(NDIM1), jqq,
     *    MVK(NZIQMA,NDIM1), IZ, JZ,
     *    KZAHL, LUPAUS, IQM, KZHX(0:NZIQMA)

      COMMON /LUPR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)

      COMMON /SKLAVE/ QO(NZVMAX,NZVMAX), VR(NZTMAX,NZCMAX-1)

      COMMON VZ(NZVMAX,NZVMAX), WERT(NDIM4,NDIM4)
C
      COMMON /SCH/ WERT2(NDIM4,NDIM4)
C
      COMMON /ww/w
C
      DIMENSION S(NZAVMA*(NZAVMA+1)+2)
      DIMENSION WERTT(900),F(NDIM1)
      EQUIVALENCE (WERT,WERTT)
C--------------------- loop seriell ---------------------------------
c      DO 4   LR = 1,JZ
c      DO 4   LL = 1,IZ
c    4 WERT(LL,LR) = .0
c      IF(LUPAUS.GT.3) WRITE(NOUT,*) ' S ',(S(N),N=1,20),' H ',H
c      N=1
c      IF(KZAHL.EQ.0) GOTO 1
c      IF(ABS(H).LT.1.D-15)  GOTO 1
c      DO 20 K=1,KZAHL
c20    F(K)=EPO(K)*H
c      IF(KZHX(0).GT.KZHX(1)) GOTO 41
c      DO 40 M=1,IQM
c      DO 30 K=KZHX(M),KZAHL
c30    F(K)=F(K)*S(MVK(K,M))
c40    CONTINUE
c41    DO 50 K=1,KZAHL
c      WERTT(KVK(K)) = WERTT(KVK(K)) + F(K)
c      IF(LUPAUS.LT.3) GOTO 50
c      IF(ABS(F(K)).LT.1.D-15) GOTO 50
c      WRITE(NOUT,100) KVK(K),
c     ,K,KZAHL,WERTT(KVK(K)),EPO(K),F(K)
c50    CONTINUE
C--------------------------------------------------------------------
c
C------------------------ loop parallel -----------------------------
c----------- MVK umdrehen wenn aktiv! -------------------------------
      N=1
      IF(KZAHL.EQ.0) GOTO 1
      IF(ABS(H).LT.1.E-15)  GOTO 1
      w=0.
            IF(KZHX(0).GT.KZHX(1)) GOTO 41
      goto (21,22,23,24,25,26,27,28,29,30),jqq
21    DO 121 K=1,KZAHL
121    W=W+H*EPO(K)*S(MVK(1,K))
      goto 40
22    DO 122 K=1,KZAHL
122    W=W+H*EPO(K)*S(MVK(1,K))*S(MVK(2,K))
      goto 40
23    DO 123 K=1,KZAHL
123    W=W+H*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))
      goto 40
24    DO 124 K=1,KZAHL
124    W=W+H*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
      goto 40
25    DO 125 K=1,KZAHL
125    W=W+H*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
     + *S(MVK(5,K))
      goto 40
26    DO 126 K=1,KZAHL
126    W=W+H*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
     + *S(MVK(5,K))*S(MVK(6,K))
      goto 40
27    DO 127 K=1,KZAHL
127    W=W+H*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
     + *S(MVK(5,K))*S(MVK(6,K))*S(MVK(7,K))
      goto 40
28    DO 128 K=1,KZAHL
128    W=W+H*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
     + *S(MVK(5,K))*S(MVK(6,K))*S(MVK(7,K))*S(MVK(8,K))
      goto 40
29    DO 129 K=1,KZAHL
129    W=W+H*EPO(K)*S(MVK(1,K))*S(MVK(2,K))*S(MVK(3,K))*S(MVK(4,K))
     + *S(MVK(5,K))*S(MVK(6,K))*S(MVK(7,K))*S(MVK(8,K))*S(MVK(9,K))
      goto 40
30    stop 'jq zu gross'
41    DO 141 K=1,KZAHL
141    W=W+H*EPO(K)
40    CONTINUE
c
C-------------------------- ende loop parallel ----------------------
100   FORMAT(' KVK,K,K2,W,E,F',3I4,E17.10,2E13.6)
    1 CONTINUE
      RETURN
      END
c--------------------------------------------------------------------
      SUBROUTINE ORDSRT(KOBEH)
      IMPLICIT REAL*8 (A-H,O-Z)
C     ORDSRT SORTIERT DEN GESAMTEN COMMONBLOCK LUP NACH AUFSTEIGENDER
C     ANZAHL VON SIGMAFAKTOREN
      INCLUDE 'par.h'
C
      PARAMETER (NZAVMA=2*NZCMAX-1, NDIM4=NZPOMA*NZLWMA,
     *           NZLRH=(2*NZCMAX-3)*(NZCMAX-1))
C
      COMMON /LUPI/ KVK(NDIM1), jqq,
     *    MVK(NZIQMA,NDIM1), IZ, JZ,
     *    KZAHL, LUPAUS, IQM, KZHX(0:NZIQMA)

      COMMON /LUPR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)

      COMMON /UPROG/ JQ(NDIM1), INDPO(NDIM1)
C
C    KZHX(IQ) GIBT DIE NIEDRIGSTE ELEMENT NUMMER AN MIT MINDESTENS
C                 IQ FAKTOREN IM RECORD
C
      KUNTH=1
      KZAHL=KOBEH
      IQMH=IQM
      DO 10 IQ=0,IQMH
10    KZHX(IQ)=0
C
      DO 100, IQ=0, IQMH
C        LOOP ANZAHL SIGMAFAKTOREN
C
         KZHX(IQ)=KUNTH
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
      IF(LUPAUS.GT.1) WRITE(NOUT,999) (KZHX(IH),IH=0,IQMH)
999   FORMAT(' KZHX ',20I4)
      RETURN
      END
