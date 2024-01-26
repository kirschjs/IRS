      PROGRAM QUELMAnoo
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C                                                                          
C      PROGRAMM QUAFORM                                                 
C      FUER ELEKTROMAGNETISCHE UEBERGAENGE
C
C      INTEGERS IM FORMAT 24I3, REALS IM FORMAT 6E12.4                  
C                                                                       
C                                                                       
      INCLUDE 'par/jquelma'
C      
      PARAMETER (NZAVMA=2*(NZCMAX-1), NZVMAX=NZTMAX-1,
     *           NDIM4=NZPOMA*NZLWMA)
C
      COMMON /LUP/ KVK(NDIM1), JQ(NDIM1), INDPO(NDIM1),
     *             MVK(NZIQMA,NDIM1), EPO(NDIM1), 
     *             IZ, JZ, KZHV(2,2*NZAVMA), LUPAUS, IQM,
c    *             IZ, JZ, KZHV(2,2*NZAVMA), LUPAUS, IQM(2*NZAVMA),
     *             WERTL(NDIM4,NDIM4), KZHX(NZIQMA,2*NZAVMA)
C
      COMMON /LUR/ IENT(NZOPLU+1),NBAND3,NBAND7
      COMMON /BIG/ DM(NDIM,NDIM)
      COMMON /BEG/ IZLURE
      COMMON /CSH/ NSH1(NDIM5), NSH2(NDIM5), SH(NDIM5), NSH
C
      COMMON  MKC, NZV, QO(NZVMAX,NZVMAX), QOR(NZVMAX,NZVMAX),
     *        VZ(NZVMAX,NZVMAX), WERT(NDIM4,NDIM4), LPARR, MC1,
     *        MD1, MFL, NCC, NREB, NZAV, NZV1, PHI2,
     *        RPAR(NZPARM,MZPARM,NZFMAX),VR(NZTMAX,NZCMAX),
     *        WR(NZTMAX),KPAR
C
      DIMENSION COF(NZPARM,NZRHOM,NZFMAX), NREG(NZOPOB),
     *             RVEC(NZTMAX,NZTMAX), SVEC(NZTMAX,NZTMAX),
     *    NUM(4,NZRHOM,NZFMAX), VW(NZTMAX,NZCMAX)
      DIMENSION NZC(NZFMAX), MZG(NZFMAX), NOL(NZFMAX),
     *          NGRV(2,NZCMAX,NZFMAX), NTE(NZOPOB),
     *             CPAR(NZAVMA,NZPARM,NZFMAX),QFCL(NZVMAX,2*NZCMAX-1),
     *          NZPAR(NZFMAX), MZPAR(NZFMAX), MS(MZGMAX,NZFMAX),
     *          POLL(NZTMAX,NZTMAX), PORR(NZTMAX,NZTMAX)

      DIMENSION  QFCR(NZVMAX,NZVMAX,2*NZCMAX-1),
     *          POR(NZVMAX,NZVMAX),POL(NZVMAX,NZVMAX)

      DIMENSION LREG(NZOPER), KOM(NZOPER,NZFMAX,NZFMAX)
      DIMENSION MMASSE(2,MZGMAX,NZFMAX), MLAD(2,MZGMAX,NZFMAX),
     *          MSS(2,MZGMAX,NZFMAX)
C
      DIMENSION NZLW(NZFMAX), LW(2*NZCMAX-3,NZLWMA,NZFMAX),
     *          LZWERT(5,NZLWMA,NZFMAX), NZRHO(NZFMAX),
     *          NZC1(NZFMAX), NZPO(NZFMAX), IVEK(NZFMAX),
     *          KP(NZCMAX-1,NZPOMA,NZFMAX), LC(NZTMAX,NPDC,NZOPOB)
       DIMENSION LREG1(NZOPER), NZRHO1(NZFMAX), ITPO(NZFMAX),
     *          NT1(NPDC,NZOPOB), U(MZGMAX,MZGMAX,NPDC,NZOPOB)
C
      DIMENSION LC1(NZTMAX)
      OPEN(UNIT=3,STATUS='SCRATCH',FORM='UNFORMATTED')
      OPEN(UNIT=4,STATUS='SCRATCH',FORM='UNFORMATTED')
      OPEN(UNIT=13,STATUS='SCRATCH',FORM='UNFORMATTED')
      OPEN(UNIT=5,FILE='INQUA',STATUS='OLD')
      OPEN(UNIT=6,FILE='OUTPUT')
      OPEN(UNIT=8,FILE='OBOUT',STATUS='OLD',FORM='UNFORMATTED')
      OPEN(UNIT=9,FILE='LUOUT',STATUS='OLD',FORM='UNFORMATTED')
      OPEN(UNIT=10,FILE='QUAOUT',STATUS='UNKNOWN',FORM='UNFORMATTED')
      OPEN(UNIT=12,FILE='QUAALT',STATUS='UNKNOWN',FORM='UNFORMATTED')
C
      INPUT=5
      PHI=3.1415926536                                                  
      PHI2 =  SQRT(PHI)                                                 
      READ (INPUT,1002) NBAND1,NBAND2,NBAND3,idun,NBAND5,idum
     1 ,NAUS,MOBAUS,LUPAUS
      NBAND7=4
      READ(INPUT,1002)(LREG(K),K=1,NZOPER)
C     EINLESEN DER FUNKTIONSEIGENSCHAFTEN UND PARAMETER                 
      REWIND  NBAND2                                                    
      READ(NBAND2) NZF,NZT,NZV,(NREG(K),K=1,NZOPOB)
      IF(NBAND5.LE.0) GOTO 90
      DO 92 KL=1,NZF                                                    
      DO 92 KR=1,KL                                                     
   92 READ(INPUT,1002) (KOM(L,KL,KR),L=1,8)
      REWIND NBAND5                                                     
      GO TO 93                                                          
   90 CONTINUE                                                          
      DO 94 KL=1,NZF                                                    
      DO 94 KR=1,KL                                                     
      DO 94 L=1,NZOPER
   94 KOM(L,KL,KR)=0                                                    
   93 CONTINUE                                                          
      NZV1 = NZV - 1                                                    
      DO 20  K = 1,NZF                                                  
      READ (NBAND2) NZC(K),MZG(K),NOL(K)                                
      M = MZG(K)                                                        
      IF(M.LE.MZGMAX) GOTO 6002
      N1=1                                                              
      N2=M                                                              
      N3=MZGMAX
      GO TO 6040                                                        
6002  NOL(K)=NOL(K)+1                                                   
      READ (NBAND2) ((MMASSE(N,L,K),MLAD(N,L,K),MSS(N,L,K),N=1,2),      
     1     MS(L,K),L=1, M)                                              
      M = NZC(K)                                                        
      READ (NBAND2) ((NGRV(N,L,K),L=1,M),N=1,2)                         
   20 CONTINUE                                                          
      REWIND NBAND3                                                     
      READ(NBAND3) NZF1,MUL,(NZLW(K),NZC1(K),K=1,NZF1),
     1(IENT(K),K=1,NZOPLU), (NZPO(MH),MH=1,NZF1)
      IF(NZF.NE.NZF1) STOP 251
      DO 252 K=1,NZF
      ITPO(K)=0
      IF(NZPO(K).NE.0) ITPO(K)=2
      NZPO(K)=MAX0(1,NZPO(K))
      IF(NZC(K).NE.NZC1(K)) STOP 252
  252 CONTINUE                                                          
      IF(LREG(1).LE.0) GOTO 253
      IF(NREG(1)*IENT(1).LE.0) STOP 253
253   IF(LREG(2).LE.0) GOTO 254
      IF(NREG(2)*IENT(1).LE.0) STOP 254
254   IF(LREG(3).LE.0) GOTO 255
      IF(NREG(3)*IENT(2).LE.0) STOP 255
255   IF(LREG(4).LE.0) GOTO 256
      IF(NREG(4)*IENT(2).LE.0) STOP 256
256   IF(LREG(5).LE.0) GOTO 257
      IF(NREG(1)*IENT(3).LE.0) STOP 257
257   IF(LREG(6).LE.0) GOTO 258
      IF(NREG(2)*IENT(3).LE.0) STOP 258
258   IF(LREG(7).LE.0) GOTO 259
      IF(NREG(3)*IENT(4).LE.0) STOP 259
259   IF(LREG(8).LE.0) GOTO 260
      IF(NREG(4)*IENT(4).LE.0) STOP 260
260   IF(LREG(9).LE.0) GOTO 261
      IF(NREG(1)*IENT(2).LE.0) STOP 261
261   IF(LREG(10).LE.0) GOTO 262
      IF(NREG(2)*IENT(2).LE.0) STOP 262
262   CONTINUE
C     CHECK LUELMA OBELMA ZUENDE
C     KONSTRUKTION DER BAHNDREHIMPULSE
      DO 264 K=1,NZF                                                    
      M1=2*NZC(K)-3                                                     
      M2=NZLW(K)                                                        
      IF(M2.LE.NZLWMA) GOTO 6000
      N1=2                                                              
      N2=M2                                                             
      N3=NZLWMA                                                         
      GO TO 6040                                                        
 6000 CONTINUE                                                          
      M3=NZC(K)-1
      MPO=NZPO(K)
      READ(NBAND3) ((LW(M,L,K),M=1,M1),L=1,M2),((KP(M,KH,K),M=1,M3),
     1 KH=1,MPO)
      DO 265 L=1,M2                                                     
      DO 266 M=1,3                                                      
  266 LZWERT(M,L,K)=0                                                   
      LZWERT(4,L,K)=LW(M3,L,K)                                          
      LZWERT(5,L,K)=LW(M1,L,K)                                          
      IF(M3-2) 265,267,268                                              
  267 LZWERT(3,L,K)=LW(1,L,K)                                           
      IF(NOL(K).GE.NZC(K)) GOTO 271
          LZWERT(2,L,K)=LW(1,L,K)
      GO TO 265                                                         
  271 LZWERT(1,L,K)=LW(1,L,K)                                           
      GO TO 265                                                         
  268 IF(NOL(K).LT.NZC(K)) GOTO 272
      LZWERT(3,L,K)=LW(4,L,K)                                           
      LZWERT(1,L,K)=LZWERT(3,L,K)                                       
      GO TO 265                                                         
  272 IF(NOL(K).GT.2) GOTO 275
      LZWERT(2,L,K)=LW(4,L,K)                                           
      LZWERT(3,L,K)=LZWERT(2,L,K)
      GO TO 265                                                         
  275 LZWERT(3,L,K)=LW(4,L,K)                                           
      LZWERT(1,L,K)=LW(1,L,K)                                           
      LZWERT(2,L,K)=LW(2,L,K)                                           
  265 CONTINUE                                                          
  264 CONTINUE                                                          
C      BAHNDREHIMPULSE ENDE
C      KONSTRUKTION DER BASISVEKTOREN
      I=0                                                               
      DO 22  K = 1,NZF                                                  
      NZPAR(K) = 0                                                      
      MZPAR(K) = 0                                                      
       READ(INPUT,1002) NZRHO(K)
      KK=NZRHO(K)                                                       
      IF(KK.LT.NZRHOM) GOTO 388
      N1=8
      N2=KK
      N3=NZRHOM
      GOTO 6040
388   IVEK(K)=I
      IF(KK.EQ.0) GOTO 22
      M = 2*NZC(K) - 2                                                  
      READ (INPUT,1002)  NZPAR(K) ,MZPAR(K)                             
 1002 FORMAT(20I3)                                                      
      LM = NZPAR(K)                                                     
      KM=MZPAR(K)
      IF(LM.LE.NZPARM) GOTO 390
      N1=7
      N2=LM
      N3=NZPARM
      GOTO 6040
c
390   DO 24  L = 1,LM                                                   
c
      READ(INPUT,1003)  (CPAR(N,L,K),N=1,M)                            
c
      READ(INPUT,1003)  (RPAR(L,N,K),N=1,KM)
c
   24 CONTINUE                          
c 
      DO 240 N=1,KK                                                     
      READ(INPUT,1002) NUM(1,N,K),NUM(2,N,K),NUM(4,N,K)
      NUM(4,N,K)=MAX(1,NUM(4,N,K))
      NUM(3,N,K)=IVEK(K)+N                                              
      READ(INPUT,'(6E12.4)') (COF(L,N,K),L=1,LM)                              

 1003 FORMAT(6E12.4)                                                    
 1004 FORMAT(/I3,25H TER SATZ INNERER WEITEN  /8F12.6)           
 1005 FORMAT(/22H SATZ RADIALPARAMETER   /3(8F12.6/))
 1050 FORMAT(//15H DEFINITION DER,I3,22H TEN BASISFUNKTION IST/         
     1 I3,13H TE ZERLEGUNG,I3,23H TE SPINISOSPINFUNKTION/I3,
     2 26H TE BAHNDREHIMPULSFUNKTION,I3,' TES POLYNOM',/,
     3 ' SUMMATION UEBER INNERE WEITEN MIT KOEFF',/,(1P20E12.4))
  240 CONTINUE                                                          
      I=I+KK                                                            
   22 CONTINUE                                             
C     ENDE DEFINITION BASISVEKTOREN
      REWIND NBAND1                                                     
      IF(NBAND5.LE.0) GOTO 800
      READ(NBAND5) NZF1,(LREG1(K),K=1,NZOPER),I1,
     1 (NZRHO1(K),K=1,NZF1)                                             
      MFEH1 = 1                                                         
      MFEH2 = NZF                                                       
      MFEH3 = NZF1                                                      
      IF(NZF.LT.NZF1) GOTO 890
C ECCE: 9 = NZOPER - 2
      DO 804 K=1,9
      MFEH1 = 3+ K                                                      
      MFEH2 = LREG(K)                                                   
      MFEH3 = LREG1(K)                                                  
      IF(LREG(K)-LREG1(K))890,804,806                                   
  806 DO 807 ML=1,NZF1                                                  
      DO 807 MR=ML,NZF1                                                 
  807 KOM(K,ML,MR)=0                                                    
  804 CONTINUE                                                          
      DO 805 K=1,NZF1                                                   
      MFEH1=11 + K                                                      
      MFEH2=NZRHO(K)                                                    
      MFEH3 = NZRHO1(K)                                                 
  805 CONTINUE                                                          
      DO 6020 ML=1,NZF                                                  
      DO 6020 MR=1,ML                                                   
      IF(ML.LE.NZF1) GOTO 6020
      DO 6022 MC=1,8
 6022 KOM(MC,ML,MR)=0
 6020 CONTINUE                                                          
      GO TO 800                                                         
  890 STOP 890                                                              
  800 CONTINUE                                                          
      WRITE(NBAND1) NZF,MUL,(LREG(K),K=1,NZOPER),I,(NZRHO(K),K=1,NZF)
      DO 950  K = 1,NZF                                                 
      M=MZG(K)                                                          
      N3=MZPAR(K)
      MC1=NZC(K)-1
      KK=NZRHO(K)
      IF(KK.LE.0) GOTO 950
      DO 4536 N=1,KK                                                    
      N1=NUM(1,N,K)                                                     
      N2=NUM(2,N,K)
       N4=NUM(4,N,K)
c RPAR(1,L,K) is written instead of RPAR(N,L,K), i.e., each of the NZRHO vectors are
c assumed to have identical widths; they differ in the order of the polynomial!       
      WRITE(NBAND1) N3,MMASSE(1,N1,K),MMASSE(2,N1,K),MLAD(1,N1,K),
     1 MLAD(2,N1,K),MSS(1,N1,K),MSS(2,N1,K),MS(N1,K),
     2 (LZWERT(L,N2,K),L=1,5),(RPAR(1,L,K),L=1,N3),KP(MC1,N4,K)
 4536 CONTINUE                                                          
      IF(NBAND5.LE.0) GOTO 810
       IF(K.GT.NZF1)GOTO 810
       KK1=NZRHO1(K)
      DO 813 N=1,KK1                                                    
813    READ(NBAND5)
810   CONTINUE                                                          
  950 CONTINUE                                                          
C     BESTIMMUNG DER ORTSMATRIXELEMENTE VOR EINSETZEN DER DREHIMPULSE   
      DO 40  MFL = 1,NZF                                                
      IZLW=NZLW(MFL)
      IZPW=NZPO(MFL)
      IRHO=NZRHO(MFL)                                                   
      IK1=MZPAR(MFL)
      READ(NBAND2)   ((RVEC(M,N),M=1,NZV),N=1,NZT)                      
      READ(NBAND2)   ((SVEC(N,M),M=1,NZV),N=1,NZT)                      
      MC = NZC(MFL)                                                     
      MC1= MC - 1                                                       
      MC2 = MC + MC1                                                    
      MC3 = MC2 - 1                                                     
      NC = NZPAR(MFL)                                                   
      NCC = MZPAR(MFL)                                                  
      READ(NBAND2) ((QFCL(M,K),M=1,NZV),K=1,MC2)
      DO 40 MFR=1,MFL
      REWIND NBAND7
      JZLW=NZLW(MFR)
       JZPW=NZPO(MFR)
      JRHO=NZRHO(MFR)                                                   
      JK1=MZPAR(MFR)
      NLUDZ=0                                                           
3009  FORMAT(////19H ZWISCHEN ZERLEGUNG,I3,15H UND ZERLEGUNG ,I3,
     124H WIRD ZUR ZEIT GERECHNET )                                     
      MD=NZC(MFR)                                                       
      MD1= MD - 1                                                       
      READ(NBAND2)    ((VW(M,K),M=1,NZT),K=1,MD1)                       
      MD2 = MD + MD1                                                    
      MD3 = MD2 - 1                                                     
      ND = NZPAR(MFR)                                                   
      NDD = MZPAR(MFR)                                                  
      DO 42 MKC =1,NZOPER
c     for a certain MFL-MFR block, the matrices are written on operator after the other
c     => after one operator has been calculated, the dimension counter is reset 
      NREB = 0
      IZ=(IZPW-1)*NZLWMA+IZLW
      JZ=(JZPW-1)*NZLWMA+JZLW
      NZAV=MC1+MD1+1
      IZLURE=1
      GOTO(202,201,201,202,202,201,201,202,202,202,202),MKC
201   NREB=1
      IZLURE=MC1*(1+ITPO(MFL))+MD1*(1+ITPO(MFR))+1
202   CONTINUE
3008  CONTINUE      
      NL = MZG(MFL)
      NR = MZG(MFR)
      IY=MOD(MKC-2,4)+2
      if(MKC.eq.1) IY = 1
C     IY BESTIMMT DEN OBEROPERATOR ZU MKC
C
      IF(NREG(IY).LE.0) GOTO 440
      IF(MKC.GT.5) GOTO 82
      READ(NBAND2)    MZQQ,NTE(IY),NTE(IY)
82    CONTINUE
      IF(NTE(IY).LE.0 .OR. mkc.gt.5) GOTO 440
      DO 922   MTE = 1,NTE(IY)
      READ(NBAND2)    NT1(MTE,IY),(LC(K,MTE,IY),K=1,NZT),
     1  ((U(NFL,NFR,MTE,IY),NFL=  1,NL),NFR=1,NR)
922   CONTINUE
C
C       READ LUELMA ELEMENTE
440   IX=MOD(MKC-1,2)
      if(mkc.eq.1) IX = 1
      IF(IX.EQ.0) GOTO 86
      CALL LURE(NLUDZ,ITV2)
86    IF(KOM(MKC,MFL,MFR).EQ.0) GOTO 95
      IF(MFL.NE.MFR)GOTO 97
      NNN= NZRHO1(MFL)*(NZRHO1(MFL)+1)/2                                
      GO TO 99                                                          
97    NNN=NZRHO1(MFL)*NZRHO1(MFR)                                       
   99 CONTINUE                                                          
      IF(LREG(MKC)*NNN.LE.0) GOTO 42
      DO 540 N=1,NNN                                                    
      READ (NBAND5) NUML,NUMR,II1,II2,((DM(K,L),K=1,II1),L=1,II2)
      IF(KOM(MKC,MFL,MFR).LE.0) GOTO 540
      WRITE (NBAND1) NUML,NUMR,II1,II2,((DM(K,L),K=1,II1),L=1,II2)
  540 CONTINUE                                                          
      IF(KOM(MKC,MFL,MFR).GT.0) GOTO 42
95    IF(IRHO*JRHO.LE.0) GOTO 42
      IF(LREG(MKC).LE.0) GOTO 42

      DM=.0 

      DO 490 M=1,NZT                                                    
  490 LC1(M)=0                                                          
      IF (NTE(IY)*ITV2.LE.0) GOTO 900
      DO 44 MTE=1,NTE(IY)
1491  MM=0                                                              
      DO 491 M=1,NZT                                                    
  491 MM=MM+IABS(LC(M,MTE,IY)-LC1(M))
      IF(MM.LE.0) GOTO 492
      DO 494 M=1,NZT                                                    
  494 LC1(M)=LC(M,MTE,IY) 
      DO 470 M=1,NZT                                                    
      MM=LC(M,MTE,IY) 
  470 VR(MM,1)=VW(M,MD1)                                                
   
      QFCR = .0
     
      VR = .0                                                      
      DO 290  K = 1,MD                                                  
      I1 = NGRV(1,K,MFR)                                                
      I2 = NGRV(2,K,MFR) + I1 - 1                                       
      IF(NGRV(2,K,MFR).LE.1)GOTO 290
      FZZ = 1./ FLOAT(NGRV(2,K,MFR))                                    
      I3=I2-1                                                           
      DO 52  L1 = I1,I3                                                 
      I4=L1+1                                                           
      DO 52  L2 =I4,I2                                                  
      L3 = LC(L1,MTE,IY)
      L4 = LC(L2,MTE,IY) 
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
      MM = LC(M,MTE,IY)
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
54    WR(M)=RVEC(M,NT1(MTE,IY))
      DO 60   KPAR = 1,NC                                               
      DO 62   M = 1,NZV                                                 
      DO 62   N = 1,M                                                   
   62 POL (M,N) = .0                                                    
      DO 64   M = 1,NZV                                                 
      IF(MC3.LE.0) GOTO 64
      DO 63   K = 1,MC3                                                 
63    POL (M,M)=POL (M,M) + CPAR(K,KPAR,MFL)*QFCL(M,K)
   64 CONTINUE                                                          
       DO 70 LPAR=1,ND                                                  
      I=0                                                               
      DO 101 M=1,IRHO
      KSL = NUM(1,M,MFL)
      KLL = NUM(2,M,MFL)
      KPL=NUM(4,M,MFL)
      JPL=(KPL-1)*NZLWMA+KLL
      NUML = NUM(3,M,MFL) 
      TS = COF(KPAR,M,MFL)
      IF (ABS(TS).LT.1.E-10) GOTO 101                                   
      MADL = (M-1) * IK1                                                
      DO 105 N=1,JRHO
      KSR = NUM(1,N,MFR) 
      KLR = NUM(2,N,MFR)
      KPR=NUM(4,N,MFR)
      JPR=(KPR-1)*NZLWMA+KLR
      NUMR = NUM(3,N,MFR)  
      IF (NUML.LT.NUMR) GOTO 105
      IF(WERTL(JPL,JPR).NE.1.) GOTO 105
      A = TS * U(KSL,KSR,MTE,IY)*COF(LPAR,N,MFR)
      IF(ABS(A).LT.1.E-20) GOTO 105
      MADR = (N-1) * JK1                                                
      IF (I.LT.NDIM5) GOTO 110                                          
 111  STOP 111                                                              
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
      DO 67 M=1,NZV                                                     
      DO 67  N = M,NZV                                                  
67    POLL(N,M)=POL(N,M)
      DO 76 LPARR=1,NDD                                                 

      QOR =0.0
      QO  =0.0
      DO 77 M=1,NZV                                                     
      DO 77 N=M,NZV                                                     
      PORR(N,M)=POR(N,M)+RPAR(LPAR,LPARR,MFR)*QFCR(N,M,MD2)
      QOR(N,M)=PORR(N,M)
77    QO(N,M)=POLL(N,M)+PORR(N,M)
      CALL BEGRI(NAUS)
   76 CONTINUE                                                          
C      LOOP RADIALWEITEN RECHTS
   70 CONTINUE                                                          
C     LOOP INNNERE WEITEN RECHTS
   60 CONTINUE                                                          
C      LOOP INNERE WEITEN LINKS
   44 CONTINUE                                                          
C       LOOP OBELMA
  900 CONTINUE                                                          
      DO 480 M=1,IRHO
      NUML=NUM(3,M,MFL)
      DO 481 N=1,JRHO
      NUMR=NUM(3,N,MFR)
c      IF(NUML.LT.NUMR) GOTO 481
      M1=(M-1)*IK1+1                                                    
      M2=M*IK1                                                          
      N1=(N-1)*JK1+1                                                    
      N2=N*JK1                                                          
      II1 = 1                                                           
      A = 0.                                                            
      DO 510 L=N1,N2 
      DO 510 K=M1,M2
 510  A = A + ABS(DM(K,L))
c      IF(A.GT.0.) GOTO 512
c      WRITE (NBAND1) NUML,NUMR,II1,II1,A,A                              
c      GOTO 481                                                          
512   WRITE(NBAND1) NUML,NUMR,IK1,JK1,((DM(K,L),K=M1,M2),L=N1,N2)
  481 CONTINUE                                                          

1021  FORMAT(1X,10E12.5)
  480 CONTINUE                                                          
C      LOOP BASISVECTOR
   42 CONTINUE                                                          
C        LOOP OPERATOREN
   40 CONTINUE                                                          
C         LOOP ZERLEGUNGEN
3010  FORMAT(//18H ENDE DER RECHNUNG )                                  
      goto 666                                                              
 6040 stop 6040                                         
 6100 FORMAT(/31H ES WIRD VOELLIG NEU GERECHNET /)                      
 6034 FORMAT(34H FUER DIESEN OPERATOR WIRD ERSETZT /)                   
 6035 FORMAT(40H FUER DIESEN OPERATOR WIRD NEU GERECHNET /)             
 6036 FORMAT(34H FUER DIESEN OPERATOR WIRD KOPIERT /)                   
666   continue
      END                                                               
      SUBROUTINE LURE(NLUDZ,ITV2)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'par/jquelma'
C      
      PARAMETER (NZAVMA=2*(NZCMAX-1), NZVMAX=NZTMAX-1,
     *           NDIM4=NZPOMA*NZLWMA)
C
C
      COMMON /LUP/ KVK(NDIM1), JQ(NDIM1), INDPO(NDIM1),
     *             MVK(NZIQMA,NDIM1), EPO(NDIM1), 
     *             IZ, JZ, KZHV(2,2*NZAVMA), LUPAUS, IQM,
c    *             IZ, JZ, KZHV(2,2*NZAVMA), LUPAUS, IQM(2*NZAVMA),
     *             WERTL(NDIM4,NDIM4), KZHX(NZIQMA,2*NZAVMA)
C
      COMMON /CSH/ NSH1(NDIM5), NSH2(NDIM5), SH(NDIM5), NSH
      COMMON MKC
      COMMON /BEG/ IZLURE
      COMMON /LUR/ IENT(NZOPLU+1),NBAND3,NBAND7
      IF(MKC.EQ.4.OR.MKC.EQ.10) REWIND NBAND7
      
      WERTL=0.
      kzhv(2,1)=0
      GOTO (442,443,500,442,500,443,500,442,500,442,500),MKC
442   NLUDZ=NLUDZ+1
      IENT(6)=IENT(3)
      IF(IENT(NLUDZ).EQ.0) GOTO 444
      IF(MKC.LE.9) GOTO 20
      READ(NBAND7) K1ZAHL,IQM
      GOTO 21
20    READ(NBAND3) K1ZAHL,IQM
      IF(MKC.EQ.4) WRITE(NBAND7) K1ZAHL,IQM
21    IQM=MAX0(IQM,1)
      IF(IQM.GT.NZIQMA) STOP 'NZIQMA ZU KLEIN'
      KZHV(1,1)=0                                                       
       KZHV(2,1)=K1ZAHL                                                 
      ITV2=K1ZAHL
      IF(K1ZAHL.GT.NDIM1)GOTO 6040
      IF(K1ZAHL.EQ.0) GOTO 444
      IF(MKC.LE.9) GOTO 22
      READ(NBAND7) (JQ(KW),INDPO(KW),EPO(KW),(MVK(IW,KW),IW=1,IQM),
     $KW=1,K1ZAHL)
      GOTO 23
22    READ(NBAND3) (JQ(KW),INDPO(KW),EPO(KW),(MVK(IW,KW),IW=1,IQM),
     1 KW=1,K1ZAHL)
      IF(MKC.EQ.4) WRITE(NBAND7)(JQ(KW),INDPO(KW),EPO(KW),
     $(MVK(IW,KW),IW=1,IQM),KW=1,K1ZAHL)
23    CONTINUE
      CALL FAPOR(1,K1ZAHL)
      GO TO 444                                                         
  443 NLUDZ=NLUDZ+1                                                     
      IF(IENT(NLUDZ).EQ.0) GOTO 444
      IIZ=0
      KZAHL=0
      DO 449 MIZ=1,IZLURE
      IIZ=IIZ+1                                                         
      READ(NBAND3) K2ZAHL,IQM                                           
      IQM=MAX0(IQM,1)                                                   
      IF(IQM.GT.NZIQMA) STOP 'NZIQMA ZU KLEIN'
      KZHV(1,IIZ)=KZAHL
      KZHV(2,IIZ)=K2ZAHL
      K3ZAHL=KZHV(1,IIZ)+1
       KZAHL=K2ZAHL+K3ZAHL-1
      ITV2=KZAHL
      IF(KZAHL.GT.NDIM1) GOTO 6040
      IF(K2ZAHL.EQ.0) GOTO 449
      READ(NBAND3) (JQ(KW),INDPO(KW),EPO(KW),(MVK(IW,KW),IW=1,IQM),
     1 KW=K3ZAHL,KZAHL)
      CALL FAPOR(K3ZAHL,KZAHL)
  449 CONTINUE
444   RETURN
6040   PRINT 10,K1ZAHL,KZAHL,NDIM1
10     FORMAT(' ZU VIELE LUELMAZAHLEN',3I10)
      STOP 444
500   CONTINUE
      STOP 500
       END
      SUBROUTINE FAPOR(IA,IE)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'par/jquelma'
C      
      PARAMETER (NZAVMA=2*(NZCMAX-1), NZVMAX=NZTMAX-1,
     *           NDIM4=NZPOMA*NZLWMA)
C
C
      COMMON /LUP/ KVK(NDIM1), JQ(NDIM1), INDPO(NDIM1),
     *             MVK(NZIQMA,NDIM1), EPO(NDIM1), 
     *             IZ, JZ, KZHV(2,2*NZAVMA), LUPAUS, IQM,
c    *             IZ, JZ, KZHV(2,2*NZAVMA), LUPAUS, IQM(2*NZAVMA),
     *             WERTL(NDIM4,NDIM4), KZHX(NZIQMA,2*NZAVMA)
C
      COMMON /CSH/ NSH1(NDIM5), NSH2(NDIM5), SH(NDIM5), NSH
      IF(LUPAUS.EQ.0) GOTO 100
      PRINT 10,IA,IE
10    FORMAT(' VON LUELMA ELEMENTE VON',I8,' BIS ',I10)
      IF(LUPAUS.LT.2) GOTO 100
      DO 15 I=IA,IE
15    PRINT 20,JQ(I),INDPO(I),EPO(I),(MVK(J,I),J=1,IQM)
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
      SUBROUTINE BEGRI(NAUS)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'par/jquelma'
C      
      PARAMETER (NZAVMA=2*(NZCMAX-1), NZVMAX=NZTMAX-1,
     *           NDIM4=NZPOMA*NZLWMA)
C
      COMMON /BEG/ IZLURE
      COMMON /BIG/ DM(NDIM,NDIM)
      COMMON /CSH/ NSH1(NDIM5), NSH2(NDIM5), SH(NDIM5), NSH
C
      COMMON  MKC, NZV, QO(NZVMAX,NZVMAX), QOR(NZVMAX,NZVMAX),
     *        VZ(NZVMAX,NZVMAX), WERT(NDIM4,NDIM4), LPARR, MC1,
     *        MD1, MFL, NCC, NREB, NZAV, NZV1, PHI2,
     *        RPAR(NZPARM,MZPARM,NZFMAX), VR(NZTMAX,NZCMAX),
     *        WR(NZTMAX),KPAR
C
      DIMENSION P(NZAVMA,NZVMAX),BETA(NZVMAX),S(NZCMAX*(NZAVMA+1))
      DIMENSION ZH(NZTMAX), ZW(NZTMAX), SS(NZCMAX*(NZAVMA+1))
      DIMENSION ZY(NZVMAX,NZAVMA+1), RHO(NZVMAX), Y(NZAVMA+1),
     *          GAM(NZAVMA), Z(NZVMAX,NZAVMA+1), H(3*(NZAVMA+1))
      DIMENSION HX(NZAVMA)
      DIMENSION HHH(3*(NZAVMA+1))
      DIMENSION WERTT(NDIM4*NDIM4), DMM(NDIM*NDIM)
      EQUIVALENCE (WERT(1,1),WERTT(1)), (DM(1,1),DMM(1))
      EQUIVALENCE(G,HHH(1)),(H(1),HHH(1))
      G = .0
      RHO = .0         
      P = 0.       
      ZY = 0.     
      
      
      INZ1=(NZAV*(NZAV+1))/2
      
      S = 0.
      IF(NREB.EQ.0) GOTO 82
      DO 84   N = 1,NZAV
      GAM(N) = .0          
      IF(N.LE.MC1) GOTO 84
      IF(N.EQ.NZAV) GOTO 84
      M = N - MC1
      DO 86   K = 1,NZV   
   86 GAM(N) = GAM(N) + VR(K,M)*WR(K)   
   84 CONTINUE                         
      DO 87   M = 1,NZV               
      DO 87   K = 1,NZV              
   87 RHO(M) = RHO(M) + WR(K)*(QOR(K,M)+QOR(M,K)) 
   82 CONTINUE                                   
      CALL HAUPT                                
      FFF=1                                    
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
      DO 4 N=1,MD1                    
      NN=MC1+N                       
      DO 4 M=1,NZV                  
      DO 6 K=1,M                   
    6 P(NN,M)=VR(K,N)*VZ(M,K)+P(NN,M) 
    4 P(NN,M)=P(NN,M)*BETA(M)        
      DO 8 M=1,NZV
      DO 9 K=1,M                    
    9 P(NZAV,M)=P(NZAV,M)+WR(K)*VZ(M,K)
    8 P(NZAV,M)=P(NZAV,M)*BETA(M)                                       
      DO 3  M = 1,NZAV                                                  
    3 ZH(M) = P(M,NZV)                                                  
      DO 5   M = 1,NZV                                                  
    5 ZW(M) = VZ(NZV,M)                                                 
      IF(NZV1.EQ.0) GOTO 14
      DO 11 M=1,NZV1
      DO 11 N=1,M
11    VZ(M,N)=VZ(M,N)*BETA(M)
      I=0                                                               
      DO 12 M=1,NZAV                                                    
      DO 12 N=M,NZAV                                                    
      I=I+1                                                             
      DO 12   K = 1,NZV1                                                
   12 S(I)=S(I)+P(N,K)*P(M,K)
   14 CONTINUE                                                          
      IF(NREB.EQ.0) GOTO 45
      IF (NZV1.EQ.0) GOTO 45
      DO 62   M = 1,NZV1
      DO 62   K = M,NZV1                                                
      DO 62    N = 1,NZAV                                               
   62 ZY(M,N) = ZY(M,N) + VZ(K,M)*P(N,K)                                
  45  CONTINUE                                                          
      DO 170  KPARR = 1,NCC                                             
      BETA(NZV) = 1./SQRT(QO(NZV,NZV)+RPAR(KPAR,KPARR,MFL))                 
      FFW = FFF *PHI2*BETA(NZV)                                         
      FF=FFW**3
      DO 91   M = 1,NZAV                                                
   91 P(M,NZV) = ZH(M)*BETA(NZV)                                        
      DO 92    M = 1,NZV                                                
   92 VZ(NZV,M) = ZW(M)*BETA(NZV)                                       
      I = 0                                                             
      DO 93   M = 1,NZAV                                                
      DO 93   N = M,NZAV                                                
      I = I + 1                                                         
   93 SS(I) = 2.*(S(I) + P(N,NZV)*P(M,NZV))
      I = (NZAV*(NZAV-1))/2                                             
      IF(NREB.GT.0) GOTO 19
      G = FF
      CALL MAT(HHH,SS,I)                                                
      GO TO 70                                                          
   19 DO 21 M=1,NZV                                                     
      DO 21 N=1,NZAV                                                    
      K = NZV                                                           
   21 Z(M,N)=ZY(M,N)+VZ(K,M)*P(N,K)                                     
      DO 24 N=1,NZAV                                                    
      Y(N) = .0                                                         
      DO 24 K=1,NZV                                                     
   24 Y(N) = Y(N) + RHO(K)*Z(K,N)                                       
      DO 44 N=1,NZAV
44    HX(N)=FF*(-0.5*Y(N)+GAM(N))
      II=0
      NZ=MC1+MD1
      DO 144 N=1,IZLURE
      NXH=(N-1)/NZ
      NH=N-NXH*NZ
      II=II+1
144   H(II)=HX(NH)
      H(IZLURE)=HX(NZAV)
      CALL MAT(HHH,SS,IZLURE)
   70 CONTINUE                                                          
      I0 = NDIM * LPARR + KPARR                                         
      DO 100 M=1,NSH                                                    
      I2 = NSH2(M)                                                      
      C = SH(M) * WERTT(I2)                                             
      IF (C.EQ.0.) GOTO 100                                             
      I1 = NSH1(M) + I0                                                 
      DMM(I1) = DMM(I1) + C
  100 CONTINUE                                                          
  170 CONTINUE                                                          
      RETURN                                                            
      END                                                               
      SUBROUTINE HAUPT                                                  
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'par/jquelma'
C      
      PARAMETER (NZAVMA=2*(NZCMAX-1), NZVMAX=NZTMAX-1,
     *           NDIM4=NZPOMA*NZLWMA)
C
C
      COMMON  MKC, NZV, QO(NZVMAX,NZVMAX), QOR(NZVMAX,NZVMAX),
     *        VZ(NZVMAX,NZVMAX)
C    HAUPT FUEHRT HAUPTACHSENTRANSFORMATION DURCH
C    TRANSFORMATION IN VZ, HAUPTACHSENWERTE IN QO
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
      SUBROUTINE MAT(HHH,S,NQ)                                          
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'par/jquelma'
C      
      PARAMETER (NZAVMA=2*(NZCMAX-1), NZVMAX=NZTMAX-1,
     *           NDIM4=NZPOMA*NZLWMA)
C
      COMMON /LUP/ KVK(NDIM1), JQ(NDIM1), INDPO(NDIM1),
     *             MVK(NZIQMA,NDIM1), EPO(NDIM1), 
     *             IZ, JZ, KZHV(2,2*NZAVMA), LUPAUS, IQM,
c    *             IZ, JZ, KZHV(2,2*NZAVMA), LUPAUS, IQM(2*NZAVMA),
     *             WERTL(NDIM4,NDIM4), KZHX(NZIQMA,2*NZAVMA)
C
C
      COMMON  MKC, NZV, QO(NZVMAX,NZVMAX), QOR(NZVMAX,NZVMAX),
     *        VZ(NZVMAX,NZVMAX), WERT(NDIM4,NDIM4)
      DIMENSION S(10),HHH(11)                                           
      DIMENSION WERTT(900)
      EQUIVALENCE (WERT(1,1),WERTT(1))
      
      WERT = .0                                                  
      GO TO(11,11,11,10,10,11,11,10,10,10,10),MKC
   10 NN = 1                                                            
      GO TO 12                                                          
   11 NN = NQ
12    NU = 1                                                            
      DO 1 N=NU,NN                                                      
      IF(ABS(HHH(N)).LT.1.E-15) GOTO 1
      K3ZAHL = KZHV(2,N)                                                
      IF(K3ZAHL.EQ.0) GOTO 1
      K1ZAHL = KZHV(1,N) + 1                                            
      K2ZAHL = K1ZAHL + K3ZAHL - 1                                      
      DO 111 K=K1ZAHL,K2ZAHL                                            
      MM = JQ(K)                                                        
      SIG = HHH(N)                                                      
      IF(MM.EQ.0) GOTO 6
      DO 2   M = 1,MM                                                   
      MN = MVK(M,K)                                                     
    2 SIG = SIG*S(MN)                                                   
6     M1 = KVK(K)                                                       
      WERTT(M1) = WERTT(M1) + EPO(K) * SIG
      IF(LUPAUS.LT.3) GOTO 111
      PRINT 100,NU,NN,N,K1ZAHL,K2ZAHL,K,MM,M1,HHH(N),SIG,EPO(K)
     $,WERTT(M1)
100   FORMAT(' NU,NN,N,K1,K2,K,JQ,KVK,HHH,S,E,W',7I5,I10,4E12.4)
  111 CONTINUE                                                          
    1 CONTINUE                                                          
      RETURN                                                            
      END                                                               
