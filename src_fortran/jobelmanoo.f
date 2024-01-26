      PROGRAM OBELMAnoo
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C    OBERPROGRAMM FUER EINTEILCHENOPERATOREN MIT EINTEILCHEN PUNKTIERTEN
C     ELEKTROMAGNETISCHE UEBERGAENGE
C
C                    
C JEDER INDEX I) BEZEICHNET EINE KARTE             
C INTEGERS IM FORMAT 10I3,REALS IM FORMAT E12.4   
C                                                
C    1) NBAND2,KBAND            NBAND2 WIRD AN QUAFOR    
C          ACHTUNG: NBAND3 = 4 IST EIN WEITERES ZWISCHENBAND]    
C    2) STEUERZAHLEN NREG(K),K=1,4
C       K=1,4  NREG(K)=0  OPERATOR K WIRD NICHT GERECHNET
C                     =1  OPERATOR K WIRD GERECHNET     
C    3) ZAHL DER MAXIMAL ZU PERMUTIERENDEN TEILCHEN ODER GRUPPEN     KZP
C    4) ZAHL DER ZERLEGUNGEN=NZF,ZAHL DER TEILCHEN =NZT          NZF NZT
C                                                     
C                                                    
C DER FOLGENDE DATENSATZ MUSS FUER JEDE ZERLEGUNG WIEDERHOLT WERDEN     
C    A) ZAHL DER CLUSTER,ZAHL DER SPIN-ISOSPIN-          NZC,NZG,MZG,NOL
C       PRODUKTE,ZAHL DER KOMBINATIONEN DER SPIN-ISOSPIN-PRODUKTE,      
C       NUMMER DER AUSGEZEICHNETEN RELATIVFUNKTION  
C    B) ZAHL DER TEILCHEN IM CLUSTER N, N=1,NZC              NGRV(2,N, )
C   C1) 1.SPIN-ISOSPIN-PRODUKT                                       NZH
C   C2) 2.SPIN-ISOSPIN-PRODUKT                                       NZH
C ....                                                               NZH
C CNZG) NZG.SPIN-ISOSPIN-PRODUKT                                     NZH
C                                                  
C     MIT DEN FOLGENDEN COF KOENNEN FESTGELEGT ERDEN                    
C     S,S3 UND T,T3                               
C                                                
C   D1) KOEFFIZIENT DES  1.  SPIN-ISOSPIN-PROD. ZUR  1. KOMBINATION  COF
C       KOEFFIZIENT DES  1.  SPIN-ISOSPIN-PROD. ZUR  2. KOMBINATION  COF
C                                                   ...                 
C       KOEFFIZIENT DES  1.  SPIN-ISOSPIN-PROD. ZUR MZG.KOMBINATION  COF
C   D2) KOEFFIZIENT DES  2.  SPIN-ISOSPIN-PROD. ZUR  1. KOMBINATION  COF
C       KOEFFIZIENT DES  2.  SPIN-ISOSPIN-PROD. ZUR  2. KOMBINATION  COF
C                                                   ...                 
C       KOEFFIZIENT DES  2.  SPIN-ISOSPIN-PROD. ZUR MZG.KOMBINATION  COF
C ....                                          
C DNZG) KOEFFIZIENT DES NZG. SPIN-ISOSPIN-PROD. ZUR  1. KOMBINATION  COF
C       KOEFFIZIENT DES NZG. SPIN-ISOSPIN-PROD. ZUR  2. KOMBINATION  COF
C                                                   ...                 
C       KOEFFIZIENT DES NZG. SPIN-ISOSPIN-PROD. ZUR MZG.KOMBINATION  COF
C     PROTON  SPIN AUF  = 1                    
C     PROTON  SPIN AB   = 2                   
C     NEUTRON SPIN AUF  = 3                  
C     NEUTRON SPIN AB   = 4                 
C     BEI JEDER FUNKTION MUSS DIE Z-KOMPONENTE DES SPINS MAXIMAL SEIN   
C     NOL GIBT DIE ZAHL DER CLUSTER IN DER ERSTEN GRUUPE AN             
C
      INCLUDE 'par/jobelma'
C
C     NZOPER: ANZAHL DER OPERATOREN
C     NZFMAX: MAXIMALE ANZAHL DER ZERLEGUNGEN
C     NZTMAX:    '       '     '  TEILCHEN
C     MZGMAX:    '       '     '  GEKOPPELTEN   SPIN-ISOSPIN-FUNKTIONEN
C     NZGMAX:    '       '     '  UNGEKOPPELTEN    '      '         '
C     NZCMAX:    '       '     '  CLUSTER
C     NZCMAF: NZCMAX!
C
      PARAMETER (NZTMA1=NZTMAX-1, NDIM4=NDIM1+NDIM2)
C
C 
C
      COMMON /PLATZ/ V(NDIM4)
      COMMON/COMY/D(100)                                                
      COMMON /CSPIN/ NALG(NZTMAX,3)
C
      COMMON /CUM/ NPERM(NZCMAX+1,NZCMAF),NH(7,4),NKOR(NZTMAX,7),NP(5)
C
      COMMON /CKO/ VEC(NZTMAX,NZTMA1,NZFMAX)
C
      COMMON /CPER/ NZD(NZTMAX),MAXLIQ,MAXLIS,MAXLIT,MZGS,NBAND3,NBAND4
C
      COMMON /CEL/ KBAND,MAXLI,MERKS(NZGMAX,NZGMAX),MG(NZCMAX),
     *             NDEL,NFAK(NZCMAX+1),NFL,NFR,MELAUS
C
      COMMON NGRU(2,NZCMAX,2),NZT,NZV,I1,I2,I3,MGW(NZCMAX+1),
     *       MMOEG(NDIM3),MSR(NZCMAX,NZCMAX),MVV(NDIMVV,NZCMAX),
     *       ANSP(NDIM1)
C
      DIMENSION NV(NZCMAX),NREG(NZOPER)
      DIMENSION NS(NZGMAX,NZFMAX),MS(MZGMAX,NZFMAX)
      DIMENSION MB(NZFMAX),MC(NZFMAX),MZG(NZFMAX),NOL(NZFMAX)
      DIMENSION NZC(NZFMAX),NZG(NZFMAX),NSS(2,NZGMAX,NZFMAX)
      DIMENSION NMASSE(2,NZGMAX,NZFMAX),NLAD(2,NZGMAX,NZFMAX)
      DIMENSION MSS(2,MZGMAX,NZFMAX),MMASSE(2,MZGMAX,NZFMAX)
      DIMENSION MLAD(2,MZGMAX,NZFMAX)
      DIMENSION UU(MZGMAX,MZGMAX)
      DIMENSION  ANTP(NDIM1), U(NDIM1), UV(NDIM2)
      DIMENSION COF(MZGMAX,NZGMAX,NZFMAX),NZH(NZTMAX,NZGMAX,NZFMAX)
      DIMENSION NGRV(2,NZCMAX,NZFMAX),NCOF1(MZGMAX,NZGMAX,NZFMAX),
     *         NCOF2(MZGMAX,NZGMAX,NZFMAX)
C
      EQUIVALENCE (V(1),UV(1)), (V(NDIM2+1),U(1))
      EQUIVALENCE (V(1),ANTP(1))
      OPEN(UNIT=3,STATUS='SCRATCH',FORM='UNFORMATTED')
      OPEN(UNIT=4,STATUS='SCRATCH',FORM='UNFORMATTED')
      OPEN(UNIT=5,FILE='INOB',STATUS='OLD')
      OPEN(UNIT=6,FILE='OUTPUT')
      OPEN(UNIT=8,FILE='OBOUT',STATUS='UNKNOWN',FORM='UNFORMATTED')
C
      INPUT=5                                                           
      NBAND3= 4
      KBAND=3
  611 FORMAT('0 NORM-OPERATOR')    
  612 FORMAT('0 PROTON-OPERATOREN')
  613 FORMAT('0 NEUTRON-OPERATOREN')
  614 FORMAT('0 PROTON-SPIN-OPERATOREN')
  615 FORMAT('0 NEUTRON-SPIN-OPERATOREN')
 1000 FORMAT(24I3)                                                      
 1001 FORMAT(10I3)                                                      
 1006 FORMAT (41H0 LISTE DER MATRIXELEMENTE ZWISCHEN DER  ,I3,          
     1          19H TEN STRUKTUR DER  ,I3,20H CLUSTEREINTEILUNG  /      
     2        30X, 9H UND DER ,I4,19H TEN STRUKTUR DER  ,I3,            
     3          20H CLUSTEREINTEILUNG  )                                
 1010 FORMAT (1H1)                                                      
 1011 FORMAT (16H0 DEFINITION DES   ,I3,4H TEN,
     127H SPIN-ISOSPIN-PRODUKTES ZUR    ,I3,4H TEN,
     2                                     18H CLUSTEREINTEILUNG   )    
 1012 FORMAT(9H  CLUSTER     ,I3,5X,4I3)
 1013 FORMAT  (45H0 DEFINITION DER ZUSAMMENGESETZTEN STRUKTUREN  )      
 1014 FORMAT (3H0 /,I3,2H ,,I3,4H ) =  /)
 1015 FORMAT(5(F10.4,4H  // ,I3,2H ,,I3,4H )   ))
 1020 FORMAT(22H SPEICHERPLATZ ZU ENG ,3I10)                            
2011  FORMAT   (19H0 BERECHNET WERDEN )                                 
 1090 FORMAT(1H0)                                                       
 5000 FORMAT (21H0SPEICHERPLATZBEDARF:,3I10)                            
C        
      D(1)=.0  
      D(2)=.0 
      DO 199 I=2,99
      HMH=I
  199 D(I+1) =LOG(HMH)+D(I)
      READ (INPUT,1000) MAIAUS,MELAUS
      READ(INPUT,1000) (NREG(K),K=1,NZOPER)
      READ(INPUT,1000) KZP                                              
      NFAK(1)=1                                                         
      DO 300 K=1,KZP                                                    
  300 NFAK(K+1)=K*NFAK(K)                                               
C         NFAK(I) = (I-1)!
C         DEFINITION VON NPERM, NPERM(1,K) =SIGNUM VON PERMUTATION K
C         NPERM(I,K) ENTHAELT DIE PERMUTATION
      NPERM(1,1)=1                                                      
      NPERM(2,1)=1                                                      
      DO702 NZP=2,KZP                                                   
      NBL=NFAK(NZP)                                                     
      NBH=NZP-1                                                         
      DO 702 L=1,NZP                                                    
      NZVL=L-1                                                          
      DO702 M=1,NBL                                                     
      ML=(L-1)*NBL+M                                                    
      NPERM(1,ML)=NPERM(1,M)*(-1)**(L+1)                                
      DO 704 NZL=1,NBH                                                  
      IF(NZL+L.GT.NZP) GOTO 706
      NVL=NZL
      GO TO 704                                                         
  706 NVL=NZL+1                                                         
  704 NPERM(NVL+1,ML)=NPERM(NZL+1,M)                                    
      NZVS=NZP+1-NZVL                                                   
      NPERM(NZVS,ML)=NZP                                                
  702 CONTINUE                                                          
      READ(INPUT,1000)NZF,NZT
  714 FORMAT('0NZT = ',I3,'IST ZU GROSS, MAX ',I3,' MOEGLICH')
      IF (NZT.LE.NZTMAX) GOTO 710
      STOP 710                                                             
  710 CONTINUE                                                          
      NZV = NZT-1                                                       
C        BLOCK EINGABE
      DO 1 K=1,NZF                                                      
      READ (INPUT,1001) NZC(K),NZG(K),MZG(K),NOL(K)
      M=NZC(K)                                                          
      READ (INPUT,1001) (NGRV(2,L,K), L=1,M)                            
      NGRV(1,1,K) =1                                                    
      DO 2 L=2,M                                                        
    2 NGRV(1,L,K) =NGRV(1,L-1,K)+NGRV(2,L-1,K)                          
C  NGRV(2,L,.)=ZAHL DER TEILCHEN IM CLUSTER L
C  NGRV(1,L,.)=NUMMER DES 1.TEILCHENS IM CLUSTER L
      KK = NOL(K) + 1                                                   
      KK = NGRV(1,KK,K)                                                 
      LM=NZG(K)                                                         
      DO 3 L=1,LM                                                       
    3 READ (INPUT,1000) (NZH(N,L,K),N=1,NZT)
C       EINLESEN DER ELEMENTAREN SPINFUNKTIONEN
      DO 940 L=1,LM                                                     
C       BESTIMMUNG DER GESAMTSPINS NS, FRAGMENTSPINS NSS
C        FRAGMENTMASSEN NMASSE, FRAGMENTLADUNGEN NLAD
      NS(L,K)=0                                                         
      NSS(1,L,K) = 0                                                    
      NSS(2,L,K) = 0                                                    
      NMASSE(1,L,K) = KK - 1                                            
      NMASSE(2,L,K) = NZT - KK + 1                                      
      NLAD(1,L,K) = 0                                                   
      NLAD(2,L,K) = 0                                                   
      DO 940 N=1,NZT                                                    
      NNS =           2*(NZH(N,L,K)-2*(NZH(N,L,K)/2))-1                 
      IF(N.GE.KK) GOTO 942
        KZ = 1
      GO TO 943                                                         
  942 KZ = 2                                                            
  943 NLAD(KZ,L,K) = NLAD(KZ,L,K) -(NZH(N,L,K)-1)/2 + 1                 
      NSS(KZ,L,K) = NSS(KZ,L,K) + NNS                                   
  940 NS(L,K)=NS(L,K)+NNS                                               
      N=MZG(K)                                                          
      DO 7   L = 1,LM                                                   
    7 READ (INPUT,1000) (NCOF1(LL,L,K),NCOF2(LL,L,K),LL=1,N)
C      CHECK EINGABE
      DO 950 LL=1,N                                                     
      I=0                                                               
      DO 951 L=1,LM                                                     
      IF (NCOF1(LL,L,K).EQ.0) GOTO 951
       IF(I.GT.0) GOTO 954
       I=1
C       SETZEN DER WERTE
      MS(LL,K)=NS(L,K)                                                  
      MMASSE(1,LL,K) = NMASSE(1,L,K)                                    
      MMASSE(2,LL,K) = NMASSE(2,L,K)                                    
      MLAD(1,LL,K) = NLAD(1,L,K)                                        
      MLAD(2,LL,K) = NLAD(2,L,K)                                        
      MSS(1,LL,K) = IABS(NSS(1,L,K))                                    
      MSS(2,LL,K) = IABS(NSS(2,L,K))                                    
      GO TO 951                                                         
C         CHECK DER WERTE
  954 IF((MS(LL,K)-NS(L,K))+(MMASSE(1,LL,K)- NMASSE(1,L,K))+
     1     (MMASSE(2,LL,K)-NMASSE(2,L,K))+(MLAD(1,LL,K)-NLAD(1,L,K))+
     2     (MLAD(2,LL,K)-NLAD(2,L,K)).EQ.0) GOTO 956
      STOP 1040                                                             
  956 MSS(1,LL,K) = MAX0(MSS(1,LL,K),IABS(NSS(1,L,K)))                  
      MSS(2,LL,K) = MAX0(MSS(2,LL,K),IABS(NSS(2,L,K)))
951   CONTINUE
  950 CONTINUE                                                          
C      BERECHNUNG DER COEFFIZIENTEN
      DO 1   L = 1,LM                                                   
      DO 1   LL = 1,N                                                   
      COF(LL,L,K) = .0
      IF(NCOF1(LL,L,K).EQ.0) GOTO1
      AB1 = NCOF1(LL,L,K)
      AB2 = NCOF2(LL,L,K)                                               
      COF(LL,L,K) = (AB1/ABS(AB1)) * SQRT(ABS(AB1)/AB2)
    1 CONTINUE                                                          
C      HIERNACH WERDEN NCOF1,NCOF2,NMASSE,NLAD,NSS NICHT MEHR VERWENDET
C      NEUER BLOCK
C     AUSDRUCK DER FUNKTIONSEIGENSCHAFTEN                               
      DO 6 K=1,NZF                                                      
      LM=NZG(K)                                                         
      DO 5 L=1,LM                                                       
      MM=NZC(K)                                                         
      DO 5 M=1,MM                                                       
      N1=NGRV(1,M,K)                                                    
      N2=NGRV(2,M,K)+N1-1                                               
    5 CONTINUE                          
      LM=MZG(K)                                                         
      DO 6 L=1,LM                                                       
      MM=NZG(K)                                                         
    6 CONTINUE
C       ENDE BLOCK AUSSCHREIBEN
C     NOTIEREN DER FUNKTIONSEIGENSCHAFTEN                               
      REWIND NBAND2                                                     
      WRITE(NBAND2) NZF,NZT,NZV,(NREG(K),K=1,NZOPER)
      DO 4 K=1,NZF                                                      
      M=NZC(K)                                                          
      WRITE(NBAND2) NZC(K),MZG(K),NOL(K)                                
      LL=MZG(K)                                                         
      WRITE(NBAND2) ((MMASSE(N,L,K),MLAD(N,L,K),MSS(N,L,K),N=1,2),      
     1     MS(L,K),L=1,LL)                                              
    4 WRITE (NBAND2) ((NGRV(N,L,K),L=1,M),N=1,2)                        
C      HIERNACH WERDEN MLAD,MMASSE,MSS NICHT MEHR VERWENDET
C     MATRIXELEMENTE ZWISCHEN EINFACHEN ALGEBRAISCHEN STRUKTUREN        
      DO 10 MFL=1,NZF                                                   
      I1=NZC(MFL)                                                       
      I3=I1-1                                                           
      I5=I3-1                                                           
      DO 12 M=1,I1                                                      
      NGRU(1,M,1) =NGRV(1,M,MFL)                                        
   12 NGRU(2,M,1)=NGRV(2,M,MFL)                                         
      CALL SJACK(I1,NOL(MFL),MFL)                                       
      MFRR=MFL                                                          
      DO 10 MFR=1,MFRR                                                  
      I2=NZC(MFR)                                                       
      I4=I2-1                                                           
      WRITE(NBAND2)  ((VEC (M,K,MFR),M=1,NZT),K=1,I4)                   
      DO 13 M=1,I2                                                      
      NGRU(1,M,2)=NGRV(1,M,MFR)                                         
   13 NGRU(2,M,2)=NGRV(2,M,MFR)                                         
C     ADRESSENSUCHREGISTER
C      HIER WERDEN DIE MOEGLICHEN TEILCHENAUSTAEUSCHE VORBEREITET
      DO 22 K=1,I1                                                      
      I=0                                                               
      NZ=NGRU(2,K,1)                                                    
      NZZ=(NZ+1)**I2-(NZ+1)**I4
C      NZZ ZAEHLT DIE MOEGLICHKEITEN AB NZ TEILCHEN AUF I2 CLUSTER ZU
C     VERTEILEN, OHNE TEILCHENZAHLERHALTUNG,DIESE WIRD ERST NACH LOOP 24
C     IN ORDNUNG GEBRACHT
      DO 23 L=1,NZZ                                                     
      J=0                                                               
      KK=L                                                              
      DO 24 M=1,I2                                                      
      MM=I2-M+1                                                         
      NV(MM)=KK/(NZ+1)**(MM-1)                                          
      KK=KK-NV(MM)*((NZ+1)**(MM-1))                                     
   24 J=J+NV(MM)                                                        
      IF(J.NE.NZ) GOTO 23
      DO 26 M=1,I2
      IF(NV(M).GT.NGRU(2,M,2)) GOTO 23
   26 CONTINUE                                                          
      I=I+1                                                             
      MVV(I,K) = L
C      IN MVV(K,L) STEHT DIE NUMMER NZZ,DIE TEILCHEN VON CLUSTER L LINKS
C      AUF DIE CLUSTER RECHTS ZU VERTEILEN. ES GIBT MG(L) MOEGLICHKEITEN
C      DAFUER
   23 CONTINUE                                                          
   22 MG(K)=I
C      MG(K) ENTHAELT DIE ANZAHL DER TEILCHENAUSTAEUSCHE DES K-TEN
C      CLUSTERS LINKS IN DIE CLUSTER RECHTS
      MGW(1)=1                                                          
      DO 27 K=1,I1                                                      
   27 MGW(K+1)=MGW(K)*MG(K)                                             
C        UNTERBLOCK ENDE
      MAXLIQ=MGW(I1)
C      MGW(I1) ENTHAELT EINE OBERE ABSCHAETZUNG DER ANZAHL DER DOPPEL-
C      NEBENKLASSEN SYMBOLE, NAEMLICH DIE PRODUKTE DER ZEILENAUSTAEUSCHE
      N1 = 1                                                            
       N2 = MAXLIQ                                                      
      N3 = NDIM3
      IF(N2.LE.N3) GOTO 910
 1021 STOP 1021                                                             
  910 CONTINUE                                                          
      DO 48 K=1,MAXLIQ                                                  
   48 MMOEG(K)=0                                                        
      I=0                                                               
      KK=1                                                              
   28 DO 29 K=KK,I1                                                     
   29 NV(K)=1                                                           
      DO 30 K=1,I1                                                      
      M=NV(K)                                                           
   30 MB(K)=MVV(M,K)                                                    
      DO 31 L=1,I2                                                      
      L1=I2-L+1                                                         
      M=0                                                               
      DO 32 K=1,I1                                                      
      MC(K)=MB(K)/((NGRU(2,K,1)+1)**(I2-L))                             
      MB(K)=MB(K)-MC(K)*((NGRU(2,K,1)+1)**(I2-L))                       
   32 M=M+MC(K)                                                         
      IF(M.NE.NGRU(2,L1,2)) GOTO 40
   31 CONTINUE                                                          
      I=I+1                                                             
      J=NV(1)                                                           
      IF(I5.LE.0) GOTO 1401
      DO 34    K = 1,I5
   34 J=J+(NV(K+1)-1)*MGW(K+1)                                          
 1401 MMOEG(J) = I                                                      
   40 DO 41 K=1,I1                                                      
      KK=I1-K+1                                                         
      IF(NV(KK).LT.MG(KK)) GOTO 42
   41 CONTINUE                                                          
      GO TO 44                                                          
   42 NV(KK)=NV(KK)+1                                                   
      KK=KK+1                                                           
      GO TO 28
C      UNTERBLOCK ENDE
   44 MAXLIT=I
C      MAXLIT IST DIE ANZAHL DER DOPPELNEBENKLASSEN
C    MMOEG(J)=N BEDEUTET,DAS DC-SYMBOL N HAT DIE ADRESSE J,WOBEI
C     J FESTLEGT WELCHE SPALTENAUSTAEUSCHE VORZUNEHMEN SIND VON LINKS
C     NACH RECHTS. J=NV(1)+SUMME (NV(K+1)-1)*MGW(K+1),K=1,NZCL-2
      J=1                                                               
      DO 46 K=1,I1                                                      
      DO 46 L=1,I2                                                      
      MSR(L,K)=J                                                        
C  ADDRESSEN DER DC-SYMBOLMATRIXELEMENTE ZUM KENNZEICHNEN DER WECHSELW
   46 J=J+1
      M=I1*I2
      MAXLIS=M
C      MAXLIS =ANZAHL DER MOEGLICHEN ET-WW IN EINEM DC
      MAXLI=MAXLIT*MAXLIS
C     MAXLI= ANZAHL DER PUNKTIERTEN DC
      N3 = NDIM1
      N1 = 2                                                            
      IF(MAXLI.LE.N3) GOTO 1026
      N2=MAXLI                                                          
      GO TO 1021                                                        
1026  NZGL=NZG(MFL)
      NZGR=NZG(MFR)                                                     
      MZGL=MZG(MFL)                                                     
      MZGR=MZG(MFR)                                                     
      MZGS = MZGR*MZGL                                                  
      DO 50   MKC = 1,NZOPER
C        LOOP UEBER OPERATOREN
      IF(0.EQ.NREG(MKC))GO TO 50                                        
      DO 52   K= 1,MAXLI                                                
52    ANTP(K)=0.
 4022 CONTINUE                                                          
      REWIND KBAND                                                      
      DO 60 NFL=1,NZGL                                                  
C         LOOP ELEMENTARE SPINFUNKTIONEN LINKS
      DO 64 K=1,NZT                                                     
   64 NALG(K,1)=NZH(K,NFL,MFL)                                          
      DO 60 NFR=1,NZGR                                                  
C         LOOP ELEMENTARE SPINFUNKTIONEN RECHTS
      DO 65 K=1,NZT                                                     
   65 NALG(K,2)=NZH(K,NFR,MFR)                                          
      NDEL = NS(NFL,MFL) - NS(NFR,MFR)                                  
      NDEL = NDEL/2                                                     
      CALL ELEML(MKC)                                                   
60    CONTINUE                                                          
C         ELEMENTARE LISTEN ERSTELLT
      REWIND NBAND3                                                     
      WRITE (NBAND3) (ANTP(K),K=1,MAXLI)
      CALL PERME (MKC,1,NNN)
C      MIT DIESEM AUFRUF WERDEN DIE NNN REPRESENTANTEN FUER WW INDC AUF
C      NBAND3 GESCHRIEBEN
      N1 = 3                                                            
      N2 = NNN*MZGL*MZGR                                                
      N3 = NDIM2
      IF (N2.GT.N3) GOTO 1021
      REWIND NBAND3                                                     
      WRITE(NBAND2)    NNN,NNN,NNN                                      
      IF(NNN.EQ.0) GOTO 700
C     MATRIXELEMENTE ZWISCHEN ZUSAMMENGESETZTEN STRUKTUREN              
      DO 130 KFL=1,MZGL                                                 
      KFRR=MZGR                                                         
      DO 130 KFR=1,KFRR                                                 
      NDEL2=MS(KFL,MFL)-MS(KFR,MFR)
      NDEL = NDEL2/2
C                   
      SL=MS(KFL,MFL)                                                    
      SML=.5*SL                                                         
      SR=MS(KFR,MFR)                                                    
      SMR=.5*SR                                                         
      GOTO(419,419,419,420,420),MKC
419   IX=0
      F=1.                                                              
      GOTO 422
420   IX=1
      F=1.
422   FAKTOR=0.
      Y=CLG(MS(KFR,MFR),2*IX,MS(KFL,MFL),MS(KFR,MFR),NDEL2)
      IF(Y.EQ.0.) GOTO 8913
      F = F * SQRT(2.*SML+1.)                                           
      FAKTOR=F/Y                                                        
C                                                                       
8913  CONTINUE
121    NBAND4 = (KFL - 1)*MZGR + KFR
      REWIND KBAND                                                      
      DO 132 K=1,MAXLI                                                  
132   U(K)=.0                                                           
      DO 134 NFL=1,NZGL                                                 
      DO 134 NFR=1,NZGR                                                 
      IF(MERKS(NFL,NFR).EQ.0) GOTO 134
       READ(KBAND) (ANSP(KKX),KKX=1,MAXLI)
      DO 140 K=1,MAXLI
      AKN = ANSP(K)
      U(K) = U(K) + COF(KFL,NFL,MFL)*COF(KFR,NFR,MFR)* AKN*FAKTOR
  140 CONTINUE                                                          
134   CONTINUE                                                          
      REWIND NBAND3                                                     
      READ (NBAND3) (ANSP(K),K=1,MAXLI)
C     NOTIEREN DER LISTEN ZUSAMMENGESETZTER STRUKTUREN                  
      IF (MAIAUS.EQ.0) GOTO 126
      CALL PERME(MKC,0,NNK)
126   CALL PERME(MKC,-2,NNK)
130   CONTINUE                                                          
      I = 0                                                             
      DO 810    K = 1,NNN                                               
      IK = 0                                                            
      DO 800   KFL = 1,MZGL                                             
      II = IK + I                                                       
      IK = IK + MZGR                                                    
      DO 800   KFR = 1,MZGR                                             
      JK = II + KFR                                                     
      UU(KFL,KFR) = UV(JK)
  800 CONTINUE                                                          
      I = I + MZGS                                                      
      READ (NBAND3) N1, (NZD(I6),I6=1,NZT)
      WRITE(NBAND2) N1, (NZD(I6),I6=1,NZT) ,
     1    ((UU(KFL,KFR),KFL=1,MZGL),KFR=1,MZGR)
810   CONTINUE
  700 CONTINUE                                                          
C      ENDE LOOP OPERATOREN
50    CONTINUE
   10 CONTINUE                                                          
      STOP                                                              
      END                                                               
      SUBROUTINE ELEML(MKC)                                             
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     SUBROUTINE ZUR AUFSTELLUNG DER ELEMENTAREN LISTEN                 
C     WIRD FUER JEDE ZERL. LI. U. RE., JEDEN OPERATOR, JEDES ELEMENTARE
C     SPIN-ISOSPIN-PRODUKT AUFGERUFEN
      INCLUDE 'par/jobelma'
      PARAMETER (NDIM4=NDIM1+NDIM2)
C
      COMMON /PLATZ/ V(NDIM4)
C
      COMMON /CSPIN/ NALG(NZTMAX,3)
C
      COMMON /CUM/ NPERM(NZCMAX+1,NZCMAF),NH(7,4),NKOR(NZTMAX,7),NP(5)
C
      COMMON /CEL/ KBAND,MAXLI,MERKS(NZGMAX,NZGMAX),MG(NZCMAX),
     *             NDEL,NFAK(NZCMAX+1),NFL,NFR,MELAUS
C
      COMMON NGRU(2,NZCMAX,2),NZT,NZV,I1,I2,I3,MGW(NZCMAX+1),
     *       MMOEG(NDIM3),MSR(NZCMAX,NZCMAX),MVV(NDIMVV,NZCMAX),
     *       ANSP(NDIM1)
C
      DIMENSION NV(NZCMAX),NSG(NZCMAX,NZCMAX)
      DIMENSION  ANTP(NDIM1)
      DIMENSION NW(NZCMAX)
      EQUIVALENCE (V(1),ANTP(1))
      DO 320 K=1,4                                                      
      I=0                                                               
      DO 321 L=1,NZT                                                    
      IF(NALG(L,1).NE.K) GOTO 321
          I=I+1
      NH(I+2,K)=L                                                       
  321 NH(1,K)=NFAK(I+1)                                                 
      NH(2,K)=I                                                         
  320 CONTINUE                                                          
C      NH(2,K) =  I =ANZAHL DER TEILCHEN VOM TYP K,  NH(1,K)= I!
C       NH(2+...I,K)= NUMMMER DES TEILCHEN VOM TYP K
      GOTO(301,301,301,305,305),MKC
301   LKC=1
      GOTO 20
305   LKC=3
20    DO 1 MLKC=1,LKC
      IF(MLKC.GT.1)GOTO 22
      DO 21 K=1,MAXLI
21    ANSP(K) = 0.
      ISPZ1 = 0                                                         
22    CONTINUE
C      CHECK OB MATRIXELEMENT FUER SPINDIFFERENZ MOEGLICH IST
      IF(MKC.GT.3)GOTO 200
      FAK=1.
      IF(NDEL.NE.0) GOTO 90
      GOTO 220
200   IF(IABS(NDEL).GT.1) GOTO 90
      IF(NDEL)207,208,209
207   IF(MLKC-1)90,220,90
208   IF(MLKC-2)90,220,90
209   IF(MLKC-3)90,220,90
220   ISPZ1 = ISPZ1+1
         LT = 1                                                         
      IF(MKC.LT.4) GOTO 253
      IF(MLKC-2) 250,251,252
250   FAK=1./SQRT(2.)
      GOTO 253
251   FAK=.5
      GOTO 253
252   FAK=-1./SQRT(2.)
253   CONTINUE
510   DO 29 K=1,NZT
      NKOR(K,2)=K                                                       
   29 NALG(K,3)=NALG(K,2)                                               
C      AN DIESER STELLE IST NALG(.,3) DIE SPINFUNKTION DER RECHTEN SEITE
C     JETZT ANWENDUNG DER SPINOPERATOREN
      IFT=1
      IF(MKC.LT.4)GOTO 100
      IF(MLKC-2)46,47,48
c46    CALL SPINDO(LT,I)
c      IF(I) 161,161,100
46    IF(MOD(NALG(LT,3),2).EQ.0) GOTO 161
      NALG(LT,3)=NALG(LT,3) + 1
      GOTO 100
47    IFT=2*MOD(NALG(LT,3),2) - 1
      GOTO 100
c47    CALL SPINWE(LT,I)
c      IFT=I
c      GOTO 100
48    IF(MOD(NALG(LT,3),2).NE.0) GOTO 161
      NALG(LT,3)=NALG(LT,3) - 1
c48    CALL SPINUP(LT,I)
c      IF(I.EQ.0) GOTO 161
C      NALG(.,3) IST DIE SPINFUNKTION DER RECHTEN SEITE NACH ANWENDUNG
C       DER SPINOPERATOREN
C     AUFSUCHEN DER PERMUTATION P0                                      
100   DO 101 K=1,NZT
      NKOR(K,1)=K                                                       
  101 NKOR(K,3)=K                                                       
      I=1                                                               
      DO 104 K=1,NZT                                                    
      IF(NALG(K,1).EQ.NALG(K,3)) GOTO 104
       K1=K+1
      DO 106 L=K1,NZT                                                   
      IF(NALG(K,1).EQ.NALG(L,3)) GOTO 107
  106 CONTINUE                                                          
      GO TO 161
C     SPINMATRIXELEMENT =0, NEUES PAAR WW-TEILCHEN
  107 NALG(L,3)=NALG(K,3)                                               
      NALG(K,3)=NALG(K,1)                                               
      J=NKOR(L,3)                                                       
      NKOR(L,3)=NKOR(K,3)                                               
      NKOR(K,3)=J                                                       
      I=-I                                                              
  104 CONTINUE                                                          
      NP(1)=I                                                           
C      NP(1) ENTHAELT VORZEICHEN DER PERMUTATION P0,
C      NKOR(.,3) ENTHAELT DIE PERMUTATION P0, ANGEWANDT AUF DIE FUNKTION
      DO 108 K=1,NZT                                                    
      L=NKOR(K,3)                                                       
  108 NKOR(L,4) =K
C      NKOR(.,4) INVERSE PERMUTATION ZU NKOR(.,3), P0**-1
      DO 110 K=1,NZT                                                    
      L=NKOR(K,2)                                                       
  110 NKOR(K,5)=NKOR(L,4)
C    NKOR(.,5) PERMUTATION P0**-1 * NKOR(.,2)
C    NKOR(.,2) IST HIER DIE IDENTITAET
      LT1=NKOR(LT,4)                                                    
C      CHECK OB WW-TEILCHEN PROTON ODER NEUTRON
      GOTO(117,114,115,114,115),MKC
114   IF(NALG(LT1,1)-3) 117,161,161
115   IF(NALG(LT1,1)-3) 161,117,117
  117 DO 120 NF1=1,NH(1,1)
      DO 1121 KH=1,NH(2,1)
1121  NKOR(NH(KH+2,1),1)=NH(NPERM(KH+1,NF1)+2,1)
      NP(2)=NPERM(1,NF1)*NP(1)
      DO 120 NF2=1,NH(1,2)
      DO 1122 KH=1,NH(2,2)
1122  NKOR(NH(KH+2,2),1)=NH(NPERM(KH+1,NF2)+2,2)
      NP(3)=NPERM(1,NF2)*NP(2)
      DO 120 NF3=1,NH(1,3)
      DO 1123 KH=1,NH(2,3)
1123  NKOR(NH(KH+2,3),1)=NH(NPERM(KH+1,NF3)+2,3)
      NP(4)=NPERM(1,NF3)*NP(3)
      DO 120 NF4=1,NH(1,4)
      DO 1124 KH=1,NH(2,4)
1124  NKOR(NH(KH+2,4),1)=NH(NPERM(KH+1,NF4)+2,4)
      NP(5)=NPERM(1,NF4)*NP(4)
C     DURCH DIE 112* DO LOOPS  WERDEN ALLE PERMUTAIONEN P' IN NKOR(.,1)
C     ERZEUGT UND IN NP(NZSORT+1) DAS VORZEICHEN DER PERMUTATION
      DO 124 K=1,NZT                                                    
      DO 124 L=1,2                                                      
      M=NKOR(K,L+3)                                                     
      N=NKOR(M,1)                                                       
  124 NKOR(K,L+5)=N                                                     
      LT2=NKOR(LT1,1)                                                   
C      NKOR(.,6)=P'*(P0**-1),NKOR(.,6) WIRD IM FOLGENDEN NICHT VERWENDET
C      NKOR(.,7)=P'*(P0**-1)*NKOR(.,2)
C      HIER IST NKOR(.,6)=NKOR(.,7)
C      NP(5) IST VORZEICHEN DER GESAMTPERMUTATION
C     KONSTRUKTION DES PUNKTIERTEN DC-SYMBOLS
      DO 130 K=1,I1
      K1=NGRU(1,K,1)                                                    
      K2=NGRU(2,K,1)+K1-1                                               
      DO 130 L=1,I2                                                     
      L1=NGRU(1,L,2)                                                    
      L2=NGRU(2,L,2)+L1-1                                               
      NSG(L,K)=0
      DO 132 M=K1,K2
      DO 1132 N=L1,L2
      IF(NKOR(N,7).NE.M)GOTO 135
      NSG(L,K)=NSG(L,K) +1
C      NSG(L,K) ENTHAELT DAS DC-SYMBOL FUER DAS L-TE CLUSTER RECHTS
C      UND K-TE CLUSTER LINKS
  135 IF(NKOR(N,7).EQ.LT2) MS21=L
1132   CONTINUE
      IF(M.EQ.LT2) MS11=K
  132 CONTINUE                                                          
  130 CONTINUE                                                          
C      ADRESSRECHNUNG
      DO 150 K=1,I1                                                     
      NW(K)=0                                                           
      DO 151 L=1,I2                                                     
  151 NW(K)=NW(K)+NSG(L,K)*(NGRU(2,K,1)+1)**(L-1)                       
      MM=MG(K)                                                          
      DO 154 M=1,MM                                                     
      IF(NW(K).EQ.MVV(M,K)) GOTO 150
  154 CONTINUE                                                          
  150 NV(K)=M
C     NV ENTHAELT DIE JEWEILIGE NR DES AUSTAUSCHES
      J=1                                                               
      DO 152 K=1,I3                                                     
  152 J=J+(NV(K)-1)*MGW(K)
C     J = ADRESSE DES DC-SYMBOLS,WIE IN HAUPT BEI 1401 BERECHNET
      J=MMOEG(J)
C     J=NUMMER DES DC-SYMBOLS
      J=I1*I2*(J-1)
      M=MSR(MS21,MS11)
C    KENNZEICHNUNG DER WECHSELWIRKUNG IM DC
C      DOPPELNEBENKLASSENINDEX VON TEILCHEN 1
      J=J+M
C      J=ADRESSE DES PUNKTIERTEN ET-DC,SO DASS ALLE MOEGLICHEN WW FUER E
C      DC NACHEINANDER ERFOLGEN
      ANSP(J) = ANSP(J) + FAK*FLOAT(IFT*NP(5))
  120 CONTINUE                                                          
  161 LT=LT+1                                                           
      IF(LT.LE.NZT) GOTO 510
   90 CONTINUE                                                          
C     ADDITION DER LISTEN FUER GLEICHE SPINDIFFERENZEN                  
      IF(MLKC.NE.LKC) GOTO 1
      MERKS(NFL,NFR) = 0
      IF (ISPZ1.EQ.0) GOTO 1
      AJJ = 0
      DO 53    I = 1,MAXLI
C      LOOP UEBER ALLE PUNKTIERTEN DC
      AIW3 = ABS(ANSP(I))
      AJJ = AJJ + AIW3
      ANTP(I) = ANTP(I) + AIW3
   53 CONTINUE                                                          
      IF(AJJ.EQ.0) GOTO 1
      MERKS(NFL,NFR) = 1
      WRITE(KBAND) (ANSP(I),I=1,MAXLI)
      IF (MELAUS.EQ.0) GOTO 1
      CALL PERME (MKC,-1,NNN)
    1 CONTINUE                                                          
      RETURN                                                            
      END                                                               
      SUBROUTINE PERME(MKC,KENN,NZAHL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C      PERME WIRD GERUFEN MIT KENN=-1 VON ELEML ZUM AUSDRUCK FUER JEDE
C      ELEMENTARE SPINFUNKTION
C      PERME WIRD GERUFEN MIT KENN=1 VON HAUPT IMMER,SCHREIBT REPRESENTA
C      FUER PUNKTIERTES DC AUF NBAND3
C      PERME WIRD GERUFEN MIT KENN=0 VON HAUPT FALLS AUSDRUCK FUER
C      GEKOPPELTE SPINFUNKTIONEN
C      PERME WIRD GERUFEN MIT KENN =-2 VON HAUPT IMMER,SPEICHERT DIE
C      MATRIXELEMENTE FUER GEKOPPELTE SPINFUNKTIONEN IN  UV
      INCLUDE 'par/jobelma'
      PARAMETER (NDIM4=NDIM1+NDIM2)
C
      COMMON /PLATZ/ V(NDIM4)
C
      COMMON /CPER/ NZD(NZTMAX),MAXLIQ,MAXLIS,MAXLIT,MZGS,NBAND3,NBAND4
C      MAXLIT=ANZAHL DER DC
C      MAXLIQ=ANZAHL DER ZEILENAUSTAUSCHE >= MAXLIT
C      MAXLIS=ANZAHL DER MOEGLICHEN 2-TEILCHEN WW IN EINEM DC
C
      COMMON /CADD/ NSH(4,NZCMAX,NZCMAX)
C
      COMMON NGRU(2,NZCMAX,2),NZT,NZV,I1,I2,I3,MGW(NZCMAX+1),
     *       MMOEG(NDIM3),MSR(NZCMAX,NZCMAX),MVV(NDIMVV,NZCMAX),
     *       ANSP(NDIM1)
C
      DIMENSION ANTP(NDIM1), U(NDIM1), UV(NDIM2)
C
      EQUIVALENCE (V(1),UV(1)), (V(NDIM2+1),U(1))
      EQUIVALENCE (V(1),ANTP(1))
      NZAHL=0                                                           
      DO 1 K=1,MAXLIT
C     LOOP UEBER DC
      NAD=(K-1)*MAXLIS                                                  
      A=.0                                                              
      DO 2 L=1,MAXLIS
C      LOOP UEBER WW-TERME
      N=NAD+L                                                           
      IF(KENN) 5,3,8                                                    
8     AKNM = ANTP(N)
      A=A+ AKNM
      GO TO 2                                                           
    3 A = A +  ABS(U(N))                                                
      GO TO2                                                            
5     A=A+ ANSP(N)**2
    2 CONTINUE                                                          
      IF(A.LT.0.00001) GOTO 1
C     FALLS KEIN BEITRAG FUER DIESES DC
      CALL DCADD(K)
      DO 30 K1=1,I1                                                     
      DO 30 L1=1,I2                                                     
      M1=MSR(L1,K1)
C     DC-INDEX TEILCHEN 1
       NADD=NAD+M1
C      NADD= GESAMTADRESSE
      N1=NSH(1,L1,K1)
C     N1 REPRESENTANT FUER WW TEILCHEN 1
      IF(KENN) 50,51,80
80    IF(ANTP(NADD).EQ.0) GOTO 30
       NZAHL=NZAHL+1
      WRITE(NBAND3) N1, (NZD(I6),I6=1,NZT)
      GO TO 30
51    IF(U(NADD).EQ.0.) GOTO 30

140   continue
      GO TO 30
50    IF(ANSP(NADD).EQ.0) GOTO 30
             IF((KENN+1).EQ.0) GO TO 140
      IF( ABS(U(NADD)).GT.0.00001)     GO TO 144                        
      U(NADD)=.0                                                        
  144 UV(NBAND4) = U(NADD)
      NBAND4 = NBAND4 + MZGS                                            
101   FORMAT(F12.4  ,4H (/O  ,I1,2H ( ,I2,3H )/ ,20I3,2H )  /)
   30 CONTINUE                                                          
    1 CONTINUE                                                          
      RETURN                                                            
      END                                                               
      SUBROUTINE DCADD(K)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     DCADD BERECHNET AUS DER ADRESSE DEN ENTSPRECHENDEN DOPPELNEBEN
C     KLASSENREPRESENTANTEN ZURUECK
C     DIESER STEHT IN DC-FORM IN NSH,BZW ALS PERMUTATION IN NZD
C
      INCLUDE 'par/jobelma'
C
      COMMON /CPER/ NZD(NZTMAX),MAXLIQ,MAXLIS,MAXLIT,MZGS,NBAND3,NBAND4
C
      COMMON /CADD/ NSH(4,NZCMAX,NZCMAX)
C
      COMMON NGRU(2,NZCMAX,2),NZT,NZV,I1,I2,I3,MGW(NZCMAX+1),
     *       MMOEG(NDIM3),MSR(NZCMAX,NZCMAX),MVV(NDIMVV,NZCMAX)
C
      DIMENSION NV(NZCMAX),NSG(NZCMAX,NZCMAX)
C
C       KONSTRUKTION EINES REPRESENTANTEN DES PUNKTIERTEN DC-SYMBOLS
       DO 6 L=1,MAXLIQ
      IF(K.EQ.MMOEG(L)) GOTO 10
    6 CONTINUE
   10 M=L
C     M=ADRESSE DES K-TEN DC-SYMBOLS
C     KONSTRUKTION DES DC-SYMBOLS
      DO 12 N=2,I1                                                      
      NN=I1-N+1                                                         
      NV(NN)=(M-1)/MGW(NN)+1                                            
      M=M-(NV(NN)-1)*MGW(NN)                                            
      I=NV(NN)                                                          
      II=MVV(I,NN)                                                      
      DO 16 I=1,I2                                                      
      IJ=I2-I+1                                                         
      NSG(IJ,NN)=II/(NGRU(2,NN,1)+1)**(IJ-1)                            
   16 II=II-NSG(IJ,NN)*(NGRU(2,NN,1)+1)**(IJ-1)                         
   12 CONTINUE                                                          
      DO 18 M=1,I2                                                      
      NSG(M,I1)=NGRU(2,M,2)                                             
      DO 18 N=1,I3                                                      
   18 NSG(M,I1)=NSG(M,I1)-NSG(M,N)
C     NSG(L,K) ENHAELT DAS DC,L-TES CLUSTER RECHTS,K-TES CLUSTER LINKS
C     BESTIMMEN DES REPRESENTANTEN DES DC-SYMBOLS
      I=1                                                               
      DO 20 J=1,I1                                                      
      DO 20 L=1,I2                                                      
      M=NSG(L,J)                                                        
      IF(M.LE.0) GOTO 20
       DO 21   N = 1,M
      NSH(N,L,J)=I                                                      
   21 I=I+1                                                             
   20 CONTINUE                                                          
      I=1                                                               
      DO 24 L=1,I2                                                      
      DO 24 J=1,I1                                                      
      M=NSG(L,J)                                                        
      IF(M.LE.0) GOTO 24
       DO 25   N = 1,M
      NZD(I)=NSH(N,L,J)                                                 
   25 I=I+1                                                             
   24 CONTINUE
C     REPRESENTANT BESTIMMT
      RETURN
       END
      SUBROUTINE SJACK(K1,LL,MFL)                                       
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     DIESE SUBROUTINE BERECHNET DIE JACOBI KOORDINATEN UND SCHREIBT
C     SIE AUF BAND
C     EINGABE !  K1 = ZAHL DER CLUSTER, MFL= NR DER ZERLEGUNG
C     LL = ZAHL DER CLUSTER IM ERSTEN FRAGMENT ,WIRD UM 1 ERHOEHT,ABER
C     NOL WIRD ANSCHLIESEND NICHT MEHR VERWENDET
C     SVEC ENTHAELT DIE JACOBIKOORDINATEN ALS FUNKTION DER EINTEILCHEKOO
C     DIE ORTHOGONALE TRANSFORMATION WIRD OHNE!!! DEN SCHWERPUNKT AUSGEF
C     RVEC ENTHAELT DIE DAZU TRANSPONIERTE MATRIX
C     S(I)=SUMME UEBER J (RVEC(I,J)*R(J))
C     C(.,.,K) =1, KENNZEICHNET DIE INNEREN JACOBIKOOR. VON CLUSTER K
C      C(.,.,K) =1 FUER K=NZC CHARAKTERISIERT DIE RELATIV KOORDINATEN
      INCLUDE 'par/jobelma'
      PARAMETER (NZTMA1=NZTMAX-1)
C
      COMMON /CKO/ VEC(NZTMAX,NZTMA1,NZFMAX)
C
      COMMON NGRU(2,NZCMAX,2),NZT,NZV
C
      DIMENSION RVEC(NZTMA1,NZTMAX),SVEC(NZTMAX,NZTMA1),
     *          C(NZTMA1,2*NZCMAX-1)
C
      LL = LL + 1                                                       
      L5 = NGRU(1,LL,1)                                                 
      K2=K1-1                                                           
      K4=K1 + K2                                                        
      DO 2124 K=1,NZV
      DO 2124 L=1,NZT
2124  SVEC(L,K)=0.
      DO 2125 L=1,NZV
      DO 2125 I=1,K4
2125  C(L,I)=0.
      I = 0                                                             
      DO 10   K = 1,K1                                                  
      L4 = NGRU(2,K,1) - 1                                              
      IF(L4.EQ.0) GOTO 10
       DO 12   L = 1,L4
      I = I + 1                                                         
      DO 13   M = 1,L                                                   
      L1 =      NGRU(1,K,1) + M - 1                                     
   13 SVEC(L1,I) = -1./ FLOAT(L)                                        
      L1=NGRU(1,K,1) + L                                                
   12 SVEC(L1,I) = 1.                                                   
   10 CONTINUE                                                          
      LL2 = LL - 2                                                      
      IF(LL2.LE.0) GOTO 14
       DO 16   K = 1,LL2
      I = I + 1                                                         
      L1 = NGRU(1,K+1,1) - 1                                            
      L2 = L1 + 1                                                       
      L3 = NGRU(2,K+1,1)                                                
      L4 = L1 + L3                                                      
      DO 17   M = 1,L1                                                  
   17 SVEC(M,I) = -1./ FLOAT(L1)                                        
      DO 16   M = L2,L4                                                 
   16 SVEC(M,I) =  1./ FLOAT(L3)                                        
   14 IF(K1.LE.LL) GOTO 24
       LL1 = LL + 1
      DO 21   K = LL1,K1                                                
      I = I + 1                                                         
      L1 = NGRU(1,K,1) - 1                                              
      L2 = L1 + 1                                                       
      L3 = NGRU(2,K,1)                                                  
      L4 = L1 + L3                                                      
      DO 22   M = L5,L1                                                 
   22 SVEC(M,I) =-1./ FLOAT(L1-L5+1)                                    
      DO 23   M = L2,L4                                                 
   23 SVEC(M,I) = 1./ FLOAT(L3)                                         
   21 CONTINUE                                                          
   24  L4 = L5 - 1                                                      
      DO 26   M  = 1,L4                                                 
      SVEC(M,NZV) = -1./ FLOAT(L4)                                      
   26 SVEC(M,NZT) = 1./ FLOAT(NZT)                                      
      DO 27   M = L5,NZT                                                
      SVEC(M,NZV) = 1./ FLOAT(NZT-L4)                                   
   27 SVEC(M,NZT) = 1./ FLOAT(NZT)                                      
      DO 60   K = 1,NZT                                                 
      A = .0                                                            
      DO 61   L = 1,NZT                                                 
   61 A = A + SVEC(L,K)**2                                              
      A = 1./ SQRT(A)                                                   
      DO 62  L = 1,NZT                                                  
   62 SVEC(L,K) = A*SVEC(L,K)                                           
   60 CONTINUE                                                          
      DO 30   K = 1,NZT                                                 
      DO 30   L = 1,NZT                                                 
      RVEC(L,K) = SVEC(K,L)                                             
   30 CONTINUE                                                          
       DO 34   K = 1,NZV                                                
      L1 = NZV - K2 + K                                                 
      IF(K.GE.K1) GOTO 34
        DO 70    M = 1,NZT
   70 VEC(M,K,MFL) = SVEC(M,L1)
C      VEC ENTHAELT DIE CLUSTERRELATIVKOORDINATEN
C     DIE K-TE RELATIVKOORD. IST SUMME UEBER M (VEC(M,K)*R(M))
   34 CONTINUE                                                          
   40 FORMAT(//28H DEFINITION DER KOORDINATEN      //)                  
   41 FORMAT(3H S(,I2,3H) =,10(F6.3,3H R(,I2,1H)  ) / 7X,               
     1         2(F6.3,3H R(,I2,1H)))                                    
   42 FORMAT(3H R(,I2,3H) =,10(F6.3,3H S(,I2,1H)  ) / 7X,               
     1         2(F6.3,3H S(,I2,1H)))                                    
   43 FORMAT(//)                                                        
      IIZ = 0                                                           
      DO 2130 K=1,K1                                                    
      NNZ   = NGRU(2,K,1)  - 1                                          
      IF(NNZ.EQ.0) GOTO 2130
       DO 2132    M = 1,NNZ
      IIZ = IIZ + 1                                                     
 2132 C(IIZ,K) = 1.
   44 FORMAT(10F10.4)                                                   
 2130 CONTINUE                                                          
      DO 50   K = 1,K2                                                  
      KK=NZV-K2+K                                                       
      KI = K1 + K                                                       
   50 C(KK ,KI) = 1.
      DO 51 K=1,K4
51    CONTINUE
      WRITE(NBAND2) ((RVEC(M,N),M=1,NZV),N=1,NZT)                       
      WRITE(NBAND2) ((SVEC(N,M),M=1,NZV),N=1,NZT)                       
      WRITE(NBAND2) ((C(N,K),N=1,NZV),K=1,K4)
C      DIE UEBERGABE VON C KANN AUF C(M,M,K) EINGESCHRAENKT WERDEN
      RETURN 
      END  
      DOUBLE PRECISION FUNCTION CLG(J1,J2,J3,M1,M2)
C
C     CLG BERECHNET DIE CLEBSCH-GORDAN-KOEFFIZIENTEN
C     (J1/2,M1/2;J2/2,M2/2|J3/2,(M1+M2)/2) NACH
C     EDMONDS 'ANGULAR MOMENTUM IN QUANTUM MECHANICS',
C     PRINCETON, 1960 GLEICHUNGEN (3.10.60), (3.7.3)
C     UND TABELLE 2 (1. GLEICHUNG)
C
C     BENUTZT COMMON /COMY/ MIT DEN LOGRITHMEN DER
C     FAKULTAETEN
C
C     M. UNKELBACH 1989
C     LETZTE AENDERUNG: 06.02.89
C
C
      INTEGER JW1, JW2, JW3, MW1, MW2, MW3, JSUM, JSUM1,
     *        JDIF1, JDIF2, JDIF3, JMSUM1, JMSUM2, JMSUM3,
     *        JMDIF1, JMDIF2, JMDIF3, JJM1, JJM2, IMAX, IMIN,
     *        I, J1, J2, J3, M1, M2
C
      DOUBLE PRECISION FAKLN, CLGH
C
      COMMON /COMY/ FAKLN(0:99)
C     FAKLN(I) = LOG(I!)
C
C
C
C
      JW1=J1
      JW2=J2
      JW3=J3
      MW1=M1
      MW2=M2
C
C     CHECK, OB CLG = 0
      CLG=0.
      IF (JW1.LT.IABS(MW1)) RETURN
      IF (JW2.LT.IABS(MW2)) RETURN
      IF (JW3.GT.JW1+JW2.OR.JW3.LT.IABS(JW1-JW2)) RETURN
      MW3=MW1+MW2
      IF (JW3.LT.IABS(MW3)) RETURN
      JMSUM1=JW1+MW1
      JMSUM2=JW2+MW2
      JMSUM3=JW3+MW3
      IF (MOD(JMSUM1,2).EQ.1) RETURN
      IF (MOD(JMSUM2,2).EQ.1) RETURN
      IF (MOD(JMSUM3,2).EQ.1) RETURN
C
C
      JSUM=(JW1+JW2+JW3)/2
      JSUM1=JSUM+1
      JDIF1=JSUM-JW1
      JDIF2=JSUM-JW2
      JDIF3=JSUM-JW3
C
      IF (IABS(MW1)+IABS(MW2).EQ.0) GOTO 100
C
C     NORMALE CLEBSCH-GORDAN-KOEFFIZIENTEN
      JMSUM1=JMSUM1/2
      JMDIF1=JMSUM1-MW1
      JMSUM2=JMSUM2/2
      JMDIF2=JMSUM2-MW2
      JMSUM3=JMSUM3/2
      JMDIF3=JMSUM3-MW3
      JJM1=JDIF1+JMDIF1
      JJM2=JDIF3-JMDIF1
      IMIN=MAX0(0,-JJM2)
      IMAX=MIN0(JMDIF1,JMDIF3)
C
      CLGH=0.
      DO 50, I=IMIN, IMAX
       CLGH=CLGH+DBLE(1-2*MOD(I,2))*
     *     EXP(FAKLN(JMSUM1+I)+FAKLN(JJM1-I)-FAKLN(I)-FAKLN(JMDIF1-I)-
     *         FAKLN(JMDIF3-I)-FAKLN(JJM2+I))
50    CONTINUE
C
      IF (IMIN.GT.IMAX) CLGH=1.
      CLGH=CLGH*EXP((FAKLN(JDIF3)+FAKLN(JMDIF1)+FAKLN(JMDIF2)+
     *             FAKLN(JMDIF3)+FAKLN(JMSUM3)-FAKLN(JSUM1)-
     *             FAKLN(JDIF1)-FAKLN(JDIF2)-FAKLN(JMSUM1)-
     *             FAKLN(JMSUM2)+FAKLN(JW3+1)-FAKLN(JW3))*.5)
      CLG=CLGH*DBLE(1-2*MOD(JMDIF1,2))
C
C     ENDE DER BERECHNUNG FUER NORMALE CLEBSCH-GORDAN-KOEFFIZIENTEN
      RETURN
C
C
C
100   CONTINUE
C     PARITAETSCLEBSCH
C
      IF (MOD(JSUM,2).EQ.1) RETURN
C
      CLGH=EXP((FAKLN(JDIF1)+FAKLN(JDIF2)+FAKLN(JDIF3)-FAKLN(JSUM1)+
     *         FAKLN(JW3+1)-FAKLN(JW3))*.5+
     *        FAKLN(JSUM/2)-FAKLN(JDIF1/2)-FAKLN(JDIF2/2)-
     *        FAKLN(JDIF3/2))
      CLG=CLGH*DBLE(1-2*MOD((JSUM+JW1-JW2)/2,2))
C
C
C     ENDE DER RECHNUNG FUER PARITAETSCLEBSCH
      RETURN
      END
      FUNCTION F6J(JD1,JD2,JD3,LD1,LD2,LD3)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     VERSION I F6J FUNCTION CALLS S6J  FORTRAN IV
C     VEREINFACHT 27.9.95 H.M.H
      J1=JD1
      J2=JD2
      J3=JD3
      L1=LD1
      L2=LD2
      L3=LD3
C     ANGULAR MOMENTUM COUPLING TESTS FOR 6J COEFFICIENT
      F6J=0.0
      IF(J1.LT.0 .OR. J2.LT.0 .OR. J3.LT.0) RETURN
      IF(L1.LT.0 .OR. L2.LT.0 .OR. L3.LT.0) RETURN
      IF(MOD(J1+J2+J3,2).NE.0) RETURN
      IF(J3.GT.J1+J2 .OR. J3.LT.ABS(J1-J2)) RETURN
      IF(MOD(J1+L2+L3,2).NE.0) RETURN
      IF(L3.GT.J1+L2 .OR. L3.LT.ABS(J1-L2)) RETURN
      IF(MOD(L1+J2+L3,2).NE.0) RETURN
      IF(L3.GT.L1+J2 .OR. L3.LT.ABS(L1-J2)) RETURN
      IF(MOD(L1+L2+J3,2).NE.0) RETURN
      IF(J3.GT.L1+L2 .OR. J3.LT.ABS(L1-L2)) RETURN
      F6J=S6J(J1,J2,J3,L1,L2,L3)
      RETURN
      END
      FUNCTION F9J(JD1,JD2,JD3,JD4,JD5,JD6,JD7,JD8,JD9)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     F9J VERSION I  CALLS S6J  FORTRAN IV
C     VEREINFACHT 27.9.95 H.M.H.
      DIMENSION KN(6),KX(6),NN(6)
      J1=JD1   
      J2=JD2  
      J3=JD3 
      J4=JD4
      J5=JD5
      J6=JD6
      J7=JD7
      J8=JD8
      J9=JD9
      F9J= 0.0
C     ANGULAR MOMENTUM COUPLING TESTS FOR 9J COEFFICIENT 
      IF(MOD(J1+J2+J3,2).NE.0) RETURN
      IF(J3.GT.J1+J2 .OR. J3.LT.ABS(J1-J2)) RETURN
      IF(MOD(J4+J5+J6,2).NE.0) RETURN
      IF(J6.GT.J4+J5 .OR. J6.LT.ABS(J4-J5)) RETURN
      IF(MOD(J7+J8+J9,2).NE.0) RETURN
      IF(J9.GT.J7+J8 .OR. J9.LT.ABS(J7-J8)) RETURN
      IF(MOD(J1+J4+J7,2).NE.0) RETURN
      IF(J7.GT.J1+J4 .OR. J7.LT.ABS(J1-J4)) RETURN
      IF(MOD(J2+J5+J8,2).NE.0) RETURN
      IF(J8.GT.J2+J5 .OR. J8.LT.ABS(J2-J5)) RETURN
      IF(MOD(J3+J6+J9,2).NE.0) RETURN
      IF(J9.GT.J3+J6 .OR. J9.LT.ABS(J3-J6)) RETURN
      KN(1)=MAX0(IABS(J2-J6),IABS(J1-J9),IABS(J4-J8))
      KN(2)=MAX0(IABS(J2-J7),IABS(J5-J9),IABS(J4-J3))
      KN(3)=MAX0(IABS(J6-J7),IABS(J5-J1),IABS(J8-J3))
      KN(4)=MAX0(IABS(J6-J1),IABS(J2-J9),IABS(J5-J7))
      KN(5)=MAX0(IABS(J2-J4),IABS(J3-J7),IABS(J6-J8))
      KN(6)=MAX0(IABS(J3-J5),IABS(J1-J8),IABS(J4-J9))
      KX(1)=MIN0(J2+J6,J1+J9,J4+J8)
      KX(2)=MIN0(J2+J7,J5+J9,J4+J3)
      KX(3)=MIN0(J6+J7,J5+J1,J8+J3)
      KX(4)=MIN0(J1+J6,J2+J9,J5+J7)
      KX(5)=MIN0(J2+J4,J3+J7,J6+J8)
      KX(6)=MIN0(J3+J5,J1+J8,J4+J9)
      DO 35 K=1,6
   35 NN(K)=KX(K)-KN(K)
      KSIGN=1
      I=MIN0(NN(1),NN(2),NN(3),NN(4),NN(5),NN(6))
      DO 40 K=1,6
      IF(I-NN(K))40,50,40
   40 CONTINUE
   50 KMIN=KN(K)+1
      KMAX=KX(K)+1
      GO TO(130,52,53,54,55,56),K
   52 J=J1
      J1=J5
      J5=J
      J=J3
      J3=J8
      J8=J
      J=J6
      J6=J7
      J7=J
      GO TO 130
   53 J=J2
      J2=J7
      J7=J
      J=J3
      J3=J4
      J4=J
      J=J5
      J5=J9
      J9=J
      GO TO 130
   54 J=J1
      J1=J2
      J2=J
      J=J4
      J4=J5
      J5=J
      J=J7
      J7=J8
      J8=J
      GO TO 120
   55 J=J1
      J1=J3
      J3=J
      J=J4
      J4=J6
      J6=J
      J=J7
      J7=J9
      J9=J
      GO TO 120
   56 J=J2
      J2=J3
      J3=J
      J=J5
      J5=J6
      J6=J
      J=J8
      J8=J9
      J9=J
  120 KSIGN=(1-MOD(J1+J2+J3+J4+J5+J6+J7+J8+J9,4))
C     SUMMATION OF SERIES OF EQUATION (2)  
  130 SUM=0.0                             
      SIG=(-1)**(KMIN-1)*KSIGN
      FLK=KMIN                           
      DO 200 K=KMIN,KMAX,2              
      TERM=FLK*S6J(J1,J4,J7,J8,J9,K-1)*S6J(J2,J5,J8,J4,K-1,J6)
     1*S6J(J3,J6,J9,K-1,J1,J2)
      FLK=FLK+2.0                      
  200 SUM=SUM+TERM                    
      F9J=SUM*SIG
      RETURN                       
      END                         
      FUNCTION S6J(JD1,JD2,JD3,LD1,LD2,LD3)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     VERSION I  FORTRAN IV
      DIMENSION MA(4),MB(3),MED(12)
      COMMON /FACT/FL(322),NCALL
      DATA NCALL/1/
      J1=JD1
      J2=JD2
      J3=JD3
      L1=LD1
      L2=LD2
      L3=LD3
C     DETERMINE WHETHER TO CALCULATE FL(N) S
      IF(NCALL.EQ.0) GOTO 15
      NCALL=0
C     CALCULATE FL(N) S
      FL(1)=0.0
      FL(2)=0.0
      DO 50 N= 3,322
      FN=N-1
   50 FL(N)=FL(N-1)+LOG(FN)
   15 MED(1)=(-J1+J2+J3)/2
      MED(2)=(+J1-J2+J3)/2
      MED(3)=(+J1+J2-J3)/2
      MED(4)=(-J1+L2+L3)/2
      MED(5)=(+J1-L2+L3)/2
      MED(6)=(+J1+L2-L3)/2
      MED(7)=(-L1+J2+L3)/2
      MED(8)=(+L1-J2+L3)/2
      MED(9)=(+L1+J2-L3)/2
      MED(10)=(-L1+L2+J3)/2
      MED(11)=(+L1-L2+J3)/2
      MED(12)=(+L1+L2-J3)/2
      MA(1)=MED(1)+MED(2)+MED(3)
      MA(2)=MED(4)+MED(5)+MED(6)
      MA(3)=MED(7)+MED(8)+MED(9)
      MA(4)=MED(10)+MED(11)+MED(12)
      MB(1)=MA(1)+MED(12)
      MB(2)=MA(1)+MED(4)
      MB(3)=MA(1)+MED(8)
C     DETERMINE MAXIMUM OF (J1+J2+J3),(J1+L2+L3),(L1+J2+L3),(L1+L2+J3)
      MAX=MA(1)
      DO 30 N=2,4
      IF (MAX-MA(N)) 20,30,30
   20 MAX=MA(N)
   30 CONTINUE
C     DETERMINE MINIMUM OF (J1+J2+L1+L2), (J2+J3+L2+L3),(J3+J1+L3+L1)
      MIN=MB(1)
      DO 51 N=2,3
      IF (MIN-MB(N)) 51,51,40
   40 MIN=MB(N)
   51 CONTINUE
      MINH=MIN
      KMAX=MIN-MAX
      MINP1=MIN+1
      MINI  =MINP1-MA(1)
      MIN2=MINP1-MA(2)
      MIN3=MINP1-MA(3)
      MIN4=MINP1-MA(4)
      MIN5=MINP1+1
      MIN6=MB(1)-MIN
      MIN7=MB(2)-MIN
      MIN8=MB(3)-MIN
      UK=1.E-15
      S=1.0E-15
      IF (KMAX) 65,65,55
   55 DO 60 K=1,KMAX
      UK=-UK*DBLE(MINI-K)*DBLE(MIN2-K)*DBLE(MIN3-K)*DBLE(MIN4-K)/
     1 (DBLE(MIN5-K)*DBLE(MIN6+K)*DBLE(MIN7+K)*DBLE(MIN8+K))
C     CUT OFF SERIES AT 1.0D-25
      IF(ABS(UK)-1.E-25) 65,60,60
   60 S=S+UK
   65 S=S*1.0E+15
C     CALCULATE DELTA FUNCTIONS
      DELOG=0.0
      DO 70 N=1,12
      NUM=MED(N)
   70 DELOG=DELOG+FL(NUM+1)
      NUM1=MA(1)+2
      NUM2=MA(2)+2
      NUM3=MA(3)+2
      NUM4=MA(4)+2
      DELOG=DELOG-FL(NUM1)-FL(NUM2)-FL(NUM3)-FL(NUM4)
      DELOG=0.5*DELOG
      ULOG=FL(MIN5)-FL(MINI)-FL(MIN2)-FL(MIN3)-FL(MIN4)-FL(MIN6+1)-
     1   FL(MIN7+1)-FL(MIN8+1)
      PLOG=DELOG+ULOG
      P=EXP (PLOG)
      S6J=P*S
      IF(MOD(MINH,2).NE.0)  S6J=-S6J
      RETURN
      END
