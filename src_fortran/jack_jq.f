      SUBROUTINE SJACK(K1,LL,MFL)                                       
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     DIESE SUBROUTINE BERECHNET DIE JACOBI KOORDINATEN UND SCHREIBT
C     SIE AUF BAND
C     EINGABE !  K1 = ZAHL DER CLUSTER, MFL= NR DER ZERLEGUNG
C     LL = ZAHL DER CLUSTER IM ERSTEN FRAGMENT ,WIRD UM 1 ERHOEHT,ABER
C     NOL WIRD ANSCHLIESEND NICHT MEHR VERWENDET
C     SVEC ENTHAELT DIE JACOBIKOORDINATEN ALS FUNKTION DER EINTEILCHEKOO
C     DIE ORTHOGONALE TRANSFORMATION WIRD OHNE! DEN SCHWERPUNKT AUSGEF
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
      WRITE(6,*) 'r: ',((RVEC(M,N),M=1,NZV),N=1,NZT)                       
      WRITE(6,*) 's: ',((SVEC(N,M),M=1,NZV),N=1,NZT)                       
C      DIE UEBERGABE VON C KANN AUF C(M,M,K) EINGESCHRAENKT WERDEN
      RETURN 
      END
