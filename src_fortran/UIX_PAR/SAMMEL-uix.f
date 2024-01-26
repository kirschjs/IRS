      PROGRAM SAMMEL
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)

C 
C      INTEGERS IM FORMAT 24I3, REALS IM FORMAT 6E12.4
C

      INCLUDE 'par.h'

      
C     NZOPER: ANZAHL DER OPERATOREN IN QUAF
C     NZFMAX:     "      "     "  ZERLEGUNGEN
C     NZRHOM:     "      "     "  BASISVEKTOREN PRO ZERLEGUNG
C     MZPARM:     "      "     "  RADIALPARAMETER
C
      parameter (lind=(nzfmax*(nzfmax+1))/2*nzoper)
      integer nteil
C
      dimension index(lind),mop(lind),mmfr(lind)
      DIMENSION   NUM(NZRHOM,NZFMAX),kmax(nzfmax),
     *           MZPAR(NZFMAX), NZRHO(NZFMAX),kmin(nzfmax)
      COMMON    DM(NDIM,NDIM,2,nzfmax,nzoper)
      DIMENSION IOP(0:NZFMAX)
      DIMENSION LREG(NZOPER)
C
      character*80  fnumber, qname
 
         OPEN(UNIT=10,FILE='DRQUAOUT',STATUS='OLD',
     *        FORM='UNFORMATTED')
         OPEN(UNIT=14,FILE='DRFINDEX',STATUS='OLD',
     *        FORM='UNFORMATTED')
         OPEN(UNIT=5 ,FILE='INSAM',STATUS='OLD')
         OPEN(UNIT=16 ,FILE='OUTPUT',
     *        STATUS='REPLACE', FORM='FORMATTED')

      NOUT=16
      WRITE(NOUT, 1111)
      
1111  FORMAT('1 DRSAMMEL VERSION VOM 29.05.2009')

      INPUT=5
      nband1=10
      read(input,1002) jfilmax,naus,ntest
      I=0
      WRITE (NOUT,*) ' Matrizen in files DRDMOUT.',jfilmax        
      read(input,1002) mflu,mflo
      write(nout,*) 'Es wird von Zerlegung ',mflu,' bis ',mflo,
     *    ' aufgesammelt'
      REWIND NBAND1
            close(unit=nout, status='keep')
         OPEN(UNIT=nout ,FILE='OUTPUT',
     $        POSITION='APPEND', FORM='FORMATTED')
      READ(NBAND1) NZF,(LREG(K),K=1,NZOPER),NZBASV2,(NZRHO(K),K=1,NZF)
      DO 22  MFL = 1,NZF
      MZPAR(MFL) = 0
      KK=NZRHO(MFL)
 1002 FORMAT(2I3,i6)
      I=I+KK
      
   22 CONTINUE
      I=0
      DO 950  K = 1,NZF
      DO 4536 N=1,NZRHO(K)
      i=i+1
      NUM(N,K)=i
      READ(NBAND1) MZPAR(K)
 4536 CONTINUE
  950 CONTINUE
            close(unit=nband1, status='keep')
      REWIND 14
      if(ntest.eq.0) then
            READ(14) NGER,(INDEX(I),I=1,NGER-1)
      else
          read(14) nger,(index(i),i=1,ntest)
          write(nout,*) ' Von ',nger,' gerechneten operatoren werden',
     *    ntest,' eingelesen'
          if(naus.gt.0) write(nout,309) (index(i),i=1,ntest)
309   format(10i12)
          nger=ntest
      endif
      write(nout,*) 'nger',  NGER
      nteil = 0
      iop(0)=0
      istop=0
      kstop=0
      DO 140  MFL = 1,MFLO
       kmin(mfl)=jfilmax
       kmax(mfl)=1
      DO 141 MFR=1,MFL
      DO 142 MKC=1, NZOPER
         nnn = nzrho(mfl)*nzrho(mfr)
      if(lreg(mkc)*nnn .eq. 0) goto 142
      nteil=nteil+1
      IF(MFL.LT.MFLU) GOTO 142
      if(index(nteil).eq.-1) then
         write(nout,*) ' Fuer Zerlegung links ',mfl,' und rechts ',mfr,
     *   'ist operator ',mkc,' nicht gerechnet, nteil=',nteil
          if(naus.eq.0) stop 'dm fehlt'
          istop=istop+1
      endif
      if(index(nteil).gt.jfilmax) then
         write(nout,*) ' Fuer Zerlegung links ',mfl,' und rechts ',mfr,
     *   'ist operator ',mkc,' nicht korrekt, nteil=',nteil
          if(naus.lt.2) stop 'dm falsch'
          kstop=kstop+1
      endif
      mop(nteil)=mkc
      mmfr(nteil)=mfr
      if(index(nteil).gt.0) then
      kmin(mfl)=min(index(nteil),kmin(mfl))
      kmax(mfl)=max(index(nteil),kmax(mfl))
      endif
142   CONTINUE
141   CONTINUE
        iop(mfl)=nteil
140   CONTINUE
       if(istop.gt.0) then
            write(nout,*) ' Es fehlen ',istop,' operatoren'
            stop ' dm fehlt'
       endif

       if(kstop.gt.0) then
            write(nout,*) ' Es sind ',kstop,' operatoren inkorrekt'
            stop ' dm falsch'
       endif

c       print *,'iop',(iop(ix),ix=0,nzf)
c       print *,'kmin',(kmin(ix),ix=1,nzf)
c       print *,'kmax',(kmax(ix),ix=1,nzf)
c       print *,'mop',(mop(ix),ix=1,nger)
c       print *,'mmfr',(mmfr(ix),ix=1,nger)
C
       DO 447 MFL=MFLU,MFLO
       CLOSE(UNIT=11,STATUS='KEEP')
            write(fnumber,*) MFL
            do i=1, 255
               if(fnumber(i:i).ne.' ') goto  297
            end do
  297       do j=i, 255
               if(fnumber(j:j).eq.' ') goto 298
            end do
  298       qname = 'TDQUAOUT.' // fnumber(i:j)
            open(unit=11, file=qname, form='unformatted',
     $           STATUS='REPLACE')
      
      do 809 mkc=1,nzoper
      do 808 mfr=1,mfl
      do 807 i=1,2
      do 806 j1=1,ndim
      do 805 i1=1,ndim
      dm(i1,j1,i,mfr,mkc)=0.
805   continue
806   continue
807   continue
808   continue
809   continue

       IFERTIG=0
       DO 312 JLIM=IOP(MFL-1)+1,IOP(MFL)
            IF(INDEX(JLIM).GT.0)IFERTIG=IFERTIG+1
312    CONTINUE

      DO 317 IDMCOUNT=KMIN(MFL),KMAX(MFL)
      NBAND=40+IDMCOUNT
      CLOSE(UNIT=NBAND,STATUS='KEEP')
            write(fnumber,*) IDMCOUNT
            do i=1, 255
               if(fnumber(i:i).ne.' ') goto  997
            end do
  997       do j=i, 255
               if(fnumber(j:j).eq.' ') goto 998
            end do
  998       qname = 'DRDMOUT.' // fnumber(i:j)
            open(unit=NBAND, file=qname, form='unformatted',
     $           STATUS='OLD')
131      READ(NBAND,END=317,err=317) MTEIL,JCOUNT,INDEXR
c        print *, MTEIL,JCOUNT,INDEXR
         IF(INDEXR.EQ.0)  GOTO 131
         IF(MTEIL.LT.(IOP(MFL-1)+1) .OR. MTEIL.GT.IOP(MFL)) THEN
            READ (NBAND,END=131)
            GOTO 131
         ELSE
        read(NBAND,end=317,err=317)(((dm(i,j,k,mmfr(mteil),mop(mteil)),
     $        i=1,ndim),j=1,ndim),k=1,2)
           ifertig= ifertig - 1
           if(KMIN(MFL).EQ.KMAX(MFL) .AND. ifertig.eq.0) goto 318
           GOTO 131
      ENDIF  
317        CONTINUE

318   IRHO=NZRHO(MFL)
      IK1=MZPAR(MFL)

      jteil=iop(mfl-1)
      DO 341 MFR=1,MFL
      JRHO=NZRHO(MFR)
      JK1=MZPAR(MFR)
      DO 342 MKC=1, NZOPER
         nnn = nzrho(mfl)*nzrho(mfr)
      if(lreg(mkc)*nnn .eq. 0) goto 342
      jteil = jteil + 1
         IF(INDEX(JTEIL).EQ.0) then
         II1=1
         A=0.
         DO 580 M=1,IRHO
            NUML=NUM(M,MFL)
            DO 581 N=1,JRHO
               NUMR=NUM(N,MFR)
               IF( NUML.LT.NUMR) GOTO 581   
               WRITE (11)NUML,NUMR,II1,II1,A,A
 581        CONTINUE
 580     CONTINUE
         goto 342
         endif

      DO 480 M=1,IRHO
      NUML=NUM(M,MFL)
      DO 481 N=1,JRHO
      NUMR=NUM(N,MFR)
      IF(NUML.LT.NUMR) GOTO 481
      M1=(M-1)*IK1+1
      M2=M*IK1
      N1=(N-1)*JK1+1
      N2=N*JK1
      II1 = 1
      A = 0.
      DO 510 I=1,2
      DO 510 L=N1,N2
      DO 510 K=M1,M2
 510  A = A + ABS(DM(K,L,I,mfr,mkc))
      IF(A.GT.0.) GOTO512
      WRITE (11) NUML,NUMR,II1,II1,A,A
      WRITE (NOUT,*) NUML,NUMR,II1,II1,A,A
      GOTO 481
512   WRITE(11) NUML,NUMR,IK1,JK1,(((DM(K,L,I,mfr,mkc),
     1 K=M1,M2), L=N1,N2), I=1,2)
      IF (NAUS.EQ.0) GOTO 481
      WRITE (NOUT,*) 'mfl,mfr,mkc',mfl,mfr,mkc,NUML, NUMR, IK1, JK1
      IF(NAUS.LT.2) GOTO 481
      WRITE (NOUT,1021) (((DM(K,L,I,mfr,mkc), K=M1,M2),L=N1,N2),I=1,2)
1021  FORMAT(1X,10E12.5)
  481 CONTINUE
  480 CONTINUE

342        continue
341        continue
C        LOOP OPERATOREN

  447 CONTINUE
      WRITE(NOUT,3011)
3011  FORMAT(//,' ENDE DER RECHNUNG VON SAMMEL')
      CLOSE(UNIT=6)
      CLOSE(UNIT=12,STATUS='KEEP')


      STOP

      END
