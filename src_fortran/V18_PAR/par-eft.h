      PARAMETER (NZOPER=9, NZOPOB=8, NZOPLU=7, NZTMAX=11,
     *           NZFMAX=100, NZCMAX=8, MZGMAX=2, NZLWMA=2,
     *           NZRHOM=16, NZPARM=12, MZPARM=20, NZPOMA=1,
     *           NZIQMA=9, NZPOTM=20, NZSIOP=4, NDIM=270,
     *           NDIM1=3000, NDIM5=1900, NPDC=11000)

      PARAMETER (NZAVMA=2*NZCMAX-1, NZVMAX=NZTMAX-1,
     *           NZMAT=MZGMAX*MZGMAX*NZSIOP,
     *           NDIM4=NZPOMA*NZLWMA, NZLRH=(2*NZCMAX-3)*(NZCMAX-1),
     *   NZLREC=2*NZCMAX  + NZLRH+(NZLRH*(NZLRH+1))/2)
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

