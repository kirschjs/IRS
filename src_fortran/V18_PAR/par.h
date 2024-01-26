      PARAMETER (NZOPER=14, NZOPOB=8, NZOPLU=7, NZTMAX=12,
     *           NZFMAX=50, NZCMAX=6, MZGMAX=14, NZLWMA=20,
     *           NZRHOM=100, NZPARM=100, MZPARM=50, NZPOMA=3,
     *           NZIQMA=9, NZPOTM=20, NZSIOP=4, NDIM=840,
     *           NDIM1=48000, NDIM5=9000, NPDC=8040)

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

