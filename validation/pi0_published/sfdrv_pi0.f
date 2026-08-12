c-----------------------------------------------------------------------
c  sfdrv -- driver to dump the EXCLURAD eta structure-function model
c           (dvmpx.F, Table B.1 eta column of pi0eta.par) so that the
c           Python port in model_tableB1.py can be verified against it.
c
c  stdin  : mode, then one record per line
c     mode = 1 :  del2  xb  Q2  E        -> XSIGMA_T/L/TT/LT (dsigma_i/dt)
c     mode = 2 :  cost  W   Q2  E        -> dvmpw S_T,S_L,S_TT,S_LT + del2,xb
c  a record with del2 = 999 terminates.
c-----------------------------------------------------------------------
      program sfdrv
      implicit none
      real a1,a2,a3,a4
      real xsigma_t,xsigma_l,xsigma_tt,xsigma_lt
      external xsigma_t,xsigma_l,xsigma_tt,xsigma_lt
      logical xcheck_kine,check_kine
      external xcheck_kine,check_kine
      real tminq
      external tminq
      real mesonmass
      integer mode,heli
      real s0,st,sl,stt,slt,sltp
      real del2,xb,jacg,jacr

      mesonmass = 0.134976
      call xsinit(mesonmass)
      heli = 1

      read(*,*) mode

 10   continue
      read(*,*,end=99) a1,a2,a3,a4
      if (a1.gt.900.) goto 99
      if (mode.eq.1) then
         if (xcheck_kine(a1,a2,a3,a4)) then
            write(*,'(5(1x,1pe24.16))')
     *           xsigma_t(a1,a2,a3,a4), xsigma_l(a1,a2,a3,a4),
     *           xsigma_tt(a1,a2,a3,a4), xsigma_lt(a1,a2,a3,a4),
     *           tminq(a3,a2)
         else
            write(*,'(5(1x,1pe24.16))') 0.,0.,0.,0.,tminq(a3,a2)
         endif
      else
         call dvmpw(a1,a2,a3,0.0,a4,heli,mesonmass,
     *              s0,st,sl,stt,slt,sltp)
         if (check_kine(a1,a2,a3,a4,del2,xb,jacg,jacr)) then
            continue
         else
            del2 = 0.
            xb   = 0.
         endif
         write(*,'(6(1x,1pe24.16))') st,sl,stt,slt,del2,xb
      endif
      goto 10

 99   continue
      end
