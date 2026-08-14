      program drv
      implicit none
      real del2,xb,q2,e
      real xsigma_t,xsigma_l,xsigma_tt,xsigma_lt
      external xsigma_t,xsigma_l,xsigma_tt,xsigma_lt
      call xsinit(0.5473)
 10   read(*,*,end=99) del2,xb,q2,e
      if (del2.gt.900.) goto 99
      write(*,'(4(1x,1pe24.16))') xsigma_t(del2,xb,q2,e),
     *  xsigma_l(del2,xb,q2,e), xsigma_tt(del2,xb,q2,e),
     *  xsigma_lt(del2,xb,q2,e)
      goto 10
 99   end
