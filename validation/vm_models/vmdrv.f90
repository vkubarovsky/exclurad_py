!     Dump the diffrad_vm vector-meson model (two structure functions)
!     so the Python ports can be verified against it.
      program vmdrv
      implicit real*8(a-h,o-z)
      common/rhomodel/rho_AT,rho_bT,rho_AL,rho_bL
      common/phimodel/phi_alf1,phi_alf2,phi_alf3,phi_nuT,phi_bt,phi_cR
      common/jpsimodel/pj_alf1,pj_alf2,pj_alf3,pj_nuT,pj_mg2,pj_cR
      real*8 sigma_T_rho,sigma_L_rho,sigma_T_phi,sigma_L_phi
      real*8 sigma_T_jpsi,sigma_L_jpsi
      external sigma_T_rho,sigma_L_rho,sigma_T_phi,sigma_L_phi
      external sigma_T_jpsi,sigma_L_jpsi
!     table defaults, exactly as the main program sets them
      rho_AT=15d0; rho_bT=2.0d0; rho_AL=25d0; rho_bL=4.0d0
      phi_alf1=400d0; phi_alf2=2.000d0; phi_alf3=0.200d0
      phi_nuT=2.253d0; phi_bt=1.326d0; phi_cR=1.000d0
      pj_alf1=2.33d-8; pj_alf2=0.500d0; pj_alf3=7.800d0
      pj_nuT=3.114d0;  pj_mg2=2.637d0;  pj_cR=0.400d0
 10   read(*,*,end=99) q2,xb,t
      if(q2.gt.900d0) goto 99
      write(*,'(6(1x,1pe24.16))') &
        sigma_T_rho(q2,xb,t), sigma_L_rho(q2,xb,t), &
        sigma_T_phi(q2,xb,t), sigma_L_phi(q2,xb,t), &
        sigma_T_jpsi(q2,xb,t), sigma_L_jpsi(q2,xb,t)
      goto 10
 99   continue
      end
