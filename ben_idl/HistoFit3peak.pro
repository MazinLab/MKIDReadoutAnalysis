FUNCTION TwinGaussDIFF, p, X=x, Y=y, ERR=err
  ;p[0] = center of line 1
  ;p[1] = width of line 1
  ;p[2] = height of line 1
  ;p[3] = center of line 2
  ;p[4] = width of line 2
  ;p[5] = height of line 2
  ;p[6] = center of line 3
  ;p[7] = width of line 3
  ;p[8] = height of line 3

  z1 = (x-p[0])/p[1]
  z2 = (x-p[3])/p[4]  
  z3 = (x-p[6])/p[7]  
  f = p[2]*exp(-z1^2/2.0) + p[5]*exp(-z2^2/2.0) + p[8]*exp(-z3^2/2.0)
  dev = (y-f)/err
 
return, dev
end

; fit the optimally filtered pulse data
PRO HistoFit3peak,datapath,outpath,pulsename,StartT,atten,res,pm,phr

path=outpath
filename = 'pulses.dat'
data = read_ascii(path+filename)
data = data.field01

device,filename=path+'histfit.ps'
!p.multi=0

bs = 0.5

; Loop and repeat for ch1 and ch2
for i=0,3 do begin

;if(i EQ 1 OR i EQ 3) then continue

if( i EQ 0 ) then begin 
  ch1 = data[4,*]  ;parabolic fit data, ch1
  ;ch1 = data[4,*]  ;shift fit data, ch1
  chisq = data[8,*] ; chisq
  Pmax = pm[0]
  Psd = pm[1]
  histzr = fix(pm[5]*1.0*180.0/!Pi/bs)
endif

if( i EQ 1 ) then begin 
  ch1 = data[5,*]  ;parabolic fit data, ch2
  ;ch1 = data[5,*]  ;shift fit data, ch2
  chisq = data[9,*] ; chisq
  Pmax = pm[2]
  Psd = pm[3]
  histzr = fix(pm[4]*1.0*180.0/!Pi/bs)
endif

if( i EQ 2 ) then begin 
  ch1 = data[6,*]  ;parabolic fit data, ch1
  ;ch1 = data[4,*]  ;shift fit data, ch1
  chisq = data[8,*] ; chisq
  Pmax = pm[0]
  Psd = pm[1]
  histzr = fix(pm[5]*1.0*180.0/!Pi/bs)
endif

if( i EQ 3 ) then begin 
  ch1 = data[7,*]  ;parabolic fit data, ch2
  
  ;ch1 = data[5,*]  ;shift fit data, ch2
  chisq = data[9,*] ; chisq
  
  Pmax = pm[2]
  Psd = pm[3]
  histzr = fix(pm[4]*1.0*180.0/!Pi/bs)
endif

; screen for chisq
for j=0,n_elements(chisq)-1 do begin
  ;if( i EQ 2 AND chisq[j] LE 1.2 ) then ch1[j] = 0.0
  ;if( i EQ 1 AND chisq[j] GE 1.5 ) then ch1[j] = 0.0
  ;if( ( chisq[j] GE 1.2 OR chisq[j] LE 0.8) ) then ch1[j] = 0.0
endfor

hist = histogram( ch1, MIN = 0, MAX = 300, BINSIZE = bs)
;if( histzr GE n_elements(hist) ) then return
if( histzr GE n_elements(hist) ) then histzr=0

;hist[0:histzr] = 0.0 ; remove false trigger tail
hist[0:(Pmax*0.18)/bs] = 0.0 ; remove false trigger tail

bins = FINDGEN(N_ELEMENTS(hist))*bs
plot, bins,hist,psym=10,/ystyle,yr=[0,max(hist)*1.1],xr=[0,Pmax*2.0],/xstyle,xtitle='Pulse Height (degrees)',title=strcompress('Channel' + string(((i mod 2)+1)))

x = double(bins)
y = double(hist)
err = replicate(4.0, (size(x))[1] )

parinfo = replicate({value:0.D, fixed:0, step:0, limited:[1,1], limits:[0.D,0.D],mpmaxstep:0.D}, 9)

;p[0] = center of line 1
;p[1] = width of line 1, sigma
;p[2] = height of line 1
;p[3] = center of line 2
;p[4] = width of line 2
;p[5] = height of line 2
;p[6] = center of line 3
;p[7] = width of line 3
;p[8] = height of line 3

Pmax1 = Pmax
;Pmax2 = Pmax*0.71
;Pmax3 = Pmax*0.52
Pmax2 = Pmax*0.57
Pmax3 = Pmax*0.4


parinfo[0].value = Pmax1
;parinfo[0].value = 155.0
;parinfo[0].limits  = [140.0,160.0]
parinfo[0].limits  = [Pmax1-10.0,Pmax1+10.0]

;parinfo[1].value = Psd
parinfo[1].value = 1.0
parinfo[1].limits  = [0.5,10.0]

;parinfo[2].value = max(hist)
parinfo[2].value = 60.0
;parinfo[2].limits  = [max(hist)/10.0,max(hist)*3.0]
parinfo[2].limits  = [10.0,200.0]

parinfo[3].value = Pmax2
;parinfo[3].value = 95.0
;parinfo[3].limits  = [85.0,105.0]
parinfo[3].limits  = [Pmax2-10.0,Pmax2+10.0]

;parinfo[4].value = Psd*2.0
parinfo[4].value = 1.0
parinfo[4].limits  = [0.8,10.0]

parinfo[5].value = 50.0
parinfo[5].limits  = [10.0,200.0]

parinfo[6].value = Pmax3
;parinfo[6].value = 55.0
;parinfo[6].limits  = [45.0,65.0]
parinfo[6].limits  = [Pmax3-10.0,Pmax3+10.0]

;parinfo74].value = Psd*2.0
parinfo[7].value = 1.0
parinfo[7].limits  = [0.8,10.0]

parinfo[8].value = 110.0
parinfo[8].limits  = [20.0,200.0]

; Do the optimization
fa = {X:x, Y:y, ERR:err}
bestnorm=0.0
covar=0.0
perror=0.0

p = mpfit('TwinGaussDIFF',functargs=fa,BESTNORM=bestnorm,COVAR=covar,PARINFO=parinfo,PERROR=perror,AUTODERIVATIVE=1,FTOL=1D-18,XTOL=1D-18,GTOL=1D-18,FASTNORM=0,STATUS=status,/QUIET)

print,'Status of Fit= ',status
print,bestnorm

DOF     = N_ELEMENTS(X)*2.0 - N_ELEMENTS(PARMS) ; deg of freedom
PCERROR = PERROR * SQRT(BESTNORM / DOF)   ; scaled uncertainties

bestp = p
bn = bestnorm

; monte carlo fit parameters
for j=0,19 do begin
  parinfo[0].value = Pmax1 + randomu(seed)*5.0
  if( parinfo[0].value LE 5.0) then parinfo[0].value=5.0

  parinfo[3].value = Pmax2 + randomu(seed)*5.0
  if( parinfo[3].value LE 5.0) then parinfo[3].value=5.0  
  
  parinfo[6].value = Pmax3 + randomu(seed)*5.0
  if( parinfo[6].value LE 5.0) then parinfo[6].value=5.0
  
  ;parinfo[1].value = 2.0 + randomu(seed)*2.0
  ;parinfo[2].value = max(hist)*randomu(seed)*2.0
 
;  fa = {X:x, Y:y, ERR:err}
  bestnorm=0.0
  covar=0.0
  perror=0.0
  p = mpfit('TwinGaussDIFF',functargs=fa,BESTNORM=bestnorm,COVAR=covar,PARINFO=parinfo,PERROR=perror,AUTODERIVATIVE=1,FTOL=1D-18,XTOL=1D-18,GTOL=1D-18,FASTNORM=0,STATUS=status,/QUIET)
  ;print,bestnorm,status,p[0]
  if( status NE 0 AND bn GT bestnorm) then begin
    bn = bestnorm
    bestp = p
  endif

endfor

p = bestp

z1 = (bins-p[0])/p[1]
z2 = (bins-p[3])/p[4]  
z3 = (bins-p[6])/p[7]
f = p[2]*exp(-z1^2/2.0) + p[5]*exp(-z2^2/2.0) + p[8]*exp(-z3^2/2.0)

oplot,bins,f,line=0,color=150

r1 = string(p[0]/(p[1]*2.355), format='(F6.2)' )
r2 = string(p[3]/(p[4]*2.355), format='(F6.2)' )
r3 = string(p[6]/(p[7]*2.355), format='(F6.2)' )
c1 =  string(p[0], format='(F6.2)' )
c2 =  string(p[3], format='(F6.2)' )
c3 =  string(p[6], format='(F6.2)' )
h1 =  string(p[2], format='(F6.2)' )
h2 =  string(p[5], format='(F6.2)' )
h3 =  string(p[8], format='(F6.2)' )

al_legend,/top,/right,[strcompress('R1 = ' + r1),strcompress('Center1 = ' + c1),strcompress('Height1 = ' + h1),strcompress('R2 = ' + r2),strcompress('Center2 = ' + c2),strcompress('Height2 = ' + h2),strcompress('R3 = ' + r3),strcompress('Center3 = ' + c3),strcompress('Height3 = ' + h3)]
;print,r1,r2

phr[i] = p[0]

endfor

device,/close

end