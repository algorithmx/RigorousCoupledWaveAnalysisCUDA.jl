using RigorousCoupledWaveAnalysis,LinearAlgebra
using Test
include("analytical.jl")
@testset "VerticalIncidence" begin
n1=rand()*10+10im*rand()
n2=rand()*10+10im*rand()
mdl=RCWAModel([],ConstantPerm(n1^2),ConstantPerm(n2^2))
grd=rcwagrid(0,0,100,100,1E-3,rand()*360,1000,ConstantPerm(n1^2))
ψte,~=rcwasource(grd,n1)
R,T=srcwa_reftra(ψte,mdl,grd,1000)
R0,~,T0,~=fresnelPower(n1,n2,0)
@test isapprox(R,R0,atol=1E-5)
@test isapprox(T,T0,atol=1E-5)
@test isapprox(R+T,1,atol=1E-5)
end

@testset "GrazingIncidence" begin
n1=rand()*10+10im*rand()
n2=n1+rand()*10+10im*rand()
θ=1+88rand()
mdl=RCWAModel([],ConstantPerm(n1^2),ConstantPerm(n2^2))
grd=rcwagrid(0,0,100,100,θ,rand()*360,1000,ConstantPerm(n1^2))
ψte,ψtm=rcwasource(grd,n1)
Rte,Tte=srcwa_reftra(ψte,mdl,grd,1000)
Rtm,Ttm=srcwa_reftra(ψtm,mdl,grd,1000)
R0s,R0p,T0s,T0p=fresnelPower(n1,n2,θ)
@test isapprox(Rte,R0s,atol=1E-5)
@test isapprox(Tte,T0s,atol=1E-5)
@test isapprox(Rtm,R0p,atol=1E-5)
@test isapprox(Ttm,T0p,atol=1E-5)
@test isapprox(Rte+Tte,1,atol=1E-5)
@test isapprox(Rtm+Ttm,1,atol=1E-5)
end

@testset "Fabryperot" begin
n1=1+1rand()+1im*rand()
n2=n1+rand()+1im*rand() #n2>n1 against TIR
n3=n1+rand()+1im*rand() #n3>n1 against TIR
θ=88rand()+1
λ=1000rand()
thickness=100rand()
#θ2=asind.(real.(n1)./real.(n2).*sind.(θ))
#thickness=λ/real(n2)/cosd(θ2)
mdl=RCWAModel([SimpleLayer(thickness,ConstantPerm(n2^2))],ConstantPerm(n1^2),ConstantPerm(n3^2))
grd=rcwagrid(0,0,100,100,θ,rand()*360,λ,ConstantPerm(n1^2))
ψte,ψtm=rcwasource(grd)
Rte,Tte=srcwa_reftra(ψte,mdl,grd,λ)
Rtm,Ttm=srcwa_reftra(ψtm,mdl,grd,λ)
R0s,R0p,T0s,T0p=fabryperot(n1,n2,n3,2π/λ,thickness,θ)
@test isapprox(Rte,R0s,atol=1E-5)
@test isapprox(Rtm,R0p,atol=1E-5)
@test isapprox(Tte,T0s,atol=1E-5)
@test isapprox(Ttm,T0p,atol=1E-5)
end

@testset "unitySource" begin
n1=1+10rand()+10rand()*1im
λ=1000rand()
θ=88rand()+1
α=360rand()
grd=rcwagrid(0,0,100rand(),100rand(),θ,α,λ,ConstantPerm(n1^2))
ste,stm=rcwasource(grd,real(n1))
ref=RigorousCoupledWaveAnalysis.halfspace(grd.Kx,grd.Ky,ConstantPerm(n1^2),λ)
P1=-RigorousCoupledWaveAnalysis.a2p(ste,0ste,ref.V,I,grd.k0[3])
P2=-RigorousCoupledWaveAnalysis.a2p(stm,0ste,ref.V,I,grd.k0[3])
@test P1≈1
@test P2≈1
end

@testset "RT_ScatVSETM_simple" begin
eps1=10rand()+10im*rand()
eps2=10rand()+10im*rand()
eps3=10rand()+10im*rand()
eps4=10rand()+10im*rand()
λ=1000rand()
mdl=RCWAModel([PatternedLayer(100rand(),[ConstantPerm(eps2),ConstantPerm(eps3)],[Circle(rand())])],ConstantPerm(eps1),ConstantPerm(eps4))
grd=rcwagrid(1,1,100rand(),100rand(),88rand()+.1,360rand(),λ,ConstantPerm(eps1))
ste,stm=rcwasource(grd,real(√eps1))
Rte1,Tte1=etm_reftra(ste,mdl,grd,λ) 
Rtm1,Ttm1=etm_reftra(stm,mdl,grd,λ) 
Rte2,Tte2=srcwa_reftra(ste,mdl,grd,λ) 
Rtm2,Ttm2=srcwa_reftra(stm,mdl,grd,λ) 
@test isapprox(Rte1,Rte2,atol=1E-5)
@test isapprox(Rtm1,Rtm2,atol=1E-5)
@test isapprox(Tte1,Tte2,atol=1E-5)
@test isapprox(Ttm1,Ttm2,atol=1E-5)
end


@testset "RT_ScatVSETM_conservation_simple" begin
eps1=10rand()+10im*rand()
eps2=10rand()
eps3=10rand()
eps4=10rand()+10im*rand()
λ=1000rand()
mdl=RCWAModel([PatternedLayer(100rand(),[ConstantPerm(eps2),ConstantPerm(eps3)],[Circle(rand())])],ConstantPerm(eps1),ConstantPerm(eps4))
grd=rcwagrid(1,1,100rand(),100rand(),88rand()+.1,360rand(),λ,ConstantPerm(eps1))
ste,stm=rcwasource(grd,real(√eps1))
Rte1,Tte1=etm_reftra(ste,mdl,grd,λ) 
Rtm1,Ttm1=etm_reftra(stm,mdl,grd,λ) 
Rte2,Tte2=srcwa_reftra(ste,mdl,grd,λ) 
Rtm2,Ttm2=srcwa_reftra(stm,mdl,grd,λ) 
@test isapprox(Rte1,Rte2,atol=1E-5)
@test isapprox(Rtm1,Rtm2,atol=1E-5)
@test isapprox(Tte1,Tte2,atol=1E-5)
@test isapprox(Ttm1,Ttm2,atol=1E-5)
@test Rte1+Tte1≈1
@test Rtm1+Ttm1≈1
@test Rte2+Tte2≈1
@test Rtm2+Ttm2≈1
end



@testset "ETM_algorithm_with_GPU_ma2018" begin
#required materials
ge=InterpolPerm(RigorousCoupledWaveAnalysis.ge_nunley) #Ge from interpolated measured values
ox=ModelPerm(RigorousCoupledWaveAnalysis.sio2_malitson) #SiO2 from dispersion formula
air=ConstantPerm(1.0) #superstrate material is air
#parameters of structure and kgrid
N=4 #accuracy
wls=1300:100:2000.0 #wavelength axis, 20x step
a=900.0 #cell size
lmid=615/a #length of main arm
larm=410/a #length of side arms
w=205/a #width of arms
function zshapegeo(lmid,larm,w) #parametrized z-shaped unit cell
    cent=Rectangle(w,lmid) #center arm
    top=Shift(Rectangle(larm-w,w),.5larm,lmid/2-w/2) #top arm
    bottom=Shift(Rectangle(larm-w,w),-.5larm,-lmid/2+w/2) #bottom arm
    return Combination([cent,top,bottom]) #combine them
end
#forward propagation
geo=zshapegeo(lmid,larm,w) #call the method
act=PatternedLayer(500,[air,ge],[geo]) #the active layer is air with Ge structures, 500 nm thick
mdl=RCWAModel([act],air,ox) #build the model, with superstrate and substrate material
Rrf=zeros(length(wls)) #Forward rcp reflectivity
Trf=zeros(length(wls)) #Forward rcp transmissivity
Rlf=zeros(length(wls)) #Forward lcp reflectivity
Tlf=zeros(length(wls));#Forward lcp transmissivity
test_mode = true
# iterate over all wavelengths
for i=1:length(wls)
    λ       = wls[i] # wavelength
    grd     = rcwagrid(N,N,a,a,1E-5,0,λ,air;use_gpu=true) # build a reciprocal space grid
    ste,stm = rcwasource(grd,1) #define source
    Rlf[i], Tlf[i] = etm_reftra(sqrt(.5)*(stm+1im*ste),mdl,grd,λ) #lcp propagation
    Rrf[i], Trf[i] = etm_reftra(sqrt(.5)*(1im*stm+ste),mdl,grd,λ) #rcp propagation
    if test_mode
        grd          = rcwagrid(N,N,a,a,1E-5,0,λ,air;use_gpu=false) # build a reciprocal space grid
        ste,stm      = rcwasource(grd,1) #define source
        Rlf_i, Tlf_i = etm_reftra(sqrt(.5)*(stm+1im*ste),mdl,grd,λ) #lcp propagation
        Rrf_i, Trf_i = etm_reftra(sqrt(.5)*(1im*stm+ste),mdl,grd,λ) #rcp propagation
        @test Rlf[i]≈Rlf_i
        @test Tlf[i]≈Tlf_i
        @test Rrf[i]≈Rrf_i
        @test Trf[i]≈Trf_i
    end
end
#now invert it
function zshapegeo(lmid,larm,w) #parametrized z-shaped unit cell
    cent=Rectangle(w,lmid) #center arm
    top=Shift(Rectangle(larm-w,w),.5larm,-lmid/2+w/2) #top arm
    bottom=Shift(Rectangle(larm-w,w),-.5larm,lmid/2-w/2) #bottom arm
    return Combination([cent,top,bottom]) #combine them
end
#backward propagation, flip device
geo=zshapegeo(lmid,larm,w) #inverted structure
act=PatternedLayer(500,[air,ge],[geo])
mdl=RCWAModel([act],ox,air) #model is now inverted
Rrb=zeros(length(wls))#Backward rcp reflectivity
Trb=zeros(length(wls))#Backward rcp transmissivity
Rlb=zeros(length(wls))#Backward lcp reflectivity
Tlb=zeros(length(wls))#Backward lcp transmissivity
for i=1:length(wls)
    λ=wls[i]
    grd=rcwagrid(N,N,a,a,1E-5,0,λ,air;use_gpu=true)
    ste,stm=rcwasource(grd,1)
    Rlb[i],Tlb[i]=etm_reftra(sqrt(.5)*(stm+1im*ste),mdl,grd,λ)
    Rrb[i],Trb[i]=etm_reftra(sqrt(.5)*(1im*stm+ste),mdl,grd,λ)
    if test_mode
        grd=rcwagrid(N,N,a,a,1E-5,0,λ,air;use_gpu=false)
        ste,stm=rcwasource(grd,1)
        Rlb_i,Tlb_i=etm_reftra(sqrt(.5)*(stm+1im*ste),mdl,grd,λ)
        Rrb_i,Trb_i=etm_reftra(sqrt(.5)*(1im*stm+ste),mdl,grd,λ)
        @test Rlb[i]≈Rlb_i
        @test Tlb[i]≈Tlb_i
        @test Rrb[i]≈Rrb_i
        @test Trb[i]≈Trb_i
    end
end
end


@testset "ETM_algorithm_with_GPU_augel2018" begin
Si=InterpolPerm(RigorousCoupledWaveAnalysis.si_schinke) #Si from interpolated literature values
Ge=InterpolPerm(RigorousCoupledWaveAnalysis.ge_nunley) #Ge from interpolated literature values
SiO2=ModelPerm(RigorousCoupledWaveAnalysis.sio2_malitson) #SiO2 from literature dispersion formula
n_H2O=1.321
n_CH3COOH=1.353 #Constant refractive indices
Al=ModelPerm(RigorousCoupledWaveAnalysis.al_rakic) #Al from dispersion formula
N=6 #one needs much larger N (~11 is good, 15 is better) here for accurate results
wls=1100:100:1600 #wavelength array, 20x step size
p=950 #pitch
d=480 #hole diameter
function build_model(n_sup)
    nha=PatternedLayer(100,[Al,ConstantPerm(n_sup^2)],[Circle(d/p)])#patterned NHA layer
    spa=SimpleLayer(50,SiO2)
    nsi=SimpleLayer(20,Si)
    nge=SimpleLayer(20,Ge)
    ige=SimpleLayer(480,Ge)
    return RCWAModel([nha,spa,nsi,nge,ige],ConstantPerm(n_sup^2),Si)
end
A_H2O=zeros(size(wls)) #array for absorption
R_H2O=zeros(size(wls)) #array for reflection
T_H2O=zeros(size(wls)) #array for transmission
R_CH3COOH=zeros(size(wls))
T_CH3COOH=zeros(size(wls))
A_CH3COOH=zeros(size(wls))
test_mode = true
for i=1:length(wls)
    λ=wls[i] #get wavelength from array
    grd=rcwagrid(N,N,p,p,1E-5,0,λ,ConstantPerm(n_H2O^2); use_gpu=true) #reciprocal space grid
    ste,stm=rcwasource(grd) #te and tm source amplitudes
    #compute for H2O
    # compute ref, tra and power flows for te
    R_H2O[i],T_H2O[i],flw=etm_reftra_flows(ste,build_model(n_H2O),grd,λ) 
    # absorption is the power entering the last layer minus the power leaving the device
    if test_mode
        grd=rcwagrid(N,N,p,p,1E-5,0,λ,ConstantPerm(n_H2O^2); use_gpu=false) #reciprocal space grid
        ste,stm=rcwasource(grd) #te and tm source amplitudes
        #compute for H2O
        # compute ref, tra and power flows for te
        R_H2O_i,T_H2O_i,flw_c=etm_reftra_flows(ste,build_model(n_H2O),grd,λ) 
        # absorption is the power entering the last layer minus the power leaving the device
        @test R_H2O_i ≈ R_H2O[i]
        @test T_H2O_i ≈ T_H2O[i]
        @test flw_c ≈ flw
    end
    #now same for CH3COOH
    grd=rcwagrid(N,N,p,p,1E-5,0,λ,ConstantPerm(n_CH3COOH^2); use_gpu=true) #reciprocal space grid
    ste,stm=rcwasource(grd) #te and tm source amplitudes
    R_CH3COOH[i],T_CH3COOH[i],flw=etm_reftra_flows(ste,build_model(n_CH3COOH),grd,λ)
    if test_mode
        grd=rcwagrid(N,N,p,p,1E-5,0,λ,ConstantPerm(n_CH3COOH^2); use_gpu=false)
        ste,stm=rcwasource(grd)
        R_CH3COOH_i,T_CH3COOH_i,flw_c=etm_reftra_flows(ste,build_model(n_CH3COOH),grd,λ)
        @test R_CH3COOH_i ≈ R_CH3COOH[i]
        @test T_CH3COOH_i ≈ T_CH3COOH[i]
        @test flw_c ≈ flw
    end
end # for
end #@test
