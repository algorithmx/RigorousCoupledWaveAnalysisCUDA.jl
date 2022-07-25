using RigorousCoupledWaveAnalysisCUDA
using CUDA

function performance_test(N; use_gpu=true)
    println("Timing for N=$(N)")
    #required materials
    ge=InterpolPerm(RigorousCoupledWaveAnalysisCUDA.ge_nunley) #Ge from interpolated measured values
    ox=ModelPerm(RigorousCoupledWaveAnalysisCUDA.sio2_malitson) #SiO2 from dispersion formula
    air=ConstantPerm(1.0) #superstrate material is air
    #parameters of structure and kgrid
    wls=1400:100:2000.0 #wavelength axis
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
    Tlf=zeros(length(wls)) #Forward lcp transmissivity
    for i = 1:length(wls) #iterate over all wavelengths
        λ=wls[i] #get wavelength from array
        grd=rcwagrid(N,N,a,a,1E-5,0,λ,air; use_gpu=use_gpu) #build a reciprocal space grid
        ste,stm=rcwasource(grd,1) #define source
        Rlf[i],Tlf[i]=etm_reftra(sqrt(.5)*(stm+1im*ste),mdl,grd,λ) #lcp propagation
        Rrf[i],Trf[i]=etm_reftra(sqrt(.5)*(1im*stm+ste),mdl,grd,λ) #rcp propagation
    end
end

CUDA.@time performance_test(4; use_gpu=false)
CUDA.@time performance_test(4; use_gpu=true)

CUDA.@time performance_test(6; use_gpu=false)
CUDA.@time performance_test(6; use_gpu=true)

CUDA.@time performance_test(8; use_gpu=false)
CUDA.@time performance_test(8; use_gpu=true)

CUDA.@time performance_test(10; use_gpu=false)
CUDA.@time performance_test(10; use_gpu=true)

CUDA.@time performance_test(12; use_gpu=false)
CUDA.@time performance_test(12; use_gpu=true)


0


#=

Timing for N=4
  2.074030 seconds (68.29 k CPU allocations: 549.457 MiB, 1.85% gc time)

Timing for N=4
  1.500811 seconds (47.49 k CPU allocations: 140.397 MiB, 1.02% gc time) (1.39 k GPU allocations: 375.357 MiB, 0.65% memmgmt time)

Timing for N=6
  9.619789 seconds (128.01 k CPU allocations: 2.252 GiB, 0.84% gc time), 0.00% GPU memmgmt time

Timing for N=6
  7.054304 seconds (49.93 k CPU allocations: 583.366 MiB, 0.30% gc time) (1.39 k GPU allocations: 1.570 GiB, 0.17% memmgmt time)

Timing for N=8
 43.713885 seconds (208.36 k CPU allocations: 6.509 GiB, 1.98% gc time), 0.00% GPU memmgmt time

Timing for N=8
 25.613362 seconds (51.25 k CPU allocations: 1.638 GiB, 0.20% gc time) (1.39 k GPU allocations: 4.575 GiB, 0.06% memmgmt time)

Timing for N=10
161.055947 seconds (310.71 k CPU allocations: 15.023 GiB, 1.47% gc time), 0.00% GPU memmgmt time

Timing for N=10
 61.472516 seconds (51.96 k CPU allocations: 3.785 GiB, 0.14% gc time) (1.39 k GPU allocations: 10.642 GiB, 0.04% memmgmt time)

Timing for N=12
485.634121 seconds (435.24 k CPU allocations: 30.132 GiB, 0.59% gc time), 0.00% GPU memmgmt time

Timing for N=12
139.412966 seconds (52.22 k CPU allocations: 7.570 GiB, 0.10% gc time) (1.39 k GPU allocations: 21.365 GiB, 0.03% memmgmt time)

=#


using Plots

scale = [4,6,8,10,12]
timing_cpu = [2.074030, 9.619789, 43.713885, 161.055947, 485.634121]
timing_gpu = [1.500811, 7.054304, 25.613362,  61.472516, 139.412966]

plot(scale, timing_cpu, color="black", lw=2, labels=nothing)
scatter!(scale, timing_cpu, color="black", labels="CPU", legend=:topleft)
plot!(scale, timing_gpu, color="lightgreen", lw=2, labels=nothing)
scatter!(scale, timing_gpu, color="lightgreen", lw=2, labels="GPU")
xlabel!("Precision")
ylabel!("Time (seconds)")
savefig("performance.png")