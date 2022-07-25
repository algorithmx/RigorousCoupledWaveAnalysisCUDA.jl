# Enhanced transmission matrix algorithm by Moharam et al.
module ETM
using LinearAlgebra
using SparseArrays
using CUDA
using ..Common
export etm_reftra, etm_amplitudes, etm_propagate,etm_flow, etm_reftra_flows
#utility method for dealing with eigenmodes
@inline F(em) = [em.W   em.W;  em.V   -em.V]
@inline showcumat(A,name="") = println("name = $(name) , size = $(A.dims) , offset = $(A.offset) , maxsize = $(A.maxsize)\nstorage = $(A.storage)")
"""
    etm_propagate(sup,sub,em,ψin)
    etm_propagate(sup,sub,em,ψin,get_r)
The propagation of waves according to the ETM method
# Arguments
* `sup` :  superstrate halfspace object
* `sub` :  substrate halfspace object
* `em` :  array with eigenmodes for all layers
* `ψin` :  incoming (source) amplitude vector
* `get_r` : set false if you do not need the internal backward propagating waves (only required for absorption calculation)
# Outputs
* `ψref` : reflected amplitude vector
* `ψtra` : transmitted amplitude vector
* `ψp` : array of internal back propagating wave vectors in each layer
* `ψm` : array of internal forward propagating wave vectors in each layer
"""
function etm_propagate(
    sup::Halfspace,
    sub::Halfspace,
    em::Vector{Eigenmodes}, 
    sp_ψin::SparseVector,
    get_r::Bool=true
    )
    etm_propagate(sup, sub, em, Vector(sp_ψin), get_r)
end
function etm_propagate(
    sup::Halfspace,
    sub::Halfspace,
    em::Vector{Eigenmodes}, 
    ψin::Vector,
    get_r::Bool=true
    )
    @assert ! (sup.V isa CuArray)
    @assert ! (sub.V isa CuArray)
    @assert all([!(emi.X isa CuArray) for emi in em])

    T = eltype(ψin)

    ψm = Array{Array{T,1},1}(undef,length(em))
    ψp = Array{Array{T,1},1}(undef,length(em))
    if length(em)>0
        # 1. [backward iteration]
        a = Array{Array{T,2},1}(undef,length(em)) 
        b = Array{Array{T,2},1}(undef,length(em))
        # transmission matrix for a wave in the last layer into the substrate
        a[end],b[end] = slicehalf(F(em[end])\Matrix([I;-sub.V])) 
        for cnt = length(em):-1:2  # [backward iteration]
            # successively compute the transmission matrix from the i-th layer into the substrate
            a[cnt-1],b[cnt-1] = slicehalf(F(em[cnt-1])\F(em[cnt])*[em[cnt].X*(a[cnt]/b[cnt])*em[cnt].X ; I])
        end
        # 2. [forward iteration]
        # compute the reflected wave and the forward wave in the first layer
        # bottleneck
        ψref,ψm1 = slicehalf( -cat([I;sup.V],F(em[1])*[em[1].X*(a[1]/b[1])*em[1].X;I],dims=2) \ ([I;-sup.V]*ψin) )
        ψm[1] = vec(ψm1) 
        for cnt = 1:length(em)-1  # [forward iteration]
            # successively compute the forward wave in the i-th layer
            ψm[cnt+1] = b[cnt]\I*(em[cnt].X*ψm[cnt]) 
            # if required, compute the backward wave in the i-th layer
            get_r && (ψp[cnt] = em[cnt].X*a[cnt]*ψm[cnt+1]) 
        end
        # if required, computethe backward wave in the first layer
        get_r && (ψp[1] = em[1].X*(a[1]/b[1])*em[1].X*ψm[1]) 
        # compute the transmitted wave from the forward wave in the last layer
        ψtra = b[end]\I*em[end].X*ψm[end] 
        # compute the backward wave in the last layer, if required
        get_r && (ψp[end] = em[end].X*a[end]*ψtra) 
    else 
        # the case for "empty" model (single interface, no layers)
        ψref,ψtra = slicehalf([-I I;sup.V sub.V]\[I*ψin;sup.V*ψin])
    end
    #TODO normalize this:  size of ψref, ψtra are not the same but equivalent   
    return ψref,ψtra,ψp,ψm
end
function etm_propagate(
    sup::Halfspace,
    sub::Halfspace,
    ems_gpu::Vector{Eigenmodes}, 
    cu_ψin::CuArray,
    get_r=true
    )
    @assert sup.V isa CuArray
    @assert sub.V isa CuArray
    @assert all([(emi.X isa CuArray) for emi in ems_gpu])

    T = eltype(cu_ψin)

    dim_block = size(sup.V,1)
    @assert dim_block==size(sup.V,2)==size(sub.V,1)==size(sub.V,2)==length(cu_ψin)
    @assert all(size(em.X)==(dim_block,dim_block) for em in ems_gpu)

    cuI   = CuArray(Diagonal{T}(I,dim_block))

    if length(ems_gpu)==0
        CUDA.synchronize()
        # the case for "empty" model (single interface, no layers)
        ψref, ψtra = slicehalf([-cuI  cuI; sup.V  sub.V] \ [cu_ψin; sup.V * cu_ψin])
    else
        # temporary vector
        _b = [-cu_ψin; sup.V*cu_ψin]
        CUDA.synchronize()

        # 0> [initialize]
        ab = Vector{CuArray{T,2}}(undef,length(ems_gpu))
        # transmission matrix for a wave in the last layer into the substrate
        ab[end] = F(ems_gpu[end]) \ vcat(cuI,-sub.V)

        # 1> [backward iteration]
        for cnt = length(ems_gpu):-1:2  # [backward iteration]
            # successively compute the transmission matrix from the i-th layer into the substrate
            ab[cnt-1] = F(ems_gpu[cnt-1]) \ (F(ems_gpu[cnt])*[ems_gpu[cnt].X*a_over_b(ab[cnt])*ems_gpu[cnt].X ; cuI])
        end

        # 2> [intermediate step]
        # compute the reflected wave and the forward wave in the first layer
        # CUDA version : Array(CuArray(A) \ CuArray(b))
        # bottleneck #2, optimized
        # TODO align the vector [-cu_ψin; v_tmp] and the vector [cu_ψin; v_tmp]
        # in the length(ems_gpu)==0 such that the minus sign agrees
        tmp = F(ems_gpu[1])*vcat(ems_gpu[1].X*a_over_b(ab[1])*ems_gpu[1].X, cuI)
        ψref, ψm1 = slicehalf(( cat(vcat(cuI,sup.V), tmp, dims=2) \ _b ))

        # containers
        ψm = Vector{CuVector{Complex{Float64}}}(undef,length(ems_gpu)) 
        ψp = Vector{CuVector{Complex{Float64}}}(undef,length(ems_gpu))
        ψm[1] = vec(ψm1) # preserves location

        # 3> [forward iteration]
        for cnt = 1:length(ems_gpu)-1  # [forward iteration]
            # successively compute the forward wave in the i-th layer
            ψm[cnt+1] = take_b(ab[cnt]) \ (ems_gpu[cnt].X*ψm[cnt])
            # if required, compute the backward wave in the i-th layer
            get_r && (ψp[cnt] = ems_gpu[cnt].X*take_a(ab[cnt])*ψm[cnt+1])
        end
        # if required, computethe backward wave in the first layer
        get_r && (ψp[1] = ems_gpu[1].X*a_over_b(ab[1])*ems_gpu[1].X*ψm[1])
        # compute the transmitted wave from the forward wave in the last layer
        ψtra = take_b(ab[end]) \ (ems_gpu[end].X*ψm[end])
        # compute the backward wave in the last layer, if required
        get_r && (ψp[end] = ems_gpu[end].X*take_a(ab[end])*ψtra)

    end

    CUDA.synchronize()

    #TODO normalize this:  size of ψref, ψtra are not the same but equivalent
    return ψref,ψtra,ψp,ψm
end

"""
    etm_reftra(ψin,m,grd,λ)
    etm_reftra(ψin,m,grd,λ,em,sup,sub)
Computes reflection and transmission according to the ETM method
# Arguments
* `ψin` :  incoming (source) amplitude vector
* `m` :  RCWA model object
* `grd` : RCWA grid object
* `λ` : free-space wavelength
* `em` :  array with eigenmodes for all layers (computed from m if not given)
* `sup` :  superstrate halfspace object (computed from m if not given)
* `sub` :  substrate halfspace object (computed from m if not given)
lation)
# Outputs
* `R` : reflection by the device 
* `T` : transmission by the device 
"""
function etm_reftra(ψin,grd::RCWAGrid,λ,ems,sup,sub)
    kzin=grd.k0[3] # * real(sqrt(get_permittivity(m.εsup,λ)))
    # propagate amplitudes
    ro,to,r,t = etm_propagate(sup,sub,ems,ψin,false)
    R =  a2p(0ro,ro,sup.V,I,kzin) # compute amplitudes to powers
    T = -a2p(to,0to,sub.V,I,kzin)
    return R,T
end
function etm_reftra(ψin,m::RCWAModel,grd::RCWAGrid,λ)
    # println("etm_reftra():")
    CUDA.@time ems = eigenmodes(grd,λ,m.layers)
    ug = (grd.V0 isa CuArray)
    tra = halfspace(grd.Kx,grd.Ky,m.εsub,λ;use_gpu=ug)
    ref = halfspace(grd.Kx,grd.Ky,m.εsup,λ;use_gpu=ug)
    CUDA.@time R, T = etm_reftra(ψin,grd,λ,ems,ref,tra)
    # println("---")
    return R, T
end
"""
    etm_reftra_flows(ψin,m,grd,λ)
    etm_reftra_flows(ψin,m,grd,λ,em,sup,sub)
Computes reflection and transmission, as well as the net power flows between layers according to the ETM method
# Arguments
* `ψin` :  incoming (source) amplitude vector
* `m` :  RCWA model object
* `grd` : RCWA grid object
* `λ` : free-space wavelength
* `em` :  array with eigenmodes for all layers (computed from m if not given)
* `sup` :  superstrate halfspace object (computed from m if not given)
* `sub` :  substrate halfspace object (computed from m if not given)
lation)
# Outputs
* `R` : reflection by the device 
* `T` : transmission by the device 
* `flw` : array containing the power flows between layers 
"""
function etm_reftra_flows(s,m::RCWAModel,grd::RCWAGrid,λ,ems,sup,sub)
    kzin=grd.k0[3]#*real(sqrt(get_permittivity(m.εsup,λ)))
    ro,to,b,a=etm_propagate(sup,sub,ems,s) #propagate waves
    R =  a2p(0ro,ro,sup.V,I,kzin) # reflected power
    T = -a2p(to,0to,sub.V,I,kzin) # transmitted power
    #intermediate power flows 
    flw=[etm_flow(a[i],b[i],ems[i],kzin) for i=1:length(a)]
    return R,T,flw
end
function etm_reftra_flows(s,m::RCWAModel,grd::RCWAGrid,λ)
    # println("etm_reftra_flows()")
    CUDA.@time  ems=eigenmodes(grd,λ,m.layers)
    ug = (grd.V0 isa CuArray)
    ref=halfspace(grd.Kx,grd.Ky,m.εsup,λ;use_gpu=ug)
    tra=halfspace(grd.Kx,grd.Ky,m.εsub,λ;use_gpu=ug)
    CUDA.@time  R,T,flw=etm_reftra_flows(s,m,grd,λ,ems,ref,tra)
    # println("---")
    return R,T,flw
end
"""
    etm_amplitudes(ψin,m,grd,λ)
    etm_amplitudes(ψin,m,grd,λ,em,sup,sub)
Computes the amplitude vectors of forward and backward propagating waves throught the ETM method
# Arguments
* `ψin` :  incoming (source) amplitude vector
* `m` :  RCWA model object
* `λ` : free-space wavelength
* `grd` : RCWA grid
* `em` :  array with eigenmodes for all layers (computed from m if not given)
* `sup` :  superstrate halfspace object (computed from m if not given)
* `sub` :  substrate halfspace object (computed from m if not given)  
lation)
# Outputs
* `a` : amplitude vectors of forward waves
* `b` : amplitude vectors of backward waves
"""
function etm_amplitudes(ψin,m::RCWAModel,grd::RCWAGrid,λ::Real,em,sup,sub)
    ro,to,r,t=etm_propagate(sup,sup,em,ψin) #propagate wave
    return cat([ψin],t,[to],dims=1),cat([ro],r,[0ro],dims=1) #put in order
end	
function etm_amplitudes(ψin,m::RCWAModel,grd::RCWAGrid,λ::Real)
    # println("etm_amplitudes()")
    CUDA.@time ems = eigenmodes(grd,λ,m.layers) 	#layer eigenmodes
    ug = (grd.V0 isa CuArray)
    ref=halfspace(grd.Kx,grd.Ky,m.εsup,λ;use_gpu=ug) #superstrate
    tra=halfspace(grd.Kx,grd.Ky,m.εsub,λ;use_gpu=ug) #substrate
    CUDA.@time a,b = etm_amplitudes(ψin,m,grd,λ,ems,ref,tra)
    # println("---")
    return a,b
end
"""
    etm_flow(a,b,em,kz0)
Computes the power flow in z direction at a single location
# Arguments
* `a` :  forward wave amplitude vector
* `b` :  backward wave amplitude vector
* `em` :  eigenmode object
* `kz0` :  z-component of the impinging wave vector (for normalization)
# Outputs
* `flow` : power flow in the z direction
"""
function etm_flow(a, b, em, kz0)
    ex, ey = a2e2d(a+b, em.W)  # electric field
    hx, hy = a2e2d(a-b, em.V)  # magnetic field
    # poynting vector z coordinate
    return imag(sum(ex.*conj.(hy)-ey.*conj.(hx)))/kz0 
end

end
