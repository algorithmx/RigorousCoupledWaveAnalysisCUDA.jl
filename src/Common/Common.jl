#Common methodes for computations with eigenmodes and transforms

module  Common
using LinearAlgebra
using CUDA
include("ft2d.jl")
include("materials.jl")
include("models.jl")
include("grids.jl")
export Eigenmodes, Halfspace
export eigenmodes, halfspace
export a2e, a2e2d, a2p, e2p, getfields, a2p2
export a_over_b, take_a, take_b, slicehalf
export upload, download
@inline showcumat(A,name="") = println("name = $(name) , size = $(A.dims) , offset = $(A.offset) , maxsize = $(A.maxsize)\nstorage = $(A.storage)")

"""
    Eigenmodes(V,W,X,q)

Structure to store the eigenmodes of a layer
# Attributes
* `V` : Magnetic field eigenmode
* `W` : Electric field eigenmode
* `X` : Factor for propagating amplitude vector
* `q` : Pseudo wave vector
"""
mutable struct Eigenmodes
    V::AbstractArray{Complex{Float64},2} #Transform towards magnetic fields
    W::AbstractArray{Complex{Float64},2} #Transform towards electric fields
    X::AbstractArray{Complex{Float64},2} #Propagation of the amplitude vector through the layer
    q::AbstractArray{Complex{Float64},2} #pseudo wave vector
end
upload(em::Eigenmodes) = Eigenmodes(CuArray(em.V),CuArray(em.W),CuArray(em.X),CuArray(em.q))
download(em::Eigenmodes) = Eigenmodes(Array(em.V),Array(em.W),Array(em.X),Array(em.q))
"""
    Halfspace(V,Kz)

Structure to store the eigenmodes of a layer
# Attributes
* `V` : Magnetic field eigenstate
* `Kz` : z component of the wave vector in the medium
"""
mutable struct Halfspace
    Kz::AbstractArray{Complex{Float64},2} #z-component of the wave vector in the medium
    V::AbstractArray{Complex{Float64},2}  #Transform towards magnetic fields. W is identity anyway.
end
"""
    eigenmodes(dnx,dny,Kx,Ky,λ,l::Layer)
    eigenmodes(g::RCWAGrid,λ,l::Layer)
    eigenmodes(g::RCWAGrid,λ,l::Array{Layer,1})
Compute the eigenmodes of a layer
# Arguments
* `g` :  grid object
* `λ` :  free space wavelength
* `l` :  layer object
* `dnx` : reciprocal space grid in x
* `dny` : reciprocal space grid in y
* `Kx` : kx component of the wave vector in reciprocal space
* `Ky` : ky component of the wave vector in reciprocal space
# Outputs
* `em` : Eigenmode object
"""
function eigenmodes(dnx,dny,Kx,Ky,λ,l::PatternedLayer;use_gpu=false)::Eigenmodes
    k0=2π/real(λ)
    # get the base permittivity
    εxx = get_permittivity(l.materials[1],λ,1)*I
    if typeof(l.materials[1])<:Isotropic
    else
    end
    #add the permittivity for all inclusions
    if minimum([typeof(m)<:Isotropic for m in l.materials])
        εxx=get_permittivity(l.materials[1],λ)*I
        for ct=1:length(l.geometries)
            rec=reciprocal(l.geometries[ct],dnx,dny)
            εxx+=rec*(get_permittivity(l.materials[ct+1],λ)-get_permittivity(l.materials[ct],λ))
        end
        εzz=εyy=εxx
        εxy=εyx=0I
    else
        εxx=get_permittivity(l.materials[1],λ,1)*I
        εxy=get_permittivity(l.materials[1],λ,2)*I
        εyx=get_permittivity(l.materials[1],λ,3)*I
        εyy=get_permittivity(l.materials[1],λ,4)*I
        εzz=get_permittivity(l.materials[1],λ,5)*I
        for ct=1:length(l.geometries)
            rec=reciprocal(l.geometries[ct],dnx,dny)
            εxx += rec * ( get_permittivity(l.materials[ct+1],λ,1) - get_permittivity(l.materials[ct],λ,1) )
            εxy += rec * ( get_permittivity(l.materials[ct+1],λ,2) - get_permittivity(l.materials[ct],λ,2) )
            εyx += rec * ( get_permittivity(l.materials[ct+1],λ,3) - get_permittivity(l.materials[ct],λ,3) )
            εyy += rec * ( get_permittivity(l.materials[ct+1],λ,4) - get_permittivity(l.materials[ct],λ,4) )
            εzz += rec * ( get_permittivity(l.materials[ct+1],λ,5) - get_permittivity(l.materials[ct],λ,5) )
        end
    end
    #reciprocal of permittivity
    η=I/εzz

    ## --------- BEWARE OF CuArray BELOW  ## ---------
    array_converter = use_gpu ? CuArray : x->x

    #Maxwell equations transformed
    Q = [(Kx*Ky+εyx)  (εyy-Kx*Kx);  (Ky*Ky-εxx)  (-εxy-Ky*Kx)] |> array_converter
    if use_gpu
        P = CuArray([(Kx*η*Ky)  (I-Kx*η*Kx);  (Ky*η*Ky-I)  (-Ky*η*Kx)])
        M = P*Q
    else
        # analytic multiplication can speed things up:
        A = η*(Ky*εyx+Kx*εxx)
        B = η*(Ky*εyy+Kx*εxy)
        # non-hermitian, dense matrix of dimension 2*(2N)^2 where N is the precsion 
        M = [(Ky.^2-εxx+Kx*A)  (-Ky*Kx-εxy+Kx*B); (-Kx*Ky-εyx+Ky*A)  (Kx.^2-εyy+Ky*B)]
    end

    ev = eigen(Matrix(M)) # bottleneck #1, CUDA.jl cannot help
    q = sqrt.(Complex.(ev.values))
    q[real.(q).>0].*=-1  # select negative root

    #W is transform between amplitude vector and E-Field
    W = ev.vectors |> array_converter
    #V is transform between amplitude vector and H-Field
    V = (Q*W)/array_converter(Diagonal(q)) # TODO bottleneck #3
    #X the factor applied to the amplitudes when propagatin through the layer
    X = Diagonal(exp.(q*k0*l.thickness)) |> array_converter
    #use_gpu && (synchronize())
    return Eigenmodes(V,W,X,Diagonal(q))
end
function eigenmodes(dnx,dny,Kx,Ky,λ,l::SimpleLayer;use_gpu=false)::Eigenmodes
    k0=2π/real(λ)
    #permittivity tensor for SimpleLayer
    ε = get_permittivity(l.material,λ)*I
    #z component
    Kz = sqrt.(Complex.(ε - Kx*Kx - Ky*Ky))
    q = [1im*diag(Kz) ; 1im*diag(Kz)]
    q[real.(q).>0].*=-1
    # 
    array_converter = use_gpu ? CuArray : Array # prevent sparse
    Q = [Kx*Ky  ε-Kx*Kx;  Ky*Ky-ε  -Ky*Kx] |> array_converter
    #magnetic field eigenmodes
    V = Q / array_converter(Diagonal(q)) # not diagonal 
    # W is identity
    W = array_converter(one(V)) # diagonal 
    # amplitude propagation
    X = Diagonal(exp.(q*k0*l.thickness)) |> array_converter
    #use_gpu && (synchronize())
    return Eigenmodes(V,W,X,Diagonal(q))
end
function eigenmodes(dnx,dny,Kx,Ky,λ,l::AnisotropicLayer;use_gpu=false)::Eigenmodes
    k0=2π/real(λ)
    #permittivity tensor
    εxx=get_permittivity(l.material,λ,1)*I
    #anisotropy
    if typeof(l.material)<:Isotropic
        εzz=εyy=εxx
        εxy=εyx=0I
    else
        εxy=get_permittivity(l.material,λ,2)*I
        εyx=get_permittivity(l.material,λ,3)*I
        εyy=get_permittivity(l.material,λ,4)*I
        εzz=get_permittivity(l.material,λ,5)*I
    end
    η=I/εzz
    #Maxwell equations transformed
    if use_gpu
        P = CuArray([(Kx*η*Ky)   (I-Kx*η*Kx); (Ky*η*Ky-I)  (-Ky*η*Kx)])
        Q = CuArray([(Kx*Ky+εyx) (εyy-Kx*Kx); (Ky*Ky-εxx)  (-εxy-Ky*Kx)])
        M = P*Q
    else
        # analytic multiplication can speed things up:
        Q = [(Kx*Ky+εyx) (εyy-Kx*Kx); (Ky*Ky-εxx)  (-εxy-Ky*Kx)]
        A = η*(Ky*εyx+Kx*εxx)
        B = η*(Ky*εyy+Kx*εxy)
        # non-hermitian, dense matrix of dimension 2*(2N)^2 where N is the precsion 
        M = [(Ky.^2-εxx+Kx*A)  (-Ky*Kx-εxy+Kx*B); (-Kx*Ky-εyx+Ky*A)  (Kx.^2-εyy+Ky*B)]
    end

    ev = eigen(Matrix(M)) # bottleneck #1, CUDA.jl cannot help
    q = sqrt.(Complex.(ev.values))
    q[real.(q).>0].*=-1  #  select negative root

    if use_gpu
        #W is transform between amplitude vector and E-Field
        W = CuArray(ev.vectors)
        #V is transform between amplitude vector and H-Field
        V = (Q*W)/CuArray(Diagonal(q)) # TODO bottleneck #3
        #X the factor applied to the amplitudes when propagatin through the layer
        X = CuArray(Diagonal(exp.(q*(k0*l.thickness))))
    else
        #W is transform between amplitude vector and E-Field
        W = ev.vectors
        #V is transform between amplitude vector and H-Field
        V = Q*W/Diagonal(q) # TODO bottleneck #3
        #X the factor applied to the amplitudes when propagatin through the layer
        X = Diagonal(exp.(q*(k0*l.thickness)))
    end
    return Eigenmodes(V,W,X,Diagonal(q))
end
function eigenmodes(g::RCWAGrid,λ,l::Layer)::Eigenmodes
    return eigenmodes(g.dnx,g.dny,g.Kx,g.Ky,λ,l; use_gpu=(g.V0 isa CuArray))
end
function eigenmodes(g::RCWAGrid,λ,l::Vector{Layer})::Vector{Eigenmodes}
    #initialize array
    rt=Array{Eigenmodes,1}(undef,length(l))
    #iterate through layers
    for cnt=1:length(l)
        rt[cnt]=eigenmodes(g,λ,l[cnt])
    end
    return rt
end
"""
    halfspace(Kx,Ky,material,λ)

Compute the eigenmodes of a halfspace
# Arguments
* `Kx` : kx component of the wave vector in reciprocal space
* `Ky` : ky component of the wave vector in reciprocal space
* `material` : medium
* `λ` : wavelength
* `use_gpu` : true if use GPU
# Outputs
* `em` : halfspace eigenmode object
"""
function halfspace(Kx,Ky,material,λ; use_gpu=false)::Halfspace
    #Base value
    εxx=real(sqrt(get_permittivity(material,λ,1)))^2*I
    #probably add off-diagonal elements
    if typeof(material)<:Isotropic
        εzz=εyy=εxx
        εxy=εyx=0I
    else
        εxy=get_permittivity(material,λ,2)*I
        εyx=get_permittivity(material,λ,3)*I
        εyy=get_permittivity(material,λ,4)*I
        εzz=get_permittivity(material,λ,5)*I
    end
    #z component of wave vector
    Kz = sqrt.(Complex.(εzz-Kx*Kx-Ky*Ky))
    Kz[imag.(Kz).<0].*=-1
    #simple solution like free space
    q0 = [1im*diag(Kz); 1im*diag(Kz)]
    Q0 = [(Kx*Ky+εyx)  (εyy-Kx*Kx);  (Ky*Ky-εxx)  (-εxy-Ky*Kx)]
    #Magnetic field eigenvalues
    array_converter = use_gpu ? x->CuArray(Matrix(x)) : x->x
    V = array_converter(Q0) / array_converter(Diagonal(q0))
    return Halfspace(Kz, V)
end

"""
    slicehalf(e)

Utility function, just slices a vector in two vectors of half length
# Arguments
* `v` : vector to be halfed
# Outputs
* `v1` : upper half of v
* `v2` : lower half of v
"""
# function slicehalf(v)
#     mylength=convert(Int64,size(v,1)/2)
#     return v[1:mylength,:],v[mylength+1:end,:]
# end
@inline slicehalf(v) = v[1:size(v,1)÷2,:], v[size(v,1)÷2+1:end,:]
@inline take_a(v) = v[1:size(v,1)÷2,:]
@inline take_b(v) = v[size(v,1)÷2+1:end,:]
@inline a_over_b(v) = take_a(v)/take_b(v) # calls \ via generic.jl
"""
    e2p(ex,ey,ez,Kz,kz0)

Converts the reciprocal-space electric field (in a substrate or superstrate) into a Poynting power flow in z direction
# Arguments
* `ex` : x component of the electric field in reciprocal space
* `ey` : y component of the electric field in reciprocal space
* `ez` : z component of the electric field in reciprocal space
* `Kz` : z component of the wavevector in the medium
* `kz0` : z component of the 0th-order wavevector in the superstrate
# Outputs
* `P` : power flow
"""
function e2p(ex,ey,ez,Kz,kz0)
    #amplitudes squared
    P=abs.(ex).^2+abs.(ey).^2+abs.(ez).^2
    #weight by z component
    P=sum(real.(Kz)*P/real(kz0))
    return P
end


"""
    a2p2(a,W,Kx,Ky,Kz,kz0)

Converts an amplitude vector (in substrate or superstrate) to Poynting power flow in z direction
# Arguments
* `a` : amplitude vector
* `W` : eigenmodes of the halfspace
* `Kx` : x component of the wavevector in the medium
* `Ky` : y component of the wavevector in the medium
* `Kz` : z component of the wavevector in the medium
* `kz0` : z component of the plane wave wavevector in the superstrate
# Outputs
* `P` : power flow
"""
function a2p2(a,W,Kx,Ky,Kz,kz0)
    ex,ey,ez=a2e(a,W,Kx,Ky,Kz)
    return e2p(ex,ey,ez,Kz,kz0)
end
"""
    a2p(a,b,W,Kx,Ky,kz0)

Converts an amplitude vector (in substrate or superstrate) to Poynting power flow in z direction
# Arguments
* `a` : forward amplitude vector
* `b` : backward amplitude vector
* `V` : H-field eigenmodes of the halfspace
* `W` : E-fieldeigenmodes of the halfspace
* `kz0` : z component of the plane wave wavevector in the superstrate
# Outputs
* `P` : power flow
"""
function a2p(a,b,V,W,kz0)
    ex,ey=a2e2d(a+b,W)
    hx,hy=a2e2d(a-b,V)
    return imag(sum(ex.*conj.(hy)-ey.*conj.(hx)))/kz0
end


"""
    a2e2d(a,W)

Converts an amplitude vector to reciprocal-space electric fields Ex and Ey.
This is a "light" version of the "a2e" method.
# Arguments
* `a` : amplitude vector
* `W` : eigenmodes of the halfspace
# Outputs
* `ex` : x-component of the electric field
* `ey` : y-component of the electric field
"""
a2e2d(a::Vector, W::AbstractArray) = slicehalf(Matrix(W) * a)
a2e2d(a::Matrix, W::AbstractArray) = slicehalf(Matrix(W) * a)
function a2e2d(a::CuArray, W::CuArray)
    ex,ey = slicehalf(W * a)
    return Array(ex), Array(ey)
end
function a2e2d(a, X::UniformScaling)
    ex,ey = slicehalf(X.λ .* Array(a))
    return Array(ex), Array(ey)
end

"""
    a2e(a,W,Kx,Ky,Kz)

converts an amplitude vector (in substrate or superstrate) to reciprocal-space electric fields
# Arguments
* `a` : x amplitude vector
* `W` : eigenmodes of the halfspace
* `Kx` : x component of the wavevector in the medium
* `Ky` : y component of the wavevector in the medium
* `Kz` : z component of the wavevector in the medium
# Outputs
* `ex` : x-component of the electric field
* `ey` : y-component of the electric field
* `ez` : z-component of the electric field
"""
function a2e(a,W,Kx,Ky,Kz)
    ex,ey=a2e2d(a,W)
    #Plane wave, E⊥k, E*k=0
    ez=-Kz\(Kx*ex+Ky*ey)
    return ex,ey,ez
end

"""
    getfields(ain,bout,thi,em::Eigenmodes,grd::RCWAGrid,sz,λ)

computes the electric and magnetic fields within a layer
# Arguments
* `ain` : amplitude vector entering the layer from the previous layer
* `bout` : amplitude vector leaving the layer towards the previous layer
* `thi` : thickness of the layer
* `em` : eigenmodes of the layer
* `grd` : reciprocal space grid object
* `sz` : three-element vector specifying the number of points in x, y, and z for which the fields are to be computed
* `λ` : wavelength
# Outputs
* `efield` : 4D tensor for the electric field (dimensions are x, y, z, and the component (E_x or E_y or E_z)
* `hfield` : 4D tensor for the magnetic field (dimensions are x, y, z, and the component (E_x or E_y or E_z)
"""
function getfields(ain,bout,thi,em::Eigenmodes,grd::RCWAGrid,sz,λ)
    #create the x and y components of the real-space rectilinear grid
    x=[r  for r in -sz[1]/2+.5:sz[1]/2-.5, c in -sz[2]/2+.5:sz[2]/2-.5]/sz[1]
    y=[c  for r in -sz[1]/2+.5:sz[1]/2-.5, c in -sz[2]/2+.5:sz[2]/2-.5]/sz[2]
    #initialize the fields
    efield=zeros(size(x,1),size(y,2),sz[3],3)*1im
    hfield=zeros(size(x,1),size(y,2),sz[3],3)*1im
    #loop through z constants
    for zind=1:sz[3]
        #propagation of the waves
        a=exp(Matrix(em.q*2π/λ*thi*(zind-1)/sz[3]))*ain
        b=exp(-Matrix(em.q*2π/λ*thi*(zind-1)/sz[3]))*bout
        #convert amplitude vectors to electric fields
        ex,ey,ez=a2e(a+b,em.W,grd.Kx,grd.Ky,grd.Kz0)
        hx,hy,hz=a2e(a-b,em.V,grd.Kx,grd.Ky,grd.Kz0)
        #convert from reciprocal lattice vectors to real space distribution
        efield[:,:,zind,1]=recipvec2real(grd.nx,grd.ny,ex,x,y)
        efield[:,:,zind,2]=recipvec2real(grd.nx,grd.ny,ey,x,y)
        efield[:,:,zind,3]=recipvec2real(grd.nx,grd.ny,ez,x,y)

        hfield[:,:,zind,1]=recipvec2real(grd.nx,grd.ny,hx,x,y)
        hfield[:,:,zind,2]=recipvec2real(grd.nx,grd.ny,hy,x,y)
        hfield[:,:,zind,3]=recipvec2real(grd.nx,grd.ny,hz,x,y)
    end
    return efield,hfield
end


end  # module  eigenmodes
