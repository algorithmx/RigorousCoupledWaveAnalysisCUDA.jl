
using LinearAlgebra

export ScatterMatrix,scattermatrix_ref,scattermatrix_tra,scattermatrix_layer,concatenate,scatMatrices
"""
    ScatterMatrix(S11,S12,S21,S22)

Structure to store the scattering matrix of a layer or halfspace
# Attributes
* `S11` : reflection matrix of port 1
* `S12` : transmission matrix port 2 to port 1
* `S21` : transmission matrix port 1 to port 2
* `S22` : reflection matrix of port 2
"""
mutable struct ScatterMatrix
    S11::AbstractArray{Complex{Float64},2}
    S12::AbstractArray{Complex{Float64},2}
    S21::AbstractArray{Complex{Float64},2}
    S22::AbstractArray{Complex{Float64},2}
end
"""
    scatMatrices(m::RCWAModel,g::RCWAGrid,λ)
Computes the scattering matrices of the device (all layers, superstrate, and substrate)
# Arguments
* `m` : RCWA model object 
* `g` : RCWA grid object 
* `λ` : free-space wavelength 
# Outputs
* `s` : array of scattering matrices
"""
function scatMatrices(m::RCWAModel,g::RCWAGrid,λ;use_gpu=false)
    s = Vector{ScatterMatrix}(undef,length(m.layers)+2) # preallocate
    s[1]   = scattermatrix_ref(halfspace(g.Kx,g.Ky,m.εsup,λ;use_gpu=use_gpu),g.V0) # superstrate
    s[end] = scattermatrix_tra(halfspace(g.Kx,g.Ky,m.εsub,λ;use_gpu=use_gpu),g.V0) # substrate
    for cnt=2:length(m.layers)+1
        # layers in between
        s[cnt] = scattermatrix_layer(eigenmodes(g,λ,m.layers[cnt-1]), g.V0) 
    end
    return s
end
"""
    scattermatrix_ref(sup::Halfspace,V0)
Computes the scattering matrix of the superstrate halfspace
# Arguments
* `sup` : superstrate halfspace eigenmode object 
* `V0` :  Magnetic eigenmodes of free space
# Outputs
* `S` : scattering matrix
"""
function scattermatrix_ref(sup::Halfspace,V0::Union{Matrix,SparseMatrixCSC})
    use_gpu = (V0 isa CuArray)
    # boundary conditions, W=W0=I
    tmp = V0 \ Matrix(sup.V)
	A = I + tmp
    B = I - tmp
    Ai = I/Matrix(A)
	S11 = -Ai*B
   	S12 = 2*Ai
   	S21 = .5*(A-B*Ai*B)
   	S22 = B*Ai
	return ScatterMatrix(S11,S12,S21,S22)
end
"""
    scattermatrix_tra(sub::Halfspace,V0)
Computes the scattering matrix of the substrate halfspace
# Arguments
* `sub` : superstrate halfspace eigenmode object 
* `V0` :  Magnetic eigenmodes of free space
# Outputs
* `S` : scattering matrix
"""
function scattermatrix_tra(sub::Halfspace,V0::Union{Matrix,SparseMatrixCSC})
    # boundary conditions, W=W0=I
    tmp = V0 \ Matrix(sub.V)
    A=I + tmp 
    B=I - tmp
    Ai=I/Matrix(A) #precompute inverse of A for speed
    S11 = B*Ai
    S12 = .5*(A-B*Ai*B)
    S21 = 2*Ai
    S22 = -Ai*B
    return ScatterMatrix(S11,S12,S21,S22)
end
"""
    scattermatrix_layer(e::Eigenmodes,V0)
Computes the scattering matrix of a layer
# Arguments
* `e` : layer eigenmode object 
* `V0` :  Magnetic eigenmodes of free space
# Outputs
* `S` : scattering matrix
"""
function scattermatrix_layer(e::Eigenmodes,V0::Union{Matrix,SparseMatrixCSC})
    # boundary conditions, W0=I
    # The inverse of a sparse matrix can often be dense
    eWinv = Matrix(e.W)\I
    eVinv = Matrix(e.V)\I
    A = eWinv + eVinv * V0
    B = eWinv - eVinv * V0
    Ai = I/A
    C = (A-e.X*B*Ai*e.X*B)\I
    S11 = S22 = C*(e.X*B*Ai*e.X*A-B)
    S12 = S21 = C*e.X*(A-B*Ai*B)
    return ScatterMatrix(S11,S12,S21,S22)
end
"""
	concatenate(S11a,S12a,S21a,S22a,S11b,S12b,S21b,S22b)
	concatenate(S1::ScatterMatrix,S2::ScatterMatrix)
	concatenate(Sin::Array{ScatterMatrix,1})

Computes the total scattering matrix for combined layers through concatenation
# Arguments
* `S11a` : S11 component of the first scattering matrix
* `S12a` : S12 component of the first scattering matrix
* `S21a` : S21 component of the first scattering matrix
* `S22a` : S22 component of the first scattering matrix
* `S11b` : S11 component of the second scattering matrix
* `S12b` : S12 component of the second scattering matrix
* `S21b` : S21 component of the second scattering matrix
* `S22b` : S22 component of the second scattering matrix
* `S1` : first scattering matrix
* `S2` : second scattering matrix
* `Sin` : array of scattering matrices (chain concatenation)
# Outputs
* `Sout` : total scattering matrix
"""
function concatenate(S11a,S12a,S21a,S22a,S11b,S12b,S21b,S22b)
    # direct reflection (S11a) plus infinite internal passes
    S11=S11a+(S12a/(I-S11b*S22a))*S11b*S21a 
    # infinite internal passes -> geometric series
    S12=(S12a/(I-S11b*S22a))*S12b 
    S21=(S21b/(I-S22a*S11b))*S21a
    S22=S22b+(S21b/(I-S22a*S11b))*S22a*S12b
    return S11,S12,S21,S22
end
function concatenate(S1::ScatterMatrix,S2::ScatterMatrix)
    S11,S12,S21,S22=concatenate(S1.S11,S1.S12,S1.S21,S1.S22,S2.S11,S2.S12,S2.S21,S2.S22)
    return ScatterMatrix(S11,S12,S21,S22)
end
function concatenate(Sin::Array{ScatterMatrix,1})
    Sout=Sin[1]
    for i=2:length(Sin)
        Sout=concatenate(Sout,Sin[i])
    end
    return Sout
end
