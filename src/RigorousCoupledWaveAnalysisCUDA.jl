module RigorousCoupledWaveAnalysisCUDA
export InterpolPerm,InterpolPermA,ConstantPerm,ConstantPermA,ModelPerm,get_permittivity
export Circle,Rectangle,Ellipse,Custom
export Combination,Rotation,Shift
export Layer,SimpleLayer,PatternedLayer,Material,AnisotropicLayer

export RCWAModel
export ngrid,kgrid,rcwagrid,rcwasource,getfields
export etm_reftra,etm_propagate
export etm_amplitudes,etm_flow,etm_reftra_flows


export ScatterMatrix,scattermatrix_ref,scattermatrix_tra,scattermatrix_layer,concatenate,scatMatrices
export srcwa_reftra,srcwa_amplitudes,srcwa_flow
export recip2real,real2recip,recipvec2real


include("Common/Common.jl")
include("SRCWA/SRCWA.jl")
include("ETM/ETM.jl")
include("BasicMaterials/BasicMaterials.jl")


using .Common
using .SRCWA
using .ETM
using .BasicMaterials


end # module
