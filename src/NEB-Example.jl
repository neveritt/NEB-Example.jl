cd("../NEB-Example.jl/src")

addprocs(2)

@everywhere using PolynomialMatrices
@everywhere using SystemsBase
@everywhere using ControlToolbox

# Import functions for overloading
@everywhere using DataFrames, Polynomials, Optim
#@everywhere using GeneralizedSchurAlgorithm
@everywhere using IdentificationToolbox
@everywhere using NetworkEmpiricalBayes
@everywhere import NetworkEmpiricalBayes.impulse

@everywhere include("smpe.jl")
@everywhere include("two-stage-bayesian.jl")
@everywhere include("create_data.jl")

# load true sensitivity
dfs = readtable("../networkdatas.dat");
s0 = _readdata(dfs)

n  = 60
N  = 150
Ts = 1.0
m  = 2
nᵤ = 2
nᵣ = 2
nₛ = nᵣ*nᵤ

# true parameters
Θ₀    = [.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.5, 0.15]
Θ₂    = [0.4, -0.5, 0.5, 0.2]

# noise to signal ratios
nsru  = 0.1
nsry  = 0.1
nsry2 = 0.01

# nebx options
neboptions  = IdentificationToolbox.IdOptions(iterations = 10, autodiff=:forward, estimate_initial=false, x_tol=1e-6)
nebxoptions = IdentificationToolbox.IdOptions(iterations = 5, autodiff=:forward, estimate_initial=false, x_tol=1e-6)

# two stage settings
fir_m    = n
orders   = [n, m, nᵤ, nᵣ, N]
firmodel = IdentificationToolbox.FIR(fir_m*ones(Int,1,nᵣ),0*ones(Int,1,nᵣ), 1, nᵣ)
options  = IdentificationToolbox.IdOptions(iterations = 10, autodiff=:forward, estimate_initial=false, x_tol=1e-6)
oemodel  = IdentificationToolbox.OE(m*ones(Int,1,nᵤ), m*ones(Int,1,nᵤ), ones(Int,1,nᵤ), 1, nᵤ)

# Monte Carlo Arrays
MC = 4
nebres      = SharedArray{Float64}((2m*nᵤ, MC))
nebxres     = SharedArray{Float64}((2m*nᵤ, MC))
smperes     = SharedArray{Float64}((2m*nᵤ, MC))
twores      = SharedArray{Float64}((2m*nᵤ, MC))
twobayesres = SharedArray{Float64}((2m*nᵤ, MC))
@sync @parallel for i = 1:MC
  y,u,r,y0,u2 = _create_data(N,m,nᵤ,nᵣ,s0,nsru,nsry,nsry2,Θ₀,Θ₂)

  z    = vcat(y[1:1,:],u)
  data = iddata(z,r)
  NEBtrace, zₛ = NetworkEmpiricalBayes.neb(data, n, m; outputidx=1, options=neboptions)
  nebres[:,i]  = last(NEBtrace).Θ

  NEBXtrace, z = NetworkEmpiricalBayes.nebx(y[1:1,:],u,r,y[2:2,:], orders, Ts; options=nebxoptions)
  nebxres[:,i] = last(NEBXtrace).Θ

  xfir, xG, xσ, opt, xinit = smpe(y[1:1,:],u,r,Ts,orders, firmodel, oemodel, options)
  A,B,F,C,D    = IdentificationToolbox._getpolys(oemodel, xG[:])
  smperes[:,i] = vcat(coeffs(B[1])[2:1+m], coeffs(F[1])[2:1+m], coeffs(B[2])[2:1+m], coeffs(F[2])[2:1+m])

  xtwoG       = view(xinit,nₛ*fir_m+(1:nᵤ*2m))
  A,B,F,C,D   = IdentificationToolbox._getpolys(oemodel, xtwoG[:])
  twores[:,i] = vcat(coeffs(B[1])[2:1+m], coeffs(F[1])[2:1+m], coeffs(B[2])[2:1+m], coeffs(F[2])[2:1+m])

  twobayesres[:,i] = twostagebayesian(y[1:1,:],u,r,Ts,orders, firmodel, oemodel, options)[1]
end

nebfit      = zeros(nᵤ,MC)
nebxfit     = zeros(nᵤ,MC)
smpefit     = zeros(nᵤ,MC)
twofit      = zeros(nᵤ,MC)
twobayesfit = zeros(nᵤ,MC)

impulseg(Θ,m::Int,Ts,N::Int) = impulse(vcat(zeros(1), Θ[1:m]),vcat(ones(1), Θ[m+1:2m]),Ts,N)

function fitg(Θ₀,Θ₁,m,Ts,N)
  g₀ = impulseg(Θ₀,m,Ts,N)
  g₁ = impulseg(Θ₁,m,Ts,N)
  sum(abs2, g₁-g₀)/sum(abs2, g₀)
end

for i in 1:MC
  # evaluate
  nebfit[1,i]       += fitg(Θ₀[1:2m],nebres[1:2m,i],m,Ts,N)
  nebxfit[1,i]      += fitg(Θ₀[1:2m],nebxres[1:2m,i],m,Ts,N)
  smpefit[1,i]      += fitg(Θ₀[1:2m],smperes[1:2m,i],m,Ts,N)
  twofit[1,i]       += fitg(Θ₀[1:2m],twores[1:2m,i],m,Ts,N)
  twobayesfit[1,i]  += fitg(Θ₀[1:2m],twobayesres[1:2m,i],m,Ts,N)

  nebfit[2,i]       += fitg(Θ₀[2m+1:4m],nebres[2m+1:4m,i],m,Ts,N)
  nebxfit[2,i]      += fitg(Θ₀[2m+1:4m],nebxres[2m+1:4m,i],m,Ts,N)
  smpefit[2,i]      += fitg(Θ₀[2m+1:4m],smperes[2m+1:4m,i],m,Ts,N)
  twofit[2,i]       += fitg(Θ₀[2m+1:4m],twores[2m+1:4m,i],m,Ts,N)
  twobayesfit[2,i]  += fitg(Θ₀[2m+1:4m],twobayesres[2m+1:4m,i],m,Ts,N)
end

# save results
res = vcat(1-smpefit,1-nebfit,1-nebxfit,1-twofit,1-twobayesfit)
write("results2-$(size(res,1))-$(size(res,2))-N$(N)-n$(n)-nsr$(nsru).dat", res)
write("theta2-$(size(res,1))-$(size(res,2))-N$(N)-n$(n)-nsr$(nsru).dat", hcat(smperes,nebres,nebxres,twores))
