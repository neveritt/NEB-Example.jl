cd("../NEB-Example.jl/src/")

addprocs(3)

# Import functions for overloading
using DataFrames
using Polynomials
using Optim
using NetworkEmpiricalBayes
using IdentificationToolbox
import NetworkEmpiricalBayes.impulse

@everywhere include("two-stage.jl")
@everywhere include("create_data.jl")

dfs = readtable("../networkdatas.dat")
s0 = _readdata(dfs)

N  = 200
Ts = 1.0
m  = 2
nᵤ = 2
nᵣ = 2
nₛ = nᵣ*nᵤ

Θ₀ = [.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.5, 0.15]
Θ₂ = [0.4, -0.5, 0.5, 0.2]
nsru = nsry = 0.1
nsry2 = 0.1
n = 100

# nebx options
neboptions = IdOptions(iterations = 20, autodiff=true, estimate_initial=false)
nebxoptions = IdOptions(iterations = 5, autodiff=true, estimate_initial=false)
# two stage settings
fir_m  = n
orders = [n, m, nᵤ, nᵣ, N]
firmodel = FIR(fir_m*ones(Int,1,nᵣ),0*ones(Int,1,nᵣ), 1, nᵣ)
options = IdOptions(iterations = 100, autodiff=true, estimate_initial=false)
oemodel = OE(m*ones(Int,1,nᵤ), m*ones(Int,1,nᵤ), ones(Int,1,nᵤ), 1, nᵤ)

MC = 100
nebres  = SharedArray(Float64, (2m*nᵤ, MC))
nebxres = SharedArray(Float64, (2m*nᵤ, MC))
twostageres = SharedArray(Float64, (2m*nᵤ, MC))
@sync @parallel for i = 1:MC
  y,u,r,y0,u2 = _create_data(N,m,nᵤ,nᵣ,s0,nsru,nsry,nsry2,Θ₀,Θ₂)

  z = vcat(y[1:1,:],u)
  data = iddata(z,r)
  NEBtrace, zₛ = NetworkEmpiricalBayes.neb(data, n, m; outputidx=1, options=neboptions)
  nebres[:,i] = last(NEBtrace).Θ

  NEBXtrace, z = NetworkEmpiricalBayes.nebx(y[1:1,:],u,r,y[2:2,:]- r[2:2,:], orders, Ts; options=nebxoptions)
  nebxres[:,i] = last(NEBXtrace).Θ

  xfir, xG, xσ = two_stage(y[1:1,:],u,r,Ts,orders, firmodel, oemodel, options)
  A,B,F,C,D = IdentificationToolbox._getpolys(oemodel, xG[:])
  twostageres[:,i] = vcat(coeffs(B[1])[2:1+m], coeffs(F[1])[2:1+m], coeffs(B[2])[2:1+m], coeffs(F[2])[2:1+m])
end

nebfit      = zeros(nᵤ,MC)
nebxfit     = zeros(nᵤ,MC)
twostagefit = zeros(nᵤ,MC)

impulseg(Θ,m::Int,Ts,N::Int) = impulse(vcat(zeros(1), Θ[1:m]),vcat(ones(1), Θ[m+1:2m]),Ts,N)

function fitg(Θ₀,Θ₁,m,Ts,N)
  g₀ = impulseg(Θ₀,m,Ts,N)
  g₁ = impulseg(Θ₁,m,Ts,N)
  sumabs2(g₁-g₀)
end

for i in 1:MC
  # evaluate
  nebfit[1,i]      += fitg(Θ₀[1:2m],nebres[1:2m,i],m,Ts,N)
  nebxfit[1,i]     += fitg(Θ₀[1:2m],nebxres[1:2m,i],m,Ts,N)
  twostagefit[1,i] += fitg(Θ₀[1:2m],twostageres[1:2m,i],m,Ts,N)

  nebfit[2,i]      += fitg(Θ₀[2m+1:4m],nebres[2m+1:4m,i],m,Ts,N)
  nebxfit[2,i]     += fitg(Θ₀[2m+1:4m],nebxres[2m+1:4m,i],m,Ts,N)
  twostagefit[2,i] += fitg(Θ₀[2m+1:4m],twostageres[2m+1:4m,i],m,Ts,N)
end

# save results
res = vcat(1-twostagefit,1-nebfit,1-nebxfit)
write("results-$(size(res,1))-$(size(res,2))-N$(N)-n$(n)-nsr$(nsru).dat", res)


i = 1
y,u,r,y0,u2 = _create_data(N,m,nᵤ,nᵣ,s0,nsru,nsry,nsry2,Θ₀,Θ₂)
using BenchmarkTools
z = vcat(y[1:1,:],u)
data = iddata(z,r)
@benchmark NEBtrace, zₛ = NetworkEmpiricalBayes.neb(data, n, m; outputidx=1, options=neboptions)
nebres[:,i] = last(NEBtrace).Θ

@time NEBXtrace, z = NetworkEmpiricalBayes.nebx(y[1:1,:],u,r,y[2:2,:]- r[2:2,:], orders, Ts; options=nebxoptions)
nebxres[:,i] = last(NEBXtrace).Θ

@time xfir, xG, xσ = two_stage(y[1:1,:],u,r,Ts,orders, firmodel, oemodel, options)
A,B,F,C,D = IdentificationToolbox._getpolys(oemodel, xG[:])
twostageres[:,i] = vcat(coeffs(B[1])[2:1+m], coeffs(F[1])[2:1+m], coeffs(B[2])[2:1+m], coeffs(F[2])[2:1+m])
