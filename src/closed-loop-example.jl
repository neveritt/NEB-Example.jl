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

N  = 100
Ts = 1.0
m  = 2
nᵤ = 1
nᵣ = 1
nₛ = nᵣ*nᵤ
n = 100

Θ₀ = [.2, 0.3, 0.5, 0.15]
Θ₂ = [0.4, -0.5, 0.5, 0.2]
nsru = nsry = 1.0

@everywhere function _create_data(N,m,nsru,nsry,Θ₀,Θ₂)
  b1 = vcat(0.0, Θ₀[1:m])
  b2 = vcat(0.0, Θ₂[1:m])
  a1 = vcat(1.0, Θ₀[m+1:2m])
  a2 = vcat(1.0, Θ₂[m+1:2m])
  Sb = conv(a1,a2)
  Sa = conv(a1,a2) - conv(b1,b2)

  r = randn(1,N)
  u0 = filt(Sb,Sa,transpose(r))  |> transpose
  y0 = filt(b1,a1,transpose(u0)) |> transpose
  u  = u0 + sqrt(sumabs2(u0)/N*nsru).*randn(1,N)
  y  = y0 + sqrt(sumabs2(y0)/N*nsry).*randn(1,N)
  return y,u,r,y0,u0
end

# nebx options
neboptions = IdOptions(iterations = 100, autodiff=true, estimate_initial=false)
# two stage settings
fir_m  = n
orders = [n, m, nᵤ, nᵣ, N]
firmodel = FIR(fir_m*ones(Int,1,nᵣ),0*ones(Int,1,nᵣ), 1, nᵣ)
options = IdOptions(iterations = 100, autodiff=true, estimate_initial=false)
oemodel = OE(m*ones(Int,1,nᵤ), m*ones(Int,1,nᵤ), ones(Int,1,nᵤ), 1, nᵤ)

MC = 100
nebres  = SharedArray(Float64, (2m*nᵤ, MC))
twostageres = SharedArray(Float64, (2m*nᵤ, MC))
@sync @parallel for i = 1:MC
  y,u,r,y0,u2 = _create_data(N,m,nsru,nsry,Θ₀,Θ₂)

  z = vcat(y[1:1,:],u)
  zdata = iddata(z,r)
  NEBtrace, zₛ = NetworkEmpiricalBayes.neb(zdata, n, m; outputidx=1, options=neboptions)
  nebres[:,i] = last(NEBtrace).Θ

  xfir, xG, xσ = two_stage(y[1:1,:],u,r,Ts,orders, firmodel, oemodel, options)
  A,B,F,C,D = IdentificationToolbox._getpolys(oemodel, xG[:])
  twostageres[:,i] = vcat(coeffs(B[1])[2:1+m], coeffs(F[1])[2:1+m])
end

nebfit      = zeros(nᵤ,MC)
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
  twostagefit[1,i] += fitg(Θ₀[1:2m],twostageres[1:2m,i],m,Ts,N)
end

# save results
res = vcat(1-twostagefit,1-nebfit)
write("results-CL-$(size(res,1))-$(size(res,2))-N$(N)-n$(n)-nsr$(nsru).dat", res)
