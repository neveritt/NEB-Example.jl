function twostagebayesian{T}(y::AbstractMatrix{T},u,r,Ts,orders, firmodel, OEmodel, options)
  nᵤ,N = size(u)
  nᵣ   = size(r,1)
  nₛ   = nᵤ*nᵣ
  Ts   = 1.0

  λᵥ = zeros(T,nₛ)
  βᵥ = zeros(T,nₛ)
  σᵥ = zeros(T,nᵤ+1)
  sᵥ = zeros(T,n,nₛ)
  û  = zeros(T,nᵤ,N)
  λ₀, β₀, σ₀ = 100*one(T), 0.9*one(T), 100*one(T)
  for k = 1:nᵤ, j = 1:nᵣ
    i = nᵣ*(k-1) + j
    λᵥ[i], βᵥ[i], σᵥ[k], sᵥ[:,i] = basicEB(u[k,:], r[j,:], n, λ₀, β₀, σ₀)
    û[k,:] += filt(sᵥ[:,i],1,r[j,:])
  end

  # fir
  fir_m    = n
  nk       = 0*ones(Int,1,nᵣ)
  firmodel = FIR(fir_m*ones(Int,1,nᵣ), nk, 1, nᵣ)
  options  = IdOptions(iterations = 100, estimate_initial=false)
  û  = zeros(T,nᵤ,N)
  for k = 0:nᵤ-1
    zdata     = IdentificationToolbox.iddata(u[k+1:k+1,:], r, Ts)
    A,B,F,C,D,info = IdentificationToolbox.pem(zdata,firmodel,zeros(T,nᵣ*fir_m),options)
    û[k+1:k+1,:] += filt(B,F,r)
    σᵥ[k+1] = info.mse[1]
  end

  ΘG      = zeros(nᵤ*2m)
  zdata   = IdentificationToolbox.iddata(y, û, Ts)
  ΘG[:],_ = IdentificationToolbox._morsm(zdata, OEmodel, options)
  A,B,F,C,D,info = pem(zdata, OEmodel, ΘG[:], options)
  ΘG[:]   = info.opt.minimizer

  return ΘG, σᵥ, λᵥ, βᵥ, sᵥ
end
