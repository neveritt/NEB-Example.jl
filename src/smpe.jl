function smpe{T}(y::AbstractMatrix{T},u,r,Ts,orders, firmodel, oemodel, options)
  size(y,1) == 1 || throw(DomainError())
  n, m, nᵤ, nᵣ, N = orders
  nₛ    = nᵤ*nᵣ
  fir_m = firmodel.orders.nb[1]

  Θ = _initial_smpe(y,u,r,Ts,orders, firmodel, oemodel, options)
  xinit = copy(Θ)
  x     = Θ
  local opt::Optim.OptimizationResults
  try
    opt = Optim.optimize(x->cost_smpe(y,u,r,Ts, x, orders, firmodel, oemodel, options),
          Θ, Optim.Newton(;linesearch = LineSearches.morethuente!), options.OptimizationOptions)
    x = opt.minimizer
  catch y
    println("failed")
    println(y)
  end

  xfir = view(x,1:nₛ*fir_m)
  xG   = view(x,nₛ*fir_m+(1:nᵤ*2m))
  xσ   = view(x,nₛ*fir_m+nᵤ*2m+(1:nᵤ))

  return xfir, xG, xσ, opt, xinit
end

function _initial_smpe{T}(y::AbstractMatrix{T},u,r,Ts,orders, firmodel, oemodel, options)
  n, m, nᵤ, nᵣ, N = orders
  nₛ = nᵤ*nᵣ
  fir_m = firmodel.orders.nb[1]
  nk = firmodel.orders.nk

  Θ    = zeros(nₛ*fir_m + nᵤ*2m + nᵤ+1)
  ϴfir = view(Θ,1:nₛ*fir_m)
  ΘG   = view(Θ,nₛ*fir_m+(1:nᵤ*2m))
  ϴσ   = view(Θ,nₛ*fir_m+nᵤ*2m+(1:nᵤ+1))

  # fir
  û  = zeros(T,nᵤ,N)
  for k = 0:nᵤ-1
    zdata     = IdentificationToolbox.iddata(u[k+1:k+1,:], r, Ts)
    A,B,F,C,D,info = IdentificationToolbox.pem(zdata,firmodel,zeros(T,nᵣ*fir_m),options)
    for j = 0:nᵣ-1
      i = nᵣ*k + j
      ϴfir[i*fir_m+(1:fir_m)] = Polynomials.coeffs(B[j+1])[nk[j+1]+1:fir_m+nk[j+1]]
    end
    ϴσ[k+1] = info.mse[1]
    û[k+1:k+1,:] += filt(B,F,r)
  end

  zdata   = IdentificationToolbox.iddata(y, û, Ts)
  options = IdentificationToolbox.IdOptions(iterations = 20, autodiff=:forward, estimate_initial=false)
  OEmodel = IdentificationToolbox.OE(m*ones(Int,1,nᵤ), m*ones(Int,1,nᵤ), ones(Int,1,nᵤ), 1, nᵤ)
  ΘG[:],_ = IdentificationToolbox._morsm(zdata, OEmodel, options)
  A,B,F,C,D,info = pem(zdata, OEmodel, ΘG[:], options)
  ΘG[:]   = info.opt.minimizer
  ϴσ[end] = info.mse[1]

  return Θ
end

function cost_smpe{T}(y,u,r,Ts, x::AbstractVector{T}, orders, firmodel, oemodel, options)
  n, m, nᵤ, nᵣ, N = orders
  nₛ = nᵤ*nᵣ
  fir_m = firmodel.orders.nb[1]

  xfir = view(x,1:nₛ*fir_m)
  xG   = view(x,nₛ*fir_m+(1:nᵤ*2m))
  xσ   = view(x,nₛ*fir_m+nᵤ*2m+(1:nᵤ+1))

  û  = zeros(T,nᵤ,N)
  costsum = zeros(T,1)
  for k = 0:nᵤ-1
    uₖ = u[k+1:k+1,:]
    fdata = IdentificationToolbox.iddata(uₖ, r, Ts)
    û[k+1,:] = IdentificationToolbox.predict(fdata, firmodel, xfir[k*nᵣ*fir_m+(1:nᵣ*fir_m)], options)
    costsum[:] += IdentificationToolbox.cost(uₖ, û[k+1:k+1,:], N, options)/abs(xσ[k+1])   #ϴσ[k+1]
  end
  zdata = IdentificationToolbox.iddata(y[1:1,:], û, Ts)
  ŷ = IdentificationToolbox.predict(zdata, oemodel, xG[:], options)
  costsum[:] += IdentificationToolbox.cost(y, ŷ, N, options)/abs(xσ[nᵤ+1])    #ϴσ[nᵤ+1]

  return costsum[1] + log(prod(abs(xσ)))
end
