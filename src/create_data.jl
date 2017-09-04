function _readdata(df)
  k = size(df,2)
  s = Matrix{Float64}(size(df,1),k)
  for i = 1:k
    s[:,i] = dfs[:,i]
  end
  return s
end

function _create_data(N,m,nᵤ,nᵣ,s0,nsru,nsry,nsry2,Θ₀,Θ₂)
  r = randn(nᵣ,N)
  u0 = zeros(nᵤ,N)
  for i in 0:nᵤ-1
    for j in 0:nᵣ-1
      idx = i*nᵤ+j+1
      u0[i+1,:] += filt(s0[:,idx],[1],r[j+1,:])
    end
  end
  u0

  y0 = zeros(2,N)
  y0[1,:] = filt(vcat(zeros(1), Θ₀[1:m]),vcat(ones(1), Θ₀[m+1:2m]), u0[1,:]) +
            filt(vcat(zeros(1), Θ₀[2m+1:3m]),vcat(ones(1), Θ₀[3m+1:4m]), u0[2,:])
  y0[2,:] = filt(vcat(zeros(1), Θ₂[1:m]),vcat(ones(1), Θ₂[m+1:2m]), y0[1,:])
  u       = u0 + sqrt.(sum(abs2,u0,2)/N*nsru).*randn(2,N)
  y       = zeros(2,N)
  σ₁      = sqrt.(sum(abs2,y0[1,:])/N*nsry)
  σ₂      = sqrt.(sum(abs2,y0[2,:])/N*nsry2)
  y[1,:]  = y0[1:1,:] + σ₁.*randn(1,N)
  y[2,:]  = y0[2:2,:] + σ₂.*randn(1,N)
  return y,u,r,y0,u0,σ₁,σ₂
end
