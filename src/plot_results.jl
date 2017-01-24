N = 200
MC = 100
nᵤ = 2
methods = 3
nsr = 0.1

x = read("results-$(nᵤ*methods)-$(MC)-N$(N)-nsr$(nsr).dat", Float64, (nᵤ*methods,MC))

twostagefit = x[1:2,:]
nebfit      = x[3:4,:]
nebxfit     = x[5:6,:]

avnebfit      = mean(nebfit)
avnebxfit     = mean(nebxfit)
avtwostagefit = mean(twostagefit[2,:])

using Plots
pyplot()


miny = 0 #1-maximum(twostagefit)
maxy = 1 #-minimum(nebxfit)

x = vcat(nebfit[:,:],nebxfit[:,:])

scatter(x[[1,3],:].',twostagefit[1:1,:].',
    xlims=(0.95,maxy),
    ylims=(miny,maxy),
    linestyle = :dot)

scatter(x[[2,4],:].',twostagefit[2:2,:].',
    xlims=(0.7,maxy),
    ylims=(miny,maxy),
    linestyle = :dot)
