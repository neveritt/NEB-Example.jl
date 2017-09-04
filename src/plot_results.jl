cd("src")
N = 150
MC = 100
nᵤ = 2
n = 60
m = 2
nsr = 0.1

x = read("results-$(nᵤ*5)-$(MC)-N$(N)-n$(n)-nsr$(nsr).dat",
  Float64, (nᵤ*5,MC))

x = read("results-CL-2-$(MC)-N$(N)-n$(n)-nsr$(nsr).dat",
  Float64, (2,MC))

Θvec = read("theta-$(nᵤ*5)-$(MC)-N$(N)-n$(n)-nsr$(nsr).dat",
  Float64, (m*4,5MC))
Θvec = read("theta-CL-2-$(MC)-N$(N)-n$(n)-nsru$(nsr).dat",
    Float64, (2m,2MC))

Θsmpe = Θvec[:,1:MC]
Θneb = Θvec[:,MC+(1:MC)]
Θnebx = Θvec[:,2MC+(1:MC)]

mean(Θsmpe,2)
mean(Θneb,2)
mean(Θnebx,2)

var(Θsmpe,2)
var(Θneb,2)
var(Θnebx,2)

open("test.txt", "w") do f
  for num in mean(Θneb,2)
  write(f, @sprintf(""), num)
end

writecsv("neb-mean.csv", mean(Θneb,2))
writecsv("nebx-mean.csv", mean(Θnebx,2))
writecsv("neb-var.csv", N*var(Θneb,2))
writecsv("nebx-var.csv", N*var(Θnebx,2))

writecsv("neb-CL-mean.csv", mean(Θneb,2))
writecsv("neb-CL-var.csv", N*var(Θneb,2))

smpefit = x[1:2,:]
nebfit  = x[3:4,:]
nebxfit = x[5:6,:]
twofit  = x[7:8,:]
bayfit  = x[9:10,:]

avnebfit      = mean(nebfit[1,:])
avnebxfit     = mean(nebxfit[1,:])
avsmpefit = mean(smpefit[1,:])

minimum(nebxfit[2,:])
minimum(smpefit[1,:])

using Plots
using Colors
kthcolorpalette = [colorant"#1954a6"; colorant"#9d102d";
    colorant"#62922e"; colorant"#24a0d8"; colorant"#e4363e";
    colorant"#b0c92b"; colorant"#fab919"; colorant"#d85497";
    colorant"#65656c"; colorant"#bdbcbc"; colorant"#e3e5e3"]
#pyplot(color_palette = kthcolorpalette)

smpef = zeros(2,size(smpefit,2))
thetasmpe = zeros(size(Θsmpe))
k = 0
clip = 0.0
for col in 1:size(smpefit,2)
  if all(smpefit[:,col:col] .> clip )
    smpef[:,k+1] = smpefit[:,col]
    thetasmpe[:,k+1] = Θsmpe[:,col]
    k += 1
  else
    smpefit[:,col] = zeros(2,1)
  end
end
smpef = smpef[:,1:k]
smpefit
thetasmpe = thetasmpe[:,1:k]

writecsv("smpe-mean.csv", mean(thetasmpe,2))
writecsv("smpe-var.csv", N*var(thetasmpe,2))

writecsv("smpe-CL-mean.csv", mean(thetasmpe,2))
writecsv("smpe-CL-var.csv", N*var(thetasmpe,2))

miny = 0 #1-maximum(smpefit)
maxy = 1 #-minimum(nebxfit)

z = vcat(nebfit[:,:],nebxfit[:,:])

scatter(z[[1,3],:].',smpefit[1:1,:].',
    xlims=(0.9,1.01maxy),
    ylims=(0.9,1.01maxy),
    linestyle = :dot)

scatter(z[[2,4],:].',smpefit[2:2,:].',
    xlims=(0.7,1.01maxy),
    ylims=(0.7,1.01maxy),
    linestyle = :dot)

scatter(z[1:1,:].',z[3:3,:].',
        xlims=(0.8,1.01maxy),
        ylims=(0.8,1.01maxy),
        linestyle = :dot)
plot!(0.8:0.01:1.0,0.8:0.01:1.0)

scatter(z[2:2,:].',z[4:4,:].',
        xlims=(0.80,1.01maxy),
        ylims=(0.80,1.01maxy),
        linestyle = :dot)


y = zeros(3,MC)
y[1,:] = (x[1:1,:].' + x[2:2,:].')/2
y[2,:] = (x[3:3,:].' + x[4:4,:].')/2
y[3,:] = (x[5:5,:].' + x[6:6,:].')/2

scatter(y[1:1,:].', y[2:2,:].',
        xlims=(0.4,1.01maxy),
        ylims=(0.4,1.01maxy),
        linestyle = :dot)

scatter!(y[1:1,:].', y[3:3,:].',
                xlims=(0.4,1.01maxy),
                ylims=(0.4,1.01maxy),
                linestyle = :dot)


using StatPlots
using DataFrames
using LaTeXStrings

Fit = vcat(nebfit[1,:], smpefit[1,:])

idmethods = [L"neb"]
for i = 1:MC-1
  push!(idmethods,L"neb")
end
for i = 1:length(smpefit[1,:])
  push!(idmethods,L"smpe")
end
idmethods

df = DataFrame(method = idmethods,
  fit = Fit)

smpefit[1,:]

violin(df,:method,:fit,marker=(0.2,stroke(0)),alpha=0.6, lab="")
boxplot!(df,:method,:fit,marker=(3,stroke(2)),color=5,alpha=0.6, lab="")


df2 = DataFrame(method = [L"smpe"], fit = [0.1])
yclip = clip + 0.003*randn(MC-size(smpef,2))
xclip = 2.5 + 0.025*randn(MC-size(smpef,2))
scatter!(xclip,yclip,marker=(0.3,stroke(0)),color=1,lab="")


savefig("boxplot.eps")
Plots.color_palette

immutable Boxplot{T}
  data::Vector{T}
end

nzerostwo = MC-size(smpef,2)
twodata1 = Boxplot(smpefit[1,:])()
twodata2 = Boxplot(smpefit[2,:])()
twodata  = Boxplot((smpef[1,:]+smpef[2,:])/2)()
append!(twodata1[2],clip*ones(nzerostwo))
append!(twodata2[2],clip*ones(nzerostwo))
append!(twodata[2],clip*ones(nzerostwo))
nebdata1 = Boxplot(nebfit[1,:])()
nebdata2 = Boxplot(nebfit[2,:])()
nebdata  = Boxplot((nebfit[1,:]+nebfit[2,:])/2)()
nebxdata1 = Boxplot(nebxfit[1,:])()
nebxdata2 = Boxplot(nebxfit[2,:])()
nebxdata  = Boxplot((nebxfit[1,:]+nebxfit[2,:])/2)()

tikzfigureout("smpe-1.txt", twodata1)
tikzfigureout("smpe-2.txt", twodata2)
tikzfigureout("smpe.txt", twodata)
tikzfigureout("neb-1.txt", nebdata1)
tikzfigureout("neb-2.txt", nebdata2)
tikzfigureout("neb.txt", nebdata)
tikzfigureout("nebx-1.txt", nebxdata1)
tikzfigureout("nebx-2.txt", nebxdata2)
tikzfigureout("nebx.txt", nebxdata)

writecsv("twovsnebcl.csv", vcat(smpefit[1:1,:],nebfit[1:1,:]).')
writecsv("smpecl.csv", transpose(smpefit[1:1,:]))
writecsv("nebcl.csv", transpose(nebfit[1:1,:]))

writecsv("smpenl1.csv", transpose(smpefit[1:1,:]))
writecsv("smpenl2.csv", transpose(smpefit[2:2,:]))
writecsv("nebnl1.csv", transpose(nebfit[1:1,:]))
writecsv("nebnl2.csv", transpose(nebfit[2:2,:]))
writecsv("nebxnl1.csv", transpose(nebxfit[1:1,:]))
writecsv("nebxnl2.csv", transpose(nebxfit[2:2,:]))
writecsv("twonl1.csv", transpose(twofit[1:1,:]))
writecsv("twonl2.csv", transpose(twofit[2:2,:]))
writecsv("baynl1.csv", transpose(bayfit[1:1,:]))
writecsv("baynl2.csv", transpose(bayfit[2:2,:]))
writecsv("twovsneb.csv", vcat(smpefit[1:1,:],nebfit[1:1,:]).')
writecsv("twovsnebx.csv", vcat(smpefit[1:1,:],nebxfit[1:1,:]).')
writecsv("twovsneb2.csv", vcat(smpefit[2:2,:],nebfit[2:2,:]).')
writecsv("twovsnebx2.csv", vcat(smpefit[2:2,:],nebxfit[2:2,:]).')

function tikzfigureout(s, bdata)
  d, o = bdata
  open(s, "w") do f
    write(f, "\\addplot\n")
    write(f, "  [boxplot prepared={\n")
    write(f, "    lower whisker=$(d[1]),\n")
    write(f, "    lower quartile=$(d[2]),\n")
    write(f, "    median=$(d[3]),\n")
    write(f, "    upper quartile=$(d[4]),\n")
    write(f, "    upper whisker=$(d[5])}]\n")
    # if length(0) > 0
    #   write(f, "  table[row sep=\\\\, y index=0,mark=*] { ")
    #   for elem in o
    #     write(f, "$(elem)\\\\ ")
    #   end
    #   write(f, "};\n")
    # else
      write(f, "  coordinates{};")
    #end
  end
end

function _boxplotinfo{T}(r::Boxplot{T})
  rdata = r.data
  Q1,Q2,Q3 = quantile(rdata,[0.25,0.5,0.75])
  IQ = Q3-Q1
  lif = max(Q1-1.5IQ, minimum(rdata)) # lower inner fence
  uif = min(Q3+1.5IQ, maximum(rdata)) # upper inner fence
  lof = max(Q1-3IQ, minimum(rdata))   # lower outer fence
  uof = min(Q3+3IQ, maximum(rdata))   # upper outer fence

  lo = filter(x->x<lif,rdata)         # lower outliers
  uo = filter(x->x>uif,rdata)         # upper outliers
  return [lif, Q1, Q2, Q3, uif], vcat(lo,uo)
end

using Compat
@compat (r::Boxplot)() = return _boxplotinfo(r)
