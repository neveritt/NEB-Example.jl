N = 100
MC = 21
nᵤ = 1
n = 50
m = 2
nsr = 1.0

x = read("results-CL-$(nᵤ*m)-$(MC)-N$(N)-n$(n)-nsr$(nsr).dat",
  Float64, (nᵤ*m,MC))

twostagefit = x[1:1,:]
nebfit      = x[2:2,:]
nebxfit     = x[5:6,:]

avnebfit      = mean(nebfit[1,:])
avnebxfit     = mean(nebxfit[1,:])
avtwostagefit = mean(twostagefit[1,:])

minimum(nebfit[1,:])
minimum(twostagefit[1,:])

using Plots
using Colors
kthcolorpalette = [colorant"#1954a6"; colorant"#9d102d";
    colorant"#62922e"; colorant"#24a0d8"; colorant"#e4363e";
    colorant"#b0c92b"; colorant"#fab919"; colorant"#d85497";
    colorant"#65656c"; colorant"#bdbcbc"; colorant"#e3e5e3"]
pyplot(color_palette = kthcolorpalette)

twostagef = zeros(2,size(twostagefit,2))
k = 0
clip = 0.0
for col in 1:size(twostagefit,2)
  if all(twostagefit[:,col:col] .> clip )
    twostagef[:,k+1] = twostagefit[:,col]
    k += 1
  else
    twostagefit[:,col] = zeros(2,1)
  end
end
twostagef = twostagef[:,1:k]
twostagefit

miny = 0 #1-maximum(twostagefit)
maxy = 1 #-minimum(nebxfit)

x = vcat(nebfit[:,:],nebxfit[:,:])

scatter(x[[1,3],:].',twostagefit[1:1,:].',
    xlims=(0.9,1.01maxy),
    ylims=(0.9,1.01maxy),
    linestyle = :dot)

scatter(x[[2,4],:].',twostagefit[2:2,:].',
    xlims=(0.7,1.01maxy),
    ylims=(0.7,1.01maxy),
    linestyle = :dot)

scatter(x[1:1,:].',x[2:2,:].',
        xlims=(0.95,1.01maxy),
        ylims=(0.95,1.01maxy),
        linestyle = :dot)
plot!(0.96:0.01:1.0,0.96:0.01:1.0)

scatter(x[2:2,:].',x[4:4,:].',
        xlims=(0.80,1.01maxy),
        ylims=(0.80,1.01maxy),
        linestyle = :dot)


y = zeros(2,MC)
y[1,:] = (x[1:1,:].' + x[2:2,:].')/2
y[2,:] = (x[3:3,:].' + x[4:4,:].')/2
y

scatter(y[1:1,:].', y[2:2,:].',
        xlims=(0.98,1.01maxy),
        ylims=(0.98,1.01maxy),
        linestyle = :dot)


using StatPlots
using DataFrames
using LaTeXStrings

Fit = vcat(nebfit[1,:], twostagefit[1,:])

idmethods = [L"neb"]
for i = 1:MC-1
  push!(idmethods,L"neb")
end
for i = 1:length(twostagefit[1,:])
  push!(idmethods,L"smpe")
end
idmethods

df = DataFrame(method = idmethods,
  fit = Fit)

twostagefit[1,:]

violin(df,:method,:fit,marker=(0.2,stroke(0)),alpha=0.6, lab="")
boxplot!(df,:method,:fit,marker=(3,stroke(2)),color=5,alpha=0.6, lab="")


df2 = DataFrame(method = [L"smpe"], fit = [0.1])
yclip = clip + 0.003*randn(MC-size(twostagef,2))
xclip = 2.5 + 0.025*randn(MC-size(twostagef,2))
scatter!(xclip,yclip,marker=(0.3,stroke(0)),color=1,lab="")


savefig("boxplot.eps")
Plots.color_palette

immutable Boxplot{T}
  data::Vector{T}
end

nzerostwo = MC-size(twostagef,2)
twodata1 = Boxplot(twostagefit[1,:])()
twodata2 = Boxplot(twostagefit[2,:])()
twodata  = Boxplot((twostagef[1,:]+twostagef[2,:])/2)()
append!(twodata1[2],clip*ones(nzerostwo))
append!(twodata2[2],clip*ones(nzerostwo))
append!(twodata[2],clip*ones(nzerostwo))
nebdata1 = Boxplot(nebfit[1,:])()
nebdata2 = Boxplot(nebfit[2,:])()
nebdata  = Boxplot((nebfit[1,:]+nebfit[2,:])/2)()
nebxdata1 = Boxplot(nebxfit[1,:])()
nebxdata2 = Boxplot(nebxfit[2,:])()
nebxdata  = Boxplot((nebxfit[1,:]+nebxfit[2,:])/2)()

tikzfigureout("twostage-1.txt", twodata1)
tikzfigureout("twostage-2.txt", twodata2)
tikzfigureout("twostage.txt", twodata)
tikzfigureout("neb-1.txt", nebdata1)
tikzfigureout("neb-2.txt", nebdata2)
tikzfigureout("neb.txt", nebdata)
tikzfigureout("nebx-1.txt", nebxdata1)
tikzfigureout("nebx-2.txt", nebxdata2)
tikzfigureout("nebx.txt", nebxdata)

writecsv("twovsnebcl.csv", vcat(twostagefit[1:1,:],nebfit[1:1,:]).')
writecsv("smpecl.csv", transpose(twostagefit[1:1,:]))
writecsv("nebcl.csv", transpose(nebfit[1:1,:]))

writecsv("smpenl1.csv", transpose(twostagefit[1:1,:]))
writecsv("smpenl2.csv", transpose(twostagefit[2:2,:]))
writecsv("nebnl1.csv", transpose(nebfit[1:1,:]))
writecsv("nebnl2.csv", transpose(nebfit[2:2,:]))
writecsv("nebxnl1.csv", transpose(nebxfit[1:1,:]))
writecsv("nebxnl2.csv", transpose(nebxfit[2:2,:]))
writecsv("twovsneb.csv", vcat(twostagefit[1:1,:],nebfit[1:1,:]).')
writecsv("twovsnebx.csv", vcat(twostagefit[1:1,:],nebxfit[1:1,:]).')
writecsv("twovsneb2.csv", vcat(twostagefit[2:2,:],nebfit[2:2,:]).')
writecsv("twovsnebx2.csv", vcat(twostagefit[2:2,:],nebxfit[2:2,:]).')

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
