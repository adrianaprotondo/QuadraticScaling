### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ fb54ebe4-0f2d-4cae-9b73-93f03a08b391
import Pkg; Pkg.add("Plots")

# ╔═╡ b8e6ab2c-1be5-4524-8504-265d8462bc2a
Pkg.add("LaTeXStrings")

# ╔═╡ da36fef5-42f1-4a70-943c-e2b292456565
Pkg.add("Measures")

# ╔═╡ 9ba0b30f-8fdb-49a3-a179-53f7bfe03a62
Pkg.add("PlutoUI")

# ╔═╡ f7a7df1b-d246-4cd5-ac46-7eee94c78037
Pkg.add("PlutoUI")

# ╔═╡ 43619fa2-ea61-419d-b86e-c673301bc685
using Plots

# ╔═╡ 0d17a04e-0334-4ff1-8c9b-bad141346414
using Statistics

# ╔═╡ cbdeb143-b778-4b32-9434-bb630bd891d5
using LaTeXStrings

# ╔═╡ d428d47b-e171-479c-8d88-e18a4ab51ccd
using Measures

# ╔═╡ d1fd22e3-93ba-4483-ad90-00389fe747db
using PlutoUI

# ╔═╡ ab9b01ff-c53f-40d6-817e-07e837762cfb
md"""
The aim of this notebook is to illustrate the effect of adding dimensions to a loss landscape for learning performance. We will consider quadratic functions and to start with going from a 1D to 2D quadratic function"""

# ╔═╡ 7c5a27bc-2da7-4a4e-a957-7dbc80e0e556
Plots.default(titlefont = (20, "times"), legendfontsize = 18, guidefont = (18, :black), tickfont = (12, :black), grid=false, guide = "x", framestyle = :zerolines, yminorgrid = false, markersize=6)

# ╔═╡ 2b7d5e2d-3b9b-42dc-bc08-3fd5106b7ebe
begin 
	width=1200
	height=800
end

# ╔═╡ 38125f75-7244-4526-9a37-9bd1af217e72
md"# Function definition"

# ╔═╡ cea61646-34cf-42b4-acf8-2226fde52146
md"We define a 1D quadratic function $f(x)=x^2$"

# ╔═╡ 247f31f4-dc34-11ec-0584-752289f2b41d
begin
	f_1(w) = w^2 # 1d quadratic function
	df_1(w) = [2w]
	eig1 = [2]
end

# ╔═╡ edadadec-a987-4661-add6-f0094bb9934c
function quadraticN_Zero(N,Tr_N)
	# generate a N-dim quad function with one 
	# non-zero eig such that Tr(H)/N = Tr_N
	# return function computing the value 
	# derivative and eigenvalues
	eig = Tr_N*N # non-zero eig 
	eigs = zeros(N)
	eigs[1] = eig
	f(w) = eig/2*w[1]^2
	df(w) = eigs.*w
	return f, df, eigs
end

# ╔═╡ 766b31e0-ecf8-4d01-9a8e-f1bf3d03efd7
function quadraticN_NonZero(N,Tr_N)
	# generate a N-dim quad function with  
	# all eq eig such that Tr(H)/N = Tr_N
	# return function computing the value 
	# derivative and eigenvalues
	eig = Tr_N # non-zero eig 
	eigs = eig*ones(N)
	f(w) = sum(w.^2 .*eig/2)
	df(w) = eigs.*w
	return f, df, eigs
end

# ╔═╡ 0e1a0954-c70d-473d-8cf4-7c07449befbf
function gradDescent(f,df,mu,w0,epochs=1)
	W = zeros(epochs,length(w0))
	F = zeros(epochs)
	w=w0
	for i=1:epochs
		W[i,:] = w
		F[i] = f(w)
		w = w -mu*df(w)
	end
	return F,W
end

# ╔═╡ 7ed096fa-fe5b-400e-af6a-45fb34c1226a
function gradDescentNoise(f,df,mu,w0,epochs=1,gamma=0,Wset=false)
	W = zeros(epochs,length(w0))
	dF = zeros(epochs,length(w0))
	grad = zeros(epochs)
	N = zeros(epochs,length(w0))
	F = zeros(epochs)
	w=w0
	for i=1:epochs
		W[i,:] = w
		F[i] = f(w)
		grad[i] = sqrt(sum(df(w).^2))
		dF[i,:] = df(w)
		N[i,:] = randn(length(w))
		if Wset==false
			w = w -mu*dF[i,:] + gamma*N[i,:]
		else
			w = Wset[i,:]
		end
	end
	return F,W,dF,N,grad
end

# ╔═╡ 18251608-65c1-414c-a9cd-734916c4b200
function ls(F,l=10)
	dF = (F[2:l+1]-F[1:l])
	return mean(-dF)
end

# ╔═╡ bfb47a2e-592d-433f-b701-685af896d659
function ss(F,l=20)
	return mean(F[end-l:end])
end

# ╔═╡ 831e1216-f083-4994-9934-254a480a2aef
function plotLsSS(Fs,xlbl,t,Fss=false)
	lsA = [ls(i) for i in Fs]
	if Fss==false
		ssA = [ss(i) for i in Fs]
	else
		ssA = [ss(i) for i in Fss]
	end
	lsP = plot(xlbl,lsA,lw=3,ylabel="learning speed",title=t)
	ssP = plot(xlbl,ssA,lw=3, ylabel="ss value")
	plot(lsP, ssP, layout = (2, 1), legend = false)
	# plot!(title=t)
end

# ╔═╡ 54b4f704-1bc2-40d0-9c25-af7e2440d401
function computeHessianProj(dW,eig)
	# compute hessian projection of normalized dW given eig the eigenvalues of hess
	P = zeros(size(dW,1))
	for i=1:size(dW,1)
		P[i] = sum(dW[i,:].^2 .*eig)/sum(dW[i,:].^2)
	end
	return P
end

# ╔═╡ 5dbdd7b7-0075-4e97-8e71-fa7d8371e5fc
function TrH_N(eig)
	return sum(eig)/length(eig)
end

# ╔═╡ ca0b97fe-d673-490a-b5cc-f940fe79100c
function TrH3_TrH2(eig)
	return sum(eig.^3)/sum(eig.^2)
end

# ╔═╡ 39fd38b7-097f-4693-a8e8-aa5a8613abbc
function sim(f,df,mu,w0,epochs,gamma,lls=10,lss=200,w0ss=0.00001)
	# train
	if gamma==0
		F,W = gradDescent(f,df,mu,w0,epochs)
	else
		F,W,dF,N,grad=gradDescentNoise(f,df,mu,w0,epochs,gamma)
		Dls = Dict(:F => F, :W => W, :dF => dF, :Noise => N, :grad => grad)
	end
	# get ls 
	ls1 = ls(F,lls)
	# train again for ss
	w0ssV = w0ss*ones(length(w0)) #point close to minimum 0
	FSS,WSS,dFSS,NSS,gradSS=gradDescentNoise(f,df,mu,w0ssV,epochs,gamma)
	Dss = Dict(:F => FSS, :W => WSS, :dF => dFSS, :Noise => NSS, :grad => gradSS)
	ss1 = ss(FSS,lss)
	return ls1 ,ss1, Dls, Dss
end

# ╔═╡ 2e288a30-0ac6-4a3e-8eb4-b3e7cc737cf2
function simulateN(Ns,quadF,Tr_N,w0,musVar,SNR,epochs,analyseF=false,args=[])
	val = map(1:length(Ns)) do i
		N = Ns[i]
		f,df,eigs = quadF(N,Tr_N)
		w0V = (w0/sqrt(N))*ones(N) # scale the initial weight to match initial val
		map(musVar[i]) do mu
			gamma = mu/SNR
			sim(f,df,mu,w0V,epochs,gamma) # get learning parameters
		end
	end
	if analyseF == false
		return val
	else
		return analyseF(val,args...)
	end
end

# ╔═╡ 6256c83c-2ee5-4b0e-865a-186649b1bbda
function scatterPlot(indeces,ssS,lsS)
	namedColors = ["Blues","Oranges","Greens","Purples","Reds","Grays"]
	# namedColors = Plots.palette(:inferno, 8)
	p = scatter(ssS[indeces[1]],lsS[indeces[1]], c=colormap(namedColors[1], length(ssS[indeces[1]])+2)[3:end], label=string("N = ",indeces[1]))
	for j in 2:length(indeces)
		itr = indeces[j]
		scatter!(ssS[itr],lsS[itr],c=colormap(namedColors[j],length(ssS[itr])+2)[3:end], label="N = $itr")
	end
	plot!(xlabel="steady state value",ylabel="learning speed")
	return p
end

# ╔═╡ 979f4a5f-12cd-4d5b-91dc-e26ef174ad6b
function swapOrder(v)
	vv = zeros(length(v[1]))
	for j in 1:length(v[1])
		vv[j] = [v[i][j] for i=1:length(v)]
	end
	return vv
end

# ╔═╡ 70018fc9-d67a-4fc0-a902-b3c4ff4ecd25
function getVal(v,i,sym=false)
	map(v) do vv
		map(vv) do vvv
			if sym == false
				vvv[i]
			else
				vvv[i][sym]
			end
		end
	end
end

# ╔═╡ 8b0bb7b7-c45e-483a-bcf0-46cc3d0db07d
function getLs_SS(vals_zero,args=[])
	lsS = getVal(vals_zero,1)
	ssS = getVal(vals_zero,2)
	return lsS,ssS
end

# ╔═╡ c9b98516-2620-4c81-8b3e-11f69feb5b62
function my_std(ls)
    numSim = length(ls)
    lsM = hcat(ls...)
    map(1:size(lsM,1)) do m
        std(lsM[m,:])./numSim
    end
end

# ╔═╡ fc03f090-0912-4875-a2a8-f663620d26d6
md"# Gradient descent"

# ╔═╡ 230d6234-ce7e-41ea-9f24-fc2d5be65e1c
begin
	epochs=400;
	Ns = 1:1:100;
	mus=collect(0.001:0.005:0.1);
	Tr_N=3;
	SNR=5;
	w0 = 5;
	musVar = [mus./N^1 for N in Ns]
	musC =  [mus for N in Ns]
	sims = 1
	# musVar = [mus for N in Ns]
end

# ╔═╡ b6eacae8-902d-46ad-8c21-916c3aef7776
mus

# ╔═╡ 8ed79bf6-adff-41ed-83b3-5a5921b2526e
p = Plots.palette(:inferno, 20)

# ╔═╡ 665e6a7a-0aa6-4e15-8e76-90d81167a4fd
begin
	lblN = string.(L"N=",Ns)
	lblN = reshape(lblN,(1,length(lblN)))
		lblMuV = map(1:length(Ns)) do i
		string.(L"\gamma=",round.(musVar[i],digits=4))
	end
	lblMu = string.(L"\gamma=",mus)
	lblMu = reshape(lblMu,(1,length(lblMu)))
	# lblMu = reshape(lblMu,(1,length(lblN)))
end

# ╔═╡ c11ab6c9-c799-405d-b76b-0b194e22990f
ls_ssAll = map(1:sims) do i
	simulateN(Ns,quadraticN_Zero,Tr_N,w0,musVar,SNR,epochs,getLs_SS)
end

# ╔═╡ 2c3cdc13-dce0-4c43-8eaa-53ca08551e8e
begin
	lsAll = map(ls_ssAll) do l
		l[1]
	end
	ssAll = map(ls_ssAll) do l
		l[2]
	end
	lsAllM = mean(lsAll)
	lsAllStd = my_std(lsAll)
	ssAllM = mean(ssAll)
	ssALlStd = my_std(ssAll)
end

# ╔═╡ 14d09075-61fd-4c93-a75f-d3ba99e82f70
begin
	int = 1:5:length(Ns)
	plot(musVar[int],lsAllM[int],lw=3,xlabel="learning step", ylabel="learning speed",label=lblN[:,int],palette=p)
end

# ╔═╡ 4173bd81-a370-4936-87e4-c3ce5f2205f9
begin
	N = Ns[10]
	f,df,eigs = quadraticN_Zero(N,Tr_N)
	w0V = (w0/sqrt(N))*ones(N) # scale the initial weight to match initial val
	mu = musVar[10][end]
	gamma = mu/SNR
	lsVal,ssVal,lsD,ssD=sim(f,df,mu,w0V,50,gamma,10,10)
	plot(lsD[:F],lw=3)
end

# ╔═╡ aaf2e829-4232-453a-914b-6c628406fc98
begin
	plot(musVar[int],ssAllM[int],lw=3,xlabel="learning step", ylabel="steady state value",label=lblN[:,int],palette=p)
end

# ╔═╡ 5b738eb7-ae34-41a7-81bc-00738f3a6ebf
begin
	indeces = [5,10,30,100]
	scatterPlot(indeces,ssAllM,lsAllM)
end

# ╔═╡ 927847fb-8ecf-4248-a131-32dc4f845779
begin
	lsMax = [maximum(lsAllM[i]) for i=1:length(lsAllM)]
	lsMaxInd = [findfirst(lsAllM[i].==lsMax[i]) for i=1:length(lsAllM)]
	muMax = [musVar[i][lsMaxInd[i]] for i=1:length(lsMaxInd)]
	ssMin = [minimum(ssAllM[i]) for i=1:length(ssAllM)]
	ssAtLsMax = [ssAllM[i][lsMaxInd[i]] for i=1:length(ssAllM)]
end

# ╔═╡ a5fe4cfb-9a8c-4c15-8f00-f0d6242874b9
plot(Ns,lsMax,lw=3,xlabel="N",ylabel="Max learning speed",legend=false)

# ╔═╡ b8d10a77-6392-46c4-9513-3cdbf6cc07b9
begin
	p1=plot(Ns, ssMin , lw=3, xlabel="N", ylabel="ss value", label="min")
	p2=plot(Ns, ssAtLsMax, lw=3, label="ss at max ls", xlabel="N", ylabel="ss value")
	plot(p1,p2,layout=(2,1))
end

# ╔═╡ f551874c-8033-48c4-b20d-787072dd92ef
md"## Non-zero eigenvalue"

# ╔═╡ acae4567-5a77-439b-be3b-ab422c47a1e5
ls_ssAllN = map(1:sims) do i
	simulateN(Ns,quadraticN_NonZero,Tr_N,w0,musC,SNR,epochs,getLs_SS)
end

# ╔═╡ 97ebb932-6ff2-46f1-84da-26affa0dd5d9
# vals_Nzero = map(1:length(Ns)) do i
# 	N = Ns[i]
# 	f,df,eigs = quadraticN_NonZero(N,Tr_N)
# 	w0V = (w0/sqrt(N))*ones(N) # scale the initial weight to match initial val
# 	map(mus) do mu # use same ls for all nets because no problem of scaling
# 		gamma = mu/SNR
# 		sim(f,df,mu,w0V,epochs,gamma)
# 	end
# end

# ╔═╡ e26d05aa-1d5e-47af-9285-a35afb616e45
begin
	lsAllN = map(ls_ssAllN) do l
		l[1]
	end
	ssAllN = map(ls_ssAllN) do l
		l[2]
	end
	lsAllNM = mean(lsAllN)
	lsAllNStd = my_std(lsAllN)
	ssAllNM = mean(ssAllN)
	ssALlNStd = my_std(ssAllN)
end

# ╔═╡ b1aedaf9-8b55-4583-b6a8-f688019a9021
plot(mus,lsAllNM,lw=3,xlabel="learning step", ylabel="learning speed",label=lblN,palette=p)

# ╔═╡ b08bd2de-0222-4413-a03b-aa8fa256b926
begin
	plot(mus,ssAllNM[int],lw=3,xlabel="learning step", ylabel="steady state value",label=lblN[:,int],palette=p)
end

# ╔═╡ 62ebb223-130e-4c22-a967-dab914374dd6
scatterPlot(indeces,ssAllNM,lsAllNM)

# ╔═╡ 8bbf893d-5f3e-491c-8723-52e605af7322
begin
	lsMaxN = [maximum(lsAllNM[i]) for i=1:length(lsAllNM)]
	lsMaxIndN = [findfirst(lsAllNM[i].==lsMaxN[i]) for i=1:length(lsAllNM)]
	muMaxN = [mus[lsMaxIndN[i]] for i=1:length(lsMaxIndN)]
	ssMinN = [minimum(ssAllNM[i]) for i=1:length(ssAllNM)]
	ssAtLsMaxN = [ssAllNM[i][lsMaxIndN[i]] for i=1:length(ssAllNM)]
end

# ╔═╡ dd70856e-a9dc-47d3-941e-79bfa4d65565
plot(Ns,lsMaxN,lw=3,xlabel="N",ylabel="Max learning speed")

# ╔═╡ ad1a4a0b-2750-48f2-9751-af448c84a208
begin
	p1N=plot(Ns, ssMinN ,lw=3, xlabel="N", ylabel="ss value", label="min")
	p2N=plot(Ns, ssAtLsMaxN,lw=3,label="ss at max ls",xlabel="N", ylabel="ss value")
	plot(p1N,p2N,layout=(2,1))
end

# ╔═╡ 9b2e008e-2243-4fc8-81c4-29f5f31d70c9
md"""
We have done gradient descent on the three different functions starting at the same value. We observe that the 2D with one zero eigenvalue has larger learning speed. We expected this because the gradient of this function is larger. The steady state loss is the same because in all cases gradient descent reaches the minimum $0$."""

# ╔═╡ 51995ba1-8ac3-444c-be53-95212904347c
md""" ## Comparison figures"""

# ╔═╡ 11e6db80-d008-43c2-805a-93174bd8e2ff
begin 
	pls = plot(musVar[int],lsAllM[int],lw=3,xlabel="", ylabel="learning speed",label=lblN[:,int],palette=p)
	plot!(title="1 non-zero",legend=false)
	plsN = plot(mus,lsAllNM[int],lw=3,xlabel="", ylabel="",label=lblN[:,int],palette=p,legend=:topleft)
	plot!(title="all non-zero")
	pss = plot(musVar[int],ssAllM[int],lw=3,xlabel="learning step", ylabel="steady state value",label=lblN[:,int],palette=p)
	plot!(legend=false)
	plot!(left_margin=10mm,bottom_margin=10mm)
	pssN = plot(mus,ssAllNM[int],lw=3,xlabel="learning step", ylabel="",label=lblN[:,int],palette=p)
	# plot!(title="1 non-zero eig")
	plot!(legend=false)
	# psp=scatterPlot(indeces,ssAllM,lsAllM)
	# plot!(legend=false)
	# pspN=scatterPlot(indeces,ssAllNM,lsAllNM)
	# plot!(legend=false)
	# pLP= plot(pls,plsN,pss,pssN,psp,pspN, layout=(3,2), size = (width, height+400))
	# savefig("./Figures/lp_sim_all2.pdf")
	pLP= plot(pls,plsN,pss,pssN, layout=(2,2), size = (width, height))
	savefig("./Figures/lp_sim_all.pdf")
	pLP
end

# ╔═╡ 3123b4c1-1dc8-4ecb-beb1-2599ac3265b1
begin
	psp=scatterPlot(indeces,ssAllM,lsAllM)
	plot!(title="1 non-zero",legend=false)
	plot!(legend=false)
	pspN=scatterPlot(indeces,ssAllNM,lsAllNM)
	plot!(title="all non-zero")
	plsM = plot(Ns,lsMax,lw=3,xlabel="",ylabel="Max learning speed",legend=false,right_margin=10mm)
	# plot!(title="1 non-zero",legend=false)
	plot!(twinx(),Ns,muMax,lw=3,axis=:right,color=:orange,legend=false,ylabel="", xlabel="")
	plsMN = plot(Ns,lsMaxN,lw=3,xlabel="",ylabel="",legend=false,right_margin=30mm)
	plot!(twinx(),Ns, muMaxN, lw=3, axis=:right, color=:orange,ylabel="learning step at max",legend=false, xlabel="", ylims=(0, muMaxN[1]+muMaxN[1]))
	# plot!(title="all non-zero")
	pssM =plot(Ns, ssAtLsMax, lw=3, legend=false, xlabel="N", ylabel="ss at max ls")
	pssMN =plot(Ns, ssAtLsMaxN, lw=3, legend=false, xlabel="N", ylabel="")
	pMax = plot(psp,pspN,plsM,plsMN,pssM,pssMN, layout=(3,2), size = (width+50, height+400))
	savefig("./Figures/lp_max_sim_all2.pdf")
	pMax
end

# ╔═╡ b3e607b5-ee76-46fe-9af0-3e2f5dc7c2ff
md"""
We now evaluate our theoretical measures of learning speed and steady-state loss to understand the results above."""

# ╔═╡ 248290dc-d84c-4f1a-9959-c95888c48d5c
md" # Theoretical analysis"

# ╔═╡ ad5b0b3f-f423-4d43-8d3e-b861de19904e
md"""The expected learning speed over teh dirstribution of $\mathbf{\epsilon}$ is 
$$\mathbb{E}[\nu_t] = -\frac{1}{F_t\delta t} (-\gamma ||\nabla F||^2+\gamma^2\nabla F^T\nabla^2F\nabla F+\mathbb{E}[\eta^2 \epsilon^T \nabla^2 F \epsilon])$$"""

# ╔═╡ cf9bea4e-f659-46c6-861b-42c1476bb3df
function learningSpeed(F,mu,dF,grad,gamma,Noise,eig,dt=1)
	val= -1 ./(F.*dt).*(-mu.*grad.^2 .+ mu^2 .*grad.^2 .*computeHessianProj(dF,eig) .+gamma^2 .*sum(Noise.^2 ;dims=2).*computeHessianProj(Noise,eig))
	return val[:,1]
end

# ╔═╡ 7f202ff4-0342-4e62-a57f-ec2f702db5d3
function learningSpeed(dict,mu,gamma,eig,dt=1)
	learningSpeed(dict[:F],mu,dict[:dF],dict[:grad],gamma,dict[:Noise],eig,dt)
end

# ╔═╡ 5266893d-fb62-40df-a90d-c84a08ecdfbe
md"""The expected local task difficulty over  dirstribution of $\mathbf{\epsilon}$ is 
$$\mathbb{E}[\nu_t] = \gamma \hat{\nabla F}^T\nabla^2F\hat{\nabla F}+\mathbb{E}[\frac{\eta^2}{\gamma} \frac{1}{||\nabla F||^2} \epsilon^T \nabla^2 F \epsilon])$$"""

# ╔═╡ edb982ce-9b84-4d61-bdc2-fb0794db7ae4
function localTask(dF,N,eig,grad,mu,gamma)
	# val= mu*computeHessianProj(dF,eig)+gamma^2/mu.*(sum(N.^2; dims=2)./grad.^2).*computeHessianProj(N,eig)
	val= mu*computeHessianProj(dF,eig)+gamma^2/mu.*(sum(N.^2; dims=2)./grad.^2).*computeHessianProj(N,eig).*grad.^2
	return val[:,1]
	# 1/2*computeHessianProj(dW21NSet.+N21Set,eig21).*(sum((dW21NSet.+N21Set).^2;dims=2).*grad21NSet.^2)
	# return 1/2*computeHessianProj(dW,eig).*(dW.^2 ./grad.^2) + 1/2*computeHessianProj(N,eig).*(N.^2 ./grad.^2)
end

# ╔═╡ 4e95640a-9111-43fc-b4c0-b7bae236f020
function localTask(dict,eig,mu,gamma)
	localTask(dict[:dF],dict[:Noise],eig,dict[:grad],mu,gamma)
end

# ╔═╡ 82119ec9-bc81-467d-a5d8-97165f525fbe
function getLpVar(vals_Nzero,Ns,quadF,mus,SNR,index=false)
	lp = map(1:length(vals_Nzero)) do i
		N = Ns[i]
		f,df,eigs = quadF(N,Tr_N)
		# w0V = (w0/sqrt(N))*ones(N) # scale the initial weight to match initial val
		map(1:length(vals_Nzero[i])) do j
			mu = mus[j]
			gamma = mu/SNR
			dict = vals_Nzero[i][j][3]
			dictSS = vals_Nzero[i][j][4]
			lsTheory = learningSpeed(dict,mu,gamma,eigs)
			ltTheory = localTask(dictSS,eigs,mu,gamma)
			if index==false
				return lsTheory, ltTheory
			else 
				return lsTheory[index], ltTheory[index]
			end
		end
	end
	return lp
end

# ╔═╡ 64feb896-d966-492d-9acf-020b32da2c54
function plotLs(ls,Ns,mus,epoch,inds,xlbl,ylbl,lbl,dims=1)
	p = Plots.palette(:inferno, length(inds)+2)
	if dims==1 # plot with dims 1 as x axis
		y = map(inds) do j
				map(1:length(ls)) do i
					ls[i][j][epoch]
				end
		end	
		return plot(Ns,y,lw=3,xlabel=xlbl,ylabel=ylbl,label=lbl[:,inds],palette=p)
	elseif dims==2
		y = map(inds) do j
				map(1:length(mus)) do i
					ls[j][i][epoch]
				end
		end	
		return plot(mus,y,lw=3,xlabel=xlbl,ylabel=ylbl,label=lbl[:,inds],palette=p)
	end
end


# ╔═╡ 28832ed8-3269-4391-b000-7b7538c3a1aa
md"## Zero eigenvalues"

# ╔═╡ d7b7e609-4058-4535-9824-1097e8ef2af3
md"### Learning speed"

# ╔═╡ 58506658-4ef1-43e0-b3cb-a19808bf1315
# lp_zero = getLpVar(vals_zero,Ns,quadraticN_Zero,musVar,SNR)

# ╔═╡ db584b37-1aa3-48ea-ac4f-2c309cbe5d84
# begin
# 	lsTheory = getVal(lp_zero,1)
# 	ltTheory = getVal(lp_zero,2)
# end

# ╔═╡ fabfe05d-062e-4f13-a429-af8264090995
# begin 
# 	intmus = 1:5
# 	plot(1:epochs,lsTheory[1][intmus],lw=3,xlabel="epoch", ylabel="learning speed theory",label=lblMuT[:,intmus])
# end

# ╔═╡ e3b2da8f-ed0b-455b-a80e-ce2b2716c309
# begin 
# 	plot(1:epochs,ltTheory[10][1:3],lw=3,xlabel="epoch", ylabel="local task diff",label=lblMuT[:,1:3])
# end

# ╔═╡ c273ae38-c4e9-492a-b502-c092f0b6b1c9
simsT = 10

# ╔═╡ 5f24d9f3-7920-4dc3-b907-02710669fdd4
begin 
	# compute the initial ls and lt for different simulations
	musT = [mus./Ns[10] for i=1:length(Ns)] # mus should be fixed and small enough to guarantee learning
 	# even on large net	
	lp_zeroAll = map(1:simsT) do i
		args = [Ns,quadraticN_Zero,musT[1],SNR,1]
		simulateN(Ns,quadraticN_Zero,Tr_N,w0,musT,SNR,epochs,getLpVar,args)
	end
	# vals_zeroT = map(1:length(Ns)) do i
	# 	N = Ns[i]
	# 	f,df,eigs = quadraticN_Zero(N,Tr_N)
	# 	w0V = (w0/sqrt(N))*ones(N) # scale the initial weight to match initial val
	# 	map(musT) do mu
	# 		gamma = mu/SNR
	# 		sim(f,df,mu,w0V,epochs,gamma)
	# 	end
	# end
end

# ╔═╡ 9e901c15-d976-4855-94a6-a27c83a8f313
begin
	# labels for the new values of mu
	lblMuT = string.(L"\gamma=",musT[1])
	lblMuT = reshape(lblMuT,(1,length(lblMuT)))
end

# ╔═╡ 354e3fcc-6d2f-4306-8aa7-2c15c20d1de0
lp_zeroAll[1][1][1]

# ╔═╡ 2c10f156-9f86-4246-8f74-27adc0879641
begin
	lsTAll = [getVal(lp_zeroAll[i],1) for i=1:length(lp_zeroAll)]
	ltAll = [getVal(lp_zeroAll[i],2) for i=1:length(lp_zeroAll)]
end

# ╔═╡ 6e0edaa8-0681-4cc9-a114-07ba0b54bf9f
ltAll

# ╔═╡ 318c76c9-7502-4f2e-8ea7-bc94687028fb
begin 
	lsTAllM = mean(lsTAll)
	lsTAllStd = my_std(lsTAll)
	ltAllM = mean(ltAll)
	ltALlStd = my_std(ltAll)
end

# ╔═╡ e3fd0e82-8e3b-4ddf-8d3e-8485394ca503
# plot learning speed from theoretical value vs learning step for different fixed N
plotLs(lsTAllM,Ns,musT[1],1,[1,10,20,30,40,50,60,70],"learning step","learning speed theory",lblN,2)

# ╔═╡ 263f7cdd-f50a-4a0c-8cf9-12cd2a7d00b4
# plot learning speed from theoretical value vs learning step for different fixed N
plotLs(lsTAllM,Ns,musT[1],1,[1,10,20,30,40,50,60,70,80,90,100],"learning step","learning speed theory",lblN,2)

# ╔═╡ 0188d095-a88a-4129-8bc6-f3128bcecd26
# plot learning speed from theoretical value vs N
plotLs(lsTAllM,Ns,musT[1],1,[1,10,15],"N","learning speed theory",lblMuT,1)

# ╔═╡ 3ef486bc-c0e3-460e-83c6-85456966dd9e
@bind muIndex Slider(1:length(musT[1]))

# ╔═╡ 933ec83b-3fbf-4605-a349-2a193293c5b8
begin 	
	p1LsVsN = plotLs(lsTAllM,Ns,musT[1],1,[muIndex],"N","learning speed theory",lblMuT,1)
	# plot!(title="1 non-zero eig")
	p2LsVsN = plotLs(ltAllM,Ns,musT[1],1,[muIndex],"N","local task diff",lblMuT,1)
	plot!(left_margin=10mm)
	# plot!(title="1 non-zero eig")
	plot(p1LsVsN,p2LsVsN, layout=(2,1),legend=:topleft,size = (width, height))
end

# ╔═╡ 0fae950c-59c1-4a05-abe8-b54891e9b749
@bind NIndex Slider(1:length(Ns))

# ╔═╡ b6b410ff-e05a-4331-8977-b832386723d4
begin 	
	p1LsVsM = plotLs(lsTAllM,Ns,musT[1],1,[NIndex],"learning step","learning speed theory",lblN,2)
	# plot!(title="1 non-zero eig")
	p2LsVsM = plotLs(ltAllM,Ns,musT[1],1,[NIndex],"learning step","local task diff",lblN,2)
	plot!(left_margin=10mm)
	# plot!(title="1 non-zero eig")
	plot(p1LsVsM,p2LsVsM, layout=(2,1),legend=:topleft,size = (width, height))
end
# plotLs(lsTAllM,Ns,musT[1],1,[NIndex],"learning step","learning speed theory",lblN,2)

# ╔═╡ 90d982f4-d1bd-44f1-a431-23f74db1ccb0
md"""
## Non-zero eigenvalues"""

# ╔═╡ e354e071-7484-4489-af92-29986983ea6a
begin 
	# compute the initial ls and lt for different simulations
	lp_NzeroAll = map(1:simsT) do i
		args = [Ns,quadraticN_NonZero,mus,SNR,1]
		simulateN(Ns,quadraticN_NonZero,Tr_N,w0,musC,SNR,epochs,getLpVar,args)
	end
end
# lp_Nzero = getLpVar(vals_Nzero,Ns,quadraticN_NonZero,mus,SNR)

# ╔═╡ 1d07cb9b-9e6b-4bf0-9067-3b37a02442f9
begin
	lsTAllN = [getVal(lp_NzeroAll[i],1) for i=1:length(lp_NzeroAll)]
	ltAllN = [getVal(lp_NzeroAll[i],2) for i=1:length(lp_NzeroAll)]
	lsTAllNM = mean(lsTAllN)
	lsTAllNStd = my_std(lsTAllN)
	ltAllNM = mean(ltAllN)
	ltALlNStd = my_std(ltAllN)
end

# ╔═╡ 9c824288-c1d4-413f-86d6-3213b2003ab0
plotLs(lsTAllNM,Ns,mus,1,[1,10,100],"learning step","learning speed theory",lblN,2)

# ╔═╡ f2458826-bcf0-4876-994f-cec6a0972a33
# plot learning speed from theoretical value vs N
plotLs(lsTAllNM,Ns,mus,1,[1,5,10],"Ns","learning speed theory",lblMu,1)

# ╔═╡ bfab4844-7afe-49c7-ab63-fcc6bba030f0
# plot learning speed from theoretical value vs learning step for different fixed N
plotLs(ltAllNM,Ns,mus,1,[1,10,100],"learning step","local task diff",lblN,2)

# ╔═╡ 59b74b14-e9b6-4b7e-b729-6a7eb493dd6a
# plot learning speed from theoretical value vs N
plotLs(ltAllNM,Ns,mus,1,[1,5,10],"N","local task diff",lblMuT,1)

# ╔═╡ ea32abef-c3bb-4de4-bfa2-a58645b3dc9c
md"""### Comparison of 2 types of quadratic functions"""

# ╔═╡ 69e82e04-a556-4eb6-a6a7-266f636e40f0
@bind muIndex1 Slider(1:length(musT[1]))

# ╔═╡ 17554255-18a7-4054-bd7a-3b993724f4f3
begin 
	p1Ls = plotLs(lsTAllM,Ns,musT[1],1,[muIndex1],"N","learning speed \n theory",lblMuT,1)
	plot!(title="1 non-zero eig")
	p2Ls = plotLs(lsTAllNM,Ns,musT[1],1,[muIndex1],"N","learning speed \n theory",lblMuT,1)
	plot!(title="all non-zero eig",left_margin=10mm)
	plot(p1Ls,p2Ls, layout=(2,1),legend=:topleft,size=(width-200,height))
end

# ╔═╡ 7fc16a05-d2df-405b-917f-8db0a5ccfff7
begin 
	p1Lt = plotLs(ltAllM,Ns,musT[1],1,[muIndex1],"N","local task diff",lblMuT,1)
	plot!(title="1 non-zero eig")
	p2Lt = plotLs(ltAllNM,Ns,musT[1],1,[muIndex1],"N","local task diff",lblMuT,1)
	plot!(title="all non-zero eig",left_margin=10mm)
	plot(p1Lt,p2Lt, layout=(2,1),legend=:topleft,size=(width-200,height))
end

# ╔═╡ 0afb022b-8192-4e17-9d86-38553c9af041
begin 
	# musInt = [1,10,20,30,40]
	musInt = [1,5,10]
	p1Ls1 = plotLs(lsTAllM,Ns,musT[1],1,musInt,"","ls \n theory",lblMuT,1)
	plot!(title="1 non-zero",legend=false)
	p2Ls1 = plotLs(lsTAllNM,Ns,musT[1],1,musInt,"","",lblMuT,1)
	plot!(title="all non-zero",legend=:topleft)
	# plot(p1Ls,p2Ls, layout=(2,1),legend=:topleft)
	p1Lt1 = plotLs(ltAllM,Ns,musT[1],1,musInt,"N","local task diff",lblMuT,1)
	# plot!title="1 non-zero eig")
	plot!(left_margin=10mm)
	plot!(legend=false)
	p2Lt1 = plotLs(ltAllNM,Ns,musT[1],1,musInt,"N","",lblMuT,1)
	# plot!(title="1 non-zero eig")
	plot!(legend=false)
	pTheoryAll = plot(p1Ls1,p2Ls1,p1Lt1,p2Lt1, layout=(2,2),size = (width, height))
	savefig("./Figures/theoryVals_all.pdf")
	pTheoryAll
end

# ╔═╡ d867584c-e6cb-4dc0-9ce8-a343423555b1
begin 
	NInt = [1,5,10,15,20,30,40,50]
	p1LsN = plotLs(lsTAllM,Ns,musT[1],1,NInt,"","ls \n theory",lblN,2)
	plot!(title="1 non-zero",legend=false)
	p2LsN = plotLs(lsTAllNM,Ns,musT[1],1,NInt,"","",lblN,2)
	plot!(title="all non-zero",legend=:topleft)
	# plot(p1Ls,p2Ls, layout=(2,1),legend=:topleft)
	p1LtN = plotLs(ltAllM,Ns,musT[1],1,NInt,"","local task diff",lblN,2)
	plot!(left_margin=10mm)
	# plot!title="1 non-zero eig")
	plot!(legend=false)
	p2LtN = plotLs(ltAllNM,Ns,musT[1],1,NInt,"","",lblN,2)
	# plot!(title="1 non-zero eig")
	plot!(legend=false)
	pTheoryAllN = plot(p1LsN,p2LsN,p1LtN,p2LtN, layout=(2,2),xlabel="learning step", size = (width, height))
	savefig("./Figures/theoryVals_all_learningstep.pdf")
	pTheoryAllN
end

# ╔═╡ e1094e57-7ec5-4c73-86dd-7696db75ce81
md"""## Optimal values over learning step"""

# ╔═╡ 622b6b8a-3817-4019-aee8-fba6f0890656
function optimalMu(trH_N,w0,N,SNR,quadCase=1)
	if quadCase == 1
		return (trH_N*w0^2)/(2*(w0^2*trH_N^2*N-1/SNR^2))
	elseif quadCase== 2
		return (trH_N*w0^2)/(2*(w0^2*trH_N^2-N/SNR^2))
	end
end

# ╔═╡ 5e6b7221-3c6c-483e-aa8b-b04f0b998106
muOpt = [optimalMu(Tr_N,w0,N,SNR,1) for N in Ns]

# ╔═╡ ef002880-d873-43be-a4ec-7a84cfbbc6f3
begin
	NInt2=[40,50,60,70,80,90,100]
	plotLs(lsTAllM,Ns,musT[1],1,NInt2,"","ls \n theory",lblN,2)
	plot!(title="1 non-zero",legend=false)
	plot!(muOpt[NInt2],seriestype=:vline,lw=3,linecolor = 1:length(NInt2),linestyle=:dash,palette=Plots.palette(:inferno, length(NInt2)+2))
end

# ╔═╡ 3fbfd4fd-c280-45b1-aca7-2a8477c4216a
muOpt[NInt2]

# ╔═╡ a9f9ed0d-1102-4df9-904d-b2233f6f3eb3
Plots.palette(:inferno, length(NInt2)+2)[5]

# ╔═╡ cad45de9-2b4d-4a74-ae62-413a9d1f29dd
md" # Hessian Projection"

# ╔═╡ 80e35cb1-64c4-45c2-ae62-f881432b28c3
function plotHessProj(dW,eig,t)
	# plot hessian proj of dW and traces
	dwHdw = computeHessianProj(dW,eig)
	plot(dwHdw,lw=3,label="proj")
	plot!([TrH3_TrH2(eig)], seriestype=:hline,lw=3, label="TrH^3/TrH^2", linestyle=:dash)
	plot!([TrH_N(eig)],seriestype=:hline,lw=3,label="TrH/N",linestyle=:dash)
	plot!(title=t)
end

# ╔═╡ da129ba2-55cf-4097-8322-5ddea094aba7
function plotHessianParam(eigs,t="")
	TrH3s = [TrH3_TrH2(eig) for eig in eigs]
	TrHs = [TrH_N(eig) for eig in eigs]
	plot(["1D","2D zero eig","2d non-zero eig"],TrH3s,seriestype=:bar,ylabel="Trace terms",title=t,label="TrH3_TrH2",bar_position=:dodge)
	plot!(["1D","2D zero eig","2d non-zero eig"],TrHs,seriestype=:bar,ylabel="Trace terms",title=t,label="TrH_N",bar_position=:dodge)
	# ssP = plot(["1D","2D zero eig","2d non-zero eig"],ssA, seriestype=:bar, ylabel="ss value")
	# plot(lsP, ssP, layout = (2, 1), legend = false)
	# plot!(title=t)
end

# ╔═╡ 59bcc767-65c9-4400-90a5-28f86c8243a4
md"## Test near steady state"

# ╔═╡ b2fc0546-046b-443a-9736-5b9fe6e30bc0
md"""
We have shown how a network expansion can increase learning speed and steady-state value. Furthermore, we have compared two different expansions. One expansion doesn't increase learning performance. The difference is in the change in loss landscape. 
Both expansions maintain the average curvature. However, one expansion adds a zero eigenvalue whereas the other adds an eigenvalue with same value as the original one.

The only issue is that we can't quite recover the result that the local task difficulty predicts steady-state value. If it were so we would expect the local task diff of the expanded net with the zero eigenvalue to have the smallest local task difficulty. It is not the case.

How can we predict or measure the expected steady state value as a measure of the loss landscape alone?
"""


# ╔═╡ Cell order:
# ╠═fb54ebe4-0f2d-4cae-9b73-93f03a08b391
# ╠═b8e6ab2c-1be5-4524-8504-265d8462bc2a
# ╟─ab9b01ff-c53f-40d6-817e-07e837762cfb
# ╠═43619fa2-ea61-419d-b86e-c673301bc685
# ╠═0d17a04e-0334-4ff1-8c9b-bad141346414
# ╠═cbdeb143-b778-4b32-9434-bb630bd891d5
# ╠═7c5a27bc-2da7-4a4e-a957-7dbc80e0e556
# ╠═2b7d5e2d-3b9b-42dc-bc08-3fd5106b7ebe
# ╟─38125f75-7244-4526-9a37-9bd1af217e72
# ╠═cea61646-34cf-42b4-acf8-2226fde52146
# ╠═247f31f4-dc34-11ec-0584-752289f2b41d
# ╠═edadadec-a987-4661-add6-f0094bb9934c
# ╠═766b31e0-ecf8-4d01-9a8e-f1bf3d03efd7
# ╠═0e1a0954-c70d-473d-8cf4-7c07449befbf
# ╠═7ed096fa-fe5b-400e-af6a-45fb34c1226a
# ╠═18251608-65c1-414c-a9cd-734916c4b200
# ╠═bfb47a2e-592d-433f-b701-685af896d659
# ╠═831e1216-f083-4994-9934-254a480a2aef
# ╠═54b4f704-1bc2-40d0-9c25-af7e2440d401
# ╠═5dbdd7b7-0075-4e97-8e71-fa7d8371e5fc
# ╠═ca0b97fe-d673-490a-b5cc-f940fe79100c
# ╠═39fd38b7-097f-4693-a8e8-aa5a8613abbc
# ╠═2e288a30-0ac6-4a3e-8eb4-b3e7cc737cf2
# ╠═8b0bb7b7-c45e-483a-bcf0-46cc3d0db07d
# ╠═6256c83c-2ee5-4b0e-865a-186649b1bbda
# ╠═979f4a5f-12cd-4d5b-91dc-e26ef174ad6b
# ╠═70018fc9-d67a-4fc0-a902-b3c4ff4ecd25
# ╠═c9b98516-2620-4c81-8b3e-11f69feb5b62
# ╟─fc03f090-0912-4875-a2a8-f663620d26d6
# ╠═230d6234-ce7e-41ea-9f24-fc2d5be65e1c
# ╠═b6eacae8-902d-46ad-8c21-916c3aef7776
# ╠═8ed79bf6-adff-41ed-83b3-5a5921b2526e
# ╠═665e6a7a-0aa6-4e15-8e76-90d81167a4fd
# ╠═c11ab6c9-c799-405d-b76b-0b194e22990f
# ╠═2c3cdc13-dce0-4c43-8eaa-53ca08551e8e
# ╠═14d09075-61fd-4c93-a75f-d3ba99e82f70
# ╠═4173bd81-a370-4936-87e4-c3ce5f2205f9
# ╠═aaf2e829-4232-453a-914b-6c628406fc98
# ╠═5b738eb7-ae34-41a7-81bc-00738f3a6ebf
# ╠═927847fb-8ecf-4248-a131-32dc4f845779
# ╠═a5fe4cfb-9a8c-4c15-8f00-f0d6242874b9
# ╠═b8d10a77-6392-46c4-9513-3cdbf6cc07b9
# ╠═f551874c-8033-48c4-b20d-787072dd92ef
# ╠═acae4567-5a77-439b-be3b-ab422c47a1e5
# ╠═97ebb932-6ff2-46f1-84da-26affa0dd5d9
# ╠═e26d05aa-1d5e-47af-9285-a35afb616e45
# ╠═b1aedaf9-8b55-4583-b6a8-f688019a9021
# ╠═b08bd2de-0222-4413-a03b-aa8fa256b926
# ╠═62ebb223-130e-4c22-a967-dab914374dd6
# ╠═8bbf893d-5f3e-491c-8723-52e605af7322
# ╠═dd70856e-a9dc-47d3-941e-79bfa4d65565
# ╠═ad1a4a0b-2750-48f2-9751-af448c84a208
# ╟─9b2e008e-2243-4fc8-81c4-29f5f31d70c9
# ╠═51995ba1-8ac3-444c-be53-95212904347c
# ╠═11e6db80-d008-43c2-805a-93174bd8e2ff
# ╠═da36fef5-42f1-4a70-943c-e2b292456565
# ╠═d428d47b-e171-479c-8d88-e18a4ab51ccd
# ╠═3123b4c1-1dc8-4ecb-beb1-2599ac3265b1
# ╠═b3e607b5-ee76-46fe-9af0-3e2f5dc7c2ff
# ╠═248290dc-d84c-4f1a-9959-c95888c48d5c
# ╠═ad5b0b3f-f423-4d43-8d3e-b861de19904e
# ╠═cf9bea4e-f659-46c6-861b-42c1476bb3df
# ╠═7f202ff4-0342-4e62-a57f-ec2f702db5d3
# ╠═5266893d-fb62-40df-a90d-c84a08ecdfbe
# ╠═edb982ce-9b84-4d61-bdc2-fb0794db7ae4
# ╠═4e95640a-9111-43fc-b4c0-b7bae236f020
# ╠═82119ec9-bc81-467d-a5d8-97165f525fbe
# ╠═64feb896-d966-492d-9acf-020b32da2c54
# ╠═28832ed8-3269-4391-b000-7b7538c3a1aa
# ╠═d7b7e609-4058-4535-9824-1097e8ef2af3
# ╠═58506658-4ef1-43e0-b3cb-a19808bf1315
# ╠═db584b37-1aa3-48ea-ac4f-2c309cbe5d84
# ╠═fabfe05d-062e-4f13-a429-af8264090995
# ╠═e3b2da8f-ed0b-455b-a80e-ce2b2716c309
# ╠═c273ae38-c4e9-492a-b502-c092f0b6b1c9
# ╠═5f24d9f3-7920-4dc3-b907-02710669fdd4
# ╠═9e901c15-d976-4855-94a6-a27c83a8f313
# ╠═354e3fcc-6d2f-4306-8aa7-2c15c20d1de0
# ╠═2c10f156-9f86-4246-8f74-27adc0879641
# ╠═6e0edaa8-0681-4cc9-a114-07ba0b54bf9f
# ╠═318c76c9-7502-4f2e-8ea7-bc94687028fb
# ╠═e3fd0e82-8e3b-4ddf-8d3e-8485394ca503
# ╠═263f7cdd-f50a-4a0c-8cf9-12cd2a7d00b4
# ╠═0188d095-a88a-4129-8bc6-f3128bcecd26
# ╠═9ba0b30f-8fdb-49a3-a179-53f7bfe03a62
# ╠═d1fd22e3-93ba-4483-ad90-00389fe747db
# ╠═3ef486bc-c0e3-460e-83c6-85456966dd9e
# ╠═933ec83b-3fbf-4605-a349-2a193293c5b8
# ╠═0fae950c-59c1-4a05-abe8-b54891e9b749
# ╠═b6b410ff-e05a-4331-8977-b832386723d4
# ╟─90d982f4-d1bd-44f1-a431-23f74db1ccb0
# ╠═e354e071-7484-4489-af92-29986983ea6a
# ╠═1d07cb9b-9e6b-4bf0-9067-3b37a02442f9
# ╠═9c824288-c1d4-413f-86d6-3213b2003ab0
# ╠═f2458826-bcf0-4876-994f-cec6a0972a33
# ╠═bfab4844-7afe-49c7-ab63-fcc6bba030f0
# ╠═59b74b14-e9b6-4b7e-b729-6a7eb493dd6a
# ╠═ea32abef-c3bb-4de4-bfa2-a58645b3dc9c
# ╠═69e82e04-a556-4eb6-a6a7-266f636e40f0
# ╠═17554255-18a7-4054-bd7a-3b993724f4f3
# ╠═7fc16a05-d2df-405b-917f-8db0a5ccfff7
# ╠═0afb022b-8192-4e17-9d86-38553c9af041
# ╠═d867584c-e6cb-4dc0-9ce8-a343423555b1
# ╠═e1094e57-7ec5-4c73-86dd-7696db75ce81
# ╠═622b6b8a-3817-4019-aee8-fba6f0890656
# ╠═5e6b7221-3c6c-483e-aa8b-b04f0b998106
# ╠═ef002880-d873-43be-a4ec-7a84cfbbc6f3
# ╠═3fbfd4fd-c280-45b1-aca7-2a8477c4216a
# ╠═a9f9ed0d-1102-4df9-904d-b2233f6f3eb3
# ╟─cad45de9-2b4d-4a74-ae62-413a9d1f29dd
# ╠═80e35cb1-64c4-45c2-ae62-f881432b28c3
# ╠═da129ba2-55cf-4097-8322-5ddea094aba7
# ╟─59bcc767-65c9-4400-90a5-28f86c8243a4
# ╟─b2fc0546-046b-443a-9736-5b9fe6e30bc0
# ╠═f7a7df1b-d246-4cd5-ac46-7eee94c78037
