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

# ╔═╡ 56e55984-9a63-40cf-97d3-1c76f870fe57
Pkg.add("PlutoUI")

# ╔═╡ da36fef5-42f1-4a70-943c-e2b292456565
Pkg.add("Measures")

# ╔═╡ 43619fa2-ea61-419d-b86e-c673301bc685
using Plots

# ╔═╡ 0d17a04e-0334-4ff1-8c9b-bad141346414
using Statistics

# ╔═╡ cbdeb143-b778-4b32-9434-bb630bd891d5
using LaTeXStrings

# ╔═╡ 91880772-db1d-4f7f-a32b-9e6f34b5ce88
using PlutoUI

# ╔═╡ d428d47b-e171-479c-8d88-e18a4ab51ccd
using Measures

# ╔═╡ edadadec-a987-4661-add6-f0094bb9934c
include("utils.jl")

# ╔═╡ ab9b01ff-c53f-40d6-817e-07e837762cfb
md"""
The aim of this notebook is to illustrate the effect of adding parameters to a loss landscape for learning performance. We will consider the expanding quadratic functions. 

In particular, we consider going from a 1D quadratic function $f(x)=x^2$ to an N-dimensional quadratic function. 
If $f$ is the loss function for training a network with a single parameter $w$ , how does the loss function change if we add a second parameter to the network?


The network expansion leads to a new loss function in N-dimensional space $\mathbb{R}^N$. The exact form of this loss function is determined by the network architecture and the nature of the learning problem –-how the loss function is defined.

Abstractly, there are infinitely many quadratic N-dim functions. However, there will generally be constrains on the possible N-dim functions that can arise from a network expansion.

Assume that the N-dim function $f_N$ satisfies: (1) $f_N$ is quadratic adn (2) $f_N$ has the same average curvature as the 1D function. (i.e. it the average eigenvalue of the Hessian of the N-dim and 1D function are equal) 

To illustrate how a network expansion can help or hurt learnign, we consider two different types of quadratic functions $f_N$ that satisfy both of these constraints.

- **Single non-zero eigenvalue of the hessian:** the first type of function is given by  
$f_{N1}:\mathbb{R}^N\to\mathbb{R}$ with 

$$f_{N1}(\mathbf{w}) =  \frac{\rho N}{2} w_1^2$$

where $N$ is the number of parameters, $\rho$ is scalar denoting the constant value of the average eigenvalue and $\mathbf{w} = (w_1,w_2,\cdots,w_N)^T\in \mathbb{R}^N$.  

- **All equal eigenvalues:** the second type of function is 
$f_{N2}:\mathbb{R}^N\to\mathbb{R}$ with 

$$f_{N2}(\mathbf{w}) =  \frac{\rho}{2} ||\mathbf{w}||^2 = \frac{\rho}{2} (w_1^2+w_2^2+\cdots+w_N^2)$$ 

where $\mathbf{w} = (w_1,w_2,\cdots,w_N)^T\in \mathbb{R}^N$.  
"""

# ╔═╡ 7c5a27bc-2da7-4a4e-a957-7dbc80e0e556
Plots.default(titlefont = (20, "times"), legendfontsize = 18, guidefont = (18, :black), tickfont = (12, :black), grid=false, guide = "x", framestyle = :zerolines, yminorgrid = false, markersize=6)

# ╔═╡ 2b7d5e2d-3b9b-42dc-bc08-3fd5106b7ebe
begin 
	width=1200
	height=800
end

# ╔═╡ 38125f75-7244-4526-9a37-9bd1af217e72
md"# Function definition
We define all the utility functions in the julia file `utils.jl`
"

# ╔═╡ fc03f090-0912-4875-a2a8-f663620d26d6
md"## Gradient descent with noise
It is implausible that biological neural networks learn with pure gradient descent. Indeed, computing gradients can be difficult (credit assignement problem) and applying exact change in weigths with biological components in sysnapses is inherently noisy. 

We consider a more general learning rule: gradient descent with noise 

$$\delta w_t = -\mu \nabla_{w} F[w_t] + \gamma \epsilon$$ 

where $$\epsilon$$ is the noise term (usually drawn from a random normal distribution).

We run gradient descent with noise on the functions defined above for different dimensions $N$. 

We train with gradient descent with noise for a range of $\mu$. $\gamma$ is chosen to satisfy  a constant ratio $SNR=\frac{\mu}{\gamma}$ accross dimensions and $\mu$.
The range of $\mu$ used varies for the different $N$. Indeed the norm of the gradient $||\nabla F||$ grows with $N$. Hence for the same $\mu$, learning will be much faster for a larger $N$. Furthermore, for larger Ns training will stop convering for smaller $\mu$.
To make a 'fair' comparison of learning we decrease the range of tested $\mu$ with the number of dimensions $N$. 

We compute learning speed and steady state loss for the range N and $\mu$. Learning speed is computed from the trajectory of the loss during training with initial weight 
$$\mathbf{w} = \frac{w0}{\sqrt{N}} (1, 1, ... ,1)^T$$, with $w0>1$ This choice of initial weight guarantees the initial value of the loss is equal for all dimensions.

The steady state is computed from loss trajectory starting at the point $w = w_{ss}(1, 1,..., 1)^T$ with $w_{ss}$ very close to 0. Initialising very close to the local minimum allows for training to reach the steady state value without having to train for a very large (and unknown) number of epochs. 
"

# ╔═╡ 230d6234-ce7e-41ea-9f24-fc2d5be65e1c
begin
	epochs=400; # number of epochs
	Ns = 1:1:100; # number of weights to test
	mus=collect(0.001:0.005:0.1); # learning step range
	Tr_N=2; # constant value of trace/N
	SNR=2; # signal to noise ratio mu/gamma
	w0 = 5; # initial weight for training (1D case)
	musVar = [mus./N^1 for N in Ns] # vary range of mus to test as a function of N (to account for increase in gradient norm as N increases)
	musC =  [mus for N in Ns] # constant mus
	sims = 3 # number of simulations 
end

# ╔═╡ 665e6a7a-0aa6-4e15-8e76-90d81167a4fd
begin
	# labels for plotting
	lblN = string.(L"N=",Ns) 
	lblN = reshape(lblN,(1,length(lblN)))
	lblMuV = map(1:length(Ns)) do i
		string.(L"\gamma=",round.(musVar[i],digits=4))
	end
	lblMu = string.(L"\gamma=",mus)
	lblMu = reshape(lblMu,(1,length(lblMu)))
	# lblMu = reshape(lblMu,(1,length(lblN)))
end

# ╔═╡ 473e8dc7-730a-4674-9879-bb48b783e3e0
md"### Single non-zero eigenvalue
Consider  the first type of function

$$f_{N1}(\mathbf{w}) =  \frac{\rho N}{2} w_1^2$$
with $\rho$ the value of the average eigenvalue set above by `Tr_N`
"

# ╔═╡ c11ab6c9-c799-405d-b76b-0b194e22990f
# compute learning speed and steady state loss for all Ns and all musVar
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
	ssAllM = mean(ssAll)
	if sims>1
		lsAllStd = my_std(lsAll)
		ssALlStd = my_std(ssAll)
	end
end

# ╔═╡ acbc5a4b-36df-4580-a178-ec3467391eb6
md"First we plot the loss during training
We can vary N and $\mu$ with the slider
" 

# ╔═╡ bc611e16-be4a-4671-a05e-e9c61fb4fe56
md"Slider for N"

# ╔═╡ 26f62803-5633-4dc8-9ca1-893dd008fa2d
@bind Nind Slider(1:length(Ns))

# ╔═╡ e3118f1b-d009-43f3-afd3-b10b8abb7d97
md"Slider for $\mu$"

# ╔═╡ 35b434db-9acf-4a1c-b133-8315a9964587
@bind muInd Slider(1:10)

# ╔═╡ a82f460d-a3ab-44f6-8e94-48b6ab923af4
begin
	N = Ns[Nind]
	f,df,eigs = quadraticN_Zero(N,Tr_N)
	w0V = (w0/sqrt(N))*ones(N) # scale the initial weight to match initial val
	# mu = musVar[Nind][muInd]
	# musL=collect(0.00001:0.005:0.1); # learning step range
	mu = mus[muInd]
	gamma = mu/SNR
	lsVal,ssVal,lsD,ssD=sim(f,df,mu,w0V,100,gamma,10,10)
	plot(lsD[:F],lw=3,xlabel="epochs",ylabel="loss",label=string(lblN[1,Nind], " ,", lblMuV[Nind][muInd]))
end

# ╔═╡ c73230f6-3cb1-4813-aa30-0b93506f2162
md" We observe that as we increase $\mu$ for fixed $N$, the loss decreases faster. Increasing $N$ for a fixed $\mu$ can have the same effect. As expected for larger $N$, training will stop convergin for a smaller $\mu$.

We now look at the learning speed and steady state loss for different $N$ and $\mu$. "

# ╔═╡ 14d09075-61fd-4c93-a75f-d3ba99e82f70
begin
	int = 1:5:length(Ns)
	plot(musVar[int],lsAllM[int],lw=3,xlabel=L"\mu", ylabel="learning speed",label=lblN[:,int],palette=p)
end

# ╔═╡ 7b8ff523-98a1-4235-a46d-2be2418d1c49
md"The learning speed is a quadratic fucntion in $\mu$. It increases until it reaches a maximum. The rate of increase and the value at which the maximum is reached depends on $N$. Indeed, the larger $N$ is, the faster the learning speed increases with $\mu$. It also seems that the maximal value of the learning speed is equal for different $N$."

# ╔═╡ aaf2e829-4232-453a-914b-6c628406fc98
begin
	plot(musVar[int],ssAllM[int],lw=3,xlabel=L"\mu", ylabel="steady state value",label=lblN[:,int],palette=p)
end

# ╔═╡ 36c748de-8fa7-4c4c-b259-e3eaa91af0e6
md"The steady state loss increases with $\mu$, decreasing the learning precision. The rate of increase seems to be independent of $N$."

# ╔═╡ cf9a8bdb-8788-4381-b847-27feb326c9a6
md"We observe that for any $N$, increasing $\mu$ (while keeping small enough to assure learning convergence) increases learning speed and steady state loss. 
If $\mu$ is kept constant during learning, there is a trade-off between learning speed and accuracy (steady state value). A larger learning step leads to larger learning speed but at the expense of worse learning accuracy. 

We can compute the optimal learning speed for each $N$, over all $\mu$."

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

# ╔═╡ 70f57b49-2705-41d1-8a09-595785c05459
md"The optimal learning speed is equal for all $N$. If we increase the number of parameters $N$, we can always find a $\mu$ such that the learning speed matches the learning speed of the original function with fewer parameters.

What about the steady state loss? From the plot above, we see that the minimal steady state loss will be achieved for the smallest $\mu$. We expect the minimal ss value over all $\mu$ to be relatively similar accross $N$, it will mainly depend on the smallest $\mu$ tested for each $N$.  

A measure that is more relevant is the steady state loss for the $\mu$ that gives optimal learning speed. If $\mu$ is set to optimise learning speed, what steady state loss can be attained? 
"

# ╔═╡ b8d10a77-6392-46c4-9513-3cdbf6cc07b9
begin
	p1=plot(Ns, ssMin , lw=3, xlabel="N", ylabel="ss value", label="min")
	p2=plot(Ns, ssAtLsMax, lw=3, label="ss at max ls", xlabel="N", ylabel="ss value")
	plot(p1,p2,layout=(2,1))
end

# ╔═╡ 13fb87f1-c65e-4319-a31e-2131c21d8597
md" The steady state loss for the $\mu$ that leads to maximal learning speed decreases with $N$. Indicating that that adding parameters can help optimise learning speed while maintaining steady state performance. 

We can plot the steady state value vs the learning speed as we vary $\mu$.
Each color represents a different $N$ and the opacity represents the value of $\mu$. $\mu$ increases with the opacity. 
"

# ╔═╡ 1de8b3ca-5796-4f5d-8245-61c7a28d958d
begin
	indeces = [5,10,30,100]
	scatterPlot(indeces,ssAllM,lsAllM)
end

# ╔═╡ 6fb78ef1-c54d-4256-ae7c-2ad76f0f7906
md" 
We observe that for each $N$, increasing $\mu$ increases both learning speed (ls) and steady state value (ss). 
However, for larger $N$, the steady state value increases slower with $\mu$.
This shifts the ss vs ls curves towards the y-axis as we increase N. 

Adding parameters navigates the learning speed to steady state loss trade-off. 
Adding parameters allows for better learning performance. We can always find a $\mu$ for which the learning is as dast or faster and more precise (larger learning speed and smaller steady state loss).
"

# ╔═╡ f551874c-8033-48c4-b20d-787072dd92ef
md"## Non-zero eigenvalues
Consider the second type of function, with all equal eigenvalues (i.e. no zero eigenvalues)

$$f_{N2}(\mathbf{w}) =  \frac{\rho}{2} ||\mathbf{w}||^2 = \frac{\rho}{2} (w_1^2+w_2^2+\cdots+w_N^2)$$ 

We do the same plots as above
"

# ╔═╡ acae4567-5a77-439b-be3b-ab422c47a1e5
ls_ssAllN = map(1:sims) do i
	simulateN(Ns,quadraticN_NonZero,Tr_N,w0,musC,SNR,epochs,getLs_SS)
end

# ╔═╡ e26d05aa-1d5e-47af-9285-a35afb616e45
begin
	lsAllN = map(ls_ssAllN) do l
		l[1]
	end
	ssAllN = map(ls_ssAllN) do l
		l[2]
	end
	lsAllNM = mean(lsAllN)
	ssAllNM = mean(ssAllN)
	if sims>1
		lsAllNStd = my_std(lsAllN)
		ssALlNStd = my_std(ssAllN)
	end
end

# ╔═╡ b1aedaf9-8b55-4583-b6a8-f688019a9021
plot(mus,lsAllNM[int],lw=3,xlabel=L"\mu", ylabel="learning speed",label=lblN[:,int],palette=p)

# ╔═╡ b08bd2de-0222-4413-a03b-aa8fa256b926
begin
	plot(mus,ssAllNM[int],lw=3,xlabel=L"\mu", ylabel="steady state value",label=lblN[:,int],palette=p)
end

# ╔═╡ 17cbc4a5-0746-4845-b166-e7f603ab3a3f
md"As for the previous type of function, the learning speed is a quadratic function in $\mu$ and a the steady state loss is a linear function in $\mu$. 
However, the effect of adding parameters $N$ is the opposite in this case. The learning speed is indepent of $N$, and the steady state loss gets worse with $N$.

We plot the max learnign speed and minimum steady state value as before." 

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

# ╔═╡ 9f210dcf-eb0c-4088-b5e5-f0a6610f54d9
scatterPlot(indeces,ssAllNM,lsAllNM)

# ╔═╡ 9b2e008e-2243-4fc8-81c4-29f5f31d70c9
md"""
For this type of function, increasing $N$ is not beneficial to learning. 
As in the previous case, for any fixed $N$ there is a trade-off between learning speed and steady state loss as $\mu$ varies. 

The difference is that in this case, for larger $N$, the steady state value gets worse. The slope of the steady state loss vs $\mu$ line increases with $N$.

The scatter plot in this case, is the opposite as before. Indicating, that the larger $N$ is, the worse the learning speed to steady state value is. 

We have considered two different types of quadratic functions. For each type, adding parameters has the opposite effect. 
"""

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
	plot!(twinx(),Ns, muMaxN, lw=3, axis=:right, color=:orange, ylabel=L"$\mu$ at max", legend=false, xlabel="", ylims=(0, muMaxN[1]+muMaxN[1]), yguidefontcolor=:orange)
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
md"## Theoretical analysis"

# ╔═╡ ad5b0b3f-f423-4d43-8d3e-b861de19904e
md"""
From previous work we defined two quantities that determine learning speed and steady state loss when optimising $F$ with gradient descent with noise with parameters $\mu$ and $\gamma$.  

The expected learning speed over the dirstribution of $\mathbf{\epsilon}$ is 

$$\mathbb{E}[\nu_t] = -\frac{1}{F_t\delta t} (-\mu ||\nabla F||^2+\mu^2\nabla F^T\nabla^2F\nabla F+\mathbb{E}[\gamma^2 \epsilon^T \nabla^2 F \epsilon])$$

The expected local task difficulty over  dirstribution of $\mathbf{\epsilon}$ is 

$$\mathbb{E}[\nu_t] = \mu \hat{\nabla F}^T\nabla^2F\hat{\nabla F}+\mathbb{E}[\frac{\gamma^2}{\mu} \frac{1}{||\nabla F||^2} \epsilon^T \nabla^2 F \epsilon])$$


We compute these values for the different type functions. We expect them to match the simulation results above.
"""

# ╔═╡ 28832ed8-3269-4391-b000-7b7538c3a1aa
md"## Single non-zero eigenvalue"

# ╔═╡ c273ae38-c4e9-492a-b502-c092f0b6b1c9
simsT = 10 # number of simulations to average over (to average noise term out)

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
	lblMuT = string.(L"\mu=",musT[1])
	lblMuT = reshape(lblMuT,(1,length(lblMuT)))
end

# ╔═╡ 2c10f156-9f86-4246-8f74-27adc0879641
begin
	# theory values of ls and ss
	lsTAll = [getVal(lp_zeroAll[i],1) for i=1:length(lp_zeroAll)]
	ltAll = [getVal(lp_zeroAll[i],2) for i=1:length(lp_zeroAll)]
end

# ╔═╡ 318c76c9-7502-4f2e-8ea7-bc94687028fb
begin 
	lsTAllM = mean(lsTAll)
	lsTAllStd = my_std(lsTAll)
	ltAllM = mean(ltAll)
	ltALlStd = my_std(ltAll)
end

# ╔═╡ e3fd0e82-8e3b-4ddf-8d3e-8485394ca503
# plot learning speed from theoretical value vs learning step for different fixed N
plotLs(lsTAllM,Ns,musT[1],1,[1,10,20,30,40,50,60,70],L"\mu","learning speed theory",lblN,2)

# ╔═╡ 3bd3e281-f94f-4ed7-bb87-f54d35edfc71
md"We observe that the leaarning speed is a quadratic function of $\mu$. This plot matches the simulated plot above. 
With the theoretical values, we obtain a decrease of the learning speed wiht larger $\mu$. This regime is not very relevant for the simulations as, for large $\mu$ learning can stop converging."

# ╔═╡ 0188d095-a88a-4129-8bc6-f3128bcecd26
# plot learning speed from theoretical value vs N
plotLs(lsTAllM,Ns,musT[1],1,[1,10,15],"N","learning speed theory",lblMuT,1)

# ╔═╡ d3405a47-08c5-48d4-827a-cae380bde621
md"We plot the learning speed now as a function of $N$ for different $\mu$. As expected ls increases with N. We observe that this relationship is quadratic."

# ╔═╡ 4836dddb-d914-40ba-8bb2-2aae5fda5ab9
# plot learning speed from theoretical value vs learning step for different fixed N
plotLs(ltAllM,Ns,musT[1],1,[1,10,20,30,40,50,60,70],L"\mu","local task difficulty",lblN,2)

# ╔═╡ 8ff3a736-8002-4483-927a-4635ddebf9f5
md" The local task difficulty (a proxy of steady state value) has the same dependence on $\mu$ and $N$ as observed with the simulations." 

# ╔═╡ 3ef486bc-c0e3-460e-83c6-85456966dd9e
@bind muIndex Slider(1:length(musT[1]))

# ╔═╡ 933ec83b-3fbf-4605-a349-2a193293c5b8
begin 	
	p1LsVsN = plotLs(lsTAllM,Ns,musT[1],1,[muIndex],"N","learning speed theory",lblMuT,1)
	plot!(ylims=(0, maximum(lsTAllM[end])*1.1)) # fix y axis
	# plot!(title="1 non-zero eig")
	p2LsVsN = plotLs(ltAllM,Ns,musT[1],1,[muIndex],"N","local task diff",lblMuT,1)
	plot!(left_margin=10mm)
	plot!(ylims=(0, maximum(ltAllM[end])*1.1)) # fix y axis
	# plot!(title="1 non-zero eig")
	plot(p1LsVsN,p2LsVsN, layout=(2,1),legend=:topleft,size = (width, height))
end

# ╔═╡ 0fae950c-59c1-4a05-abe8-b54891e9b749
@bind NIndex Slider(1:length(Ns))

# ╔═╡ b6b410ff-e05a-4331-8977-b832386723d4
begin 	
	p1LsVsM = plotLs(lsTAllM,Ns,musT[1],1,[NIndex],"learning step","learning speed theory",lblN,2)
	plot!(ylims=(0, maximum(lsTAllM[end])*1.1)) # fix y axis
	# plot!(title="1 non-zero eig")
	p2LsVsM = plotLs(ltAllM,Ns,musT[1],1,[NIndex],"learning step","local task diff",lblN,2)
	plot!(left_margin=10mm)
	plot!(ylims=(0, maximum(ltAllM[end])*1.1)) # fix y axis
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
plotLs(lsTAllNM,Ns,mus,1,[1,10,100],L"\mu","learning speed theory",lblN,2)

# ╔═╡ f2458826-bcf0-4876-994f-cec6a0972a33
# plot learning speed from theoretical value vs N
plotLs(lsTAllNM,Ns,mus,1,[1,5,10],"N","learning speed theory",lblMu,1)

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
	plot!(ylims=(0, maximum(lsTAllM[end])*1.1)) # fix y axis
	p2Ls = plotLs(lsTAllNM,Ns,musT[1],1,[muIndex1],"N","learning speed \n theory",lblMuT,1)
	plot!(title="all non-zero eig",left_margin=10mm)
	plot!(ylims=(0, maximum(lsTAllNM[end])*1.1)) # fix y axis
	plot(p1Ls,p2Ls, layout=(2,1),legend=:topleft,size=(width-200,height))
end

# ╔═╡ 7fc16a05-d2df-405b-917f-8db0a5ccfff7
begin 
	p1Lt = plotLs(ltAllM,Ns,musT[1],1,[muIndex1],"N","local task diff",lblMuT,1)
	plot!(title="1 non-zero eig")
	plot!(ylims=(0, maximum(ltAllM[end])*1.1)) # fix y axis
	p2Lt = plotLs(ltAllNM,Ns,musT[1],1,[muIndex1],"N","local task diff",lblMuT,1)
	plot!(title="all non-zero eig",left_margin=10mm)
	plot!(ylims=(0, maximum(ltAllNM[end])*1.1)) # fix y axis
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

# ╔═╡ b2fc0546-046b-443a-9736-5b9fe6e30bc0
md"""
We have shown how a network expansion can increase learning speed and steady-state value. Furthermore, we have compared two different expansions. One expansion doesn't increase learning performance. The difference is in the change in loss landscape. 
Both expansions maintain the average curvature. However, one expansion adds a zero eigenvalue whereas the other adds an eigenvalue with same value as the original one.
"""


# ╔═╡ Cell order:
# ╠═fb54ebe4-0f2d-4cae-9b73-93f03a08b391
# ╠═b8e6ab2c-1be5-4524-8504-265d8462bc2a
# ╠═56e55984-9a63-40cf-97d3-1c76f870fe57
# ╟─ab9b01ff-c53f-40d6-817e-07e837762cfb
# ╠═43619fa2-ea61-419d-b86e-c673301bc685
# ╠═0d17a04e-0334-4ff1-8c9b-bad141346414
# ╠═cbdeb143-b778-4b32-9434-bb630bd891d5
# ╠═91880772-db1d-4f7f-a32b-9e6f34b5ce88
# ╠═7c5a27bc-2da7-4a4e-a957-7dbc80e0e556
# ╠═2b7d5e2d-3b9b-42dc-bc08-3fd5106b7ebe
# ╟─38125f75-7244-4526-9a37-9bd1af217e72
# ╠═edadadec-a987-4661-add6-f0094bb9934c
# ╟─fc03f090-0912-4875-a2a8-f663620d26d6
# ╠═230d6234-ce7e-41ea-9f24-fc2d5be65e1c
# ╠═665e6a7a-0aa6-4e15-8e76-90d81167a4fd
# ╟─473e8dc7-730a-4674-9879-bb48b783e3e0
# ╠═c11ab6c9-c799-405d-b76b-0b194e22990f
# ╠═2c3cdc13-dce0-4c43-8eaa-53ca08551e8e
# ╟─acbc5a4b-36df-4580-a178-ec3467391eb6
# ╟─bc611e16-be4a-4671-a05e-e9c61fb4fe56
# ╠═26f62803-5633-4dc8-9ca1-893dd008fa2d
# ╟─e3118f1b-d009-43f3-afd3-b10b8abb7d97
# ╠═35b434db-9acf-4a1c-b133-8315a9964587
# ╠═a82f460d-a3ab-44f6-8e94-48b6ab923af4
# ╟─c73230f6-3cb1-4813-aa30-0b93506f2162
# ╠═14d09075-61fd-4c93-a75f-d3ba99e82f70
# ╟─7b8ff523-98a1-4235-a46d-2be2418d1c49
# ╠═aaf2e829-4232-453a-914b-6c628406fc98
# ╠═36c748de-8fa7-4c4c-b259-e3eaa91af0e6
# ╟─cf9a8bdb-8788-4381-b847-27feb326c9a6
# ╠═927847fb-8ecf-4248-a131-32dc4f845779
# ╠═a5fe4cfb-9a8c-4c15-8f00-f0d6242874b9
# ╟─70f57b49-2705-41d1-8a09-595785c05459
# ╠═b8d10a77-6392-46c4-9513-3cdbf6cc07b9
# ╟─13fb87f1-c65e-4319-a31e-2131c21d8597
# ╠═1de8b3ca-5796-4f5d-8245-61c7a28d958d
# ╟─6fb78ef1-c54d-4256-ae7c-2ad76f0f7906
# ╟─f551874c-8033-48c4-b20d-787072dd92ef
# ╠═acae4567-5a77-439b-be3b-ab422c47a1e5
# ╠═e26d05aa-1d5e-47af-9285-a35afb616e45
# ╠═b1aedaf9-8b55-4583-b6a8-f688019a9021
# ╠═b08bd2de-0222-4413-a03b-aa8fa256b926
# ╟─17cbc4a5-0746-4845-b166-e7f603ab3a3f
# ╠═8bbf893d-5f3e-491c-8723-52e605af7322
# ╠═dd70856e-a9dc-47d3-941e-79bfa4d65565
# ╠═ad1a4a0b-2750-48f2-9751-af448c84a208
# ╠═9f210dcf-eb0c-4088-b5e5-f0a6610f54d9
# ╟─9b2e008e-2243-4fc8-81c4-29f5f31d70c9
# ╟─51995ba1-8ac3-444c-be53-95212904347c
# ╠═11e6db80-d008-43c2-805a-93174bd8e2ff
# ╠═da36fef5-42f1-4a70-943c-e2b292456565
# ╠═d428d47b-e171-479c-8d88-e18a4ab51ccd
# ╠═3123b4c1-1dc8-4ecb-beb1-2599ac3265b1
# ╟─b3e607b5-ee76-46fe-9af0-3e2f5dc7c2ff
# ╟─248290dc-d84c-4f1a-9959-c95888c48d5c
# ╟─ad5b0b3f-f423-4d43-8d3e-b861de19904e
# ╟─28832ed8-3269-4391-b000-7b7538c3a1aa
# ╠═c273ae38-c4e9-492a-b502-c092f0b6b1c9
# ╠═5f24d9f3-7920-4dc3-b907-02710669fdd4
# ╠═9e901c15-d976-4855-94a6-a27c83a8f313
# ╠═2c10f156-9f86-4246-8f74-27adc0879641
# ╠═318c76c9-7502-4f2e-8ea7-bc94687028fb
# ╠═e3fd0e82-8e3b-4ddf-8d3e-8485394ca503
# ╟─3bd3e281-f94f-4ed7-bb87-f54d35edfc71
# ╠═0188d095-a88a-4129-8bc6-f3128bcecd26
# ╠═d3405a47-08c5-48d4-827a-cae380bde621
# ╠═4836dddb-d914-40ba-8bb2-2aae5fda5ab9
# ╟─8ff3a736-8002-4483-927a-4635ddebf9f5
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
# ╟─17554255-18a7-4054-bd7a-3b993724f4f3
# ╠═7fc16a05-d2df-405b-917f-8db0a5ccfff7
# ╠═0afb022b-8192-4e17-9d86-38553c9af041
# ╠═d867584c-e6cb-4dc0-9ce8-a343423555b1
# ╟─b2fc0546-046b-443a-9736-5b9fe6e30bc0
