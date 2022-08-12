### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ e2ae0aad-422b-40b6-9c78-0d5b4b4d05b8
import Pkg; Pkg.add("Plots")

# ╔═╡ 43619fa2-ea61-419d-b86e-c673301bc685
using Plots

# ╔═╡ 0d17a04e-0334-4ff1-8c9b-bad141346414
using Statistics

# ╔═╡ c1b958e5-7089-4327-9eee-59a840ea65b2
using Measures

# ╔═╡ ab9b01ff-c53f-40d6-817e-07e837762cfb
md"""
The aim of this notebook is to illustrate the effect of adding dimensions to a loss landscape for learning performance. We will consider quadratic functions and to start with going from a 1D to 2D quadratic function"""

# ╔═╡ 53b988f1-12a1-480c-ba3b-0078960deef6
Plots.default(titlefont = (20, "times"), legendfontsize = 18, guidefont = (18, :black), tickfont = (12, :black), grid=false, guide = "x", framestyle = :zerolines, yminorgrid = false, markersize=6)

# ╔═╡ 9c2bcbab-db2f-4dd3-847f-430df8e8adde
begin 
	width=1000
	height=600
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
	f_2_1(w1,w2) = 2*w1^2 # 2D quadratic function zero eigenvalue
	df_2_1(w1,w2) = [4w1,0]
	eig21 = [4,0]
	f_2_2(w1,w2) = w1^2+w2^2 # 2D quadratic function non-zero eigenvalue 
	df_2_2(w1,w2) = [2w1,2w2]
	eig22 = [2,2]
end

# ╔═╡ 0e1a0954-c70d-473d-8cf4-7c07449befbf
function gradDescent(f,df,mu,w0,epochs=1)
	W = zeros(epochs,length(w0))
	F = zeros(epochs)
	w=w0
	for i=1:epochs
		W[i,:] = w
		F[i] = f(w...)
		w = w -mu*df(w...)
	end
	return F,W
end

# ╔═╡ 18251608-65c1-414c-a9cd-734916c4b200
function ls(F,l=10)
	dF = (F[2:l+1]-F[1:l])./F[1:l]
	return mean(-dF)
end

# ╔═╡ bfb47a2e-592d-433f-b701-685af896d659
function ss(F,l=20)
	return mean(F[end-l:end])
end

# ╔═╡ 831e1216-f083-4994-9934-254a480a2aef
function plotLsSS(Fs,t)
	lsA = [ls(i) for i in Fs]
	ssA = [ss(i) for i in Fs]
	lsP = plot(["1D","2D zero eig","2d non-zero eig"],lsA,seriestype=:bar,ylabel="learning speed",title=t)
	ssP = plot(["1D","2D zero eig","2d non-zero eig"],ssA, seriestype=:bar, ylabel="ss value")
	plot(lsP, ssP, layout = (2, 1), legend = false)
	# plot!(title=t)
end

# ╔═╡ fc03f090-0912-4875-a2a8-f663620d26d6
md"# Gradient descent"

# ╔═╡ 230d6234-ce7e-41ea-9f24-fc2d5be65e1c
begin
	mu=0.1; epochs=100
	w0=10
end

# ╔═╡ a1ebb468-fca5-4ff7-a74d-49e96bbae923
F1, W1 = gradDescent(f_1,df_1,mu*2,[w0*sqrt(2)],epochs);

# ╔═╡ 02d62dbe-12e7-4c47-85a3-498d84eb7a15
F21,W21 = gradDescent(f_2_1,df_2_1,mu,[w0,w0],epochs);

# ╔═╡ ba21765c-6c76-46b4-916c-47b3f4fdc4b8
F22,W22 = gradDescent(f_2_2,df_2_2,mu,[w0,w0],epochs);

# ╔═╡ b76e6786-ec77-40ef-9442-59873eea830b
begin
	plot(F1,lw=3)
	plot!(F21,lw=3)
	plot!(F22,linestyle=:dash,lw=3)
end

# ╔═╡ d45ce8e2-a76f-4da2-bcfd-0cfea525230b
begin
	Fs = [F1,F21,F22]
	lsA = [ls(i) for i in Fs]
	ssA = [ss(i) for i in Fs]
end

# ╔═╡ a485ae5e-29f5-4801-ad4d-e4db58a5a544
plotLsSS(Fs,"Gradient descent no noise")

# ╔═╡ 9b2e008e-2243-4fc8-81c4-29f5f31d70c9
md"""
We have done gradient descent on the three different functions starting at the same value. We observe that the 2D with one zero eigenvalue has larger learning speed. We expected this because the gradient of this function is larger. The steady state loss is the same because in all cases gradient descent reaches the minimum $0$."""

# ╔═╡ b3e607b5-ee76-46fe-9af0-3e2f5dc7c2ff
md"""
In the case of gradient descent with some random noise we expect in all cases to reach some non-zero steady state value. This steady state value should be detemined by the local task difficulty."""

# ╔═╡ 248290dc-d84c-4f1a-9959-c95888c48d5c
md" # Gradient descent with noise"

# ╔═╡ d4a70361-efe5-499a-9238-f90dc70354ce
function gradDescentNoise(f,df,mu,w0,epochs=1,gamma=0,Wset=false)
	W = zeros(epochs,length(w0))
	dW = zeros(epochs,length(w0))
	grad = zeros(epochs)
	N = zeros(epochs,length(w0))
	F = zeros(epochs)
	w=w0
	for i=1:epochs
		W[i,:] = w
		F[i] = f(w...)
		grad[i] = sqrt(sum(df(w...).^2))
		dW[i,:] = -mu*df(w...)
		N[i,:] = gamma*randn(length(w))
		if Wset==false
			w = w +dW[i,:] + N[i,:]
		else
			w = Wset[i,:]
		end
	end
	return F,W,dW,N, grad
end

# ╔═╡ 84b2e25b-b8cf-4afd-869a-86c3cc164b80
gamma=mu/2;

# ╔═╡ 4fb68f21-9fed-47bc-b2da-34f23cd4932f
F1N, W1N, dW1N, N1, grad1N = gradDescentNoise(f_1,df_1,mu*2,[w0*sqrt(2)],epochs,gamma*2);

# ╔═╡ ddc8d2ea-6b5e-4868-b7c8-1c2e75a5af4a
F21N,W21N,dW21N, N21,grad21N = gradDescentNoise(f_2_1,df_2_1,mu,[w0,w0],epochs,gamma);

# ╔═╡ dae69d2c-1733-4a65-a5e9-4c9ed0e47dc3
F22N,W22N,dW22N, N22,grad22N = gradDescentNoise(f_2_2,df_2_2,mu,[w0,w0],epochs,gamma);

# ╔═╡ 1b101b38-22f4-4956-9cc5-01c4d307d700
begin
	plot(F1N,lw=3)
	plot!(F21N,lw=3)
	plot!(F22N,linestyle=:dash,lw=3)
end

# ╔═╡ c31d2ef6-4780-4e8e-901e-2a8da8821656
sequential_palette(240, 200,s=0.3,b=0,logscale=false)

# ╔═╡ b60fb647-4357-42b8-9e3c-a61e04d18787
W1N

# ╔═╡ 2412e898-66e8-474a-a11e-8956dfe05ad8
function plot1DQuad(minV, maxV, f, W=false, F=false,c=:hot)
	""" plot surface of 1d quadratic fucntion f 
	between [minV,  maxV]. If W and F are given, plot trajectory from a descent algorithm
	W is  a 1D array with the trajectory of the weights and F a  1D array 
	with the trajectory of the error 
	code adapted from: https://xavierbourretsicotte.github.io/Intro_optimization.html"""
	# x and y arrays over plotting interval
	x = minV:0.0001:maxV
	
	# contour plot
	pC = plot(x, f.(x), line_z=f.(x),lw=5,color=c, legend=false, xlabel="w1",ylabel="F")
	if W!=false
		cmap = sequential_palette(240, size(W,1),s=0.5,b=0,logscale=false)
	  	# plot weight and error trajectory from descent
	  	plot!(W,F, linecolor=:black, lw=3, markercolors=cmap, markershape=:circle)
	end
	return pC
end

# ╔═╡ 127cbc58-f5cf-4450-979a-43698605d362
begin
	p1D = plot1DQuad(-w0*sqrt(2),w0*sqrt(2),f_1,W1N,F1N,:YlOrRd_9)
	savefig("./Figures/1d_descentNoise.pdf")
	p1D
end

# ╔═╡ a4ba7d5f-d1d3-4f78-a6f6-75d107295912
function plot2DQuad(minV, maxV, f, W=false, F=false, surf=true,c=:YlOrRd_9)
	""" plot surface (if surf=True) and contour plots of 2D quadratic fucntion f 
	between [minV,  maxV]. If W and F are given, plot trajectory from a descent algorithm
	W is  a 2D array with the trajectory of the weights and F a  1D array 
	with the trajectory of the error 
	code adapted from: https://xavierbourretsicotte.github.io/Intro_optimization.html"""
	# x and y arrays over plotting interval
	dt = 0.1
	if maxV<1
		dt = 0.001
	end
	x = minV:dt:maxV
	y = minV:dt:maxV

	# contour plot
	pC = plot(x, y, f, st = :contourf, fill=c, legend=false,xlabel="w1",ylabel="w2")
	if W!=false
		cmap = sequential_palette(240, size(W,1),s=0.5,b=0,logscale=false)
	  # plot weight and error trajectory from descent
	  	plot!(W[:,1],W[:,2], linecolor=:black, lw=3, markercolors=cmap, markershape=:circle, right_margin=10mm)
	end
	if surf
		pS = plot(x, y, f, st = :surface, fill=c, xlabel="w1",ylabel="w2",zlabel="F")
		if W!=false && F!=false
			plot!(W[:,1],W[:,2],F,color=cmap, markershape=:circle,label="weight traj")
		end
		p = plot(pS,pC,layout=(2,1),size=(width,height+200))
	else
		p=pC
	end
	return p
end


# ╔═╡ 428dc44a-f818-4776-b792-540179ab3f47
begin
	p21 = plot2DQuad(-w0-5, w0+5, f_2_1 , W21N, F21N, true,:YlOrRd_9)
	savefig("./Figures/21d_descentNoise.pdf")
	p21
end

# ╔═╡ a97bf7dd-1c5d-406c-9931-3b65445b016a
begin 
	p22=plot2DQuad(-w0-2, w0+2, f_2_2 , W22N, F22N, true,:YlOrRd_9)
	savefig("./Figures/22d_descentNoise.pdf")
	p22
end

# ╔═╡ 57dedfb8-ddf7-4181-8d18-3b989a4126c3
# def desc2DPlot(l1,l2,w_initial):
#   """plot contour plot with trajectory of steepest descent 
#   of a 2D quadratic fucntion f = w^TAw with A =diag(l1,l2)"""
#   A = np.array([[l1,0],[0,l2]])
#   # define quadratic function
#   def f2(x,y):
#     # calculate the error for weights given as two vectors
#       return 1/2*(A[0,0]*x**2+A[1,1]*y**2+2*A[0,1]*x*y)
#   def f(w):
#     # calculate the error for weights given as one vector
#       return 1/2*(A[0,0]*w[0]**2+A[1,1]*w[1]**2+2*A[0,1]*w[0]*w[1])

#   # define derivative function
#   def df(w):
#     return 1/2*onp.array([2*A[0,0]*w[0]+2*A[0,1],2*A[1,1]*w[1]+2*A[0,1]])

#   # plot contour plot
#   W, F =  steepestDescent(f,df,w_initial)
#   z, ax = plot2DQuad(w_initial[0]-0.5,w_initial[1]+0.5,f2,W=W,F=F,surf=False)
#   ax.set_title('lambda1 = {}, lambda2 = {}. Steepest descent trajectory'.format(l1,l2))
#   plt.show()

# ╔═╡ db04c2a2-5ced-406d-a53a-bea66f938c7a
FsN = [F1N,F21N,F22N];

# ╔═╡ 02b3c8db-6f92-4117-a634-487b8504065d
plotLsSS(FsN,"Gradient descent noise = $gamma")

# ╔═╡ cad45de9-2b4d-4a74-ae62-413a9d1f29dd
md" # Hessian Projection"

# ╔═╡ 4f7504e4-abd1-44e3-ad87-287cc522718b
function computeHessianProj(dW,eig)
	P = zeros(size(dW,1))
	for i=1:size(dW,1)
		P[i] = sum(dW[i,:].^2 .*eig)/sum(dW[i,:].^2)
	end
	return P
end

# ╔═╡ 58456c76-8ee6-417d-af5a-dc0254dc472a
function TrH3_TrH2(eig)
	return sum(eig.^3)/sum(eig.^2)
end

# ╔═╡ ed6b88e9-894e-4819-9a21-e5cbad0a0ff9
function TrH_N(eig)
	return sum(eig)/length(eig)
end

# ╔═╡ 80e35cb1-64c4-45c2-ae62-f881432b28c3
function plotHessProj(dW,eig,t)
	dwHdw = computeHessianProj(dW,eig)
	plot(dwHdw,lw=3,label="proj")
	plot!([TrH3_TrH2(eig)], seriestype=:hline,lw=3, label="TrH^3/TrH^2", linestyle=:dash)
	plot!([TrH_N(eig)],seriestype=:hline,lw=3,label="TrH/N",linestyle=:dash)
	plot!(title=t)
end

# ╔═╡ c69c5757-0465-4748-bf92-bab08dc26991
function localTask(dW,N,eig,grad)
	dWAll = dW.+N
	return 1/2*computeHessianProj(dWAll,eig).*(sum(dWAll.^2;dims=2)./grad.^2)
	# 1/2*computeHessianProj(dW21NSet.+N21Set,eig21).*(sum((dW21NSet.+N21Set).^2;dims=2).*grad21NSet.^2)
	# return 1/2*computeHessianProj(dW,eig).*(dW.^2 ./grad.^2) + 1/2*computeHessianProj(N,eig).*(N.^2 ./grad.^2)
end

# ╔═╡ 59061c77-818b-4646-96ed-4eec3dd7bec3
plotHessProj(dW1N,eig1,"1D dW Proj")

# ╔═╡ f8e021b0-856e-460b-aec2-d41007e03c40
plotHessProj(dW21N,eig21,"2D zero eig dW Proj")

# ╔═╡ 06a64a29-00a5-4695-a9cc-68d64ca65311
plotHessProj(dW22N,eig22,"2D non-zero eig dW Proj")

# ╔═╡ 4b2ba9bb-dd48-4f19-9629-d3079761d455
plotHessProj(N21,eig21,"2D noise Proj")

# ╔═╡ da129ba2-55cf-4097-8322-5ddea094aba7
function plotHessianParam(eigs,t="")
	TrH3s = [TrH3_TrH2(eig) for eig in eigs]
	TrHs = [TrH_N(eig) for eig in eigs]
	p1 = plot(["1D","2D zero eig","2d non-zero eig"],TrH3s,seriestype=:bar,ylabel="Trace terms",title=t,label="TrH3_TrH2")
	p2 = plot(["1D","2D zero eig","2d non-zero eig"],TrHs,seriestype=:bar,ylabel="Trace terms",title=t,label="TrH_N")
	plot(p1, p2, layout = (2, 1), legend = false)
	# ssP = plot(["1D","2D zero eig","2d non-zero eig"],ssA, seriestype=:bar, ylabel="ss value")
	# plot(lsP, ssP, layout = (2, 1), legend = false)
	# plot!(title=t)
end

# ╔═╡ 96560824-f6e3-4f53-9377-59afcb1578a1
eigs=[eig1,eig21,eig22]

# ╔═╡ 5077200c-ee07-4a2e-b837-d7022cff7e55
plotHessianParam(eigs)

# ╔═╡ 53c03f5b-51b8-4534-870b-1a8b15c2c677
begin
	lt1N = localTask(dW1N,N1,eig1,grad1N)
	lt21N = localTask(dW21N,N21,eig21,grad21N)
	lt22N = localTask(dW22N,N22,eig22,grad22N)
end

# ╔═╡ 86f58454-53cf-46a6-ba19-72c571e425f3
begin
	plot(grad1N,lw=3,label="1D")
	plot!(grad21N,lw=3,label="2D zero eig")
	plot!(grad22N,lw=3,linestyle=:dash,label="2D non-zero eig")
	plot!(xlabel="epochs",ylabel="gradient norm squared")
end

# ╔═╡ a7b915d6-d620-4ade-91d3-52d1dfd1e85e
begin
	i = 200
	plot(lt1N[1:i],lw=3,label="1D",yaxis=:log10)
	plot!(lt21N[1:i],lw=3,label="2D zero eig")
	plot!(lt22N[1:i],lw=3,label="2D non-zero eig",linestyle=:dash)
	plot!(xlabel="epochs",ylabel="local task diff")
end

# ╔═╡ 59bcc767-65c9-4400-90a5-28f86c8243a4
md"## Test near steady state"

# ╔═╡ 1ddf3fce-6d48-4a9c-a40f-1c1ccacf6bb3
begin
	w0I = 0.1
	epochsI = 100
	muI = 0.05
	gammaI = muI/2
	F1NSet, W1NSet, dW1NSet, N1Set, grad1NSet = gradDescentNoise(f_1,df_1,mu,[w0I],epochsI,gammaI);
	F21NSet,W21NSet,dW21NSet, N21Set,grad21NSet = gradDescentNoise(f_2_1,df_2_1,mu,[w0I,w0I],epochsI,gammaI);
	F22NSet,W22NSet,dW22NSet, N22Set, grad22NSet = gradDescentNoise(f_2_2,df_2_2,mu,[w0I,w0I],epochsI,gammaI);
end

# ╔═╡ 98775bcd-69e3-4438-8935-b7ee23338543
begin
	p1SS= plot1DQuad(-w0I-w0I,w0I+w0I,f_1,W1NSet,F1NSet,:YlOrRd_9)
	savefig("./Figures/1d_descentNoise_SS.pdf")
	p1SS
end

# ╔═╡ 11fe02e3-8a8a-4479-9f83-d271d68cfb07
begin
	p21SS=plot2DQuad(-w0I-0.2, w0I+0.2, f_2_1 , W21NSet, F21NSet, true,:YlOrRd_9)
	savefig("./Figures/21d_descentNoise_SS.pdf")
	p21SS
end

# ╔═╡ 5c4c2d3c-e731-48ce-a188-23f5c358b658
begin
	p22SS=plot2DQuad(-w0I-0.2, w0I+0.2, f_2_2 , W22NSet, F22NSet, true,:YlOrRd_9)
	savefig("./Figures/22d_descentNoise_SS.pdf")
	p22SS
end

# ╔═╡ 0f29135b-075e-4e49-aa04-c8144d086c56
begin
	plot(F1NSet,lw=3,label="1D")
	plot!([mean(F1NSet)],lw=4,label="1D mean",seriestype=:hline,linestyle=:dot)
	plot!(F21NSet,lw=3,label="2D zero")
	plot!([mean(F21NSet)],lw=4,label="2D zero mean",seriestype=:hline,linestyle=:dot)
	plot!(F22NSet,lw=3,label="2D non-zero")
	plot!([mean(F22NSet)],lw=4,label="2D non-zero mean",seriestype=:hline,linestyle=:dot)
end

# ╔═╡ 982b19e1-0f83-4d06-85b3-f5a2fba22700
FsNSet = [F1NSet,F21NSet,F22NSet]

# ╔═╡ 6e996e99-6ebf-4eca-82bf-77c2f2067b44
ssNSet=[ss(i,150) for i in FsNSet]

# ╔═╡ b12785cc-4ee0-49e7-b2b4-431f5f68891b
plot(["1D","2D zero eig","2d non-zero eig"],ssNSet, seriestype=:bar, ylabel="ss value")

# ╔═╡ 0695c0f1-815d-45aa-b2f6-62bf53296ac1
begin
	lt1NSet = localTask(dW1NSet,N1Set,eig1,grad1NSet)
	lt21NSet = localTask(dW21NSet,N21Set,eig21,grad21NSet)
	lt22NSet = localTask(dW22NSet,N22Set,eig22,grad22NSet)
end

# ╔═╡ c7f80567-1f4b-48f9-ac8c-c0bf863db21e
begin
	plot(lt1NSet[1:i],lw=3,label="1D")
	plot!([mean(lt1NSet[1:i])],lw=3,label="1D mean",seriestype=:hline,linestyle=:dot)
	plot!(lt21NSet[1:i],lw=3,label="2D zero eig")
	plot!([mean(lt21NSet[1:i])],lw=3,label="2D zero mean",seriestype=:hline,linestyle=:dot)
	plot!(lt22NSet[1:i],lw=3,label="2D non-zero eig")
	plot!([mean(lt22NSet[1:i])],lw=3,label="2D non-zero mean",seriestype=:hline,linestyle=:dot)
	plot!(xlabel="epochs",ylabel="local task diff")
end

# ╔═╡ 360160f1-fba1-49a3-aa44-d3219d546226
begin
	plot(grad1NSet,lw=3,label="1D")
	plot!(grad21NSet,lw=3,label="2D zero eig")
	plot!(grad22NSet,lw=3,label="2D non-zero eig")
	plot!(xlabel="epochs",ylabel="gradient norm squared")
end

# ╔═╡ b2fc0546-046b-443a-9736-5b9fe6e30bc0
md"""
We have shown how a network expansion can increase learning speed and steady-state value. Furthermore, we have compared two different expansions. One expansion doesn't increase learning performance. The difference is in the change in loss landscape. 
Both expansions maintain the average curvature. However, one expansion adds a zero eigenvalue whereas the other adds an eigenvalue with same value as the original one.

The only issue is that we can't quite recover the result that the local task difficulty predicts steady-state value. If it were so we would expect the local task diff of the expanded net with the zero eigenvalue to have the smallest local task difficulty. It is not the case.

How can we predict or measure the expected steady state value as a measure of the loss landscape alone?
"""


# ╔═╡ Cell order:
# ╟─ab9b01ff-c53f-40d6-817e-07e837762cfb
# ╠═e2ae0aad-422b-40b6-9c78-0d5b4b4d05b8
# ╠═43619fa2-ea61-419d-b86e-c673301bc685
# ╠═0d17a04e-0334-4ff1-8c9b-bad141346414
# ╠═53b988f1-12a1-480c-ba3b-0078960deef6
# ╠═9c2bcbab-db2f-4dd3-847f-430df8e8adde
# ╟─38125f75-7244-4526-9a37-9bd1af217e72
# ╠═cea61646-34cf-42b4-acf8-2226fde52146
# ╠═247f31f4-dc34-11ec-0584-752289f2b41d
# ╠═0e1a0954-c70d-473d-8cf4-7c07449befbf
# ╠═18251608-65c1-414c-a9cd-734916c4b200
# ╠═bfb47a2e-592d-433f-b701-685af896d659
# ╠═831e1216-f083-4994-9934-254a480a2aef
# ╟─fc03f090-0912-4875-a2a8-f663620d26d6
# ╠═230d6234-ce7e-41ea-9f24-fc2d5be65e1c
# ╠═a1ebb468-fca5-4ff7-a74d-49e96bbae923
# ╠═02d62dbe-12e7-4c47-85a3-498d84eb7a15
# ╠═ba21765c-6c76-46b4-916c-47b3f4fdc4b8
# ╠═b76e6786-ec77-40ef-9442-59873eea830b
# ╠═d45ce8e2-a76f-4da2-bcfd-0cfea525230b
# ╠═a485ae5e-29f5-4801-ad4d-e4db58a5a544
# ╟─9b2e008e-2243-4fc8-81c4-29f5f31d70c9
# ╟─b3e607b5-ee76-46fe-9af0-3e2f5dc7c2ff
# ╟─248290dc-d84c-4f1a-9959-c95888c48d5c
# ╠═d4a70361-efe5-499a-9238-f90dc70354ce
# ╠═84b2e25b-b8cf-4afd-869a-86c3cc164b80
# ╠═4fb68f21-9fed-47bc-b2da-34f23cd4932f
# ╠═ddc8d2ea-6b5e-4868-b7c8-1c2e75a5af4a
# ╠═dae69d2c-1733-4a65-a5e9-4c9ed0e47dc3
# ╠═1b101b38-22f4-4956-9cc5-01c4d307d700
# ╠═c31d2ef6-4780-4e8e-901e-2a8da8821656
# ╠═b60fb647-4357-42b8-9e3c-a61e04d18787
# ╠═2412e898-66e8-474a-a11e-8956dfe05ad8
# ╠═127cbc58-f5cf-4450-979a-43698605d362
# ╠═c1b958e5-7089-4327-9eee-59a840ea65b2
# ╠═a4ba7d5f-d1d3-4f78-a6f6-75d107295912
# ╠═428dc44a-f818-4776-b792-540179ab3f47
# ╠═a97bf7dd-1c5d-406c-9931-3b65445b016a
# ╠═57dedfb8-ddf7-4181-8d18-3b989a4126c3
# ╠═db04c2a2-5ced-406d-a53a-bea66f938c7a
# ╠═02b3c8db-6f92-4117-a634-487b8504065d
# ╟─cad45de9-2b4d-4a74-ae62-413a9d1f29dd
# ╠═4f7504e4-abd1-44e3-ad87-287cc522718b
# ╠═58456c76-8ee6-417d-af5a-dc0254dc472a
# ╠═ed6b88e9-894e-4819-9a21-e5cbad0a0ff9
# ╠═80e35cb1-64c4-45c2-ae62-f881432b28c3
# ╠═c69c5757-0465-4748-bf92-bab08dc26991
# ╠═59061c77-818b-4646-96ed-4eec3dd7bec3
# ╠═f8e021b0-856e-460b-aec2-d41007e03c40
# ╠═06a64a29-00a5-4695-a9cc-68d64ca65311
# ╠═4b2ba9bb-dd48-4f19-9629-d3079761d455
# ╠═da129ba2-55cf-4097-8322-5ddea094aba7
# ╠═96560824-f6e3-4f53-9377-59afcb1578a1
# ╠═5077200c-ee07-4a2e-b837-d7022cff7e55
# ╠═53c03f5b-51b8-4534-870b-1a8b15c2c677
# ╠═86f58454-53cf-46a6-ba19-72c571e425f3
# ╠═a7b915d6-d620-4ade-91d3-52d1dfd1e85e
# ╟─59bcc767-65c9-4400-90a5-28f86c8243a4
# ╠═1ddf3fce-6d48-4a9c-a40f-1c1ccacf6bb3
# ╠═98775bcd-69e3-4438-8935-b7ee23338543
# ╠═11fe02e3-8a8a-4479-9f83-d271d68cfb07
# ╠═5c4c2d3c-e731-48ce-a188-23f5c358b658
# ╠═0f29135b-075e-4e49-aa04-c8144d086c56
# ╠═982b19e1-0f83-4d06-85b3-f5a2fba22700
# ╠═6e996e99-6ebf-4eca-82bf-77c2f2067b44
# ╠═b12785cc-4ee0-49e7-b2b4-431f5f68891b
# ╠═0695c0f1-815d-45aa-b2f6-62bf53296ac1
# ╠═c7f80567-1f4b-48f9-ac8c-c0bf863db21e
# ╠═360160f1-fba1-49a3-aa44-d3219d546226
# ╟─b2fc0546-046b-443a-9736-5b9fe6e30bc0
