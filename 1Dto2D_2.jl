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
The aim of this notebook is to illustrate the effect of adding parameters to a loss landscape for learning performance. We will consider the expansion from a 1D to a 2D quadratic function."""

# ╔═╡ 53b988f1-12a1-480c-ba3b-0078960deef6
begin
	# define parameters for the plots
	Plots.default(titlefont = (20, "times"), legendfontsize = 18, guidefont = (18, :black), tickfont = (12, :black), grid=false, framestyle = :zerolines, yminorgrid = false, markersize=6)
	width=1000
	height=600
end

# ╔═╡ 38125f75-7244-4526-9a37-9bd1af217e72
md"# Function definition"

# ╔═╡ cea61646-34cf-42b4-acf8-2226fde52146
md"We consider a simple 1D quadratic function $f(x)=x^2$. 
If $f$ is the loss function for training a network with a single parameter $w$, how does the loss function change if we add a second parameter to the network? 

This would correspond to a family of network expansions $\phi_a:\mathbb{R} \to \mathbb{R}^2$ such that $\phi_a(w) = (w,a)$ and the inverse mapping is defined $\phi^{-1}\left( (w_1,w_2) \right) = w_1$. 

The network expansion leads to a new loss function in 2D space. The exact form of this loss function is determined by the network architecture and the nature of the learning problem ---how the loss function is defined. 

Abstractly, there are infinitely many quadratic 2D functions. However, there will generally be constrains on the possible 2D functions that can arise from a network expansion.

Assume that the 2D functions must be quadratic and have the same average curvature as the 1D function. 
We consider two different quadratic functions that satisfy both of these constraints. 

First, we consider the 2D quadratic function $$f_{21}:\mathbb{R}^2\to\mathbb{R}$$ with 
$$f_{21}(w_1,w_2) =  2w_1^2$$
where $w_1$ and $w_2$ are the variables. 

Second, we consider the 2D quadratic function $$f_{22}:\mathbb{R}^2\to\mathbb{R}$$ with 
$$f_{22}(w_1,w_2) =  w_1^2+w_2^2$$
"

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
"""
	function to perform gradient descent on function f
	with diferential df
	train for epochs
	return array of task loss and weights during training
"""
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
"""
	compute learning speed of an array F of task loss during learning
	compute the mean of the change in loss over the first l epochs
"""
function ls(F,l=10)
	dF = (F[2:l+1]-F[1:l])./F[1:l]
	return mean(-dF)
end

# ╔═╡ bfb47a2e-592d-433f-b701-685af896d659
"""
	compute steady state loss of array F of loss during training 
	take the mean of the loss over the last l epochs
"""
function ss(F,l=20)
	return mean(F[end-l:end])
end

# ╔═╡ 831e1216-f083-4994-9934-254a480a2aef
"""
	plot a bar graph of the learning speed and steady state loss 
	for different loss arrays given by Fs
"""
function plotLsSS(Fs,t,xlbl=["1D","2D zero eig","2d non-zero eig"])
	lsA = [ls(i) for i in Fs]
	ssA = [ss(i) for i in Fs]
	lsP = plot(xlbl,lsA,seriestype=:bar,ylabel="learning speed",title=t)
	ssP = plot(xlbl,ssA, seriestype=:bar, ylabel="ss value")
	plot(lsP, ssP, layout = (2, 1), legend = false)
	# plot!(title=t)
end

# ╔═╡ fc03f090-0912-4875-a2a8-f663620d26d6
md"""# Gradient descent
First consider descending the loss function with gradient descent. The change in weights at epoch $t$ is 
$$\delta w_t = -\mu \nabla_w F[w_t]$$
"""

# ╔═╡ 230d6234-ce7e-41ea-9f24-fc2d5be65e1c
begin
	# training parameters
	mu=0.1; epochs=100
	w0=10
end

# ╔═╡ 84b2e25b-b8cf-4afd-869a-86c3cc164b80
begin 
	using Random
	Random.seed(10)
	SNR = 2 # signal to noise ratio
	gamma=mu/SNR; # noise term scalar
end

# ╔═╡ a1ebb468-fca5-4ff7-a74d-49e96bbae923
F1, W1 = gradDescent(f_1,df_1,mu*2,[w0*sqrt(2)],epochs);

# ╔═╡ 02d62dbe-12e7-4c47-85a3-498d84eb7a15
F21,W21 = gradDescent(f_2_1,df_2_1,mu,[w0,w0],epochs);

# ╔═╡ ba21765c-6c76-46b4-916c-47b3f4fdc4b8
F22,W22 = gradDescent(f_2_2,df_2_2,mu,[w0,w0],epochs);

# ╔═╡ b76e6786-ec77-40ef-9442-59873eea830b
begin
	lbl = ["1D","2D zero eig","2d non-zero eig"]
	plot(F1,lw=3,label = lbl[1])
	plot!(F21,lw=3,label=lbl[2])
	plot!(F22,linestyle=:dash,lw=3,label=lbl[3],xlabel="training epochs",ylabel="loss")
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
We have done gradient descent on the three different functions starting at the same value. We observe that the 2D function with one zero eigenvalue has larger learning speed than the 2D function with non-zero eigenvalues. We expected this because the gradient of this function is larger. The steady state loss is the same because in all cases gradient descent converges to the minimum $0$."""

# ╔═╡ b3e607b5-ee76-46fe-9af0-3e2f5dc7c2ff
md"""
In the case of gradient descent with some random noise we expect in all cases to reach some non-zero steady state value. This steady state value should be detemined by the local task difficulty."""

# ╔═╡ 248290dc-d84c-4f1a-9959-c95888c48d5c
md" # Gradient descent with noise
It is implausible that biological neural networks compute learn with pure gradient descent. Indeed, computing gradients can be difficult (credit assignement problem) and applying exact change in weigths with biological components in sysnapses is inherently noisy. 

We consider a more general learning rule: gradient descent with noise 
$$\delta w_t = -\mu \nabla_{w} F[w_t] + \gamma \epsilon$$ 
where $$\epsilon$$ is the noise term (usually drawn from a random normal distribution).
"

# ╔═╡ d4a70361-efe5-499a-9238-f90dc70354ce
"""
	run gradient decent with gaussian noise 
	mu is the learning step and gamma is the strength of the gaussina noise 
	returns the array of loss, weights, gradient decsent term, noise term and gradient norm during training
"""
function gradDescentNoise(f,df,mu,w0,epochs=1,gamma=0,Wset=false)
	W = zeros(epochs,length(w0))
	dW = zeros(epochs,length(w0))
	grad = zeros(epochs)
	N = zeros(epochs,length(w0))
	F = zeros(epochs)
	w=w0
	for i=1:epochs
		W[i,:] = w
		F[i] = f(w...) # loss value
		grad[i] = sqrt(sum(df(w...).^2)) # gradient norm
		dW[i,:] = -mu*df(w...) # gradient descent term
		N[i,:] = gamma*randn(length(w)) # noise term
		if Wset==false
			w = w +dW[i,:] + N[i,:]
		else
			w = Wset[i,:]
		end
	end
	return F,W,dW,N, grad
end

# ╔═╡ 4fb68f21-9fed-47bc-b2da-34f23cd4932f
F1N, W1N, dW1N, N1, grad1N = gradDescentNoise(f_1,df_1,mu*2,[w0*sqrt(2)],epochs,gamma*2);

# ╔═╡ ddc8d2ea-6b5e-4868-b7c8-1c2e75a5af4a
F21N,W21N,dW21N, N21,grad21N = gradDescentNoise(f_2_1,df_2_1,mu,[w0,w0],epochs,gamma);

# ╔═╡ dae69d2c-1733-4a65-a5e9-4c9ed0e47dc3
F22N,W22N,dW22N, N22,grad22N = gradDescentNoise(f_2_2,df_2_2,mu,[w0,w0],epochs,gamma);

# ╔═╡ 1b101b38-22f4-4956-9cc5-01c4d307d700
begin
	plot(F1N,lw=3,label = lbl[1])
	plot!(F21N,linestyle=:dash,lw=3,label=lbl[2])
	plot!(F22N,linestyle=:dash,lw=3,label=lbl[3],xlabel="training epochs",ylabel="loss")
end

# ╔═╡ ba75ddcd-68ce-4ddf-aede-57ff481c88ff
begin
	FsN = [F1N,F21N,F22N];
	plotLsSS(FsN,"Gradient descent noise = $gamma")
end

# ╔═╡ 3b6136d2-cb8a-47b1-adf2-8a208d83eb36
md"The 2D quadriatic function with a zero eigenvalue has the best learning performance: high learning speed and small steady state loss"

# ╔═╡ 9c266f5c-e370-4801-a1cc-dbefeed83434
md"## Loss landscape visualisation
We plot the 1D and 2D functions to visualise the change in the geometry of the loss landscapes from adding a parameter to the 1D function
"

# ╔═╡ 2412e898-66e8-474a-a11e-8956dfe05ad8
""" plot surface of 1d quadratic fucntion f 
	between [minV,  maxV]. If W and F are given, plot trajectory from a descent algorithm
	W is  a 1D array with the trajectory of the weights and F a  1D array 
	with the trajectory of the loss
"""
function plot1DQuad(minV, maxV, f, W=false, F=false,c=:hot)
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
""" plot surface (if surf=True) and contour plots of 2D quadratic fucntion f 
	between [minV,  maxV]. If W and F are given, plot trajectory from a descent algorithm
	W is  a 2D array with the trajectory of the weights and F a  1D array 
	with the trajectory of the error 
"""
function plot2DQuad(minV, maxV, f, W=false, F=false, surf=true,c=:YlOrRd_9)
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

# ╔═╡ 1ba560aa-1d96-4d0d-aac0-904cca5ed08d
md"The 2D function with a zero eigengvalue $$f_{21}$$ has the form of a valley with minimum for $$w_1=0$$. Indeed, it is essentially a parabola translated through the $$w_2$$ axis. 

By can compute the gradient and the hessian of the function, we can measure the slope and the curvature of the loss landscape. The gradient at point $$(w_1,w_2)$$ is 

$$\nabla f_{21}(w_1,w_2) = \begin{pmatrix}
        4w_1 & 0\\
    \end{pmatrix}^T$$

The hessian matrix is constant in $\mathbb{R}$ and given by 

$$\nabla^2 f_{21}(w_1,w_2) = \begin{pmatrix}
        %2a & 0\\
        4 & 0\\
        0 & 0\\
    \end{pmatrix}$$

We find the slope in the $$w_2$$ direction is zero and in the $$w_1$$ direction it is proportional to $$w_1$$. The curvature is zero in the $w_2$ direction and $4$ in the $w_1$ direction.
The average curvature given by the average eigenvalue is $\frac{Tr(\nabla^2 f_{21})}{N} = \frac{4}{2} = 2$.
The spread of the eigenvalues 
$\frac{Tr((\nabla^2 f_{21})^3)}{Tr\left( (\nabla^2 f_{21})^2 \right)} = 4$

This 2D quadratic function has a larger slope $||\nabla f_{21}(\phi_a(w))||=4w \ge \frac{df(w)}{dw} = 2|w|$, same average curvature but larger spread of eigenvalues compared to the 1D function. 



The 2D function with non-zero eigenvalue $f_{22}$ has the form of a symmetric parabola with minimum for $w_1=0, w_2=0$. 
The gradient of this function is the vector 

$$\nabla f_{22}(w_1,w_2) = \begin{pmatrix}
        %2aw_1\\
        2w_1\\
        2w_2\\
    \end{pmatrix}$$

meaning that the slope in the $w_1$ and $w_2$ directions for $w_1=w_2$ are equal.

The hessian matrix is constant in $\mathbb{R}$ and given by 
$$\nabla^2 f_{22}(w_1,w_2) = \begin{pmatrix}
        %2a & 0\\
        2 & 0\\
        0 & 2\\
    \end{pmatrix}$$
This matrix has two eigenvectors $(1 \quad 0)^T$ and  $(0 \quad 1)^T$ with corresponding eigenvalues $\lambda_1=2$ and $\lambda_2=2$. 
The curvature is equal in all directions, indeed, the loss landscape is symmetrical. 
As expected, the average curvature given by the average eigenvalue is
$\frac{Tr(\nabla^2 f_{22})}{N} = \frac{4}{2} = 2$

The spread of the eigenvalues 
$$\frac{Tr((\nabla^2 f_{22})^3)}{Tr\left( (\nabla^2 f_{22})^2 \right)} = 2$$

This 2D quadratic function has the same average curvature and spread of eigenvalues of the hessian than the 1D quadratic function. 
The slope $||\nabla f_{22}(\phi_a(w))||=2\sqrt{w^2+a^2}=2||\phi_a(w)||$ is equal to the slope of the 1D function as long as $||\phi_a(w)||=|w|$ (i.e. as long as point in 2D space is the same distance as the point in 1D space).  

We have considered two qualitative different expansions of the 1D quadratic function $f$. The first expansions adds a zero eigenvalue to the hessian (i.e. a direction of zero curvature), the second expansion adds an eigenvalue equal to the average eigenvalue. Figure \ref{fig:1D2Dfun} shows the loss landscape for the 1D and the two different 2D functions.  
We will show that these two different types of expansions have a different effect on learning performance. One, is beneficial ---it speeds up learning---, the other doesn't. 

"

# ╔═╡ b2fc0546-046b-443a-9736-5b9fe6e30bc0
md"""
We have shown how a network expansion can increase learning speed and steady-state value. Furthermore, we have compared two different expansions. One expansion doesn't increase learning performance. The difference is in the change in loss landscape. 
Both expansions maintain the average curvature. However, one expansion adds a zero eigenvalue whereas the other adds an eigenvalue with same value as the original one.

The only issue is that we can't quite recover the result that the local task difficulty predicts steady-state value. If it were so we would expect the local task diff of the expanded net with the zero eigenvalue to have the smallest local task difficulty. It is not the case.

How can we predict or measure the expected steady state value as a measure of the loss landscape alone?
"""


# ╔═╡ Cell order:
# ╠═ab9b01ff-c53f-40d6-817e-07e837762cfb
# ╠═e2ae0aad-422b-40b6-9c78-0d5b4b4d05b8
# ╠═43619fa2-ea61-419d-b86e-c673301bc685
# ╠═0d17a04e-0334-4ff1-8c9b-bad141346414
# ╠═53b988f1-12a1-480c-ba3b-0078960deef6
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
# ╠═ba75ddcd-68ce-4ddf-aede-57ff481c88ff
# ╠═3b6136d2-cb8a-47b1-adf2-8a208d83eb36
# ╟─9c266f5c-e370-4801-a1cc-dbefeed83434
# ╠═2412e898-66e8-474a-a11e-8956dfe05ad8
# ╠═127cbc58-f5cf-4450-979a-43698605d362
# ╠═c1b958e5-7089-4327-9eee-59a840ea65b2
# ╠═a4ba7d5f-d1d3-4f78-a6f6-75d107295912
# ╠═428dc44a-f818-4776-b792-540179ab3f47
# ╠═a97bf7dd-1c5d-406c-9931-3b65445b016a
# ╟─1ba560aa-1d96-4d0d-aac0-904cca5ed08d
# ╟─b2fc0546-046b-443a-9736-5b9fe6e30bc0
