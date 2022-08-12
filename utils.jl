"""
    generate a N-dim quad function with one 
	non-zero eig such that Tr(H)/N = Tr_N
	return function computing the value 
	derivative and eigenvalues
"""
function quadraticN_Zero(N,Tr_N)
	eig = Tr_N*N # non-zero eig 
	eigs = zeros(N)
	eigs[1] = eig
	f(w) = eig/2*w[1]^2
	df(w) = eigs.*w
	return f, df, eigs
end

"""
	generate a N-dim quad function with  
	all eq eig such that Tr(H)/N = Tr_N
	return function computing the value 
	derivative and eigenvalues
"""
function quadraticN_NonZero(N,Tr_N)
	eig = Tr_N # eig value
	eigs = eig*ones(N)
	f(w) = sum(w.^2 .*eig/2)
	df(w) = eigs.*w
	return f, df, eigs
end

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
		F[i] = f(w)
		w = w -mu*df(w)
	end
	return F,W
end

"""
	run gradient decent with gaussian noise 
	mu is the learning step and gamma is the strength of the gaussina noise 
	returns the array of loss, weights, gradient decsent term, noise term and gradient norm during training
"""
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

"""
	compute learning speed of an array F of task loss during learning
	compute the mean of the change in loss over the first l epochs
"""
function ls(F,l=10)
	dF = (F[2:l+1]-F[1:l])
	return mean(-dF)
end

"""
	compute steady state loss of array F of loss during training 
	take the mean of the loss over the last l epochs
"""
function ss(F,l=20)
	return mean(F[end-l:end])
end

"""
	plot learning speed and steady state loss 
	for different loss arrays given by Fs
"""
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

"""
    compute hessian projection v^THv/||v||^2
    in the particular case where the eigenvectors of  are the euclidean basis vectors 
    v is the vector eig the eigenvalues of hess in the euclidean basis (same length array)
"""
function computeHessianProj(v,eig)
	P = zeros(size(v,1))
	for i=1:size(v,1)
		P[i] = sum(v[i,:].^2 .*eig)/sum(v[i,:].^2)
	end
	return P
end

"""
    compute the average eigenvalue (i.e. trace of hessian / number of parameters)
"""
function TrH_N(eig)
	return sum(eig)/length(eig)
end

"""
    compute Tr(H³)/Tr(H²) , where H is the hessian
    eig are the eienvalues of the hessian 
"""
function TrH3_TrH2(eig)
	return sum(eig.^3)/sum(eig.^2)
end

"""
    simulate gradient descent with learning step mu and gaussian noise with coef gamma (can be zero)
    compute learning speed and steady state loss 
        learning speed is computed from loss trajecotry from training from w0
        steady state loss is computed from loss trajectory training from 
            close to minimum ( w = (w0ss,w0ss,...,w0ss) )
    returns ls and ss and dictionaries with training variables 
"""
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

"""
    simulate training of functions quadF for different number of weigths given by Ns 
        and different training parameters (learning step mu and gaussian noise gamma)
    Tr_N is the constant value of average eigenvalue of all functions 
    w0 is a scalar, the initial weight for the 1D function 
        we scale the initial weight for N-dim function to math inital value of function 
            for all sized networks
    musVar is an array of arrays lenght(Ns)xnumberOfMus 
        it gives the values of the learning steps to train each function 
    SNR constant ratio mu/gamma for each training situation
    can run a pre-analisis of the simulation results by calling analyseF
"""
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

"""
    scatter ss vs ls at indices 
"""
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

"""
    swap the order of an array of arrays 
    [[v11,v12,...],[v21,v22,...]] becomes [[v11,v21,...],[v12,v22,...]]
"""
function swapOrder(v)
	vv = zeros(length(v[1]))
	for j in 1:length(v[1])
		vv[j] = [v[i][j] for i=1:length(v)]
	end
	return vv
end

"""
    get the values v[:][i] of an array of arrays v 
    if sym is given return v[:][i][symb]
"""
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

"""
    return ls and ss values 
    vals is an array of arrays [[ls,ss],[ls,ss],...]
"""
function getLs_SS(vals,args=[])
	lsS = getVal(vals,1)
	ssS = getVal(vals,2)
	return lsS,ssS
end

"""
    compute standard error of array ls
"""
function my_std(ls)
    numSim = length(ls)
    lsM = hcat(ls...)
    map(1:size(lsM,1)) do m
        std(lsM[m,:])./sqrt(numSim)
    end
end


""" 
	Compute theoretical learning speed
"""
function learningSpeed(F,mu,dF,grad,gamma,Noise,eig,dt=1)
	val= -1 ./(F.*dt).*(-mu.*grad.^2 .+ mu^2 .*grad.^2 .*computeHessianProj(dF,eig) .+gamma^2 .*sum(Noise.^2 ;dims=2).*computeHessianProj(Noise,eig))
	return val[:,1]
end

""" 
	Compute theoretical learning speed
	dict has training variables
"""
function learningSpeed(dict,mu,gamma,eig,dt=1)
	learningSpeed(dict[:F],mu,dict[:dF],dict[:grad],gamma,dict[:Noise],eig,dt)
end

"""
	Compute the local task difficulty 
"""
function localTask(dF,N,eig,grad,mu,gamma)
	# val= mu*computeHessianProj(dF,eig)+gamma^2/mu.*(sum(N.^2; dims=2)./grad.^2).*computeHessianProj(N,eig)
	val= mu*computeHessianProj(dF,eig)+gamma^2/mu.*(sum(N.^2; dims=2)./grad.^2).*computeHessianProj(N,eig).*grad.^2
	return val[:,1]
	# 1/2*computeHessianProj(dW21NSet.+N21Set,eig21).*(sum((dW21NSet.+N21Set).^2;dims=2).*grad21NSet.^2)
	# return 1/2*computeHessianProj(dW,eig).*(dW.^2 ./grad.^2) + 1/2*computeHessianProj(N,eig).*(N.^2 ./grad.^2)
end


function localTask(dict,eig,mu,gamma)
	localTask(dict[:dF],dict[:Noise],eig,dict[:grad],mu,gamma)
end

"""
	Compute the theoretical ls and lp from training with mus and gammas=mus/SNR
	loss functions are given by quadF
"""
function getLpVar(vals,Ns,quadF,mus,SNR,index=false)
	lp = map(1:length(vals)) do i
		N = Ns[i]
		f,df,eigs = quadF(N,Tr_N)
		# w0V = (w0/sqrt(N))*ones(N) # scale the initial weight to match initial val
		map(1:length(vals[i])) do j
			mu = mus[j]
			gamma = mu/SNR
			dict = vals[i][j][3]
			dictSS = vals[i][j][4]
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

"""
	plot learning speed
"""
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

function plotHessProj(dW,eig,t)
	# plot hessian proj of dW and traces
	dwHdw = computeHessianProj(dW,eig)
	plot(dwHdw,lw=3,label="proj")
	plot!([TrH3_TrH2(eig)], seriestype=:hline,lw=3, label="TrH^3/TrH^2", linestyle=:dash)
	plot!([TrH_N(eig)],seriestype=:hline,lw=3,label="TrH/N",linestyle=:dash)
	plot!(title=t)
end


function plotHessianParam(eigs,t="")
	TrH3s = [TrH3_TrH2(eig) for eig in eigs]
	TrHs = [TrH_N(eig) for eig in eigs]
	plot(["1D","2D zero eig","2d non-zero eig"],TrH3s,seriestype=:bar,ylabel="Trace terms",title=t,label="TrH3_TrH2",bar_position=:dodge)
	plot!(["1D","2D zero eig","2d non-zero eig"],TrHs,seriestype=:bar,ylabel="Trace terms",title=t,label="TrH_N",bar_position=:dodge)
	# ssP = plot(["1D","2D zero eig","2d non-zero eig"],ssA, seriestype=:bar, ylabel="ss value")
	# plot(lsP, ssP, layout = (2, 1), legend = false)
	# plot!(title=t)
end

function optimalMu(trH_N,w0,N,SNR,quadCase=1)
	if quadCase == 1
		return (trH_N*w0^2)/(2*(w0^2*trH_N^2*N-1/SNR^2))
	elseif quadCase== 2
		return (trH_N*w0^2)/(2*(w0^2*trH_N^2-N/SNR^2))
	end
end