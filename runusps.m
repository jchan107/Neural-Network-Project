function runusps

[trainX,trainY,testX,testY] = getusps(7,9,550);

nhiddens = [5 10 50];
lambdas = logspace(-4,0,5);
lambdas = lambdas*length(trainY);
erates = nan(length(lambdas),length(nhiddens));

li = 1;
for lambda=lambdas
	ni = 1;
	for nhidden=nhiddens
		mingrad = 1e-3;
		% this function (trainneuralnet) you are to supply!
		[W1,W2] = trainneuralnet(trainX,trainY,nhidden,lambda);
		% this function (nneval) you are to supply too!
		predY = nneval(testX,W1,W2);
        %disp(size(predY));
        %disp(predY);
		predY(predY<0.5) = 0;
		predY(predY>=0.5) = 1;
		testerr = sum(predY~=testY)/length(testY);
		erates(li,ni) = testerr;
		plotit(lambdas,nhiddens,erates);
		ni = ni+1;
	end;
	li=li+1;
end;

figure(1);
print -dpdf result.pdf;


function plotit(ls,ns,errs)

figure(1);
hold off;
ll = cell(length(ns),1);
for i=1:length(ns)
     loglog(ls,errs(:,i));
     hold on;
     ll{i} = num2str(ns(i));
end;
legend(ll{:})
xlabel('lambda');
ylabel('testing error rate');
hold off;
drawnow;
