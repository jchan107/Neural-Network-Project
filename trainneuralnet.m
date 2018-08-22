%Jeremy Chan SID: 861169589 Date: 11/20/17 CS171 PS3
function [W1,W2] = trainneuralnet(X,Y,nhid,lambda)
%TRAINNEURALNET Summary of this function goes here
%   Detailed explanation goes here
    inputLayerSize = size(X,2);
    inputNumVar = size(X,1);
    deltaOut = 1;
    eta = .1;
    iteration = 0;
    
    W1 = rand(nhid,inputLayerSize+1);
    W1 = rdivide(W1,10);
    W2 = rand(1,nhid+1);
    W2 = rdivide(W2,10);
    index = 1;
    oldLoss = 1;
    loss = 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%
    %X = [ 2 -1;2 -1];
    %Y = 1;
    %W1 = [-1 2 2; 0 -1 3; 1 5 0];
    %W2 = [0 0 1 2; -2 1 -1 1];
    
    %%%%%%%%%%%%%%%%%%%%%%%
    insert = ones(inputNumVar,1);
    %insert = [1;1];
    %disp(insert);
    %disp(X);
    mGrad = 1;
    
    X = [insert X(:,:)];
    while (mGrad > (1e-3))
          [hiddenLayer,outerLayer] = forwardProp(X,W1,W2);
          
          [deltaW1,deltaW2] = backwardProp(hiddenLayer,outerLayer,Y,X,W1,W2,lambda);
          
          deltaW1 = (deltaW1) + ((2*lambda*W1)/inputNumVar);
          deltaW2 = (deltaW2) + ((2*lambda*W2)/inputNumVar);
          
          %deltaW1 = deltaW1*eta;
          %deltaW2 = deltaW2*eta;
          
          mGrad = max(max(max(abs(deltaW1))),max(max(abs(deltaW2))));
          
          if(iteration == 0)
                 loss = lossF(W1,W2,lambda,Y,outerLayer,inputNumVar);
                 oldLoss = loss;
          else
                 oldLoss = loss;
                 loss = lossF(W1,W2,lambda,Y,outerLayer,inputNumVar);
          end
          
         if(mod(iteration,1000) == 0)
             disp(loss);
             if(oldLoss <= loss)
                 eta = eta/10;
             end
         end
         W1 = W1 - eta*deltaW1;
         W2 = W2 - eta*deltaW2;
         iteration = iteration + 1;
    end

end

function [z] = sigmoid(a)
    z = 1/(1+exp(-a));
end

function [hiddenLayer,outerLayer] = forwardProp(X,W1,W2)
    hiddenLayer = W1*X';
    for i = 1:size(hiddenLayer,1)
        for j = 1:size(hiddenLayer,2)
            hiddenLayer(i,j) = sigmoid(hiddenLayer(i,j));
        end
    end
    hiddenNumVar = size(hiddenLayer,2);
    insert = ones(1,hiddenNumVar);
    hiddenLayer = [insert; hiddenLayer(:,:)];
    outerLayer = W2*hiddenLayer;
    for i = 1:size(outerLayer,1)
        for j = 1:size(outerLayer,2)
            outerLayer(i,j) = sigmoid(outerLayer(i,j));
        end
    end
end

function [deltaW1,deltaW2] = backwardProp(hiddenLayer,outerLayer,Y,X,W1,W2,lambda)
    col = ones(size(hiddenLayer));
    numVar = size(outerLayer,2);
    deltaOut = outerLayer - Y';
    delta1 = (hiddenLayer.*(col-hiddenLayer)).*(transpose(W2)*deltaOut);
    deltaW2=deltaOut*(transpose(hiddenLayer));
    deltaW1=delta1(2:end,:)*X;
    deltaW2 = deltaW2/numVar;
    deltaW1 = deltaW1/numVar;
end

function [loss] = lossF(W1,W2,lambda,Y,outerLayer,m)
    col = ones(size(Y));
    loss = sum(-(Y.*log((outerLayer')) + (col - Y).*log(col-(outerLayer'))))/m;
    W1sq = W1.^2;
    W2sq = W2.^2;
    W1sum = sum((sum(W1sq,1)),2);
    W2sum = sum(W2sq);
    regularization = ((lambda*(W1sum + W2sum))/m);
    loss = loss + regularization;
end