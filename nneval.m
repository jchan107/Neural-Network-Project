%Jeremy Chan SID: 861169589 Date: 11/20/17 CS171 PS3
function [predY] = nneval(X,W1,W2)
%NNEVAL Summary of this function goes here
%   Detailed explanation goes here
    inputNumVar = size(X,1);
    insert = ones(inputNumVar,1);
    X = [insert X(:,:)];

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
    predY = outerLayer';
end

function [z] = sigmoid(a)
    z = 1/(1+exp(-a));
end