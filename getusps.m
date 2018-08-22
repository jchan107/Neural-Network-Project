function [trainX,trainY,testX,testY] = getusps(c1,c2,ntrain)

all = load('usps_all.mat');
D = double(all.data);
X = reshape(D(:,:,[c1 c2]),[256 size(D,2)*2]);
X = X';
Y = reshape(ones(size(D,2),1)*[0 1],[1 size(D,2)*2]);
Y = Y';
randstate = rng;
rng(132857); % to make the data consistent for grading
rperm = randperm(length(Y));
rng(randstate); % restore
X = X(rperm,:);
Y = Y(rperm,:);

trainX = X(1:ntrain,:);
trainY = Y(1:ntrain,:);
testX = X(ntrain+1:end,:);
testY = Y(ntrain+1:end,:);
