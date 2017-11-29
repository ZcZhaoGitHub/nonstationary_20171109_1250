function [ learningCurve ] = LMMN(delte,stepSizeWeightVector,Inputsignal,desired_sig_noise,inputDimension,trainSize,testSize)
[trainInput,trainTarget,testInput,testTarget] = distribution(Inputsignal,desired_sig_noise,inputDimension,trainSize,testSize);
% memeory initialization
learningCurve = zeros(trainSize,1);
weightVector = zeros(inputDimension,1);
learningCurve(1)=1;
% training
%    functionError(n) = delte*aprioriErr(n) + ( 1 - delte )*(aprioriErr(n)^3);
for n = 2:trainSize
    aprioriErr = trainTarget(n) -  weightVector'*trainInput(:,n);
    functionError = delte*aprioriErr + (1-delte)*(aprioriErr^3);
    weightVector = weightVector + stepSizeWeightVector*functionError*trainInput(:,n);
    % theory
    % testing
    temperror = testTarget - testInput'*weightVector;
    learningCurve(n) = mean(temperror.^2);
end
return


