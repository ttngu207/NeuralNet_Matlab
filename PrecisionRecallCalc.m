function [ accuracy,P_R_F ] = PrecisionRecallCalc( Ytrue, Ypredict )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

accuracy =  mean(double(Ytrue == Ypredict)) * 100;

subjNum = length(unique(Ytrue));

P_R_F = [];
for subj = 1: subjNum
ytrue  = zeros(size(Ytrue)); ypredict = zeros(size(Ypredict));
ytrue(Ytrue == subj) = 1; ypredict(Ypredict == subj) = 1; 


truePos = ytrue + ypredict; truePos = length(find(truePos ==2));
falsePos = ytrue - ypredict; falsePos = length(find(falsePos == -1));
falseNeg = ypredict - ytrue; falseNeg = length(find(falseNeg == -1));
Precision = truePos /(truePos + falsePos);
Recall = truePos / (truePos + falseNeg);
F1score = 2*Precision*Recall / (Precision + Recall);

Vec = [subj,Precision*100,Recall*100,F1score*100];
P_R_F = [P_R_F;Vec];
end

end

