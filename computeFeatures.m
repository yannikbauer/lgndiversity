function [f, b] = computeFeatures(X)
% data = computeRGCFeatures(dates)
%
%   X response matrix of T by N (time by neurons)
%

assert(exist('spca')>0,'computeFeatures: sparse pca not found on path!')

% sparse chirp features
nComp = 20;         % adjust these values to yield good results
nNonZero = 10;

[b, ~] = spca(X',[],nComp,inf,-nNonZero);

fid = zeros(nComp,1);
for i=1:size(b,2)
    fid(i) = find(abs(diff(b(:,i)))>0,1,'first'); 
end
[~, sidx] = sort(fid,'ascend');

b = b(:,sidx);
f = b' * X;


