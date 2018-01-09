function [dL,dW] = mj_nnwsum(L,W,dzdy, varargin)
% [dL,dW] = mj_nnwsum(L,W,dzdy, varargin)
% Weighted sum. Useful for multi-task learning
%
% Input:
%  - W: weights (one per task)
%  - L: loss vector (one per task)
%
% Options:
%  - 'learnW': boolean to learn weights (backprop)
%
% (c) MJMJ/2016

opts.learnW = false;
%opts.thr = 0.5;

opts = vl_argparse(opts,varargin) ;
%Y2 = []; % Init

%nloss = length(L); 

if isempty(dzdy) %forward
   Y = W * L';
   dL = Y; % Output
else   
   % Init
   dW = zerosLike(W);
   dL = zerosLike(L);
   
   % Derivative wrt Loss
   dL(:) = W .* dzdy;
   
   % Derivative wrt Weights
   if opts.learnW
      dW(:) = L .* dzdy;      
   end
end


% if isempty(dzdy) %forward
%    a1 = 0.5 * sum(res(:,posIdx) .^2,1);
%    Ypos = a1* (1.0/nsamples);
%    
%    Yneg = 0.5 * (max(0, m - sqrt(sum(res(:,negIdx).^2,1)))).^2;
%    Y = sum(Ypos) + sum(Yneg);
% else
%    
%    % Init
%    Y_ = zerosLike(X2);
%    Y2_ = zerosLike(X2);
%    
%    % For positive pairs
%    Y_(1,1,:,posIdx)= -1.* res(:,posIdx);
%    Y = single (Y_ * (dzdy / n) );
%    
%    Y2_(1,1,:,posIdx)= res(:,posIdx);
%    Y2 = single (Y2_ * (dzdy / n) );
%    
%    % For negative pairs
%    resN = res(:,negIdx);
%    Yneg = zerosLike(resN);
%    N = sqrt(sum(resN.^2,1));      % Euclidean distance
%    
%    below = find(N <= m);
%    bN = repmat(N(below),[size(resN,1),1]);
%    Yneg(:,below) = + (m - bN).*(resN(:,below)./bN); % First minus is correct
%    Y(1,1,:,negIdx) = single (Yneg * (dzdy / n) );
%    
%    Y2neg = -Yneg;
%    Y2(1,1,:,negIdx) = single (Y2neg * (dzdy / n) );
% end

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
    y = gpuArray.zeros(size(x),'single') ;
else
    y = zeros(size(x),'single') ;
end
