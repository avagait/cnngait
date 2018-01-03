function [Y, Y2] = mj_nndiffloss(X,X2,c,dzdy, varargin)
% [Y, Y2] = mj_nndiffloss(X,X2,c,dzdy, varargin)
%
% This function, used in Marin-Jimenez et al, ICIP'2017, extends the one released by:
%
%Robust Optimization for Deep Regression
%V. Belagiannis, C. Rupprecht, G. Carneiro, and N. Navab,
%ICCV 2015, Santiago de Chile.
%Available loss functions: Tukey and L2 loss.
%See the paper for details on the Tukey loss.
%http://campar.in.tum.de/twiki/pub/Chair/DeepReg/deepreg.html
%Contact: Vasilis Belagiannis, vb@robots.ox.ac.uk

opts.loss = 'hadsellloss' ;
opts.thr = 0.5;

opts = vl_argparse(opts,varargin) ;
Y2 = []; % Init
nsamples = length(c);

switch lower(opts.loss)
   case {'hadsellloss'}
         X = squeeze(X);
         Y = squeeze(X2);
         
         posIdx = find(c > 0);
         negIdx = find(c < 0);
      
        res=(Y-X);
        
        m = opts.thr;
        
        n=1;
        if isempty(dzdy) %forward
           Ypos = 0.5 * sum(res(:,posIdx) .^2,1);
           %Ypos = a1* (1.0/nsamples);
            
           Yneg = 0.5 * (max(0, m - sqrt(sum(res(:,negIdx).^2,1)))).^2;
           Y = (sum(Ypos) + sum(Yneg))/nsamples;
        else
           % Init
           Y_ = zerosLike(X2);
           Y2_ = zerosLike(X2);
           
           % For positive pairs
           Y_(1,1,:,posIdx)= -1.* res(:,posIdx);
           Y = single (Y_ * (dzdy / n) );
                      
           Y2_(1,1,:,posIdx)= res(:,posIdx); 
           Y2 = single (Y2_ * (dzdy / n) );
           
           % For negative pairs
           resN = res(:,negIdx);
           Yneg = zerosLike(resN);
           N = sqrt(sum(resN.^2,1));      % Euclidean distance
           
           below = find(N <= m);
           bN = repmat(N(below),[size(resN,1),1]);
           Yneg(:,below) = + (m - bN).*(resN(:,below)./bN); % First minus is correct
           Y(1,1,:,negIdx) = single (Yneg * (dzdy / n) );
           
           Y2neg = -Yneg;
           Y2(1,1,:,negIdx) = single (Y2neg * (dzdy / n) );
        end
        
   %l2loss
   case {'l2loss'}
      if iscell(c)
         X = reshape(X(1,1,:,:),[size(c{1,1},1),size(c,2)]);
         Y = [c{1,1:size(c,2)}];
      else
         X = squeeze(X)';
         Y = c;
      end

        res=(Y-X);
        
        n=1;
        if isempty(dzdy) %forward
            Y = (sum(res(:).^2))/numel(res);
        else
            Y_(1,1,:,:)= -1.*(Y-X);
            Y = single (Y_ * (dzdy / n) );
        end
                    
   % verification error layer: {-1: different; +1: same}
   case {'acc'} % accuracy

      if isempty(dzdy) %forward
         
         X = squeeze(X);
         Y = squeeze(X2);
         % Get current threshold
         %thr_ = opts.thr;
         thr_ = mj_findBestVerifThr(X, Y, c);          
         
         res=(Y-X);
         D = sqrt(sum(res.^2,1));
         
         estimPos = D <= thr_;
         estimNeg = D > thr_;
         TP = sum(c(estimPos) == 1); % / sum(estimPos);
         TN = sum(c(estimNeg) == -1); % / sum(estimNeg);
         
         Y = (TP+TN); %/length(c);
         
      else %nothing to backprop
         Y = zerosLike(X) ;
      end
        
        %error layer
   case {'mpe'} %mean pixel error
      X_orig = X;
      if iscell(c)
         X = reshape(X(1,1,:,:),[size(c{1,1},1),size(c,2)]);
         Y = [c{1,1:size(c,2)}];
      else
         X = squeeze(X)';
         Y = c;
      end
        if isempty(dzdy) %forward
            
            %residuals
            err=abs(Y-X);
            
            %scale back to pixels
            funScale = @(A,B) A.*(B);
            err = bsxfun(funScale,err,scbox);
            Y=[];
            Y = sum(err)./size(X,1);%error per samples
            Y = sum(Y);%summed batch error
            
        else %nothing to backprop
            Y = zerosLike(X_orig) ;
        end
        
   otherwise
        error('Unknown parameter ''%s''.', opts.loss) ;
end

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
    y = gpuArray.zeros(size(x),'single') ;
else
    y = zeros(size(x),'single') ;
end
