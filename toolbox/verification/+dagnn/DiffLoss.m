% Difference loss
%
% (c) MJMJ/2016
classdef DiffLoss < dagnn.ElementWise
    properties
    loss = 'hadsellloss' %default loss
    thr = 0.50           % Threshold
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end
    
  methods
    function outputs = forward(obj, inputs, params)
      
      outputs{1} = mj_nndiffloss(inputs{1}, inputs{2}, inputs{3}, [], 'loss', obj.loss, 'thr', obj.thr);
      n = obj.numAveraged;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
      
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
       
      [derInputs{1}, derInputs{2}] = mj_nndiffloss(inputs{1}, inputs{2}, inputs{3}, derOutputs{1}, 'loss', obj.loss, 'thr', obj.thr);
      %derInputs{2} = [] ;
      derInputs{3} = [] ;
      derParams = {} ;
      
      % Update threshold
      X = squeeze(inputs{1});
      Y = squeeze(inputs{2});
      res=(Y-X);
      D = sqrt(sum(res.^2,1));
      vals = []; 
      dfmnmx = max(D)-min(D);
      stepthr = dfmnmx / (length(inputs{3})-1); % Adapt step
      for thr_ = min(D):stepthr:max(D) 
         estimPos = D <= thr_; 
         estimNeg = D > thr_; 
         TP = sum(inputs{3}(estimPos) == 1) / sum(estimPos); 
         TN = sum(inputs{3}(estimNeg) == -1) / sum(estimNeg); 
         vals = [vals; thr_, TP, TN]; 
      end
      [minval, minpos] = min(abs(vals(:,2)-vals(:,3)));
      obj.thr = vals(minpos,1);
    end
    
     function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(1)] ;
    end
    
    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

       
    function move(obj, device)
switch device
  case 'gpu'
    obj.thr = gpuArray(obj.thr);
  case 'cpu'
    obj.thr = gather(obj.thr);
  otherwise
    error('DEVICE must be either ''cpu'' or ''gpu''.') ;
end
   end

    function obj = DiffLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
