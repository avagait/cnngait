classdef MTaskLoss < dagnn.ElementWise
   % Multi-task loss layer
   %
   % 'inputs': cell-array {'loss01', 'weight01', 'loss02', 'weight02',...}
   %
   % (c) MJMJ/2016
   
    properties
    ws = 1.0           % Weights
    learnW = false     % Are weights trainable?
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end
    
  methods
    function outputs = forward(obj, inputs, params)
      % Prepare input
      L = [inputs{1:2:end}];
      W = ones(1,length(L), class(inputs{end}));
      % Fill with non-empty weights
      for i = 2:2:length(inputs)
         if ~isempty(inputs{i})
            W(i/2) = inputs{i};
         end
      end       
      outputs{1} = mj_nnwsum(L, W, [], 'learnW', obj.learnW);
      
      n = obj.numAveraged;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
      
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      % Prepare input
      L = [inputs{1:2:end}];
      W = ones(1,length(L), class(inputs{end}));
      % Fill with non-empty weights
      for i = 2:2:length(inputs)
         if ~isempty(inputs{i})
            W(i/2) = inputs{i};
         end
      end
      [dL, dW] = mj_nnwsum(L, W, derOutputs{1}, 'learnW', obj.learnW);
      
      % Prepare output
%       derInputs{2} = [] ;
%       if ~obj.learnW
%          derInputs{2} = derParams;
%          derParams = [];         
%       end
      derInputs = cell(1,length(dL)*2);
      for i = 1:length(dL)
         derInputs{1+(i-1)*2} = dL(i);
         derInputs{i*2} = dW(i);
      end
      derParams = [];
    end
    
     function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(end)] ;
    end
    
    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = MTaskLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
