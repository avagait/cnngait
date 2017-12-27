function [OF, factor] = mj_decompressOF(OF_compress, factor)
% [OF_compress, factor] = mj_decompressOF(OF_compress, factor)
% Converts data to single precission by multipliying it by a constant factor
% (e.g. 1/1000)
% 
% Nothing is done if class(OF_compress) == 'single' or class(OF_compress) == 'double'
%
% See also mj_compressOF
%
% (c) MJMJ/2011

if ~exist('factor', 'var')
   factor = 1.0/1000;
end

switch class(OF_compress)
   case {'double', 'single'}
      OF = OF_compress;
   otherwise
      OF = single(OF_compress) * factor;
end
