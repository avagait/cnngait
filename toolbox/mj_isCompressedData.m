function ic = mj_isCompressedData(data)
% ic = mj_isCompressedData(data)
% Checks whether the data is compressed (i.e. integer)
%
%
% Output:
%  - ic: boolean
%
% (c) MJMJ/2015

%ic = strcmp(class(data), 'int16');
ic = isa(data, 'integer');

