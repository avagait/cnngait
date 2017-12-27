function fc = mj_compactFilter4Draw(filter)
% fc = mj_compactFilter4Draw(filter)
% COMMENT ME!!!
% Given a multichannel filter, returns the composition of the channels
%
% Output:
%  - fc: matrix with channels pasted side by side
%
% (c) MJMJ/2015

[nr, nc, nchannels] = size(filter);

fc = zeros(nr, nc * nchannels, class(filter) );

for chix = 1:nchannels
   fc(:,1+(chix-1)*nc:chix*nc) = filter(:,:,chix);
end % chix
