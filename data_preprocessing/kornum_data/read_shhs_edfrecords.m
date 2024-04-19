function [signal, fs] = read_shhs_edfrecords(filename, channel_name, channel_index)
    [header, edf] = edfread(filename,'targetSignals_type',channel_name,'targetSignals_index',channel_index);
    assert(length(header.samples) == numel(channel_index));
    
    fs = header.frequency(channel_index{1});
    assert(length(fs) == 1);
    
%     if(length(channel_index) > 1)
%         signal = diff(edf)';
%     else
    signal = edf';
%     end
end

