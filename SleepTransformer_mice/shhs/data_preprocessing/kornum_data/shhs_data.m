clear all
close all
clc

addpath('C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\edf_reader');

data_kornum_path = 'C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum';
mat_path = './mat/';
 
if(~exist(mat_path,'dir'))
    mkdir(mat_path);
end

dirlist = dir([data_kornum_path, '/**/*.edf']);
dirlist = dirlist(arrayfun(@(x) ~strcmp(x.name(1),'.'),dirlist)); % remove hidden files
N = numel(dirlist);

for n = 1 : N
    filepath = [dirlist(n).folder, '/', dirlist(n).name];
    disp(filepath);
    process_and_save_1file_v1(filepath, n, mat_path);
end