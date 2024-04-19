clear all
close all
clc

% addpath('C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\edf_reader');

spindle_path = 'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA';
scorer=2;
mat_path = ['./mat/', 'scorer_', num2str(scorer), '/'];

if(~exist(mat_path,'dir'))
    mkdir(mat_path);
end

dirlist = dir([spindle_path, '/**/*.edf']);
% dirlist = dirlist(arrayfun(@(x) ~strcmp(x.name(1),'.'),dirlist)); % remove hidden files
N = numel(dirlist);

for n = 1 : N
    filepath = [dirlist(n).folder, '/', dirlist(n).name];
    disp(filepath);
    process_and_save_1file_v1(filepath, n, mat_path, scorer);
end