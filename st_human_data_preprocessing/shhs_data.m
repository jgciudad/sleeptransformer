clear all
close all
clc

addpath('edf_reader');

xml_path = '/scratch/s202283/data/shhs/polysomnography/annotations-events-nsrr/shhs1/';
edf_path = '/scratch/s202283/data/shhs/polysomnography/edfs/shhs1/';
mat_path = '/scratch/s202283/data/mat_human_sleeptransformer_5_classes/';
fs = 128;
n_classes = 5;

if(~exist(mat_path,'dir'))
    mkdir(mat_path);
end

dirlist = dir([edf_path, '*.edf']);
N = numel(dirlist);

counter = 1;
tic
parfor n = 1 : N
    filename = dirlist(n).name;
    disp(filename);
    [~, counter] = process_and_save_1file(filename, n, xml_path, edf_path, mat_path, fs, n_classes, counter);
end
toc