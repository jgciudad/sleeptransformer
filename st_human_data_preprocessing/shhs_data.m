clear all
close all
clc

addpath('edf_reader');

xml_path = '/scratch/s202283/data/shhs_prepro_prueba/annotations/';
edf_path = '/scratch/s202283/data/shhs_prepro_prueba/edf/';
mat_path = '/scratch/s202283/data/mat_prepro_prueba/';
fs = 128;
n_classes = 5;

if(~exist(mat_path,'dir'))
    mkdir(mat_path);
end

dirlist = dir([edf_path, '*.edf']);
N = numel(dirlist);
disp('Number of files:')
disp(N)

parfor n = 1 : N
% for n = 1 : N
    filename = dirlist(n).name;
    disp(filename);
    process_and_save_1file(filename, n, xml_path, edf_path, mat_path, fs, n_classes);
end