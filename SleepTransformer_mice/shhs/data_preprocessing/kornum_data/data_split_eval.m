clear all
close all
clc


rng(10); % for repeatable

% divide subjects into training, evaluation, and test sets for consistency
% between various networks

Nsub = 55;

subjects = randperm(Nsub);

test_sub = [11, 14, 19, 22, 28, 35, 39, 43, 47, 50, 51];
rest = setdiff(subjects, test_sub);
perm_list = randperm(numel(rest));

val = 0.12; % 12% of the training set = 5 subjects

eval_sub = sort(rest(perm_list(1 : round(val*length(perm_list)))));
train_sub = sort(setdiff(rest, eval_sub));

% % 50 subjects as eval set
% train_check_sub = sort(rest(perm_list(101:200)));
% train_sub = sort(rest(perm_list(101:end)));

% save('./data_split_eval.mat', 'train_sub','test_sub','eval_sub','train_check_sub');
save('./data_split_eval.mat', 'train_sub','test_sub','eval_sub');