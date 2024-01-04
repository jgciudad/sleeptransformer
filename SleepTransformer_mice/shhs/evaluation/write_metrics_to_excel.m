% Writes the results from all iterations into the corresponding excel
% sheet, into the corresponding tab, and saves them to the metrics.mat file
% The average over iterations is calculated in aggregate_over_iterations.m

% results_path = '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/outputs';
clear

addpath '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/Code/HUMMUSS/SleepTransformer_mice/shhs/metrics'

n_iterations = 3; % 1, 2, or 3
datasets_list = ["kornum", "spindle"]; % "kornum" or "spindle"
cohorts_list = ["a", "d"]; % 'a' or 'd'
n_scorers_spindle = 2; % 1 or 2
seq_len_list = [11, 21, 31, 41, 61, 81, 101, 157, 225];

config.filter_out_artifacts = 1;
config.nclass_model = 3;
% nchan=1;
config.Nfold = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % folders = dir(results_path);
upper_left_corners_column.best_model = 'C';
upper_left_corners_column.kornum = 'I';
upper_left_corners_column.spindle.cohort_a.scorer_1 = 'T';
upper_left_corners_column.spindle.cohort_a.scorer_2 = 'AE';
upper_left_corners_column.spindle.cohort_d.scorer_1 = 'AP';
upper_left_corners_column.spindle.cohort_d.scorer_2 = 'BA';
upper_left_corners_column.kornum_kappa = 'F';
upper_left_corners_column.spindle.cohort_a.scorer_1_kappa = 'Q';
upper_left_corners_column.spindle.cohort_a.scorer_2_kappa = 'AB';
upper_left_corners_column.spindle.cohort_d.scorer_1_kappa = 'AM';
upper_left_corners_column.spindle.cohort_d.scorer_2_kappa = 'AX';
row = 5;

for s = 1:length(seq_len_list)
    seq_len = seq_len_list(s);

    for d = 1:length(datasets_list)
        dataset = datasets_list(d);

        if strcmp(dataset, "kornum")
            mat_path = '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/THESIS_DATA/SleepTransformer_mice/kornum_data/mat/';
            data_split_path = ['/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/Code/HUMMUSS/SleepTransformer_mice/shhs/data_preprocessing/', char(dataset), '_data/data_split_eval.mat'];
            column = upper_left_corners_column.(dataset);
            column_kappa = upper_left_corners_column.([char(dataset), '_kappa']);

            for iteration = 1:n_iterations

                test_ret_path = ['/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/outputs/sleeptransformer/weighted_ce', '/iteration', num2str(iteration), '/seq_len_', int2str(seq_len), '/testing/kornum', '/test_ret.mat'];
            
                out_file_path = ['/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/outputs/sleeptransformer/weighted_ce', '/iteration', num2str(iteration), '/seq_len_', int2str(seq_len), '/it', num2str(iteration), '_', int2str(seq_len), '.out'];
                write_best_model_update(out_file_path, upper_left_corners_column, row, iteration)
                [acc, bal_acc, mysensitivity, myprecision, fscore, kappa] = evaluate_model_and_write(data_split_path, test_ret_path, mat_path, seq_len, column, row, iteration, config, column_kappa);

                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).acc = acc;
                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).bal_acc = bal_acc;
                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).sensitivity = mysensitivity;
                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).precision = myprecision;
                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).fscore = fscore;
                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).kappa = kappa;
            end
        elseif strcmp(dataset, "spindle")

            for c = 1:length(cohorts_list)
                cohort = cohorts_list(c);

                for scorer = 1:n_scorers_spindle
                    mat_path = ['/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/THESIS_DATA/SleepTransformer_mice/spindle_data/', 'cohort_', upper(char(cohort)), '/scorer_', int2str(scorer), '/'];
                    data_split_path = ['/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/Code/HUMMUSS/SleepTransformer_mice/shhs/data_preprocessing/', char(dataset), '_data/', 'data_split_eval_cohort_', upper(char(cohort)), '.mat'];
                    column = upper_left_corners_column.(dataset).(['cohort_', char(cohort)]).(['scorer_', int2str(scorer)]);
                    column_kappa = upper_left_corners_column.(dataset).(['cohort_', char(cohort)]).(['scorer_', int2str(scorer), '_kappa']);

                    for iteration = 1:n_iterations

                        test_ret_path = ['/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/outputs/sleeptransformer/weighted_ce/', 'iteration', num2str(iteration), '/seq_len_', int2str(seq_len), '/testing/spindle/cohort_', upper(char(cohort)), '/test_ret.mat'];
                        [acc, bal_acc, mysensitivity, myprecision, fscore, kappa] = evaluate_model_and_write(data_split_path, test_ret_path, mat_path, seq_len, column, row, iteration, config, column_kappa);
                    
                        metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).acc = acc;
                        metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).bal_acc = bal_acc;
                        metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).sensitivity = mysensitivity;
                        metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).precision = myprecision;
                        metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).fscore = fscore;
                        metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).kappa = kappa;
                    end
                end
            end
        end
    end
    row = row + 7;
end
save('/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/metrics.mat',"metrics")

function [acc, bal_acc, mysensitivity, myprecision, fscore, kappa] = evaluate_model_and_write(data_split_path, test_ret_path, mat_path, seq_len, column, row, iteration, config, column_kappa)
    load(data_split_path)
    load(test_ret_path)
    listing = dir([mat_path, '*_eeg1.mat']);

    yh = cell(config.Nfold,1);
    yt = cell(config.Nfold,1);
       
    for fold = 1 : config.Nfold
        % fold
        test_s = test_sub;
        sample_size = zeros(numel(test_s), 1);
        for i = 1 : numel(test_s)
            sname = listing(test_s(i)).name;
            load([mat_path,sname], 'label');
            sample_size(i) = numel(label) -  (seq_len - 1); 
            yt{fold} = [yt{fold}; double(label)];
        end
        
        score_ = cell(1,seq_len);
        for n = 1 : seq_len
            score_{n} = softmax_own(squeeze(score(:,n,:)));
        end
        score = score_;
        clear score_;

        for i = 1 : numel(test_s)
            start_pos = sum(sample_size(1:i-1)) + 1;
            end_pos = sum(sample_size(1:i-1)) + sample_size(i);
            score_i = cell(1,seq_len);
            for n = 1 : seq_len
                score_i{n} = score{n}(start_pos:end_pos, :);
                % N = size(score_i{n},1);

                score_i{n} = [ones(seq_len-1,config.nclass_model); score{n}(start_pos:end_pos, :)];
                score_i{n} = circshift(score_i{n}, -(seq_len - n), 1);
            end

            smoothing = 0;
            fused_score = log(score_i{1});
            for n = 2 : seq_len
                if(smoothing == 0)
                    fused_score = fused_score + log(score_i{n});
                else
                    fused_score = fused_score + score_i{n};
                end
            end

            yhat = zeros(1,size(fused_score,1));
            for k = 1 : size(fused_score,1)
                [~, yhat(k)] = max(fused_score(k,:));
            end

            yh{fold} = [yh{fold}; double(yhat')];
        end
    end
    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    if config.filter_out_artifacts == 1
        yh = yh(yt~=4); % filter out artifacts
        yt = yt(yt~=4); % filter out artifacts
    end

    C = confusionmat(yt, yh);
    acc = sum(yh == yt)/numel(yt);

    [mysensitivity, myprecision]  = calculate_sensitivity_selectivity(yt, yh); % THEY MADE A MISTAKE, THIS IS NOT SELECTIVITY, IS PRECISION!!! (is actually good, because is what I want)
    bal_acc = mean(mysensitivity);
    [fscore, ~, ~] = litis_class_wise_f1(yt, yh);
    kappa = kappaindex(yh,yt,config.nclass_model);

    writematrix(C, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [column, num2str(row)], 'Sheet', iteration)
    writematrix(kappa, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [column_kappa, num2str(row-1)], 'Sheet', iteration)

    

end

function write_best_model_update(out_file_path, upper_left_corners_column, row, iteration)
    out_file = importdata(out_file_path);
    index = find(contains(out_file.textdata,'Best model updated'));
    index = index(end) - 2;
    best_line = string(out_file.textdata(index));
    best_line_split = split(best_line, {',', ':'});
    step_index = find(contains(best_line_split,'step'));
    best_step = char(best_line_split(step_index));
    best_step = str2num(best_step(7:end));

    writematrix(best_step, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.best_model, num2str(row-1)], 'Sheet', iteration)
end