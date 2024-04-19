% Takes the metrics from each iteration (stored in the metrics.mat file
% produced by write_metrics_to_excel.m and calculates average and store it
% in metrics.mat

clear

load('/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/metrics.mat')

n_iterations = 3; % 1, 2, or 3
datasets_list = ["kornum", "spindle"]; % "kornum" or "spindle"
cohorts_list = ["a", "d"]; % 'a' or 'd'
n_scorers_spindle = 2; % 1 or 2
seq_len_list = [11, 21, 31, 41, 61, 81, 101, 157, 225];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % folders = dir(results_path);
upper_left_corners_column.kornum.acc = 'D';
upper_left_corners_column.kornum.bal_acc = 'E';
upper_left_corners_column.kornum.kappa = 'F';
upper_left_corners_column.kornum.sensitivity = 'H';
upper_left_corners_column.kornum.precision = 'I';
upper_left_corners_column.kornum.fscore = 'J';

upper_left_corners_column.spindle.cohort_A.acc = 'L';
upper_left_corners_column.spindle.cohort_A.bal_acc = 'M';
upper_left_corners_column.spindle.cohort_A.kappa = 'N';
upper_left_corners_column.spindle.cohort_A.sensitivity = 'P';
upper_left_corners_column.spindle.cohort_A.precision = 'Q';
upper_left_corners_column.spindle.cohort_A.fscore = 'R';

upper_left_corners_column.spindle.cohort_D.acc = 'T';
upper_left_corners_column.spindle.cohort_D.bal_acc = 'U';
upper_left_corners_column.spindle.cohort_D.kappa = 'V';
upper_left_corners_column.spindle.cohort_D.sensitivity = 'X';
upper_left_corners_column.spindle.cohort_D.precision = 'Y';
upper_left_corners_column.spindle.cohort_D.fscore = 'Z';
row = 4;

for s = 1:length(seq_len_list)
    seq_len = seq_len_list(s);

    for d = 1:length(datasets_list)
        dataset = datasets_list(d);

        if strcmp(dataset, "kornum")
            acc_ = 0;
            bal_acc_ = 0;
            sensitivity_ = 0;
            precision_ = 0;
            fscore_ = 0;
            kappa_ = 0;

            for iteration = 1:n_iterations
                acc_ = acc_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).acc;
                bal_acc_ = bal_acc_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).bal_acc;
                sensitivity_ = sensitivity_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).sensitivity;
                precision_ = precision_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).precision;
                fscore_ = fscore_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).fscore;
                kappa_ = kappa_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', num2str(iteration)]).kappa;
            end

            mean_acc = acc_ ./ 3;
            mean_bal_acc = bal_acc_ ./ 3;
            mean_sensitivity = sensitivity_ ./ 3;
            mean_precision = precision_ ./ 3;
            mean_fscore = fscore_ ./ 3;
            mean_kappa = kappa_ ./ 3;
            
            writematrix(mean_acc, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.kornum.acc, num2str(row)], 'Sheet', 4)
            writematrix(mean_bal_acc, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.kornum.bal_acc, num2str(row)], 'Sheet', 4)
            writematrix(mean_sensitivity, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.kornum.sensitivity, num2str(row+1)], 'Sheet', 4)
            writematrix(mean_precision, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.kornum.precision, num2str(row+1)], 'Sheet', 4)
            writematrix(mean_fscore, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.kornum.fscore, num2str(row+1)], 'Sheet', 4)
            writematrix(mean_kappa, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.kornum.kappa, num2str(row)], 'Sheet', 4)
            
            metrics.(['seq_len_', int2str(seq_len)]).(dataset).iterations_avg.acc = mean_acc;
            metrics.(['seq_len_', int2str(seq_len)]).(dataset).iterations_avg.bal_acc = mean_bal_acc;
            metrics.(['seq_len_', int2str(seq_len)]).(dataset).iterations_avg.sensitivity = mean_sensitivity;
            metrics.(['seq_len_', int2str(seq_len)]).(dataset).iterations_avg.precision = mean_precision;
            metrics.(['seq_len_', int2str(seq_len)]).(dataset).iterations_avg.fscore = mean_fscore;
            metrics.(['seq_len_', int2str(seq_len)]).(dataset).iterations_avg.kappa = mean_kappa;
        elseif strcmp(dataset, "spindle")

            for c = 1:length(cohorts_list)
                cohort = cohorts_list(c);

                acc_ = 0;
                bal_acc_ = 0;
                sensitivity_ = 0;
                precision_ = 0;
                fscore_ = 0;
                kappa_ = 0;

                for scorer = 1:n_scorers_spindle

                    for iteration = 1:n_iterations
                        acc_ = acc_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).acc;
                        bal_acc_ = bal_acc_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).bal_acc;
                        sensitivity_ = sensitivity_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).sensitivity;
                        precision_ = precision_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).precision;
                        fscore_ = fscore_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).fscore;
                        kappa_ = kappa_ + metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).kappa;
                    end
                end

                mean_acc = acc_ ./ 6;
                mean_bal_acc = bal_acc_ ./ 6;
                mean_sensitivity = sensitivity_ ./ 6;
                mean_precision = precision_ ./ 6;
                mean_fscore = fscore_ ./ 6;
                mean_kappa = kappa_ ./ 6;

                writematrix(mean_acc, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.spindle.(['cohort_', upper(char(cohort))]).acc, num2str(row)], 'Sheet', 4)
                writematrix(mean_bal_acc, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.spindle.(['cohort_', upper(char(cohort))]).bal_acc, num2str(row)], 'Sheet', 4)
                writematrix(mean_sensitivity, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.spindle.(['cohort_', upper(char(cohort))]).sensitivity, num2str(row+1)], 'Sheet', 4)
                writematrix(mean_precision, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.spindle.(['cohort_', upper(char(cohort))]).precision, num2str(row+1)], 'Sheet', 4)
                writematrix(mean_fscore, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.spindle.(['cohort_', upper(char(cohort))]).fscore, num2str(row+1)], 'Sheet', 4)
                writematrix(mean_kappa, '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/sleeptransformer.xlsx', 'Range', [upper_left_corners_column.spindle.(['cohort_', upper(char(cohort))]).kappa, num2str(row)], 'Sheet', 4)
                
                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.acc = mean_acc;
                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.bal_acc = mean_bal_acc;
                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.sensitivity = mean_sensitivity;
                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.precision = mean_precision;
                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.fscore = mean_fscore;
                metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.kappa = mean_kappa;
            end
        end
    end
    row = row + 6;
end
save('/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/metrics.mat',"metrics")