% Plots the metrics stored in the file metrics.mat. In SleepTransformer,
% metrics.mat is produced by write_metrics_to_excel.m and
% aggregate_over_iterations.m . In L-SeqSleepNet, metrics.mat is produced
% by evalute.py.

clear
close all

n_iterations = 3; % 1, 2, or 3
datasets_list = ["kornum", "spindle"]; % "kornum" or "spindle"
cohorts_list = ["a", "d"]; % 'a' or 'd'
n_scorers_spindle = 2; % 1 or 2
seq_len_list = [11, 21, 31, 41, 61, 81, 101, 157, 225];
metrics_list = ["acc", "bal_acc", "kappa", "sensitivity", "precision", "fscore"];
seqsleepnet = 1;

acc_color = "#0072BD";
bal_acc_color = "#D95319";
w_color = "#EDB120";
n_color = "#77AC30";
r_color = "#A2142F";


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if seqsleepnet == 1
    load('/Users/tlj258/Documents/Code/l-seqsleepnet/metric_lines.mat')
    seq_len_list = [10, 20, 30, 40, 60, 80, 100, 160, 220];
    lines.kornum = kornum;
    lines.spindle = spindle;
else
    load('/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/Results/new_results/metrics.mat')

    lines.kornum.acc = zeros(n_iterations, length(seq_len_list));
    lines.kornum.bal_acc = zeros(n_iterations, length(seq_len_list));
    lines.kornum.kappa = zeros(n_iterations, length(seq_len_list));
    lines.kornum.sensitivity = zeros(n_iterations, 3, length(seq_len_list));
    lines.kornum.precision = zeros(n_iterations, 3, length(seq_len_list));
    lines.kornum.fscore = zeros(n_iterations, 3, length(seq_len_list));
    
    lines.kornum.avg_acc = zeros(1, length(seq_len_list));
    lines.kornum.avg_bal_acc = zeros(1, length(seq_len_list));
    lines.kornum.avg_kappa = zeros(1, length(seq_len_list));
    lines.kornum.avg_sensitivity = zeros(3, length(seq_len_list));
    lines.kornum.avg_precision = zeros(3, length(seq_len_list));
    lines.kornum.avg_fscore = zeros(3, length(seq_len_list));
    
    lines.spindle.cohort_A.acc = zeros(n_iterations, n_scorers_spindle, length(seq_len_list));
    lines.spindle.cohort_A.bal_acc = zeros(n_iterations, n_scorers_spindle, length(seq_len_list));
    lines.spindle.cohort_A.kappa = zeros(n_iterations, n_scorers_spindle, length(seq_len_list));
    lines.spindle.cohort_A.sensitivity = zeros(n_iterations, n_scorers_spindle, 3, length(seq_len_list));
    lines.spindle.cohort_A.precision = zeros(n_iterations, n_scorers_spindle, 3, length(seq_len_list));
    lines.spindle.cohort_A.fscore = zeros(n_iterations, n_scorers_spindle, 3, length(seq_len_list));
    
    lines.spindle.cohort_A.avg_acc = zeros(1, length(seq_len_list));
    lines.spindle.cohort_A.avg_bal_acc = zeros(1, length(seq_len_list));
    lines.spindle.cohort_A.avg_kappa = zeros(1, length(seq_len_list));
    lines.spindle.cohort_A.avg_sensitivity = zeros(3, length(seq_len_list));
    lines.spindle.cohort_A.avg_precision = zeros(3, length(seq_len_list));
    lines.spindle.cohort_A.avg_fscore = zeros(3, length(seq_len_list));
    
    lines.spindle.cohort_D.acc = zeros(n_iterations, n_scorers_spindle, length(seq_len_list));
    lines.spindle.cohort_D.bal_acc = zeros(n_iterations, n_scorers_spindle, length(seq_len_list));
    lines.spindle.cohort_D.kappa = zeros(n_iterations, n_scorers_spindle, length(seq_len_list));
    lines.spindle.cohort_D.sensitivity = zeros(n_iterations, n_scorers_spindle, 3, length(seq_len_list));
    lines.spindle.cohort_D.precision = zeros(n_iterations, n_scorers_spindle, 3, length(seq_len_list));
    lines.spindle.cohort_D.fscore = zeros(n_iterations, n_scorers_spindle, 3, length(seq_len_list));
    
    lines.spindle.cohort_D.avg_acc = zeros(1, length(seq_len_list));
    lines.spindle.cohort_D.avg_bal_acc = zeros(1, length(seq_len_list));
    lines.spindle.cohort_D.avg_kappa = zeros(1, length(seq_len_list));
    lines.spindle.cohort_D.avg_sensitivity = zeros(3, length(seq_len_list));
    lines.spindle.cohort_D.avg_precision = zeros(3, length(seq_len_list));
    lines.spindle.cohort_D.avg_fscore = zeros(3, length(seq_len_list));
    
    
    
    
    %%
    dataset = 'kornum';
    
    
    for iteration = 1:(n_iterations+1)
    
        if iteration ~= 4
            it_ = num2str(iteration);
    
            for s = 1:length(seq_len_list)
                seq_len = seq_len_list(s);
        
                lines.(dataset).acc(iteration, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).acc;
                lines.(dataset).bal_acc(iteration, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).bal_acc;
                lines.(dataset).kappa(iteration, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).kappa;
                lines.(dataset).sensitivity(iteration, :, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).sensitivity;
                lines.(dataset).precision(iteration, :, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).precision;
                lines.(dataset).fscore(iteration, :, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).fscore;
    
            end
                
    
        else 
            it_ = 's_avg';
    
            for s = 1:length(seq_len_list)
                seq_len = seq_len_list(s);
    
                lines.(dataset).avg_acc(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).acc;
                lines.(dataset).avg_bal_acc(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).bal_acc;
                lines.(dataset).avg_kappa(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).kappa;
                lines.(dataset).avg_sensitivity(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).sensitivity;
                lines.(dataset).avg_precision(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).precision;
                lines.(dataset).avg_fscore(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['iteration', it_]).fscore;
    
            end
    
        end
    end
    
    
    %%
    
    dataset = 'spindle';
    
    for c = 1:length(cohorts_list)
        cohort = cohorts_list(c);
    
        for iteration = 1:(n_iterations+1)
        
            if iteration ~= 4
        
                for scorer = 1:n_scorers_spindle
            
                    for s = 1:length(seq_len_list)
                        seq_len = seq_len_list(s);
                
                        lines.(dataset).(['cohort_', upper(char(cohort))]).acc(iteration, scorer, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).acc;
                        lines.(dataset).(['cohort_', upper(char(cohort))]).bal_acc(iteration, scorer, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).bal_acc;
                        lines.(dataset).(['cohort_', upper(char(cohort))]).kappa(iteration, scorer, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).kappa;
                        lines.(dataset).(['cohort_', upper(char(cohort))]).sensitivity(iteration, scorer, :, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).sensitivity;
                        lines.(dataset).(['cohort_', upper(char(cohort))]).precision(iteration, scorer, :, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).precision;
                        lines.(dataset).(['cohort_', upper(char(cohort))]).fscore(iteration, scorer, :, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).(['scorer_', int2str(scorer)]).(['iteration', num2str(iteration)]).fscore;
                
                    end
        
                end
        
        
            else
                for s = 1:length(seq_len_list)
                    seq_len = seq_len_list(s);
                    lines.(dataset).(['cohort_', upper(char(cohort))]).avg_acc(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.acc;
                    lines.(dataset).(['cohort_', upper(char(cohort))]).avg_bal_acc(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.bal_acc;
                    lines.(dataset).(['cohort_', upper(char(cohort))]).avg_kappa(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.kappa;
                    lines.(dataset).(['cohort_', upper(char(cohort))]).avg_sensitivity(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.sensitivity;
                    lines.(dataset).(['cohort_', upper(char(cohort))]).avg_precision(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.precision;
                    lines.(dataset).(['cohort_', upper(char(cohort))]).avg_fscore(:, s) = metrics.(['seq_len_', int2str(seq_len)]).(dataset).(['cohort_', upper(char(cohort))]).iterations_avg.fscore;
                end
        
            end
        
        
        end
    end
end

%% kornum

dataset = 'kornum';

for m = 1:6
    metric = char(metrics_list(m));


    if m==1
       
        figure()


        plot(seq_len_list, lines.(dataset).acc, 'x', 'MarkerSize', 7, 'Color', acc_color)
        hold on
        acc_line = plot(seq_len_list, lines.(dataset).avg_acc, '.-', 'MarkerSize', 20, 'LineWidth', 4, 'Color', acc_color);
        hold on
        title([upper(dataset), ' ', metric], 'Interpreter', 'none')
        hold on
        plot(seq_len_list, lines.(dataset).bal_acc, 'x', 'MarkerSize', 7, 'Color', bal_acc_color)
        hold on
        bal_acc_line = plot(seq_len_list, lines.(dataset).avg_bal_acc, '.-', 'MarkerSize', 20, 'LineWidth', 4, 'Color', bal_acc_color);
        ylim([0.9, 1])
        legend([acc_line, bal_acc_line], 'Average acc.', 'Average bal_acc.', 'Interpreter', 'none');
        xticks(seq_len_list)
        xlabel('Input sequence length')

    elseif m==3

        % figure(figures.(dataset).(metric))
        figure()
        plot(seq_len_list, lines.(dataset).(metric), 'x', 'MarkerSize', 7, 'Color', acc_color)
        hold on
        plot(seq_len_list, lines.(dataset).(['avg_', metric]), '.-', 'MarkerSize', 20, 'LineWidth', 4, 'Color', acc_color)
        title([upper(dataset), ' ', metric], 'Interpreter', 'none')
        ylim([0.88, 1])
        xticks(seq_len_list)
        xlabel('Input sequence length')

    elseif m>3

        figure()
        hold on
        w = plot(seq_len_list, lines.(dataset).(['avg_', metric])(1,:), '.-', 'MarkerSize', 20, 'LineWidth', 4, 'Color', w_color);
        n = plot(seq_len_list, lines.(dataset).(['avg_', metric])(2,:), '.-', 'MarkerSize', 20, 'LineWidth', 4, 'Color', n_color);
        r = plot(seq_len_list, lines.(dataset).(['avg_', metric])(3,:), '.-', 'MarkerSize', 20, 'LineWidth', 4, 'Color', r_color);
        plot(seq_len_list, squeeze(lines.(dataset).(metric)(:,1,:)), 'x', 'MarkerSize', 7, 'Color', w_color)
        plot(seq_len_list, squeeze(lines.(dataset).(metric)(:,2,:)), 'x', 'MarkerSize', 7, 'Color', n_color)
        plot(seq_len_list, squeeze(lines.(dataset).(metric)(:,3,:)), 'x', 'MarkerSize', 7, 'Color', r_color)
        xticks(seq_len_list)
        ylim([0.7, 1])
        legend([w, n, r], 'WAKE', 'NREM', 'REM');
        xlabel('Input sequence length')

        title([upper(dataset), ' ', metric])
        ylim([0.7, 1])

    end
end

%% spindle

dataset = 'spindle';

for c = 1:length(cohorts_list)
    cohort = char(cohorts_list(c));

    
    for m = 1:6
        metric = char(metrics_list(m));
        
        if m==1
            figure()
    
            hold on
            s1 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).acc(:,1,:)), 'sg', 'MarkerSize', 7, 'Color', acc_color);
            s2 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).acc(:,2,:)), 'dr', 'MarkerSize', 7, 'Color', acc_color);
            s11 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).bal_acc(:,1,:)), 'sg', 'MarkerSize', 7, 'Color', bal_acc_color);
            s22 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).bal_acc(:,2,:)), 'dr', 'MarkerSize', 7, 'Color', bal_acc_color);
            avg = plot(seq_len_list, lines.(dataset).(['cohort_', upper(cohort)]).avg_acc, '-b.', 'MarkerSize', 20, 'LineWidth', 4, 'Color', acc_color);
            avg2 = plot(seq_len_list, lines.(dataset).(['cohort_', upper(cohort)]).avg_bal_acc, '-b.', 'MarkerSize', 20, 'LineWidth', 4, 'Color', bal_acc_color);
            title([upper(dataset), ' cohort ', upper(cohort), ' ', metric], 'Interpreter', 'none')
            ylim([0.9, 1])
            legend([s1(1), s2(1), avg, avg2], 'Scorer 1', 'Scorer 2', 'Average acc.', 'Average bal_acc.', 'Interpreter', 'none');
            xticks(seq_len_list)
            xlabel('Input sequence length')
        
        elseif m==3
            
            figure()
            hold on
            s1 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).(metric)(:,1,:)), 'sg', 'MarkerSize', 7, 'Color', acc_color);
            s2 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).(metric)(:,2,:)), 'dr', 'MarkerSize', 7, 'Color', acc_color);
            avg = plot(seq_len_list, lines.(dataset).(['cohort_', upper(cohort)]).(['avg_', metric]), '-b.', 'MarkerSize', 20, 'LineWidth', 4, 'Color', acc_color);
            title([upper(dataset), ' cohort ', upper(cohort), ' ', metric], 'Interpreter', 'none')
            ylim([0.88, 1])
            legend([s1(1), s2(1), avg], 'Scorer 1', 'Scorer 2', 'Average');
            xticks(seq_len_list)
            xlabel('Input sequence length')


        elseif m>3
            
            figure()
            hold on

            w_s1 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).(metric)(:,1,1,:)), 's', 'MarkerSize', 7, 'Color', w_color);
            w_s2 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).(metric)(:,2,1,:)), 'd', 'MarkerSize', 7, 'Color', w_color);

            n_s1 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).(metric)(:,1,2,:)), 's', 'MarkerSize', 7, 'Color', n_color);
            n_s2 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).(metric)(:,2,2,:)), 'd', 'MarkerSize', 7, 'Color', n_color);

            r_s1 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).(metric)(:,1,3,:)), 's', 'MarkerSize', 7, 'Color', r_color);
            r_s2 = plot(seq_len_list, squeeze(lines.(dataset).(['cohort_', upper(cohort)]).(metric)(:,2,3,:)), 'd', 'MarkerSize', 7, 'Color', r_color);

            w_avg = plot(seq_len_list, lines.(dataset).(['cohort_', upper(cohort)]).(['avg_', metric])(1,:), '.-', 'MarkerSize', 20, 'LineWidth', 4, 'Color', w_color);
            n_avg = plot(seq_len_list, lines.(dataset).(['cohort_', upper(cohort)]).(['avg_', metric])(2,:), '.-', 'MarkerSize', 20, 'LineWidth', 4, 'Color', n_color);
            r_avg = plot(seq_len_list, lines.(dataset).(['cohort_', upper(cohort)]).(['avg_', metric])(3,:), '.-', 'MarkerSize', 20, 'LineWidth', 4, 'Color', r_color);

            legend([w_avg, n_avg, r_avg], 'WAKE', 'NREM', 'REM');
            xticks([11, 21, 31, 41, 61, 81, 101, 157, 225])
            ylim([0.7, 1])
            xlabel('Input sequence length')

    
            title([upper(dataset), ' cohort ', upper(cohort), ' ', metric])
            % ylim([0.9, 1])
    
        end
    end
end
    
                
    
    
                
