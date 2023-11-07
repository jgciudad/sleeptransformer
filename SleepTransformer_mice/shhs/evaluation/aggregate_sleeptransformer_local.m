function collection = aggregate_sleeptransformer_local(nchan)
    
    filter_out_artifacts = 1;
    nchan=1;
    Nfold = 1;
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);
    mat_path = '/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/THESIS_DATA/SleepTransformer_mice/kornum_data/mat/';
    listing = dir([mat_path, '*_eeg1.mat']);
    load("/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/Code/HUMMUSS/SleepTransformer_mice/shhs/data_preprocessing/kornum_data/data_split_eval.mat");
    
    acc_novote = [];
    
    seq_len = 61;
    for fold = 1 : Nfold
        fold
        %test_s = test_sub{fold};
        test_s = test_sub;
        sample_size = zeros(numel(test_s), 1);
        for i = 1 : numel(test_s)
            sname = listing(test_s(i)).name;
            load([mat_path,sname], 'label');
            % handle the different here
            sample_size(i) = numel(label) -  (seq_len - 1); 
            yt{fold} = [yt{fold}; double(label)];
        end
        
%     if(seq_len < 100)
        load("/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/PhD/HUMMUSS_paper/outputs/n_heads/Ne_1_Ns_2/test_ret.mat");
% 	else
% 	    load(['./intepretable_sleep/sleeptransformer_simple_longseq/scratch_training_',num2str(nchan),'chan/n',num2str(fold),'/test_ret.mat']);
% 	end
        
        
        acc_novote = [acc_novote; acc];
        
        score_ = cell(1,seq_len);
        for n = 1 : seq_len
            score_{n} = softmax(squeeze(score(:,n,:)));
        end
        score = score_;
        clear score_;

        for i = 1 : numel(test_s)
            start_pos = sum(sample_size(1:i-1)) + 1;
            end_pos = sum(sample_size(1:i-1)) + sample_size(i);
            score_i = cell(1,seq_len);
            %valid_ind = cell(1,seq_len);
            for n = 1 : seq_len
                score_i{n} = score{n}(start_pos:end_pos, :);
                N = size(score_i{n},1);
                %valid_ind{n} = ones(N,1);

                score_i{n} = [ones(seq_len-1,4); score{n}(start_pos:end_pos, :)];
                %valid_ind{n} = [zeros(seq_len-1,1); valid_ind{n}]; 
                score_i{n} = circshift(score_i{n}, -(seq_len - n), 1);
                %valid_ind{n} = circshift(valid_ind{n}, -(seq_len - n), 1);
            end

            smoothing = 0;
            %fused_score = score_i{1};
            %fused_score = log(score_i{1}.*repmat(valid_ind{1},1,5));
            fused_score = log(score_i{1});
            for n = 2 : seq_len
                if(smoothing == 0)
                    %fused_score = fused_score + log(score_i{n}.*repmat(valid_ind{n},1,5));
                    fused_score = fused_score + log(score_i{n});
                else
                    %fused_score = fused_score + score_i{n}.*repmat(valid_ind{n},1,5);
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
    
    if filter_out_artifacts == 1
        yh = yh(yt~=4); % filter out artifacts
        yt = yt(yt~=4); % filter out artifacts
    end

    acc = sum(yh == yt)/numel(yt)
    C = confusionmat(yt, yh);
        
    [mysensitivity, myselectivity]  = calculate_sensitivity_selectivity(yt, yh); % THEY MADE A MISTAKE, THIS IS NOT SELECTIVITY, IS PRECISION!!! (is actually good, because is what I want)
    
    [fscore, sensitivity, specificity] = litis_class_wise_f1(yt, yh);
    mean_fscore = mean(fscore)
    mean_sensitivity = mean(sensitivity)
    mean_specificity = mean(specificity)
    kappa = kappaindex(yh,yt,4)
    
    
    str = '';
    % acc
    str = [str, '$', num2str(acc*100, '%.1f'), '$', ' & '];
    % kappa
    str = [str, '$', num2str(kappa, '%.3f'), '$', ' & '];
    % fscore
    str = [str, '$', num2str(mean_fscore*100, '%.1f'), '$', ' & '];
    % mean_sensitivity
    str = [str, '$', num2str(mean_sensitivity*100, '%.1f'), '$', ' & '];
    % mean_specificity
    str = [str, '$', num2str(mean_specificity*100, '%.1f'), '$', ' & '];
    
    % class-wise MF1
    for i = 1 : 3
        str = [str, '$', num2str(fscore(i)*100,'%.1f'), '$ & '];
    end
    str
    
    collection = [acc, mean_fscore, kappa, mean_sensitivity, mean_specificity];
    
    if filter_out_artifacts==0
        Stages = ["Stage";"WAKE";"NREM";"REM";"ART"]; % W=1, N=2, R=3, A=4
        Acc = ["Acc"; acc; "-"; "-"; "-"];
        Kappa = ["Kappa"; kappa; "-"; "-"; "-"];
    else
        Stages = ["Stage";"WAKE";"NREM";"REM"];
        Acc = ["Acc"; acc; "-"; "-"];
        Kappa = ["Kappa"; kappa; "-"; "-"];
    end
    Sens = ["Sens"; mysensitivity];
    Prec = ["Prec"; myselectivity];
    F1 = ["F1"; fscore];
    ResultsTable = table(Acc, Kappa, Stages,Sens,Prec,F1);
    openvar('ResultsTable')
end