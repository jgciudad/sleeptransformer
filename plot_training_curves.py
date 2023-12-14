import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

plt.ion()

txt_path_train = "/home/s202283/outputs/sleeptransformer/weighted_ce/iteration1/seq_len_41/train_log.txt"
txt_path_evaluation = "/home/s202283/outputs/sleeptransformer/weighted_ce/iteration1/seq_len_41/eval_result_log.txt"
evaluate_every = 1000


data_train = pd.read_csv(txt_path_train, sep=" ", header=None)
data_eval = pd.read_csv(txt_path_evaluation, sep=" ", header=None)

train_updates = np.arange(len(data_train))
eval_updates = (1+np.arange(len(data_eval)))*evaluate_every
train_loss = data_train.iloc[:, 1]
# eval_loss
train_acc = data_train.iloc[:, 3]
eval_acc = data_eval.to_numpy()[:,2:-1]
eval_acc = eval_acc[:, ::2]
eval_acc = np.mean(eval_acc, axis=1)
train_bal_acc = data_train.iloc[:, 4]
eval_bal_acc = data_eval.to_numpy()[:,2:-1]
eval_bal_acc = eval_bal_acc[:, 1::2]
eval_bal_acc = np.mean(eval_bal_acc, axis=1)

# plt.figure(figsize=[7.4, 5.6])
# plt.plot(data_train.iloc[:, 1], color="royalblue")
# plt.plot(np.arange(len(data_eval))*3825, data_eval.iloc[:, 0]/560, '.-', markersize=14, linewidth=3, color="seagreen") # /560 because in the evaluation the loss is the sum of the loss across all batches, and there are 560 batches
# plt.xlabel('Minibatches', fontsize=15)
# plt.legend(['Training', 'Evaluation'], fontsize=12)
# plt.ylabel('Cross entropy', fontsize=15)
# plt.tick_params(axis='both', which='major', labelsize=15)

plt.figure(figsize=[7.4, 5.6])
plt.plot(train_updates, train_acc)
plt.plot(eval_updates, eval_acc, '.-', markersize=9)
plt.xlabel('Minibatches', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(['Training', 'Evaluation'], fontsize=12)
plt.savefig("/home/s202283/outputs/sleeptransformer/weighted_ce/iteration1/seq_len_41/accuracy.png")

plt.figure(figsize=[7.4, 5.6])
plt.plot(train_updates, train_bal_acc)
plt.plot(eval_updates, eval_bal_acc, '.-', markersize=9)
plt.xlabel('Minibatches', fontsize=15)
plt.ylabel('Balanced accuracy', fontsize=15)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(['Training', 'Evaluation'], fontsize=12)
plt.savefig("/home/s202283/outputs/sleeptransformer/weighted_ce/iteration1/seq_len_41/balanced_accuracy.png")

# plt.figure(figsize=[9, 6.8])
# plt.plot(data_train.iloc[:, 1].rolling(window=15).mean(), color="royalblue", linewidth=1.75)
# plt.plot(np.arange(len(data_eval))*3825, data_eval.iloc[:, 0]/560, '.-', markersize=17, linewidth=3.5, color="mediumseagreen")
# plt.xlabel('Minibatches', fontsize=18)
# plt.ylabel('Cross entropy', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.plot((np.arange(len(data_eval))*3825)[9], (data_eval.iloc[:, 0]/560)[9], '.', markersize=18, color='tomato')
# plt.legend(['Training', 'Evaluation', 'Selected model'], fontsize=17)


# plt.figure(figsize=[9, 6.8])
# plt.plot(data_train.iloc[:, 3].rolling(window=15).mean(), color="royalblue", linewidth=1.75)
# plt.plot(np.arange(len(data_eval))*3825, data_eval.iloc[:, 2], '.-', markersize=17, linewidth=3.5, color="mediumseagreen")
# plt.xlabel('Minibatches', fontsize=18)
# plt.ylabel('Accuracy', fontsize=18)
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.plot((np.arange(len(data_eval))*3825)[9], (data_eval.iloc[:, 2])[9], 'r.', markersize=18, color="tomato")
# plt.legend(['Training', 'Evaluation', 'Selected model'], fontsize=17)

# d2=data.groupby(np.arange(len(data))//400).mean()
# plt.figure()
# plt.plot(d2.iloc[:, 1])
# plt.title('Training output loss')
# plt.figure()
# plt.plot(d2.iloc[:, 3])
# plt.title('Training accuracy')


#
# plt.figure()
# plt.plot(np.arange(len(data))*3825, data.iloc[:, 0], '.-', markersize=9)
# plt.title('Evaluation output loss', fontsize=15)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.xlabel('Minibatches', fontsize=15)
# plt.figure()
# plt.plot(np.arange(len(data))*3825, data.iloc[:, 2], '.-', markersize=9)
# plt.title('Evaluation accuracy', fontsize=15)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.xlabel('Minibatches', fontsize=15)
a=8