import tensorflow as tf
from nn_basic_layers import *
from config import Config
from transformer_encoder import Transformer_Encoder

class SleepTransformer(object):
    """
    SeqSleepNet implementation
    """

    def __init__(self, config):
        # Placeholders for input, output and dropout
        self.config = config
        # self.config.frame_seq_len+1 because of CLS
        self.input_x = tf.placeholder(tf.float32,[None, self.config.epoch_seq_len, self.config.frame_seq_len, self.config.ndim, self.config.nchannel], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.epoch_seq_len, self.config.nclass_data], name="input_y")
        self.istraining = tf.placeholder(tf.bool, name='istraining') # idicate training for batch normmalization


        # input for frame-level transformer, self.config.frame_seq_len+1 because of CLS
        frm_trans_X = tf.reshape(self.input_x,[-1, self.config.frame_seq_len, self.config.ndim * self.config.nchannel])
        with tf.variable_scope("frame_transformer"):
            frm_trans_encoder = Transformer_Encoder(d_model=self.config.frm_d_model,
                                                    d_ff=self.config.frm_d_ff,
                                                    num_blocks=self.config.frm_num_blocks, # +1 because of CLS
                                                    num_heads=self.config.frm_num_heads,
                                                    maxlen=self.config.frm_maxlen,
                                                    fc_dropout_rate=self.config.frm_fc_dropout,
                                                    attention_dropout_rate=self.config.frm_attention_dropout,
                                                    smoothing=self.config.frm_smoothing)
            frm_trans_out = frm_trans_encoder.encode(frm_trans_X, training=self.istraining)
            # print(frm_trans_out.get_shape())
            #[-1, frame_seq_len+1, d_model] [-1, 29, 128*3]


        with tf.variable_scope("frame_attention_layer"):
            self.attention_out, self.attention_weight = attention(frm_trans_out, self.config.frame_attention_size)
            # print(self.attention_out.get_shape())
            # attention_output1 of shape (batchsize*epoch_step, nhidden1*2)

        # unfold the data for sequence processing
        seq_trans_X = tf.reshape(self.attention_out, [-1, self.config.epoch_seq_len, self.config.frm_d_model])
        with tf.variable_scope("seq_transformer"):
            seq_trans_encoder = Transformer_Encoder(d_model=self.config.seq_d_model,
                                                    d_ff=self.config.seq_d_ff,
                                                    num_blocks=self.config.seq_num_blocks,
                                                    num_heads=self.config.seq_num_heads,
                                                    maxlen=self.config.seq_maxlen,
                                                    fc_dropout_rate=self.config.seq_fc_dropout,
                                                    attention_dropout_rate=self.config.seq_attention_dropout,
                                                    smoothing=self.config.seq_smoothing)
            seq_trans_out = seq_trans_encoder.encode(seq_trans_X, training=self.istraining)
            # print(seq_trans_out.get_shape())

        self.scores = []
        self.predictions = []
        with tf.variable_scope("output_layer"):
            seq_trans_out = tf.reshape(seq_trans_out, [-1, self.config.seq_d_model])
            fc1 = fc(seq_trans_out, self.config.seq_d_model, self.config.fc_hidden_size, name="fc1", relu=True)
            fc1 = tf.layers.dropout(fc1, rate=self.config.fc_dropout, training=self.istraining)
            fc2 = fc(fc1, self.config.fc_hidden_size, self.config.fc_hidden_size, name="fc2", relu=True)
            fc2 = tf.layers.dropout(fc2, rate=self.config.fc_dropout, training=self.istraining)
            score = fc(fc2, self.config.fc_hidden_size, self.config.nclass_model, name="output", relu=False)
            pred = tf.argmax(score, 1, name="pred")
            self.scores = tf.reshape(score, [-1, self.config.epoch_seq_len, self.config.nclass_model])
            self.predictions = tf.reshape(pred, [-1, self.config.epoch_seq_len])

        # calculate sequence cross-entropy output loss
        with tf.name_scope("output-loss"):

            if config.loss_type == 'normal_ce':
                print('heyoo')
                input_y_categorical = tf.math.argmax(self.input_y, -1) # dummy labels to numbers
                input_y_categorical = tf.reshape(input_y_categorical, [-1])
                scores = tf.reshape(self.scores, [-1, self.config.nclass_model])
                scores = tf.nn.softmax(scores)

                if self.config.nclass_model == self.config.nclass_data:
                    cce = tf.keras.metrics.sparse_categorical_crossentropy(y_true=input_y_categorical, y_pred=scores, from_logits=False)
                    n_elements_in_batch = tf.cast(tf.size(cce), dtype=tf.float32)

                elif self.config.nclass_model != self.config.nclass_data and self.config.artifacts_label != None:
                    artifacts_column = tf.zeros([tf.shape(scores)[0],1])
                    scores = tf.concat([scores, artifacts_column], 1)

                    artifact_mask = tf.not_equal(input_y_categorical, self.config.artifacts_label) # artifact mask (boolean)
                    artifact_mask = tf.where(artifact_mask, tf.ones(tf.shape(artifact_mask)), tf.zeros(tf.shape(artifact_mask))) # boolean artifact mask to binary

                    cce = tf.keras.metrics.sparse_categorical_crossentropy(y_true=input_y_categorical, y_pred=scores, from_logits=False)
                    cce = tf.multiply(cce, artifact_mask)

                    n_elements_in_batch = tf.reduce_sum(artifact_mask)


                cce = tf.reduce_sum(cce)
                self.output_loss = cce / self.config.epoch_seq_len / n_elements_in_batch # average over sequence length and (not-artifacts) elements in batch

            elif config.loss_type == 'weighted_ce':
                print('heyo2222')
                input_y_categorical = tf.math.argmax(self.input_y, -1) # dummy labels to numbers
                input_y_categorical = tf.reshape(input_y_categorical, [-1])
                scores = tf.reshape(self.scores, [-1, self.config.nclass_model])
                scores = tf.nn.softmax(scores)

                if self.config.nclass_model == self.config.nclass_data:
                    cce = tf.keras.metrics.sparse_categorical_crossentropy(y_true=input_y_categorical, y_pred=scores, from_logits=False)
                    n_elements_in_batch = tf.cast(tf.size(cce), dtype=tf.float32)
                elif self.config.nclass_model != self.config.nclass_data and self.config.artifacts_label != None:
                    artifacts_column = tf.zeros([tf.shape(scores)[0],1])
                    scores = tf.concat([scores, artifacts_column], 1)

                    artifact_mask = tf.not_equal(input_y_categorical, self.config.artifacts_label) # artifact mask (boolean)
                    artifact_mask = tf.where(artifact_mask, tf.ones(tf.shape(artifact_mask)), tf.zeros(tf.shape(artifact_mask))) # boolean artifact mask to binary

                    cce = tf.keras.metrics.sparse_categorical_crossentropy(y_true=input_y_categorical, y_pred=scores, from_logits=False)
                    cce = tf.multiply(cce, artifact_mask)

                    n_elements_in_batch = tf.reduce_sum(artifact_mask)

                class_counts = []
                def cond_function_true_wce(cce, n_elements_in_batch, n_classes_in_batch, labels_class_i_binary, labels_class_i_bool):

                    w = n_elements_in_batch / (n_classes_in_batch * tf.reduce_sum(labels_class_i_binary))
                    weights_mask = tf.where(labels_class_i_bool, w*tf.ones(tf.shape(labels_class_i_bool)), tf.ones(tf.shape(labels_class_i_bool)))

                    weighted_cce = tf.multiply(cce, weights_mask)

                    return weighted_cce 

                def cond_function_false_wce(cce):
                    
                    identical_cce = tf.multiply(cce, tf.ones(tf.shape(cce)))

                    return identical_cce
                
                for i in range(self.config.nclass_model):
                    labels_class_i = tf.equal(input_y_categorical, i)
                    labels_class_i = tf.where(labels_class_i, tf.ones(tf.shape(labels_class_i)), tf.zeros(tf.shape(labels_class_i))) # boolean artifact mask to binary
                    
                    class_counts.append(tf.reduce_sum(labels_class_i))
                
                n_classes_in_batch = tf.math.count_nonzero(class_counts, dtype=tf.dtypes.float32)

                for i in range(self.config.nclass_model):
                    labels_class_i_bool = tf.equal(input_y_categorical, i)
                    labels_class_i_binary = tf.where(labels_class_i_bool, tf.ones(tf.shape(labels_class_i_bool)), tf.zeros(tf.shape(labels_class_i_bool))) # boolean artifact mask to binary

                    cce  = tf.cond(tf.reduce_sum(labels_class_i_binary) > 0, lambda: cond_function_true_wce(cce, n_elements_in_batch, n_classes_in_batch, labels_class_i_binary, labels_class_i_bool), lambda: cond_function_false_wce(cce))

                cce = tf.reduce_sum(cce)
                self.output_loss = cce / self.config.epoch_seq_len / n_elements_in_batch # average over sequence length and elements in batch


            # add on regularization
        with tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars])
            self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss

        # self.original_accuracy_list = []
        # # Accuracy at each time index of the input sequence
        # with tf.name_scope("original_accuracy"):
        #     for i in range(self.config.epoch_seq_len):
        #         correct_prediction_i = tf.equal(self.predictions[:,i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
        #         accuracy_i = tf.reduce_mean(tf.cast(correct_prediction_i, "float"), name="accuracy-%s" % i)
        #         self.original_accuracy_list.append(accuracy_i)
        #     self.original_accuracy = sum(self.original_accuracy_list) / len(self.original_accuracy_list)

        with tf.name_scope("accuracy"):
    
            input_y_categorical = tf.math.argmax(self.input_y, -1) # dummy labels to numbers
            
            correct_prediction = tf.equal(self.predictions, input_y_categorical)
            correct_prediction = tf.where(correct_prediction, tf.ones(tf.shape(correct_prediction)), tf.zeros(tf.shape(correct_prediction)))

            if self.config.nclass_model == self.config.nclass_data:
                self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.int32)) / tf.size(correct_prediction)

            if self.config.nclass_model != self.config.nclass_data and self.config.artifacts_label != None:
                artifact_mask = tf.not_equal(input_y_categorical, self.config.artifacts_label) # artifact mask (boolean)
                artifact_mask = tf.where(artifact_mask, tf.ones(tf.shape(artifact_mask)), tf.zeros(tf.shape(artifact_mask))) # boolean artifact mask to binary
                correct_prediction = tf.multiply(correct_prediction, artifact_mask)
                self.accuracy = tf.reduce_sum(correct_prediction) / tf.reduce_sum(artifact_mask)

        with tf.name_scope("balanced_accuracy"):
            def cond_function_true(prediction_class_i, labels_class_i, recalls_sum, n_classes_in_balanced_accuracy):
                correct_prediction_class_i = tf.multiply(prediction_class_i, labels_class_i)

                recalls_sum += (tf.reduce_sum(correct_prediction_class_i) / tf.reduce_sum(labels_class_i))
                n_classes_in_balanced_accuracy += 1

                return [recalls_sum, n_classes_in_balanced_accuracy] 

            def cond_function_false(recalls_sum, n_classes_in_balanced_accuracy):
                
                recalls = tf.multiply(recalls_sum, tf.ones(tf.shape(recalls_sum)))
                n_classes = tf.multiply(n_classes_in_balanced_accuracy, tf.ones(tf.shape(n_classes_in_balanced_accuracy)))

                return [recalls, n_classes] 
        
            recalls_sum = 0.0
            n_classes_in_balanced_accuracy = 0.0
            
            input_y_categorical = tf.math.argmax(self.input_y, -1) # dummy labels to numbers

            for i in range(self.config.nclass_model):
                prediction_class_i = tf.equal(self.predictions, i)
                labels_class_i = tf.equal(input_y_categorical, i)
                prediction_class_i = tf.where(prediction_class_i, tf.ones(tf.shape(prediction_class_i)), tf.zeros(tf.shape(prediction_class_i))) # boolean artifact mask to binary
                labels_class_i = tf.where(labels_class_i, tf.ones(tf.shape(labels_class_i)), tf.zeros(tf.shape(labels_class_i))) # boolean artifact mask to binary
                
                [recalls_sum, n_classes_in_balanced_accuracy]  = tf.cond(tf.reduce_sum(labels_class_i) > 0, lambda: cond_function_true(prediction_class_i, labels_class_i, recalls_sum, n_classes_in_balanced_accuracy), lambda: cond_function_false(recalls_sum, n_classes_in_balanced_accuracy))

            self.balanced_accuracy = recalls_sum / n_classes_in_balanced_accuracy


            
