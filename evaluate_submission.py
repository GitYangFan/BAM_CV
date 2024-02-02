import tensorflow as tf
import numpy as np
import custom_layers_submission as cl
import helpers_submission as h
import testing_generator_multidim as genT
import pandas as pd
import time
import os


tf.keras.backend.set_floatx('float32')
ddtype = tf.float32


custom_objects = {
    'my_loss_crossentropy_var_size': h.my_loss_crossentropy_var_size,
    'my_loss_Frob_var_size': h.my_loss_Frob_var_size,
    'my_accuracy_var_size': h.my_accuracy_var_size,
    'cov2cor': h.cov2cor,
    'my_loss_categorical_N_d_d_c':h.my_loss_categorical_N_d_d_c,
    'my_loss_categorical_N_d_d_c_binary':h.my_loss_categorical_N_d_d_c_binary,
    'my_loss_categorical_combined':h.my_loss_categorical_combined,
    'weight_binary_loss':0.8,
    'my_loss_categorical_combined2':h.my_loss_categorical_combined(weight_binary_loss=0.8),
    'my_accuracy_categorical_N_d_d_c':h.my_accuracy_categorical_N_d_d_c,
    'my_accuracy_categorical_N_d_d_c_binary':h.my_accuracy_categorical_N_d_d_c_binary,
    'layer_N_M_d_1_to_N_M_d_C_residual': cl.layer_N_M_d_1_to_N_M_d_C_residual,
    'layer_N_M_d_C_attention_features_for_each_sample': cl.layer_N_M_d_C_attention_features_for_each_sample,
    'layer_N_M_d_C_attention_samples_for_each_feature': cl.layer_N_M_d_C_attention_samples_for_each_feature,
    #'layer_squeeze_and_excitation_N_c_d_d': cl.layer_squeeze_and_excitation_N_c_d_d,
    'layer_N_C_d_d_bilinear_attention_cov2cor_spd':cl.layer_N_C_d_d_bilinear_attention_cov2cor_spd,
    'layer_N_C_d_d_spd_activation_scaled':cl.layer_N_C_d_d_spd_activation_scaled,
    'layer_N_c_d_d_to_N_d_d_3_LogEig_softmax2':cl.layer_N_c_d_d_to_N_d_d_3_LogEig_softmax2,
    'layer_channels_dense_res_N_M_d_c':cl.layer_channels_dense_res_N_M_d_c,
    'layer_squeeze_and_excitation_shape_information_N_c_d_d':cl.layer_squeeze_and_excitation_shape_information_N_c_d_d,
    'data_N_M_d_c_to_cov_N_c_d_d':cl.data_N_M_d_c_to_cov_N_c_d_d,
    'frob': cl.frob,
    'spd_sigmoid_activation_N_d_d': cl.spd_sigmoid_activation_N_d_d,
    'lam_init_eps': cl.lam_init_eps,
    'cov2cor_N_c_d_d': cl.cov2cor_N_c_d_d,
    'l1_constraintLessEqual': cl.l1_constraintLessEqual,
    'get_off_diag_var_size':h.get_off_diag_var_size,
    'get_off_diag_var_size_N_d_d_c':h.get_off_diag_var_size_N_d_d_c,
    'class_counts0':h.class_counts0,
    'class_counts1':h.class_counts1,
    'class_counts2':h.class_counts2,
    'my_penalty':h.my_penalty,
    'my_penalty_metric':h.my_penalty_metric,
    'precisionBinary':h.precisionBinary,
    'recallBinary':h.recallBinary,
    'aucBinary':h.aucBinary,
    'my_loss_categorical_penalty':h.my_loss_categorical_penalty
}

model=tf.keras.models.load_model("/Users/philipp/Documents/Graph_Inference/curie/send/pureCheby_factorial_decrease_small_degree_M1000d100_shapeInformationNEW.hd5", custom_objects=custom_objects)

def evaluate_auc_acc(model_name,sample_fun,args,d_list,M_list,spe=10):
    if not os.path.exists("results_varyMd_{}".format(model_name)):
        os.makedirs("results_varyMd_{}".format(model_name))

    sample_name = sample_fun.__name__.split("_")[-1]
    tf.print(sample_name)
    tf.print(args)

    columns = ['mean_aucs', 'std_auc', 'elapsed_time']
    columns_accs = ['mean_accs', 'std_accs', 'elapsed_time']

    parameter_list = []

    for d in d_list:
        for M in M_list:
            parameter_list.append((1, d, M))

    for (N, d, M) in parameter_list:
        df = pd.DataFrame(columns=columns)
        df_accs = pd.DataFrame(columns=columns_accs)

        tf.print(N, d, M)
        tf.print(args)
        start_time = time.time()

        # Initialize lists to store accuracies for each step
        aucs = []
        accs = []

        # Manually control the number of steps in each epoch
        for step in range(spe):
            # Generate data for one step
            data_gen = genT.DataGeneratorEvaluate3(N, M, d, sample_fun, 1, *args)
            # Evaluate the model on this step's data
            hist = model.evaluate(data_gen, steps=1)
            # Append the accuracy to the list
            aucs.append(hist[11])  # replace 'accuracy' with the name of your accuracy metric
            accs.append(hist[1])
        # Convert the list to a numpy array
        aucs = np.array(aucs)
        accs = np.array(accs)
        # Compute the mean accuracy and variance for this epoch
        mean_aucs = np.mean(aucs)
        std_aucs = np.sqrt(np.var(aucs))

        mean_accs = np.mean(accs)
        std_accs = np.sqrt(np.var(accs))


        # Append the mean accuracy and variance to the respective lists
        # mean_values.append(mean_aucs)
        # stds.append(std_aucs)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Create a new DataFrame with your calculated values
        new_row = pd.DataFrame([[mean_aucs, std_aucs, elapsed_time]], columns=columns)
        new_row_accs = pd.DataFrame([[mean_accs, std_accs, elapsed_time]], columns=columns_accs)


        row_name = "dist={}, N={}, M={}, d={}".format(sample_name, N, M, d)
        new_row.index = [row_name]
        new_row_accs.index = [row_name]

        # Concatenate the new row to the DataFrame
        df = pd.concat([df, new_row], axis=0)
        df_accs = pd.concat([df_accs, new_row_accs], axis=0)

        # Save the DataFrame to a CSV file
        df.to_csv("results_varyMd_{}".format(model_name)+"/AUC_d{}_M{}_{}".format(d,M,sample_name) + ".csv", index=True)
        df_accs.to_csv("results_varyMd_{}".format(model_name)+"/Acc_d{}_M{}_{}".format(d,M,sample_name) + ".csv", index=True)



all_sample_fun=[]
all_sample_fun.append(genT.sample_from_dag_chebyshev)
#all_sample_fun.append(genT.sample_from_dag_MLP)
all_sample_fun.append(genT.sample_from_dag_Linear)
# all_sample_fun.append(genT.sample_from_dag_Linear2)
all_sample_fun.append(genT.sample_from_dag_x2)
all_sample_fun.append(genT.sample_from_dag_x3)
all_sample_fun.append(genT.sample_from_dag_Multidim)
#all_sample_fun.append(genT.sample_from_dag_MLP)
all_sample_fun.append(genT.sample_from_dag_sin)
# #all_sample_fun.append(genT.sample_from_dag_LinearAddMult)
all_sample_fun.append(genT.sample_from_dag_cos)

all_args=[]
all_args.append([100])
all_args.append([100])
all_args.append([100])
all_args.append([100])
all_args.append([100])
#all_args.append([100])
all_args.append([1,1,100])
all_args.append([1,1,100])


for l in range(len(all_sample_fun)):
    sample_fun = all_sample_fun[l]
    args = all_args[l]

    evaluate_auc_acc("shapeInfoBig",sample_fun,args,[10,20,50,100],[50,100,200,500,1000],50)




