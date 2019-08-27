from keras.models import load_model
import numpy as np
import time
import csv
import os

input = np.load('set_hp_1s_0.2total.npy')

path = '.'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.h5' in file:
            files.append(os.path.join(r, file))

for f in files:
    print(f)
    model = load_model(f)
    predict_start = time.time()
    predictions = model.predict(input)
    predict_end = time.time()
    predict_time = predict_end - predict_start
    print(predictions)
    print(predict_time)


with open('inference_metrics.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['device', 'device_name', 'inference_time'])
    employee_writer.writerow(['GPU', 'JetsonTX2', predict_time])

# print('EVAL')
# eval_start = time.time()
# loss_inf, mae_inf, mse_inf = model.evaluate(input, tags)
# eval_end = time.time()
# eval_time = eval_end - eval_start
# print (eval_time)
# print (mse_inf)

#SACAMOS LOS DATOS DE AQUÍ
#pARSEADO DEL DICCIONARIO A TABLA CON EL NOMBRE COMO PRIMERA COLUMNA
# fields = ['net', 'total_time', 'epoch_time', 'step_time', 'mse', 'mse_i', 'inference_time', 'parameters', 'name', 'batch_size', 'learning_rate', 'opt', 'model', 'epochs']
# with open('metrics_3_capas_950M_b100.csv', 'w', newline='') as csvfile:
#     w = csv.DictWriter(csvfile, fields )
#     for key,val in sorted(global_results.items()):
#         row = {'net': key}
#         row.update(val)
#         w.writerow(row)
