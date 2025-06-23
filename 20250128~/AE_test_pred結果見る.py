import pandas as pd

# 各foldのデータ
fold1 = [
    {"slide_id": "51-4", "predicted_survival_time": 23.331073262287326, "true_survival_time": 5.180327892303467},
    {"slide_id": "38-4", "predicted_survival_time": 28.057090709294265, "true_survival_time": 35.73770523071289},
    {"slide_id": "14-6", "predicted_survival_time": 25.262059901616535, "true_survival_time": 11.672131538391113},
    {"slide_id": "11-1", "predicted_survival_time": 24.91314054012597, "true_survival_time": 16.557376861572266},
    {"slide_id": "1-2", "predicted_survival_time": 23.488051913368043, "true_survival_time": 10.032787322998047},
    {"slide_id": "14-4", "predicted_survival_time": 25.214081132474085, "true_survival_time": 11.672131538391113},
    {"slide_id": "60-8", "predicted_survival_time": 37.23296030244316, "true_survival_time": 36.655738830566406},
    {"slide_id": "95-8", "predicted_survival_time": 32.94217883346793, "true_survival_time": 33.83606719970703},
    {"slide_id": "55-12", "predicted_survival_time": 35.75185306205302, "true_survival_time": 38.78688430786133},
    {"slide_id": "51-1", "predicted_survival_time": 9.012565113637638, "true_survival_time": 5.180327892303467},
    {"slide_id": "38-10", "predicted_survival_time": 14.277888602268316, "true_survival_time": 35.73770523071289},
]

fold2 = [
    {"slide_id": "53-7", "predicted_survival_time": 36.87143390489117, "true_survival_time": 43.573768615722656},
    {"slide_id": "112-1-B", "predicted_survival_time": 11.279321392908704, "true_survival_time": 8.065573692321777},
    {"slide_id": "122-9", "predicted_survival_time": 19.391567650520503, "true_survival_time": 35.278690338134766},
    {"slide_id": "91-2", "predicted_survival_time": 27.94147266281542, "true_survival_time": 37.573768615722656},
    {"slide_id": "21-7", "predicted_survival_time": 18.503597631429184, "true_survival_time": 18.1639347076416},
    {"slide_id": "2-4", "predicted_survival_time": 18.332884159209623, "true_survival_time": 3.9672131538391113},
    {"slide_id": "122-7", "predicted_survival_time": 24.1860488995306, "true_survival_time": 35.278690338134766},
    {"slide_id": "2-3", "predicted_survival_time": 22.091383029538278, "true_survival_time": 3.9672131538391113},
    {"slide_id": "110-3", "predicted_survival_time": 11.190415694569582, "true_survival_time": 29.57377052307129},
    {"slide_id": "21-6", "predicted_survival_time": 19.251975871334274, "true_survival_time": 18.1639347076416},
    {"slide_id": "110-2", "predicted_survival_time": 12.425336918750626, "true_survival_time": 29.57377052307129},
    {"slide_id": "91-17", "predicted_survival_time": 14.241679191844058, "true_survival_time": 37.573768615722656},
    {"slide_id": "59-19", "predicted_survival_time": 20.717668997184017, "true_survival_time": 1.9016393423080444},
]

fold3 = [
    {"slide_id": "57-10", "predicted_survival_time": 23.692158162609694, "true_survival_time": 20.68852424621582},
    {"slide_id": "32-9", "predicted_survival_time": 36.69498984658589, "true_survival_time": 37.6065559387207},
    {"slide_id": "32-4", "predicted_survival_time": 34.2046640911082, "true_survival_time": 37.6065559387207},
    {"slide_id": "34-2", "predicted_survival_time": 12.830064489174541, "true_survival_time": 9.377049446105957},
    {"slide_id": "120-7", "predicted_survival_time": 17.85550986450784, "true_survival_time": 11.180327415466309},
    {"slide_id": "121-3", "predicted_survival_time": 17.299011337822925, "true_survival_time": 25.967212677001953},
    {"slide_id": "113-21", "predicted_survival_time": 36.291623046053246, "true_survival_time": 34.09836196899414},
    {"slide_id": "113-1", "predicted_survival_time": 36.17547522220546, "true_survival_time": 34.09836196899414},
    {"slide_id": "121-2", "predicted_survival_time": 22.19422788090871, "true_survival_time": 25.967212677001953},
    {"slide_id": "34-11", "predicted_survival_time": 19.912991420769465, "true_survival_time": 9.377049446105957},
    {"slide_id": "120-2", "predicted_survival_time": 27.050554317279452, "true_survival_time": 11.180327415466309},
    {"slide_id": "13-1", "predicted_survival_time": 32.26308745770274, "true_survival_time": 25.37704849243164},
]

# 各foldのMAEを計算
mae_fold1 = pd.DataFrame(fold1)["predicted_survival_time"].sub(pd.DataFrame(fold1)["true_survival_time"]).abs().mean()
mae_fold2 = pd.DataFrame(fold2)["predicted_survival_time"].sub(pd.DataFrame(fold2)["true_survival_time"]).abs().mean()
mae_fold3 = pd.DataFrame(fold3)["predicted_survival_time"].sub(pd.DataFrame(fold3)["true_survival_time"]).abs().mean()

# 平均MAEを計算
average_mae = (mae_fold1 + mae_fold2 + mae_fold3) / 3

print(f"Fold1 MAE: {mae_fold1:.4f}")
print(f"Fold2 MAE: {mae_fold2:.4f}")
print(f"Fold3 MAE: {mae_fold3:.4f}")
print(f"Average MAE: {average_mae:.4f}")