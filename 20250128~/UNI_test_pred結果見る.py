import pandas as pd

# 各foldのデータ
fold1 = [
    {"slide_id": "51-4", "predicted_survival_time": 21.528114207671234, "true_survival_time": 5.180327892303467},
    {"slide_id": "38-4", "predicted_survival_time": 35.92401924889707, "true_survival_time": 35.73770523071289},
    {"slide_id": "14-6", "predicted_survival_time": 25.88378117239703, "true_survival_time": 11.672131538391113},
    {"slide_id": "11-1", "predicted_survival_time": 14.20683098972552, "true_survival_time": 16.557376861572266},
    {"slide_id": "1-2", "predicted_survival_time": 26.328871469150013, "true_survival_time": 10.032787322998047},
    {"slide_id": "14-4", "predicted_survival_time": 24.752272875889187, "true_survival_time": 11.672131538391113},
    {"slide_id": "60-8", "predicted_survival_time": 34.436698720446394, "true_survival_time": 36.655738830566406},
    {"slide_id": "95-8", "predicted_survival_time": 24.603172434128684, "true_survival_time": 33.83606719970703},
    {"slide_id": "55-12", "predicted_survival_time": 30.68665318359683, "true_survival_time": 38.78688430786133},
    {"slide_id": "51-1", "predicted_survival_time": 15.605341938501116, "true_survival_time": 5.180327892303467},
    {"slide_id": "38-10", "predicted_survival_time": 37.13870166137217, "true_survival_time": 35.73770523071289},
]

fold2 = [
    {"slide_id": "53-7", "predicted_survival_time": 37.38392586737783, "true_survival_time": 43.573768615722656},
    {"slide_id": "112-1-B", "predicted_survival_time": 15.148210524012027, "true_survival_time": 8.065573692321777},
    {"slide_id": "122-9", "predicted_survival_time": 17.37647500524787, "true_survival_time": 35.278690338134766},
    {"slide_id": "91-2", "predicted_survival_time": 24.0659097799123, "true_survival_time": 37.573768615722656},
    {"slide_id": "21-7", "predicted_survival_time": 18.215727452572125, "true_survival_time": 18.1639347076416},
    {"slide_id": "2-4", "predicted_survival_time": 23.824935641764583, "true_survival_time": 3.9672131538391113},
    {"slide_id": "122-7", "predicted_survival_time": 21.781956785929992, "true_survival_time": 35.278690338134766},
    {"slide_id": "2-3", "predicted_survival_time": 20.180433163314117, "true_survival_time": 3.9672131538391113},
    {"slide_id": "110-3", "predicted_survival_time": 11.335935033354449, "true_survival_time": 29.57377052307129},
    {"slide_id": "21-6", "predicted_survival_time": 21.876687435296397, "true_survival_time": 18.1639347076416},
    {"slide_id": "110-2", "predicted_survival_time": 13.583568248186323, "true_survival_time": 29.57377052307129},
    {"slide_id": "91-17", "predicted_survival_time": 21.89584494480791, "true_survival_time": 37.573768615722656},
    {"slide_id": "59-19", "predicted_survival_time": 20.40394649184425, "true_survival_time": 1.9016393423080444},
]

fold3 = [
    {"slide_id": "57-10", "predicted_survival_time": 15.217314835136193, "true_survival_time": 20.68852424621582},
    {"slide_id": "32-9", "predicted_survival_time": 35.85466771825738, "true_survival_time": 37.6065559387207},
    {"slide_id": "32-4", "predicted_survival_time": 27.294844703531975, "true_survival_time": 37.6065559387207},
    {"slide_id": "34-2", "predicted_survival_time": 13.009721260392542, "true_survival_time": 9.377049446105957},
    {"slide_id": "120-7", "predicted_survival_time": 17.00965292134201, "true_survival_time": 11.180327415466309},
    {"slide_id": "121-3", "predicted_survival_time": 30.892993215866017, "true_survival_time": 25.967212677001953},
    {"slide_id": "113-21", "predicted_survival_time": 36.46638829806287, "true_survival_time": 34.09836196899414},
    {"slide_id": "113-1", "predicted_survival_time": 35.96792747598406, "true_survival_time": 34.09836196899414},
    {"slide_id": "121-2", "predicted_survival_time": 27.848117174344196, "true_survival_time": 25.967212677001953},
    {"slide_id": "34-11", "predicted_survival_time": 15.486355852924467, "true_survival_time": 9.377049446105957},
    {"slide_id": "120-2", "predicted_survival_time": 31.747079181449653, "true_survival_time": 11.180327415466309},
    {"slide_id": "13-1", "predicted_survival_time": 35.20347836866766, "true_survival_time": 25.37704849243164},
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