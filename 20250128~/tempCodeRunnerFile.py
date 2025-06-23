
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