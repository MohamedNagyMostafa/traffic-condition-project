import pandas as pd

preds_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\codes\\traffic-condition-project\\predictions\\ls_lf_best_lstm_v3.csv'
ground_truth_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\codes\\traffic-condition-project\\predictions\\test_data.csv'

preds = pd.read_csv(preds_path).to_numpy()
ground_truth = pd.read_csv(ground_truth_path).to_numpy()

preds_stream = False
ground_truth_stream = False

preds_detection = []
ground_truth_detection = []

for idx in range(0, len(ground_truth)):
    if preds[idx] == 0:
        preds_stream = False
    elif not preds_stream:
        preds_stream = True
        preds_detection.append(idx)

    if ground_truth[idx] == 0:
        ground_truth_stream = False
    elif not ground_truth_stream:
        ground_truth_stream = True
        ground_truth_detection.append(idx)


print(preds_detection)
print(ground_truth_detection)


