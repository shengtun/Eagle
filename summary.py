import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def normalize_dataset_name(dataset_name):
    """标准化数据集名称，将DS-MVTec和MVTec-AD合并为MVTec-AD"""
    if dataset_name in ['DS-MVTec', 'MVTec-AD']:
        return 'MVTec-AD'
    return dataset_name

def caculate_accuracy_mmad(answers_json_path, normal_flag='good', show_overkill_miss=False):
    # 用于存储所有答案
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    dataset_names = []
    type_list = []
    for answer in all_answers_json:
        dataset_name = normalize_dataset_name(answer['image'].split('/')[0])
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        if question_type not in type_list:
            type_list.append(question_type)

    # 初始化统计数据结构
    question_stats = {dataset_name: {} for dataset_name in dataset_names}
    detection_stats = {dataset_name: {} for dataset_name in dataset_names}
    for dataset_name in dataset_names:
        detection_stats[dataset_name]['normal'] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}
        detection_stats[dataset_name]['abnormal'] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}
        for question_type in type_list:
            question_stats[dataset_name][question_type] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}

    for answer in all_answers_json:
        dataset_name = normalize_dataset_name(answer['image'].split('/')[0])
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"
        gpt_answer = answer['gpt_answer']
        correct_answer = answer['correct_answer']
        if correct_answer not in ['A', 'B', 'C', 'D', 'E'] or gpt_answer not in ['A', 'B', 'C', 'D', 'E']:
            all_answers_json.remove(answer)
            print("Remove error:", "correct_answer:", correct_answer, "gpt_answer:", gpt_answer)
            continue

        question_stats[dataset_name][question_type]['total'] += 1
        if answer['correct_answer'] == answer['gpt_answer']:
            question_stats[dataset_name][question_type]['correct'] += 1

        if question_type == "Anomaly Detection":
            if normal_flag in answer['image']:
                detection_stats[dataset_name]['normal']['total'] += 1
                if answer['correct_answer'] == answer['gpt_answer']:
                    detection_stats[dataset_name]['normal']['correct'] += 1
            else:
                detection_stats[dataset_name]['abnormal']['total'] += 1
                if answer['correct_answer'] == answer['gpt_answer']:
                    detection_stats[dataset_name]['abnormal']['correct'] += 1


        answers_dict = question_stats[dataset_name][question_type]['answers']
        if gpt_answer not in answers_dict:
            answers_dict[gpt_answer] = 0
        answers_dict[gpt_answer] += 1
        correct_answers_dict = question_stats[dataset_name][question_type]['correct_answers']
        if correct_answer not in correct_answers_dict:
            correct_answers_dict[correct_answer] = 0
        correct_answers_dict[correct_answer] += 1

    # 创建准确率表格
    accuracy_df = pd.DataFrame(index=dataset_names)
    for dataset_name in dataset_names:
        for question_type in type_list:
            total = question_stats[dataset_name][question_type]['total']
            correct = question_stats[dataset_name][question_type]['correct']
            cls_accuracy = correct / total if total != 0 else 0
            accuracy_df.at[dataset_name, question_type] = cls_accuracy*100

            if question_type in ['Anomaly Detection']:
                TP = detection_stats[dataset_name]['abnormal']['correct']
                FP = detection_stats[dataset_name]['normal']['total'] - detection_stats[dataset_name]['normal']['correct']
                FN = detection_stats[dataset_name]['abnormal']['total'] - detection_stats[dataset_name]['abnormal']['correct']
                TN = detection_stats[dataset_name]['normal']['correct']
                Precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                Recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                TPR = Recall
                FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
                F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) != 0 else 0
                normal_acc = detection_stats[dataset_name]['normal']['correct'] / detection_stats[dataset_name]['normal']['total'] if detection_stats[dataset_name]['normal']['total'] != 0 else 0
                anomaly_acc = detection_stats[dataset_name]['abnormal']['correct'] / detection_stats[dataset_name]['abnormal']['total'] if detection_stats[dataset_name]['abnormal']['total'] != 0 else 0
                # accuracy_df.at[dataset_name, 'normal_acc'] = normal_acc
                # accuracy_df.at[dataset_name, 'anomaly_acc'] = anomaly_acc
                accuracy_df.at[dataset_name, 'Anomaly Detection'] = (normal_acc+anomaly_acc)/2*100

        # 计算该数据集的平均准确率
        accuracy_df.at[dataset_name, 'Average'] = accuracy_df.loc[dataset_name].mean()

        accuracy_df.at[dataset_name, 'Recall'] = Recall*100
        accuracy_df.at[dataset_name, 'Precision'] = Precision*100
        accuracy_df.at[dataset_name, 'F1'] = F1*100

    # # 计算每个问题的平均准确率
    # accuracy_df['Average'] = accuracy_df.drop(columns=['Recall']).mean(axis=1)

    if show_overkill_miss:
        for dataset_name in dataset_names:
            normal_acc = detection_stats[dataset_name]['normal']['correct'] / detection_stats[dataset_name]['normal'][
                'total'] if detection_stats[dataset_name]['normal']['total'] != 0 else 0
            anomaly_acc = detection_stats[dataset_name]['abnormal']['correct'] / detection_stats[dataset_name]['abnormal'][
                'total'] if detection_stats[dataset_name]['abnormal']['total'] != 0 else 0
            accuracy_df.at[dataset_name, 'Overkill'] = (1 - normal_acc) * 100
            accuracy_df.at[dataset_name, 'Miss'] = (1 - anomaly_acc) * 100

    accuracy_df.loc['Average'] = accuracy_df.mean()

    # 数据可视化
    plt.figure(figsize=(10, 7))
    sns.heatmap(accuracy_df, annot=True, cmap='coolwarm', fmt=".1f", vmax=100, vmin=25)
    plt.title(f'Accuracy of {os.path.split(answers_json_path)[-1].replace(".json", "")}')
    # 旋转X轴标签
    plt.xticks(rotation=30, ha='right')  # ha='right'可以使标签稍微倾斜，以便更好地阅读

    # 自动调整边框，减少空白
    plt.tight_layout()
    plt.show()

    # 保存准确率表格
    accuracy_path = answers_json_path.replace('.json', '_accuracy.csv')
    accuracy_df.to_csv(accuracy_path)

    print("The statistics of {}".format(answers_json_path))
    print(accuracy_df)

    print(accuracy_df.loc['Average'])
    return question_stats

def caculate_accuracy(answers_json_path, normal_flag='good'): # for mvtec only
    # 用于存储所有答案
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    # 统计classname
    classname = []
    for answer in all_answers_json:
        cls = answer['class']
        if cls not in classname:
            classname.append(cls)
    # 初始化统计数据结构
    question_stats = {'normal': {}, 'anomaly': {}}

    for category in ['normal', 'anomaly']:
        for i in range(1, 6):
            question_stats[category][i] = {}
            for cls in classname:
                question_stats[category][i][cls] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}

    count = 0
    question_number = 1
    last_image = ''
    # 填充统计数据结构
    for answer in all_answers_json:
        cls = answer['class']
        question_text = answer['question']['text']
        if 'Question' in question_text:
            # question_number = int(question_text.split(':')[0].split(' ')[1])
            question_number = int(question_text.split('Question')[1].strip()[0])
        elif answer['image'] == last_image:
            question_number += 1
        else:
            question_number = 1
        last_image = answer['image']

        is_normal = normal_flag in answer['image']
        category = 'normal' if is_normal else 'anomaly'
        # if answer['gpt_answer'] == '' or answer['gpt_answer'] == '':
        #     count += 1
        #     continue
        # 更新总数和正确数
        question_stats[category][question_number][cls]['total'] += 1
        if answer['correct_answer'] == answer['gpt_answer']:
            question_stats[category][question_number][cls]['correct'] += 1
        gpt_answer = answer['gpt_answer']
        correct_answer = answer['correct_answer']
        if correct_answer not in ['A', 'B', 'C', 'D', 'E'] or gpt_answer not in ['A', 'B', 'C', 'D', 'E']:
            # 从all_answers_json中删除该条并保存
            all_answers_json.remove(answer)
            print("correct_answer:", correct_answer, "gpt_answer:", gpt_answer)

            continue
        # 更新答案计数
        answers_dict = question_stats[category][question_number][cls]['answers']
        if gpt_answer not in answers_dict:
            answers_dict[gpt_answer] = 0
        answers_dict[gpt_answer] += 1
        # 更新正确答案计数
        correct_answers_dict = question_stats[category][question_number][cls]['correct_answers']
        if correct_answer not in correct_answers_dict:
            correct_answers_dict[correct_answer] = 0
        correct_answers_dict[correct_answer] += 1
    # with open(answers_json_path, "w") as file:
    #     json.dump(all_answers_json, file, indent=4)

    # 异常问题：1有无 2种类 3位置 4外观 5其他
    Anomaly_Question = ["Existence", "Defect Type", "Defect Location", "Defect Appearance", "Other"]
    # 正常问题：1有无 2-5其他
    Normal_Question = ["Existence", "Other", "Other", "Other", "Other"]

    # 根据问题和类别重新统计
    Question_label = ["Existence", "Defect Type", "Defect Location", "Defect Appearance", "Other"]
    new_question_stats = {}
    for cls in classname:
        new_question_stats[cls] = {}
        for question_label in Question_label:
            new_question_stats[cls][question_label] = {'total': 0, 'correct': 0}
    for cls in classname:
        for category in ['normal', 'anomaly']:
            for i in range(1, 6):
                if category == 'normal':
                    question_label = Normal_Question[i - 1]
                else:
                    question_label = Anomaly_Question[i - 1]
                new_question_stats[cls][question_label]['total'] += question_stats[category][i][cls]['total']
                new_question_stats[cls][question_label]['correct'] += question_stats[category][i][cls]['correct']

    # 创建准确率表格
    accuracy_df = pd.DataFrame(index=classname)
    for cls in classname:
        for question_label in Question_label:
            total = new_question_stats[cls][question_label]['total']
            correct = new_question_stats[cls][question_label]['correct']
            cls_accuracy = correct / total if total != 0 else 0
            accuracy_df.at[cls, question_label] = cls_accuracy

    # 计算每个问题的平均准确率
    accuracy_df['Average'] = accuracy_df.mean(axis=1)


    for cls in classname:
        TP = question_stats['anomaly'][1][cls]['correct']
        FP = question_stats['normal'][1][cls]['total'] - question_stats['normal'][1][cls]['correct']
        FN = question_stats['anomaly'][1][cls]['total'] - question_stats['anomaly'][1][cls]['correct']
        TN = question_stats['normal'][1][cls]['correct']
        Precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        Recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        # F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) != 0 else 0
        # accuracy_df.at[cls, 'Existence (F1)'] = F1
        TPR = Recall
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
        # # auroc = (1 - FPR) * TPR
        # auroc = (1 - FPR + TPR) / 2
        # accuracy_df.at[cls, 'Existence (AUROC)'] = auroc
        # # AUPR = Precision*Recall
        # AUPR = (Precision + Recall) / 2
        # accuracy_df.at[cls, 'Existence (AUPR)'] = AUPR

        normal_acc = question_stats['normal'][1][cls]['correct'] / question_stats['normal'][1][cls]['total'] if question_stats['normal'][1][cls]['total'] != 0 else 0
        anomaly_acc = question_stats['anomaly'][1][cls]['correct'] / question_stats['anomaly'][1][cls]['total'] if question_stats['anomaly'][1][cls]['total'] != 0 else 0
        accuracy_df.at[cls, 'Overkill'] = 1 - normal_acc
        accuracy_df.at[cls, 'Miss'] = 1 - anomaly_acc
        accuracy_df.at[cls, 'Recall'] = Recall



    # 计算每个类别的平均准确率
    accuracy_df.loc['Average'] = accuracy_df.mean()
    # 数据可视化
    plt.figure(figsize=(10, 9))
    sns.heatmap(accuracy_df, annot=True, cmap='coolwarm', fmt=".2f", vmax=1, vmin=0)
    plt.title(f'Accuracy of {os.path.split(answers_json_path)[-1].replace(".json", "")}')
    # 旋转X轴标签
    plt.xticks(rotation=30, ha='right')  # ha='right'可以使标签稍微倾斜，以便更好地阅读
    plt.show()

    # 保存准确率表格
    accuracy_path = answers_json_path.replace('.json', '_accuracy.csv')
    accuracy_df.to_csv(accuracy_path)

    print("The statistics of {}".format(answers_json_path))
    print(accuracy_df)

    print(accuracy_df['Average'])
    return question_stats

def caculate_accuracy_selected_dataset(answers_json_path, dataset_name_filter = None, normal_flag='good', show_overkill_miss=False):
    """
    统计并可视化模型在指定数据集上的各类问题准确率

    参数:
    - answers_json_path (str): 回答结果的 JSON 路径
    - dataset_name_filter (str): 要分析的数据集名称，如 'mvtec', 'VisA' 等
    - normal_flag (str): 标识正常图像的关键词，默认是 'good'
    - show_overkill_miss (bool): 是否显示 Overkill / Miss 率

    返回:
    - accuracy_df: 准确率表格
    """

    # 加载 JSON 文件
    if not os.path.exists(answers_json_path):
        raise FileNotFoundError(f"文件不存在: {answers_json_path}")
    with open(answers_json_path, "r") as f:
        all_answers_json = json.load(f)

    # 仅保留选定数据集的记录
    if dataset_name_filter:
        selected_answers = [
            ans for ans in all_answers_json
            if ans['image'].split('/')[0] == dataset_name_filter
        ]

    class_names =[]
    type_list = []

    for answer in selected_answers:
        class_name = answer['image'].split('/')[1]
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"
        if class_name not in class_names:
            class_names.append(class_name)
        if type not in type_list:
            type_list.append(question_type)

    # 初始化统计结构
    question_stats = {class_name:{} for class_name in class_names}
    detection_stats = {class_name:{} for class_name in class_names}
    for class_name in class_names:
        detection_stats[class_name]['normal'] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}
        detection_stats[class_name]['abnormal'] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}
        for question_type in type_list:
            question_stats[class_name][question_type] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}

    # 开始统计
    for answer in selected_answers:
        class_name = answer['image'].split('/')[1]
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"
        gpt_answer = answer['gpt_answer']
        correct_answer = answer['correct_answer']

        if correct_answer not in ['A', 'B', 'C', 'D', 'E'] or gpt_answer not in ['A', 'B', 'C', 'D', 'E']:
            print("跳过无效答案:", answer)
            continue

        question_stats[class_name][question_type]['total'] += 1
        if gpt_answer == correct_answer:
            question_stats[class_name][question_type]['correct'] += 1

        if question_type == "Anomaly Detection":
            is_normal = normal_flag in answer['image']
            key = 'normal' if is_normal else 'abnormal'
            detection_stats[class_name][key]['total'] += 1
            if gpt_answer == correct_answer:
                detection_stats[class_name][key]['correct'] += 1

        question_stats[class_name][question_type]['answers'][gpt_answer] = \
            question_stats[class_name][question_type]['answers'].get(gpt_answer, 0) + 1
        question_stats[class_name][question_type]['correct_answers'][correct_answer] = \
            question_stats[class_name][question_type]['correct_answers'].get(correct_answer, 0) + 1

    # 创建准确率表
    accuracy_df = pd.DataFrame(index=class_names)
    for class_name in class_names:
        for qtype in type_list:
            total = question_stats[class_name][qtype]['total']
            correct = question_stats[class_name][qtype]['correct']
            acc = correct / total if total != 0 else 0
            accuracy_df.at[class_name, qtype] = acc * 100

            if qtype == "Anomaly Detection":
                # TP = detection_stats['abnormal']['correct']
                # FP = detection_stats[class_name]['normal']['total'] - detection_stats[class_name]['normal']['correct']
                # FN = detection_stats[class_name]['abnormal']['total'] - detection_stats[class_name]['abnormal']['correct']
                # TN = detection_stats[class_name]['normal']['correct']
                normal_acc = detection_stats[class_name]['normal']['correct'] / detection_stats[class_name]['normal']['total'] if detection_stats[class_name]['normal']['total'] != 0 else 0
                abnormal_acc = detection_stats[class_name]['abnormal']['correct'] / detection_stats[class_name]['abnormal']['total'] if detection_stats[class_name]['abnormal']['total'] != 0 else 0
                accuracy_df.at[class_name, 'Anomaly Detection'] = (normal_acc + abnormal_acc) / 2 * 100

    # # 计算每个类别的平均准确率
    accuracy_df['Average'] = accuracy_df.mean(axis=1)

    if show_overkill_miss:
        for class_name in class_names:
            normal_acc = detection_stats[class_name]['normal']['correct'] / detection_stats[class_name]['normal']['total'] if detection_stats[class_name]['normal']['total'] else 0
            abnormal_acc = detection_stats[class_name]['abnormal']['correct'] / detection_stats[class_name]['abnormal']['total'] if detection_stats[class_name]['abnormal']['total'] else 0
            accuracy_df.at[class_name, 'Overkill'] = (1 - normal_acc) * 100
            accuracy_df.at[class_name, 'Miss'] = (1 - abnormal_acc) * 100

    # 计算每个question的平均准确率
    accuracy_df.loc['Average'] = accuracy_df.mean()
    # 可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(accuracy_df, annot=True, cmap='coolwarm', fmt=".1f", vmax=100, vmin=25)
    plt.title(f'Accuracy of {os.path.split(answers_json_path)[-1].replace(".json", "")}')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    # plt.show()

    # 保存表格
    csv_path = answers_json_path.replace('.json', f'_{dataset_name_filter}_accuracy.csv')
    accuracy_df.to_csv(csv_path)
    print(f"准确率表格已保存至: {csv_path}")

    return accuracy_df

def caculate_accuracy_selected_dataset_v(answers_json_path, dataset_name_filter = None, normal_flag='good', show_overkill_miss=False):
    # ... (earlier unchanged code)
    """
    统计并可视化模型在指定数据集上的各类问题准确率

    参数:
    - answers_json_path (str): 回答结果的 JSON 路径
    - dataset_name_filter (str): 要分析的数据集名称，如 'mvtec', 'VisA' 等
    - normal_flag (str): 标识正常图像的关键词，默认是 'good'
    - show_overkill_miss (bool): 是否显示 Overkill / Miss 率

    返回:
    - accuracy_df: 准确率表格
    """

    # 加载 JSON 文件
    if not os.path.exists(answers_json_path):
        raise FileNotFoundError(f"文件不存在: {answers_json_path}")
    with open(answers_json_path, "r") as f:
        all_answers_json = json.load(f)

    # 仅保留选定数据集的记录
    if dataset_name_filter:
        selected_answers = [
            ans for ans in all_answers_json
            if ans['image'].split('/')[0] == dataset_name_filter
        ]

    class_names =[]
    type_list = []

    for answer in selected_answers:
        class_name = answer['image'].split('/')[1]
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"
        if class_name not in class_names:
            class_names.append(class_name)
        if type not in type_list:
            type_list.append(question_type)

    # 初始化统计结构
    question_stats = {class_name:{} for class_name in class_names}
    detection_stats = {class_name:{} for class_name in class_names}
    for class_name in class_names:
        detection_stats[class_name]['normal'] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}
        detection_stats[class_name]['abnormal'] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}
        for question_type in type_list:
            question_stats[class_name][question_type] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}

    # 开始统计
    for answer in selected_answers:
        class_name = answer['image'].split('/')[1]
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"
        gpt_answer = answer['gpt_answer']
        correct_answer = answer['correct_answer']

        if correct_answer not in ['A', 'B', 'C', 'D', 'E'] or gpt_answer not in ['A', 'B', 'C', 'D', 'E']:
            print("跳过无效答案:", answer)
            continue

        question_stats[class_name][question_type]['total'] += 1
        if gpt_answer == correct_answer:
            question_stats[class_name][question_type]['correct'] += 1

        if question_type == "Anomaly Detection":
            is_normal = normal_flag in answer['image']
            key = 'normal' if is_normal else 'abnormal'
            detection_stats[class_name][key]['total'] += 1
            if gpt_answer == correct_answer:
                detection_stats[class_name][key]['correct'] += 1

        question_stats[class_name][question_type]['answers'][gpt_answer] = \
            question_stats[class_name][question_type]['answers'].get(gpt_answer, 0) + 1
        question_stats[class_name][question_type]['correct_answers'][correct_answer] = \
            question_stats[class_name][question_type]['correct_answers'].get(correct_answer, 0) + 1
    # 创建准确率表
    accuracy_df = pd.DataFrame(index=class_names)
    for class_name in class_names:
        for qtype in type_list:
            total = question_stats[class_name][qtype]['total']
            correct = question_stats[class_name][qtype]['correct']
            acc = correct / total if total != 0 else 0
            accuracy_df.at[class_name, qtype] = acc * 100

            if qtype == "Anomaly Detection":
                # detailed detection metrics
                TP = detection_stats[class_name]['abnormal']['correct']
                FP = detection_stats[class_name]['normal']['total'] - detection_stats[class_name]['normal']['correct']
                FN = detection_stats[class_name]['abnormal']['total'] - detection_stats[class_name]['abnormal']['correct']
                TN = detection_stats[class_name]['normal']['correct']

                Precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                Recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                TPR = Recall
                FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
                F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) != 0 else 0

                normal_acc = detection_stats[class_name]['normal']['correct'] / detection_stats[class_name]['normal']['total'] if detection_stats[class_name]['normal']['total'] != 0 else 0
                abnormal_acc = detection_stats[class_name]['abnormal']['correct'] / detection_stats[class_name]['abnormal']['total'] if detection_stats[class_name]['abnormal']['total'] != 0 else 0

                # keep original Anomaly Detection cell (average of normal & abnormal accuracy)
                accuracy_df.at[class_name, 'Anomaly Detection'] = (normal_acc + abnormal_acc) / 2 * 100

                # store additional metrics as percentage where appropriate
                accuracy_df.at[class_name, 'Precision'] = Precision * 100
                accuracy_df.at[class_name, 'Recall'] = Recall * 100
                accuracy_df.at[class_name, 'F1'] = F1 * 100
                # optionally store TPR/FPR if desired (kept as ratios here)
                accuracy_df.at[class_name, 'TPR'] = TPR
                accuracy_df.at[class_name, 'FPR'] = FPR

    # 计算每个question的平均准确率
    accuracy_df['Average'] = accuracy_df.mean(axis=1)
    if show_overkill_miss:
        for class_name in class_names:
            normal_acc = detection_stats[class_name]['normal']['correct'] / detection_stats[class_name]['normal'][
                'total'] if detection_stats[class_name]['normal']['total'] else 0
            abnormal_acc = detection_stats[class_name]['abnormal']['correct'] / detection_stats[class_name]['abnormal'][
                'total'] if detection_stats[class_name]['abnormal']['total'] else 0
            accuracy_df.at[class_name, 'Overkill'] = (1 - normal_acc) * 100
            accuracy_df.at[class_name, 'Miss'] = (1 - abnormal_acc) * 100

    # 计算每个question的平均准确率
    accuracy_df.loc['Average'] = accuracy_df.mean()
    # 可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(accuracy_df, annot=True, cmap='coolwarm', fmt=".1f", vmax=100, vmin=25)
    plt.title(f'Accuracy of {os.path.split(answers_json_path)[-1].replace(".json", "")}')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

    # 保存表格
    csv_path = answers_json_path.replace('.json', f'_{dataset_name_filter}_accuracy.csv')
    accuracy_df.to_csv(csv_path)
    print(f"准确率表格已保存至: {csv_path}")
    return accuracy_df


def plot_subcategory_accuracy(answers_json_path, dataset_name, normal_flag='good'):
    """
    Draws a heatmap of accuracy rates for each subcategory (class/object) in the specified dataset.

    Parameters:
    - answers_json_path (str): Path to the answers JSON file.
    - dataset_name (str): Name of the dataset to filter.
    - normal_flag (str): Keyword to identify normal images (default: 'good').
    """
    import json
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not os.path.exists(answers_json_path):
        raise FileNotFoundError(f"File not found: {answers_json_path}")
    with open(answers_json_path, "r") as f:
        all_answers_json = json.load(f)

    # Filter answers for the specified dataset
    filtered = [ans for ans in all_answers_json if ans['image'].split('/')[0] == dataset_name]
    if not filtered:
        print(f"No records found for dataset: {dataset_name}")
        return

    # Get all subcategories (classes/objects)
    subcategories = sorted(set(ans['image'].split('/')[1] for ans in filtered))
    question_types = sorted(set(
        ans['question_type'] if ans['question_type'] not in ["Object Structure", "Object Details"] else "Object Analysis"
        for ans in filtered
    ))

    # Initialize accuracy table
    accuracy_df = pd.DataFrame(index=subcategories, columns=question_types)
    for subcat in subcategories:
        for qtype in question_types:
            subcat_answers = [
                ans for ans in filtered
                if ans['image'].split('/')[1] == subcat and
                   (ans['question_type'] if ans['question_type'] not in ["Object Structure", "Object Details"] else "Object Analysis") == qtype
            ]
            total = len(subcat_answers)
            correct = sum(ans['gpt_answer'] == ans['correct_answer'] for ans in subcat_answers)
            acc = correct / total * 100 if total > 0 else 0
            accuracy_df.at[subcat, qtype] = acc

    # Add average column
    accuracy_df['Average'] = accuracy_df.mean(axis=1)

    # Plot heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(accuracy_df.astype(float), annot=True, cmap='coolwarm', fmt=".1f", vmax=100, vmin=25)
    plt.title(f'Accuracy by Subcategory in {dataset_name}')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

    # Save accuracy table
    csv_path = answers_json_path.replace('.json', f'_{dataset_name}_subcategory_accuracy.csv')
    accuracy_df.to_csv(csv_path)
    print(f"Accuracy table saved to: {csv_path}")

    return accuracy_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers_json_path', type=str, default='/apdcephfs/private_kokijiang/project/MLLM-AD/Transformers/result/answers_1_shot_Qwen2-VL-7B-Instruct.json')
    parser.add_argument('--normal_flag', type=str, default='good')
    args = parser.parse_args()
    # caculate_accuracy(args.answers_json_path, args.normal_flag)
    # 判断args.answers_json_path是文件夹还是json文件
    if os.path.isdir(args.answers_json_path):
        for root, dirs, files in os.walk(args.answers_json_path):
            for file in files:
                if file.endswith('.json'):
                    caculate_accuracy_mmad(os.path.join(root, file), show_overkill_miss=True)
    else:
        caculate_accuracy_mmad(args.answers_json_path, show_overkill_miss=True)

