import json
import os

# 50 salads
# action_dict = {"cut_tomato": 0, "place_tomato_into_bowl": 1, "cut_cheese": 2, "place_cheese_into_bowl": 3,
#                "cut_lettuce": 4, "place_lettuce_into_bowl": 5, "add_salt": 6, "add_vinegar": 7,
#                "add_oil": 8, "add_pepper": 9, "mix_dressing": 10, "peel_cucumber": 11, "cut_cucumber": 12,
#                "place_cucumber_into_bowl": 13, "add_dressing": 14, "mix_ingredients": 15, "serve_salad_onto_plate": 16,
#                "action_start": 17, "action_end": 18}

action_dict = {"background": 0, "take_bread": 1, "put_bread": 2, "take_cheese": 3, "open_cheese": 4, "put_cheese": 5, "take_mayonnaise": 6, "shake_mayonnaise": 7, "open_mayonnaise": 8, "pour_mayonnaise": 9, "close_mayonnaise": 10, "put_mayonnaise": 11, "take_mustard": 12, "shake_mustard": 13, "open_mustard": 14, "pour_mustard": 15, "hold_mustard": 16, "close_mustard": 17, "put_mustard": 18, "compress_bread": 19, "take_cup": 20, "put_cup": 21, "take_coffee": 22, "open_coffee": 23, "put_coffee": 24, "take_spoon": 25, "scoop_coffee": 26, "pour_coffee": 27, "close_coffee": 28, "take_water": 29, "open_water": 30, "pour_water": 31, "close_water": 32, "put_water": 33, "stir_spoon": 34, "take_honey": 35, "open_honey": 36, "pour_honey": 37, "close_honey": 38, "put_honey": 39, "take_sugar": 40, "open_sugar": 41, "scoop_sugar": 42, "pour_sugar": 43, "close_sugar": 44, "put_sugar": 45, "take_hotdog": 46, "put_hotdog": 47, "take_ketchup": 48, "shake_ketchup": 49, "open_ketchup": 50, "pour_ketchup": 51, "close_ketchup": 52, "put_ketchup": 53, "fold_bread": 54, "take_jam": 55, "open_jam": 56, "put_jam": 57, "scoop_jam": 58, "spread_jam": 59, "close_jam": 60, "take_peanut": 61, "open_peanut": 62, "scoop_peanut": 63, "spread_peanut": 64, "put_spoon": 65, "close_peanut": 66, "put_peanut": 67, "take_chocolate": 68, "open_chocolate": 69, "shake_chocolate": 70, "pour_chocolate": 71, "put_chocolate": 72, "close_chocolate": 73, "hold_bread": 74, "take_tea": 75, "put_tea": 76, "open_tea": 77, "hold_tea": 78, "shake_tea": 79, "hold_sugar": 80, "shake_honey": 81, "hold_spoon": 82, "shake_cup": 83, "stir_cup": 84}

# 更改参数列表，注意后面还有几处需更改
split = "4"
split_dir = r"/disks/disk0/huangxvfeng/dataset/segmentation/gtea/splits"
annotation_dir = r"/disks/disk0/huangxvfeng/dataset/segmentation/gtea/groundTruth_grain_85"
video_fps = 15.0

training_split_txt_dir = os.path.join(split_dir, "train.split"+split+".bundle")
testing_split_txt_dir = os.path.join(split_dir, "test.split"+split+".bundle")
training_video_list = []
testing_video_list = []

with open(training_split_txt_dir, "r") as f:
    video_list = f.readlines()
    for item in video_list:
        video = item.replace("\n", "")
        training_video_list.append(video)

with open(testing_split_txt_dir, "r") as f:
    video_list = f.readlines()
    for item in video_list:
        video = item.replace("\n", "")
        testing_video_list.append(video)


training_video_list.sort()
testing_video_list.sort()

overall_videos_list = training_video_list + testing_video_list

tot_content = dict([(item.split(".")[0], None) for item in overall_videos_list])

# 训练集
for i, item in enumerate(training_video_list):
    txt_path = os.path.join(annotation_dir, item)
    action_arr = []
    # 此处需更改为第一帧标签
    pre_action = 0
    pre_transtime = 0.0
    with open(txt_path, "r") as f:
        annatations = f.readlines()
        annotation_length = len(annatations)
        line = 0
        for annatation in annatations:
            line += 1
            action = annatation.replace("\n", "")
            action_num = action_dict[action]
            if action_num != pre_action or line >= annotation_length:
                transtime_b = (line-1) / video_fps
                transtime_a = line / video_fps
                sub_action = [0 for i in range(3)]
                sub_action[0] = pre_action
                sub_action[1] = pre_transtime
                sub_action[2] = transtime_b
                if line >= annotation_length:
                    sub_action[2] = transtime_a
                pre_action, pre_transtime = action_num, transtime_a
                action_arr.append(sub_action)

    sub_content = {"subset": "training", "duration": annotation_length/video_fps,
                       "actions": list(action_arr)}

    tot_content[item.split(".")[0]] = sub_content

# 测试集
for i, item in enumerate(testing_video_list):
    txt_path = os.path.join(annotation_dir, item)
    action_arr = []
    # 此处需更改为第一帧标签
    pre_action = 0
    pre_transtime = 0.0
    with open(txt_path, "r") as f:
        annatations = f.readlines()
        annotation_length = len(annatations)
        line = 0
        for annatation in annatations:
            line += 1
            action = annatation.replace("\n", "")
            action_num = action_dict[action]
            if action_num != pre_action or line >= annotation_length:
                transtime_b = (line-1) / video_fps
                transtime_a = line / video_fps
                sub_action = [0 for i in range(3)]
                sub_action[0] = pre_action
                sub_action[1] = pre_transtime
                sub_action[2] = transtime_b
                if line >= annotation_length:
                    sub_action[2] = transtime_a
                pre_action, pre_transtime = action_num, transtime_a
                action_arr.append(sub_action)

    sub_content = {"subset": "testing", "duration": annotation_length/video_fps,
                       "actions": list(action_arr)}

    tot_content[item.split(".")[0]] = sub_content

# 保存地址需更改
save_path = os.path.join("../resources", "gtea_85_"+split+".json")
json.dump(tot_content, open(save_path, 'w'), indent=4)






