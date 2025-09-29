import pandas as pd
import random
import ast
import cv2
import os

random.seed(42)

class IssueImageGenerator:
    def __init__(self, baseline_csv_path, output_folder, issue_threshold_dict=None, save_added_issue_images=False):
        if issue_threshold_dict is None:
            issue_threshold_dict = {
                "wrong_class": 1,
                "smaller_bbox": 1,
                "larger_bbox": 1,
                "shift_bbox": 1,
                "diag_shift": 1,
                "background": 1,
            }
            
        self.save_added_issue_images = save_added_issue_images
        self.baseline_csv_path = baseline_csv_path
        self.output_folder = output_folder
        self.issue_threshold_dict = issue_threshold_dict
        self.baseline_df = pd.read_csv(baseline_csv_path)
        self.class_issue_count = self.class_issue_count()
        self.total_images = len(self.baseline_df)
        self.classes, self.total_roi, self.gt_class_distribution = self._get_gt_dataset_count()
        
        total_threshold = sum(issue_threshold_dict.values())/100
        tolal_images_to_be_modified = int(self.total_roi * total_threshold)
        if tolal_images_to_be_modified > self.total_images:
            raise ValueError('total_threshold value is too high: {}(more than total images: {}). Please reduce it')
 

    # def get_stats(self):
    #     '''Return dataset stats'''
    #     stats ={
    #         'total_images':self.total_images,
    #         'total_classes':len(self.classes),
    #         'total_bbox':self.total_roi,
    #         'gt_class_distribution': self.gt_class_distribution
    #     }
    #     return stats
    
    def class_issue_count(self):
        df = self.baseline_df
        issue_threshold_dict = self.issue_threshold_dict
        
        class_dict = {}
        a = 0
        for index, row in df.iterrows():
            ann = ast.literal_eval(row['annotation'])
            if not ann:
                continue
            for an in ann:
                class_n = an['class_name']
                class_dict[class_n] = class_dict.get(class_n, 0) + 1   
                a = a + 1
        # class_dict['all_class_count'] = a
        
        data = {}
        for k, v in class_dict.items():
            d = {}
            for issue_type, issue_threshold in issue_threshold_dict.items():
                d[issue_type] = int(v * issue_threshold/100)
            data[k] = d
        return data
        

    def _get_gt_dataset_count(self):
        '''Calculate classes, bounding boxes count and class distribution for groundtruth'''
        classes = set()
        gt_class_distribution={}
        total_roi = 0
        for index, row in self.baseline_df.iterrows():
            ann = ast.literal_eval(row['annotation'])
            for c in ann:
                total_roi += 1
                classes.add(c['class_name'])
                if c['class_name'] not in gt_class_distribution:
                    gt_class_distribution[c['class_name']]=0
                gt_class_distribution[c['class_name']]+=1
        return list(classes), total_roi, gt_class_distribution
	
    def save_issue_images(self, df, save_path):
        for index, row in df.iterrows():
            issue_class_id = row['is_issue_added']
            if issue_class_id == -1:
                continue
            issue_type = row['issue_type']
            filepath = row['filepath']
            boxes = ast.literal_eval(row['annotation'])
            image = cv2.imread(filepath)
            for box in boxes:
                if box['class_id'] == issue_class_id:
                    x, y, w, h = box['bbox']
     
                    x1 = int((x - w / 2) * image.shape[1])
                    y1 = int((y - h / 2) * image.shape[0])
                    x2 = int((x + w / 2) * image.shape[1])
                    y2 = int((y + h / 2) * image.shape[0])
                    
                    thickness = 2
                    color = (0, 0, 255)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                    class_name = str(box['class_name'])
                    cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    name = os.path.basename(filepath)
                    save_dir = os.path.join(save_path, issue_type)
                    os.makedirs(save_dir, exist_ok=True)
                    filename = f"{issue_class_id}_{class_name}_{name}"
                    save_filepath = os.path.join(save_dir, filename)
                    cv2.imwrite(save_filepath, image)
                    break

    def run(self):
        # save a copy of baseline dataframe
        data_type = self.baseline_csv_path.split('/')[-1].split('.')[0]
        os.makedirs(self.output_folder , exist_ok=True)
        self.baseline_df.to_csv(f'{self.output_folder}/{data_type}_baseline_copy.csv', index=False)
        print('**** a copy of baseline csv saved to: ', f'{self.output_folder}/{data_type}_baseline_copy.csv')
        
        # add issue bboxes in the baseline dataframe
        self.wrong_class(self.issue_threshold_dict['wrong_class'])
        self.smaller_bbox(self.issue_threshold_dict['smaller_bbox'], smaller_range = (0.3, 0.7))
        self.larger_bbox(self.issue_threshold_dict['larger_bbox'], larger_range = (1.8, 2.2))
        self.shift_bbox(self.issue_threshold_dict['shift_bbox'], shift_range = (0.2, 0.4))
        self.background(self.issue_threshold_dict['background'], box_size_range = (0.1, 0.75))
        self.diag_shift_bbox(self.issue_threshold_dict['diag_shift'], diag_shift_range = (0.2, 0.4))

        # save issue images
        if self.save_added_issue_images:
            df = self.baseline_df
            save_path =  self.output_folder+'/issue_images/'
            self.save_issue_images(df, save_path=save_path)
            print('**** issue images saved to: ', save_path)   
        
        # save baseline with issues dataframe
        self.baseline_df.to_csv(f'{self.output_folder}/{data_type}_with_issues.csv', index=False)
        print('**** baseline with issues csv saved to: ', f'{self.output_folder}/{data_type}_with_issues.csv')
        
        # save the stats
        self.save_issue_stats_csv(original_csv_path = f'{self.output_folder}/{data_type}_baseline_copy.csv')
        
    
    def save_issue_stats_csv(self, original_csv_path):
        # get issue stats
        df = self.baseline_df
        class_dict = {}
        for index, row in df.iterrows():
            ann = ast.literal_eval(row['annotation'])
            if not ann:
                continue
            
            is_issue_added = row['is_issue_added']
            if is_issue_added == -1:
                continue
            
            for an in ann:
                class_id = an['class_id']
                if class_id == is_issue_added:
                    class_n = an['class_name']
                    issue_type = row['issue_type']
                    if class_n not in class_dict:
                        class_dict[class_n] = {}
                    class_dict[class_n][issue_type] = class_dict[class_n].get(issue_type, 0) + 1
                    break
            
        dfr = pd.DataFrame(class_dict)
        dfr = dfr.transpose() 
        dfr['wrong_class'] = dfr['smaller_bbox']
        dfr.loc['all_class_count'] = dfr.sum()
        dfr['all_issue_count'] = dfr.sum(axis=1, skipna=True)
        
        
        # get original stats
        dff = pd.read_csv(original_csv_path)
        class_dict1 = {}
        a = 0
        for index, row in dff.iterrows():
            ann = ast.literal_eval(row['annotation'])
            if not ann:
                continue
            
            for an in ann:
                class_n = an['class_name']
                class_dict1[class_n] = class_dict1.get(class_n, 0) + 1   
                a = a + 1
        class_dict1['all_class_count'] = a
        
        
        # combine all the stats
        dfr['original_count'] = [class_dict1[idx] for idx in dfr.index]
        dfr['%_issue_added_from_each_class'] = round((dfr['all_issue_count'] / dfr['original_count']) * 100, 1)
        
        # save the stats
        save_path = f"{self.output_folder}/dataset_all_stats.csv"
        dfr.to_csv(save_path)
        print('**** dataset stats saved to: ', save_path)
        
        
					
    def calculate_overlap(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
        area_bbox1 = w1 * h1
        area_bbox2 = w2 * h2
        union_area = area_bbox1 + area_bbox2 - intersection_area
        overlap = intersection_area / union_area if union_area > 0 else 0
        return overlap
    
    def modify_bbox(self, box, image):
        x, y, w, h = box['bbox']
        x1 = max(int((x - w / 2) * image.shape[1]), 0)
        y1 = max(int((y - h / 2) * image.shape[0]), 0)
        x2 = min(int((x + w / 2) * image.shape[1]), image.shape[1])
        y2 = min(int((y + h / 2) * image.shape[0]), image.shape[0])
        
        w = (x2 - x1) / image.shape[1]
        h = (y2 - y1) / image.shape[0]
        x = (x1 + x2) / (2 * image.shape[1])
        y = (y1 + y2) / (2 * image.shape[0])
        
        return [x, y, w, h]
    


    def wrong_class(self, threshold):
        issue_type = 'wrong_class'
        if threshold == 0:
            return 
        
        threshold = threshold/100
        classes = self.classes
        sample_size = int(self.total_roi * threshold)
        df = self.baseline_df
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        
        n = 0
        # for each selected image, randomly select one bbox and change its class name
        for index, row in df.iterrows():
            # check if the image is not already modified
            if row['is_issue_added'] != -1:
                continue
            boxes = ast.literal_eval(row['annotation'])
            # skip if there are no objects
            if len(boxes) == 0:
                continue
            
            # chose a box
            random.shuffle(boxes)
            box = None
            for BOX in boxes:
                chosen_class_name = BOX['class_name']
                if self.class_issue_count[chosen_class_name][issue_type] > 0:
                    box = BOX
                    self.class_issue_count[chosen_class_name][issue_type] -= 1
                    break
            if box is None:
                continue
            
            
            chosen_class_id = box['class_id']  
            chosen_class_name = box['class_name']
            # chose a different class name
            dif_classes = classes.copy()
            dif_classes.remove(chosen_class_name)
            new_class_name = random.choice(dif_classes)
            df.at[index, 'is_issue_added'] = chosen_class_id
            df.at[index, 'issue_type'] = issue_type
            # df1.at[index, 'old_label'] = chosen_class_name
            box['class_name'] = new_class_name 
            # image = cv2.imread(row['filepath'])
            # box['bbox'] = self.modify_bbox(box, image)
            df.at[index, 'annotation'] = str(boxes)
            
            n += 1
            if n == sample_size:
                break
            
        self.baseline_df = df

	
    def smaller_bbox(self, threshold, smaller_range = (0.3, 0.7)):
        '''Reduce bbox dimensions'''
        issue_type = 'smaller_bbox'
        if threshold == 0:
            return 
        
        threshold = threshold/100
        sample_size = int(self.total_roi * threshold)
        df = self.baseline_df
        df = df.sample(frac=1, random_state=2).reset_index(drop=True)
        
        n = 0
        # for each selected image, randomly select one bbox and reduce its size
        a, b = smaller_range
        for index, row in df.iterrows():
            # check if the image is not already modified
            if row['is_issue_added'] != -1:
                continue
            boxes = ast.literal_eval(row['annotation'])
            if len(boxes) == 0:
                continue
            
            
            # chose a box
            random.shuffle(boxes)
            box = None
            for BOX in boxes:
                chosen_class_name = BOX['class_name']
                if self.class_issue_count[chosen_class_name][issue_type] > 0:
                    box = BOX
                    self.class_issue_count[chosen_class_name][issue_type] -= 1
                    break
            if box is None:
                continue
            
            
            chosen_class_id = box['class_id']  
            df.at[index, 'is_issue_added'] = chosen_class_id
            df.at[index, 'issue_type'] = issue_type
            box['bbox'] = [box['bbox'][0], box['bbox'][1], box['bbox'][2]*random.uniform(a, b), box['bbox'][3]*random.uniform(a, b)]
            # image = cv2.imread(row['filepath'])
            # box['bbox'] = self.modify_bbox(box, image)
            df.at[index, 'annotation'] = str(boxes) 
            
            n += 1
            if n == sample_size:
                break
        
        self.baseline_df = df


    def larger_bbox(self, threshold, larger_range = (1.8, 2.2)):
        '''Increase bbox dimensions'''
        issue_type = 'larger_bbox'
        if threshold == 0:
            return 
        
        threshold = threshold/100
        sample_size = int(self.total_roi * threshold)
        df = self.baseline_df
        df = df.sample(frac=1, random_state=3).reset_index(drop=True)

        n = 0
        # for each selected image, randomly select one bbox and increase its size
        a, b = larger_range
        for index, row in df.iterrows():
            # check if the image is not already modified
            if row['is_issue_added'] != -1:
                continue
            boxes = ast.literal_eval(row['annotation'])
            if len(boxes) == 0:
                continue
            
            # chose a box
            random.shuffle(boxes)
            box = None
            for BOX in boxes:
                chosen_class_name = BOX['class_name']
                if self.class_issue_count[chosen_class_name][issue_type] > 0:
                    box = BOX
                    self.class_issue_count[chosen_class_name][issue_type] -= 1
                    break
            if box is None:
                continue
            
            chosen_class_id = box['class_id']  
            df.at[index, 'is_issue_added'] = chosen_class_id
            df.at[index, 'issue_type'] = issue_type
            box['bbox'] = [box['bbox'][0], box['bbox'][1], box['bbox'][2]*random.uniform(a, b), box['bbox'][3]*random.uniform(a, b)]
            image = cv2.imread(row['filepath'])
            box['bbox'] = self.modify_bbox(box, image)
            df.at[index, 'annotation'] = str(boxes) 
            
            n += 1
            if n == sample_size:
                break
        
        self.baseline_df = df


    def shift_bbox(self, threshold, shift_range = (0.3, 0.5)):
        '''shift bbox left,right,top,bottom'''
        issue_type = 'shift_bbox'
        if threshold == 0:
            return 
        
        threshold = threshold/100
        sample_size = int(self.total_roi * threshold)
        df = self.baseline_df
        df = df.sample(frac=1, random_state=4).reset_index(drop=True)
        
        
        n = 0
        # for each selected image, randomly select one bbox and one direction to shift
        for index, row in df.iterrows():
            # check if the image is not already modified
            if row['is_issue_added'] != -1:
                continue
            boxes = ast.literal_eval(row['annotation'])
            if len(boxes) == 0:
                continue
            
            # chose a box
            random.shuffle(boxes)
            box = None
            for BOX in boxes:
                chosen_class_name = BOX['class_name']
                if self.class_issue_count[chosen_class_name][issue_type] > 0:
                    box = BOX
                    self.class_issue_count[chosen_class_name][issue_type] -= 1
                    break
            if box is None:
                continue
            
            chosen_class_id = box['class_id']  
            df.at[index, 'is_issue_added'] = chosen_class_id
            df.at[index, 'issue_type'] = issue_type
            # Randomly select one direction to shift
            direction = random.choice(['left', 'right', 'up', 'down'])
            a, b = shift_range
            shift_amount = random.uniform(a, b) 
            # Modify the selected bounding box based on the direction
            if direction == 'left':
                box['bbox'][0] -= shift_amount * box['bbox'][2]  # Shift left
            elif direction == 'right':
                box['bbox'][0] += shift_amount * box['bbox'][2]  # Shift right
            elif direction == 'up':
                box['bbox'][1] -= shift_amount * box['bbox'][3]  # Shift up
            elif direction == 'down':
                box['bbox'][1] += shift_amount * box['bbox'][3]  # Shift down
            # Update the annotation in the dataframe
            image = cv2.imread(row['filepath'])
            box['bbox'] = self.modify_bbox(box, image)
            df.at[index, 'annotation'] = str(boxes)
            
            n += 1
            if n == sample_size:
                break
        
        self.baseline_df = df
    
    

    def background(self, threshold, box_size_range = (0.1, 0.75), overlap_threshold = 0, tolerance = 5):
        '''add background as a class randomly take one class'''
        issue_type = 'background'
        if threshold == 0:
            return 
        
        threshold = threshold/100
        sample_size = int(self.total_roi * threshold)
        df = self.baseline_df
        df = df.sample(frac=1, random_state=5).reset_index(drop=True)
        
        n = 0
        # for each selected image, randomly create a bbox which are not already present
        a, b = box_size_range
        for index, row in df.iterrows():
            # check if the image is not already modified
            if row['is_issue_added'] != -1:
                continue
            boxes = ast.literal_eval(row['annotation'])
            if len(boxes) == 0:
                continue
            filepath = row['filepath']
            bboxes = []
            for box in boxes:
                bbox = box['bbox']
                bboxes.append(bbox)
            
            
            # Artificially create a bbox which are not already present
            # Ensure that the randomly created bbox does not overlap with existing bboxes
            new_bbox = []
            tol = 0
            while True:
                new_bbox = [random.uniform(a, b), random.uniform(a, b), random.uniform(a, b), random.uniform(a, b)]
                bbox_overlap = False
                for existing_bbox in bboxes:
                    if self.calculate_overlap(existing_bbox, new_bbox) > overlap_threshold:
                        bbox_overlap = True
                        break
                if not bbox_overlap:
                    break
                tol += 1
                if tol == tolerance:
                    break
            if tol == tolerance:
                    continue
                
            
            #take a random class from the classes present in the image
            random.shuffle(boxes)
            box = None
            for BOX in boxes:
                chosen_class_name = BOX['class_name']
                if self.class_issue_count[chosen_class_name][issue_type] > 0:
                    box = BOX
                    self.class_issue_count[chosen_class_name][issue_type] -= 1
                    break
            if box is None:
                continue

            chosen_class_id = box['class_id']
            box['bbox'] = new_bbox 
            df.at[index, 'is_issue_added'] = chosen_class_id
            df.at[index, 'issue_type'] = issue_type
            image = cv2.imread(row['filepath'])
            box['bbox'] = self.modify_bbox(box, image)
            df.at[index, 'annotation'] = str(boxes)
            
            n += 1
            if n == sample_size:
                break    
       
        self.baseline_df = df




    def diag_shift_bbox(self, threshold, diag_shift_range = (0.3, 0.5)):
        '''diagonal shift bbox'''
        issue_type = 'diag_shift'
        if threshold == 0:
            return 
        
        threshold = threshold/100
        sample_size = int(self.total_roi * threshold)
        df = self.baseline_df
        df = df.sample(frac=1, random_state=6).reset_index(drop=True)
    
        n = 0
        # for each selected image, randomly select one bbox and one diag direction to shift
        for index, row in df.iterrows():
            # check if the image is not already modified
            if row['is_issue_added'] != -1:
                continue
            boxes = ast.literal_eval(row['annotation'])
            if len(boxes) == 0:
                continue
            
            # chose a box
            random.shuffle(boxes)
            box = None
            for BOX in boxes:
                chosen_class_name = BOX['class_name']
                if self.class_issue_count[chosen_class_name][issue_type] > 0:
                    box = BOX
                    self.class_issue_count[chosen_class_name][issue_type] -= 1
                    break
            if box is None:
                continue
            
            chosen_class_id = box['class_id']  
            df.at[index, 'is_issue_added'] = chosen_class_id
            df.at[index, 'issue_type'] = issue_type
            # Randomly select one direction to shift
            direction = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
            a, b = diag_shift_range
            shift_amount = random.uniform(a, b) 
            # Modify the selected bounding box based on the direction
            if direction == 'top-left':
                box['bbox'][0] -= shift_amount * box['bbox'][2]  # Shift top-left
                box['bbox'][1] += shift_amount * box['bbox'][3]
            elif direction == 'top-right':
                box['bbox'][0] += shift_amount * box['bbox'][2]  # Shift top-right
                box['bbox'][1] += shift_amount * box['bbox'][3]
            elif direction == 'bottom-left':
                box['bbox'][0] -= shift_amount * box['bbox'][2]  # Shift bottom-left
                box['bbox'][1] -= shift_amount * box['bbox'][3]
            elif direction == 'bottom-right':
                box['bbox'][0] += shift_amount * box['bbox'][2]  # Shift bottom-right
                box['bbox'][1] -= shift_amount * box['bbox'][3]
            # Update the annotation in the dataframe
            image = cv2.imread(row['filepath'])
            box['bbox'] = self.modify_bbox(box, image)
            df.at[index, 'annotation'] = str(boxes)
            
            n += 1
            if n == sample_size:
                break
        
        self.baseline_df = df




if __name__ == '__main__':
    baseline_file = '/home/ubuntu/LQC/dataset/mscoco/mscoco_val_baseline.csv'
    output_directory = 'testing_lqc_coco_val'
    obj=IssueImageGenerator(baseline_file,output_directory)
    obj.run()
    