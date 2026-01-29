from .operators import *
import torch, json, pandas
import random
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None



def safe_open_and_resize(file_path):

        with Image.open(file_path) as img:

            max_size = (1024, 1024)
            

            if img.width > 1024 or img.height > 1024:

                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            

            return img.convert("RGB")

class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True




class IncontextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat

            incontext_metadata_path = args.incontext_metadata_path
            style_metadata_path = args.style_metadata_path
            openpose_metadata_path = args.openpose_metadata_path
            segmentation_metadata_path = args.segmentation_metadata_path
            detection_metadata_path = args.detection_metadata_path
            extension_metadata_path = args.extension_metadata_path
            incontext2_metadata_path = args.incontext2_metadata_path
            select_metadata_path = args.select_metadata_path
            derain_metadata_path = args.derain_metadata_path
            enhance_metadata_path = args.enhance_metadata_path
            gopro_metadata_path = args.gopro_metadata_path
            generate_metadata_path = args.generate_metadata_path
            general_metadata_path = args.general_metadata_path
            general2_metadata_path = args.general2_metadata_path
            omniedit_metadata_path = args.omniedit_metadata_path
            imgcluster_metadata_path = args.imgcluster_metadata_path

            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat
        self.load_from_cache = None

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        incontext_metadata = pd.read_csv(incontext_metadata_path)
        incontext2_metadata = pd.read_csv(incontext2_metadata_path)
        openpose_metadata = pd.read_csv(openpose_metadata_path)

        self.style_metadata = pd.read_csv(style_metadata_path)
        if detection_metadata_path is not None:
            self.detection_metadata = pd.read_csv(detection_metadata_path)
        

        segmentation_metadata = pd.read_csv(segmentation_metadata_path)
        extension_metadata = pd.read_csv(extension_metadata_path)
        select_metadataa = pd.read_csv(select_metadata_path)
        
        derain_metadata = pd.read_csv(derain_metadata_path)
        self.enhance_metadata = pd.read_csv(enhance_metadata_path)
        gopro_metadata = pd.read_csv(gopro_metadata_path)
        generate_metadata = pd.read_csv(generate_metadata_path)
        imgcluster_metadata = pd.read_csv(imgcluster_metadata_path)

        self.general_metadata = pd.read_csv(general_metadata_path)
        self.general2_metadata = pd.read_csv(general2_metadata_path)
        self.omniedit_metadata = pd.read_csv(omniedit_metadata_path)

        
        self.incontext_data = [incontext_metadata.iloc[i].to_dict() for i in range(len(incontext_metadata))]
        self.incontext2_data = [incontext2_metadata.iloc[i].to_dict() for i in range(len(incontext2_metadata))]
        self.openpose_data = [openpose_metadata.iloc[i].to_dict() for i in range(len(openpose_metadata))]
        self.segmentation_data = [segmentation_metadata.iloc[i].to_dict() for i in range(len(segmentation_metadata))]
        self.extension_data = [extension_metadata.iloc[i].to_dict() for i in range(len(extension_metadata))]
        self.select_data = [select_metadataa.iloc[i].to_dict() for i in range(len(select_metadataa))]
        
        self.derain_data = [derain_metadata.iloc[i].to_dict() for i in range(len(derain_metadata))]
        # self.enhance_data = [enhance_metadata.iloc[i].to_dict() for i in range(len(enhance_metadata))]
        self.gopro_data = [gopro_metadata.iloc[i].to_dict() for i in range(len(gopro_metadata))]
        self.generate_data = [generate_metadata.iloc[i].to_dict() for i in range(len(generate_metadata))]
        self.imgcluster_data = [imgcluster_metadata.iloc[i].to_dict() for i in range(len(imgcluster_metadata))]

        name_counts = self.style_metadata['etype'].value_counts()
        self.valid_names = name_counts[name_counts >= 2].index.tolist()

        obj_counts = self.detection_metadata['object'].value_counts()
        self.valid_objects = obj_counts[obj_counts >= 2].index.tolist()

        general_counts = self.general_metadata['etype'].value_counts()
        self.valid_etypes_1 = general_counts[general_counts >= 2].index.tolist()

        general2_counts = self.general2_metadata['etype'].value_counts()
        self.valid_etypes_2 = general2_counts[general2_counts >= 2].index.tolist()

        omniedit_counts = self.omniedit_metadata['etype'].value_counts()
        self.valid_etypes_3 = omniedit_counts[omniedit_counts >= 2].index.tolist()

        enhance_counts = self.enhance_metadata['etype'].value_counts()
        self.valid_etypes_4 = enhance_counts[enhance_counts >= 2].index.tolist()

        # self.style_data = [metadata.iloc[i].to_dict() for i in range(len(style_metadata))]

    def get_style_data(self):
        name_id = torch.randint(0, len(self.valid_names), (1,))[0]
        selected_name = self.valid_names[name_id]
        group = self.style_metadata[self.style_metadata['etype'] == selected_name]
        # sampled_rows = group.sample(n=2, random_state=random.randint(1, 1000)).reset_index(drop=True)
        indices = torch.randperm(len(group))[:2].numpy()
        sampled_rows = group.iloc[indices].reset_index(drop=True)
        datas = [
            sampled_rows.iloc[0].to_dict(),
            sampled_rows.iloc[1].to_dict()
        ]
        return datas

    def get_detection_data(self):
        obj_id = torch.randint(0, len(self.valid_objects), (1,))[0]
        selected_obj = self.valid_objects[obj_id]
        group = self.detection_metadata[self.detection_metadata['object'] == selected_obj]
        indices = torch.randperm(len(group))[:2].numpy()
        sampled_rows = group.iloc[indices].reset_index(drop=True)
        datas = [
            sampled_rows.iloc[0].to_dict(),
            sampled_rows.iloc[1].to_dict()
        ]
        return datas

    def get_general1_data(self):
        etype_id = torch.randint(0, len(self.valid_etypes_1), (1,))[0]
        selected_etype = self.valid_etypes_1[etype_id]
        group = self.general_metadata[self.general_metadata['etype'] == selected_etype]
        indices = torch.randperm(len(group))[:2].numpy()
        sampled_rows = group.iloc[indices].reset_index(drop=True)
        datas = [
            sampled_rows.iloc[0].to_dict(),
            sampled_rows.iloc[1].to_dict()
        ]
        return datas
    
    def get_general2_data(self):
        etype_id = torch.randint(0, len(self.valid_etypes_2), (1,))[0]
        selected_etype = self.valid_etypes_2[etype_id]
        group = self.general2_metadata[self.general2_metadata['etype'] == selected_etype]
        indices = torch.randperm(len(group))[:2].numpy()
        sampled_rows = group.iloc[indices].reset_index(drop=True)
        datas = [
            sampled_rows.iloc[0].to_dict(),
            sampled_rows.iloc[1].to_dict()
        ]
        return datas
    
    def get_omniedit_data(self):
        etype_id = torch.randint(0, len(self.valid_etypes_3), (1,))[0]
        selected_etype = self.valid_etypes_3[etype_id]
        group = self.omniedit_metadata[self.omniedit_metadata['etype'] == selected_etype]
        indices = torch.randperm(len(group))[:2].numpy()
        sampled_rows = group.iloc[indices].reset_index(drop=True)
        datas = [
            sampled_rows.iloc[0].to_dict(),
            sampled_rows.iloc[1].to_dict()
        ]
        return datas
    
    def get_enhance_data(self):
        etype_id = torch.randint(0, len(self.valid_etypes_4), (1,))[0]
        selected_etype = self.valid_etypes_4[etype_id]
        group = self.enhance_metadata[self.enhance_metadata['etype'] == selected_etype]
        indices = torch.randperm(len(group))[:2].numpy()
        sampled_rows = group.iloc[indices].reset_index(drop=True)
        datas = [
            sampled_rows.iloc[0].to_dict(),
            sampled_rows.iloc[1].to_dict()
        ]
        return datas

    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
     

    
    
    def load_image(self, file_path):
        # image = Image.open(file_path).convert("RGB")
        image = safe_open_and_resize(file_path)
        h,w = self.height, self.width
        if h is not None and w is not None:
            image = image.resize((h,w))
        else:
            image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    def load_image2(self, file_path):
        # image = Image.open(file_path).convert("RGB")
        image = safe_open_and_resize(file_path)
        h,w = self.height, self.width
        if h is not None and w is not None:
            image = image.resize((h,w))
        else:
            width, height = image.size
            image = self.crop_and_resize(image, height, width)
        return image
    
    def load_data(self, file_path):
        file_path = file_path.replace('/linear/', '/lineart/').replace('/linear_', '/lineart_')
        return self.load_image(file_path)

    def load_data2(self, file_path):
        file_path = file_path.replace('/linear/', '/lineart/').replace('/linear_', '/lineart_')
        return self.load_image2(file_path)


    def __getitem__(self, data_id):

        addtask2 = ['generate'] * 2

        restore_task = ['gray','compress'] * 2
        addtask = ['select','segmentation', 'detection','normal','openpose','derain', 'enhance'] * 3
        # tasks1 = [ 'canny', 'depth', 'linear_anime', 'linear', 'softedge', 'watermark'] * 1
        tasks1 = ['depth', 'linear', 'softedge', 'watermark','style'] * 2
        tasks = tasks1 + addtask + addtask2 + restore_task




        random.shuffle(tasks)
        task_id = torch.randint(0, len(tasks), (1,))[0]
        data = {'prompt':' '}
        if tasks[task_id] == 'style':
            style_datas = self.get_style_data()
            data['image'] = self.load_data(style_datas[0]['style_image_path'])
            edit_image_0 = self.load_data(style_datas[0]['origin_image_path'])
            in_context_0 = self.load_data(style_datas[1]['origin_image_path'])
            in_context_1 = self.load_data(style_datas[1]['style_image_path'])
            data['edit_image'] = [in_context_0,in_context_1,edit_image_0]
        
        elif tasks[task_id] == 'detection':
            detection_datas = self.get_detection_data()
            detype = 'mask'
            if torch.randint(0, 10000, (1,))[0] % 2 == 0:
                detype = 'box'
            
            data['image'] = self.load_data(detection_datas[0][f'{detype}_path'])
            edit_image_0 = self.load_data(detection_datas[0]['origin_image_path'])

            in_context_0 = self.load_data(detection_datas[1]['origin_image_path'])
            in_context_1 = self.load_data(detection_datas[1][f'{detype}_path'])
            data['edit_image'] = [in_context_0,in_context_1,edit_image_0]
        elif tasks[task_id] in ['general','general2','omniedit']:
            task = tasks[task_id]
            if task == 'general':
                gdatas = self.get_general1_data()
            elif task == 'general2':
                gdatas = self.get_general2_data()
            elif task == 'omniedit':
                gdatas = self.get_omniedit_data()
            data_1, data_2 = gdatas[0], gdatas[1]

            data['image'] = self.load_data(data_1[f'{task}_image_path'])
            edit_image_0 = self.load_data(data_1['origin_image_path'])
            in_context_0 = self.load_data(data_2['origin_image_path'])
            in_context_1 = self.load_data(data_2[f'{task}_image_path'])
            data['edit_image'] = [in_context_0,in_context_1,edit_image_0]
        
        elif tasks[task_id] == "enhance":
            task = tasks[task_id]
            gdatas = self.get_enhance_data()
            data_1, data_2 = gdatas[0], gdatas[1]

            data['image'] = self.load_data(data_1[f'{task}_image_path'])
            edit_image_0 = self.load_data(data_1['origin_image_path'])
            in_context_0 = self.load_data(data_2['origin_image_path'])
            in_context_1 = self.load_data(data_2[f'{task}_image_path'])

            data['edit_image'] = [in_context_0,in_context_1,edit_image_0]


        elif tasks[task_id] in ['generate','imgcluster']:
            if tasks[task_id] == 'generate':
                a, b = torch.randperm(len(self.generate_data))[:2].tolist()
                data_1 = self.generate_data[a]
            elif tasks[task_id] == 'imgcluster':
                a, b = torch.randperm(len(self.imgcluster_data))[:2].tolist()
                data_1 = self.imgcluster_data[a]
            if True:
                data['image'] = self.load_data(data_1['image'])
                edit_image = self.load_data(data_1['edit_image'])
                in_context_0 = self.load_data(data_1['incontext_0'])
                in_context_1 = self.load_data(data_1['incontext_1'])
                data['edit_image'] = [in_context_0,in_context_1,edit_image]

            else:
                data['image'] = self.load_data(data_1['incontext_1'])
                edit_image = self.load_data(data_1['incontext_0'])
                in_context_0 = self.load_data(data_1['edit_image'])
                in_context_1 = self.load_data(data_1['image'])
                data['edit_image'] = [in_context_0,in_context_1,edit_image]

        else:
            task = tasks[task_id]
            if task == 'openpose':
                a, b = torch.randperm(len(self.openpose_data))[:2].tolist()
                data_1, data_2 = self.openpose_data[a], self.openpose_data[b]
            elif task == 'segmentation':
                a, b = torch.randperm(len(self.segmentation_data))[:2].tolist()
                data_1, data_2 = self.segmentation_data[a], self.segmentation_data[b]
            elif task == 'extension':
                a, b = torch.randperm(len(self.extension_data))[:2].tolist()
                data_1, data_2 = self.extension_data[a], self.extension_data[b]
            elif task == 'select':
                a, b = torch.randperm(len(self.select_data))[:2].tolist()
                data_1, data_2 = self.select_data[a], self.select_data[b]
            elif task == 'derain':
                a, b = torch.randperm(len(self.derain_data))[:2].tolist()
                data_1, data_2 = self.derain_data[a], self.derain_data[b]
            # elif task == 'enhance':
            #     a, b = torch.randperm(len(self.enhance_data))[:2].tolist()
            #     data_1, data_2 = self.enhance_data[a], self.enhance_data[b]
            elif task == 'gopro':
                a, b = torch.randperm(len(self.gopro_data))[:2].tolist()
                data_1, data_2 = self.gopro_data[a], self.gopro_data[b]
            elif task in ['gray','depth','normal','watermark']:
                a, b = torch.randperm(len(self.incontext2_data))[:2].tolist()
                data_1, data_2 = self.incontext2_data[a], self.incontext2_data[b]
            else:
                a, b = torch.randperm(len(self.incontext_data))[:2].tolist()
                data_1, data_2 = self.incontext_data[a], self.incontext_data[b]
            
            if task in ['extension','select','derain','enhance','gopro']:
                data['image'] = self.load_data(data_1[f'{task}_image_path'])
                edit_image_0 = self.load_data(data_1['origin_image_path'])
                in_context_0 = self.load_data(data_2['origin_image_path'])
                in_context_1 = self.load_data(data_2[f'{task}_image_path'])
                data['edit_image'] = [in_context_0,in_context_1,edit_image_0]
            elif task in ['watermark','compress']: # 这两个任务只有反向任务
                data['image'] = self.load_data(data_1['origin_image_path'])
                edit_image_0 = self.load_data(data_1[f'{task}_image_path'])
                in_context_0 = self.load_data(data_2[f'{task}_image_path'])
                in_context_1 = self.load_data(data_2['origin_image_path'])
                data['edit_image'] = [in_context_0,in_context_1,edit_image_0]
            else:
                if a % 2 == 0:
                    data['image'] = self.load_data(data_1[f'{task}_image_path'])
                    edit_image_0 = self.load_data(data_1['origin_image_path'])
                    in_context_0 = self.load_data(data_2['origin_image_path'])
                    in_context_1 = self.load_data(data_2[f'{task}_image_path'])
                    data['edit_image'] = [in_context_0,in_context_1,edit_image_0]
                else:
                    data['image'] = self.load_data(data_1['origin_image_path'])
                    edit_image_0 = self.load_data(data_1[f'{task}_image_path'])
                    in_context_0 = self.load_data(data_2[f'{task}_image_path'])
                    in_context_1 = self.load_data(data_2['origin_image_path'])
                    data['edit_image'] = [in_context_0,in_context_1,edit_image_0]
        return data
    

    def __len__(self):
        return 3200
        # return len(self.data) * self.repeat

