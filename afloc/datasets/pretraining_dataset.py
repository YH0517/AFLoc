import re
import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import pickle
import numpy.random as random
import torch
import torch.utils.data as data

from PIL import Image
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
from afloc.constants import *
import PIL
from skimage import exposure


class MultimodalPretrainingDataset(data.Dataset):

    def __init__(self, cfg, split="train", transform=None):
        """
        A PyTorch dataset for multimodal pretraining.

        Args:
            cfg (object): The configuration object.
            split (str, optional): The split of the dataset (default: "train").
            transform (object, optional): Data augmentation for images (default: None).

        """

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        self.df = pd.DataFrame()
        if 'mimic' in self.cfg.data.dataset:
            # read MIMIC csv file
            self.df_mimic = pd.read_csv(MIMIC_MASTER_CSV)
            self.df_mimic = self.df_mimic.loc[:, MIMIC_USED_COLS]
            self.df_mimic[MIMIC_PATH_COL] = self.df_mimic[MIMIC_PATH_COL].apply(
                lambda x: os.path.join(MIMIC_IMG_DIR, x)
            )
            self.df_mimic.rename(columns={
                MIMIC_REPORT_COL: PRETRAIN_REPORT_COL,
                MIMIC_SPLIT_COL: PRETRAIN_SPLIT_COL,
                MIMIC_VIEW_COL: PRETRAIN_VIEW_COL, 
                MIMIC_PATH_COL: PRETRAIN_PATH_COL,
                MIMIC_IMPRESSION_COL: PRETRAIN_IMPRESSION_COL}, inplace=True)

            self.df = pd.concat([self.df, self.df_mimic])
            self.precess_missing_values(mode=cfg.data.missing_value_mode)
                 
        # load studies and study to text mapping
        self.filenames, self.path2sent, self.impression_num = self.load_text_data(split, self.cfg.data.view, "_".join(self.cfg.data.dataset))

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.split = split

    def precess_missing_values(self, mode=None):
        """
        Impute missing values in the df.

        Args:
            mode (int): The mode for handling missing values.

        """
        if mode == 0:
            print("missing_values mode 0")
            for col in TASKS:
                if col in ['Edema', 'Atelectasis']:
                    self.df[col].replace(-1, 1, inplace=True)  
                    self.df[col].replace(-2, 0, inplace=True) 
                else:
                    self.df[col].replace(-1, 0, inplace=True) 
                    self.df[col].replace(-2, 0, inplace=True)
        elif mode == 1:
            print("missing_values mode 1")
            for col in TASKS:
                self.df[col].replace(-1, 1, inplace=True)  
                self.df[col].replace(-2, 0, inplace=True) 
        elif mode == 2:
            print("missing_values mode 2")
            for col in TASKS:
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].replace(-2, 0, inplace=True)
        else:
            raise NotImplementedError
        
        self.df[TASKS] = self.df[TASKS].astype(float)

    def load_text_data(self, split, view, dataset):
        """
        Load text data for a given split, view, and dataset.

        Args:
            split (str): The split of the data (e.g., "train", "val", "test").
            view (str): The view of the data (e.g., "full", "frontal", "lateral").
            dataset (str): The dataset name.

        Returns:
            filenames (list): A list of filenames.
            path2sent (dict): A dictionary mapping paths to sentences.
            impression_num (dict): The number of sentences of each report.

        """
        # get study to captions mapping
        filepath = os.path.join("./", "captions_{}_{}_{}.pickle".format(PICKLE_SUFFIX, dataset, view))
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exist. Creating captions...")
            path2sent, to_remove, impression_num = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove, impression_num], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove, impression_num = pickle.load(f)

        # filter studies to use for current split
        if view == "full":
            filenames = self.df[self.df[PRETRAIN_SPLIT_COL] == split][
                PRETRAIN_PATH_COL
            ]
            print("full view")
        else:
            filenames = self.df[(self.df[PRETRAIN_SPLIT_COL] == split)
                                &(self.df[PRETRAIN_VIEW_COL] == view)][
                PRETRAIN_PATH_COL
            ]
        # remove
        filenames = filenames[~filenames.isin(to_remove)].to_list()

        return filenames, path2sent, impression_num

    def get_caption(self, path):
        """
        Get a caption for the given path.

        Args:
            path (str): The path to the image.

        Returns:
            tokens (torch.Tensor): The tokenized caption.
            x_len (int): The length of the caption.
        
        """
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            if self.cfg.data.text.random_combine is True:
                if len(series_sents) < 2:
                    sent = series_sents[0]
                else:
                    # mode 1
                    if self.cfg.data.text.random_combine_mode == 1:
                        sample_num = len(series_sents)
                        if self.cfg.data.text.random_num is True:
                            sample_num = random.randint(1, len(series_sents))
                        sent_ix = random.choice(range(len(series_sents)), sample_num, replace=False)
                        sent = ". ".join([series_sents[ix] for ix in sent_ix])
                    else:
                        # mode 2
                        findings_num = len(series_sents) - self.impression_num[path]
                        if findings_num == 0:
                            sent = ". ".join(series_sents)
                        else:
                            sample_num = findings_num
                            if self.cfg.data.text.random_num is True:
                                sample_num = random.randint(1, findings_num+1)
                            sent_ix = random.choice(findings_num, sample_num, replace=False)
                            findings = [series_sents[ix] for ix in sent_ix]
                            impression = series_sents[-self.impression_num[path]:] if self.impression_num[path]!=0 else []
                            sent = ". ".join(findings + impression)
            else:
                sent = ". ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def equalize_hist(self, data, use_mask=True):
        """
        Equalizes the histogram of the image.

        Args:
            data (ndarray): The image to be equalized.
            use_mask (bool, optional): Exclude zero values from the equalization process.

        Returns:
            img: The equalized image.

        """
        # Create a mask if use_mask is True
        if use_mask:
            mask = (np.array(data) != 0)
        else:
            mask = None

        img = np.array(data) / 255.
        img = exposure.equalize_hist(img, mask=mask)
        img = (255 * img).astype(np.uint8)

        return img

    def read_img(self, img_path):
        """
        Read an image from the given path and equalize its histogram.

        Args:
            img_path (str): The path to the image.

        Returns:
            img: The equalized image.

        """

        data = PIL.Image.open(img_path)
        img = self.equalize_hist(data, use_mask=True)
        img = PIL.Image.fromarray(img).convert('RGB')
        return img

    def get_imgs(self, img_path, transform=None):
        """
        Get the image from the given path.

        Args:
            img_path (str): The path to the image.
            transform (object, optional): Data augmentation for images (default: None).

        Returns:
            img: The image.
        """
        if self.cfg.data.image.equalize_hist:
            img = self.read_img(img_path)
        else:
            x = cv2.imread(str(img_path), 0)

            # tranform images
            # x = self._resize_img(x, self.cfg.data.image.imsize)
            img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):
        """
        Get data.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            imgs (torch.Tensor): The images.
            caps (list): The captions.
            cap_len (int): The length of the captions.
            key (str): The file name.
            label (torch.Tensor): The label for the item.
        """

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)
        label = torch.tensor(self.df[self.df[PRETRAIN_PATH_COL] == key][MIMIC_TASKS].values[0])

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key, label

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):
        """
        Create a mapping from image path to its corresponding sentences.

        Args:
            df (DataFrame): The data.
            max_word_num (int): The maximum number of words in the sentence.

        Returns:
            path2sent (dict): The mapping from image path to its corresponding sentences.
            to_remove (list): The list of paths with empty reports to remove.
            impression_num (dict): The number of sentences of each report.
        
        """
        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        impression_num = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[PRETRAIN_REPORT_COL]) == str:
                captions += row[PRETRAIN_REPORT_COL]
            impression = ""
            if type(row[PRETRAIN_IMPRESSION_COL]) == str:
                impression += row[PRETRAIN_IMPRESSION_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[PRETRAIN_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]
            impression = impression.rstrip(".")
            impression = splitter.split(impression)
            impression = [point.split(".") for point in impression]
            impression = [sent for point in impression for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            study_impression = []
            for cap in impression:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_impression.append(" ".join(included_tokens))


            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[PRETRAIN_PATH_COL]] = study_sent
                if len(study_impression) <= len(study_sent):
                    impression_num[row[PRETRAIN_PATH_COL]] = len(study_impression)
                else:
                    impression_num[row[PRETRAIN_PATH_COL]] = len(study_sent)
            else:
                to_remove.append(row[PRETRAIN_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove, impression_num

    def _resize_img(self, img, scale):
        """
        Resize the image to the desired scale.
        
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img


def multimodal_collate_fn(batch):
    """Sorts a batch of data with sequence length.

    Args:
        batch: A list of tuples, where each tuple contains the image, caption, caption length, path, and label.

    Returns:
        A dictionary containing the sorted and batched data.
    """
    imgs, cap_len, ids, tokens, attention, path, labels = [], [], [], [], [], [], []

    # Flatten the batch
    for b in batch:
        img, cap, cap_l, p, label = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        labels.append(label)

    # Stack the tensors
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    labels = torch.stack(labels)

    # Sort the caption lengths and indices
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)

    # Create a dictionary of batched data
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "labels": labels[sorted_cap_indices],
    }

    return return_dict
