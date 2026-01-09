import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Union
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from classification import metrics as Metrics
import classification.model_afloc as model_afloc
from classification.datasets import RSNAPneumonia, NIHChestXray, SIIM
from classification.datasets import CXR_LT_T2, CXR_LT_T3 
from classification.datasets import DHMCLUAD, SICAPV2, Wsss4luad3
from classification.prompts import get_all_text_prompts


class BaseEvaluateEngine:
    """
    This is a universal code structure for downstream task evaluation.
    """

    support_tasks = [
        "single_classification",
        "multi_classification",
    ]

    available_dataloaders = {
        "RSNA": RSNAPneumonia,
        "NIH": NIHChestXray,
        "SIIM": SIIM,
        "CXR-LT-T2": CXR_LT_T2,
        "CXR-LT-T3": CXR_LT_T3,
        "DHMCLUAD": DHMCLUAD,
        "SICAPV2": SICAPV2,
        "Wsss4luad3": Wsss4luad3,
    }

    def __init__(self, args) -> None:
        self.args = args
        self.split = "test"
        self.dataset, self.dataloader = self.load_dataset(args)
        self.device = args.device
        self.model = self.load_model(args)
        self.model.to(self.device)
        self.create_necessary_components()

    @staticmethod
    def get_arguments() -> argparse:
        parser = argparse.ArgumentParser(description='args')
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--bert_type", default="emilyalsentzer/Bio_ClinicalBERT")
        parser.add_argument("--multi_class", default=False, action='store_true')
        parser.add_argument("--ckpt", type=str,
                            default="weight/Pretrained_CXR.ckpt")
        parser.add_argument("--dataset", default="RSNA", 
                            choices=["RSNA", "SIIM", "CXR-LT-T2", "CXR-LT-T3", "NIH", "SICAPV2", "Wsss4luad3", "DHMCLUAD"])
        parser.add_argument("--modality", default='Chest X-ray', type=str)
        parser.add_argument("--dataset_path", default='/mnt/disk3/MIMIC-DATA-Final/', type=str)
        parser.add_argument("--redo", default=False, type=bool)
        parser.add_argument("--compute_ci", default=True, type=bool)
        parser.add_argument("--num_workers", type=int, default=2)
        args = parser.parse_args()
        print("multi_class:", args.multi_class)
        return args

    def create_necessary_components(self):
        print("Setting text prompts and cls feature...")
        all_prompts = get_all_text_prompts(self.dataset.categories, self.args)
        self.model.set_cls_feature(all_prompts)

    def load_model(self, args) -> torch.nn.Module:
        """ Load the model. 
        Just load the model and leave the post-processing in self.post_process()
        Input:
            - Necessary arguments to load the pretrained model. (e.g. ckpt_dir) 
        Return:
            - model(torch.nn.Module): Any torch model with loaded parameters
        """
        model = model_afloc.afloc(
            args.ckpt,
            multi_class=args.multi_class,
            num_classes=len(self.dataset.categories),
            device=args.device)
        model.train(False)

        return model

    def load_dataset(self, args) -> torch.utils.data.Dataset:
        """ Load the dataset
        Input:
            - Necessary arguments to load the dataset. (e.g. dataset_dir) 
        Return:
            - dataset(torch.utils.data.Dataset): A torch dataset.
        """
        transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],std=[0.5])
            ])

        if args.dataset == "NIH":
            self.task = "multi_classification"
            dataset = self.available_dataloaders["NIH"](
                transform=transform_test,
            )
        elif args.dataset == "SIIM":
            self.task = "multi_classification"
            dataset = self.available_dataloaders["SIIM"](
                transform=transform_test,
            )
        elif args.dataset == "RSNA":
            self.task = "multi_classification"
            dataset = self.available_dataloaders["RSNA"](
                transform=transform_test,
            )
        if args.dataset == "CXR-LT-T2":
            self.task = "multi_classification"
            dataset = self.available_dataloaders["CXR-LT-T2"](
                transform=transform_test,
            )
        elif args.dataset == "CXR-LT-T3":
            self.task = "multi_classification"
            dataset = self.available_dataloaders["CXR-LT-T3"](
                transform=transform_test,
            )
        elif args.dataset == "SICAPV2":
            self.task = "multi_classification"
            dataset = self.available_dataloaders["SICAPV2"](
                transform=transform_test,
            )
        elif args.dataset == "DHMCLUAD":
            self.task = "multi_classification"
            dataset = self.available_dataloaders["DHMCLUAD"](
                transform=transform_test,
            )
        elif args.dataset == "Wsss4luad3":
            self.task = "multi_classification"
            dataset = self.available_dataloaders["Wsss4luad3"](
                transform=transform_test,
            )
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        return dataset, dataloader
    
        
    def compute_metrics(self, data, softmax) -> Dict[str,float]:
        """ Compute metrics
        Input:
            - data (Dict[str,np.ndarray]): A dict contain prediction & ground-truth
                For more details about the input Dict, please refer to the instruction
                of function compute_xxx_metrics() in utils.metrics
        Return:
            - metrics (Dict): A dict with multiple metrics
        """
        if self.task == "single_classification":
            return Metrics.compute_single_classification_metrics(**data)
        elif self.task == "multi_classification":
            if self.args.compute_ci:
                return Metrics.compute_multi_classification_metrics_bootstrap(**data, softmax=softmax)
            else:
                return Metrics.compute_multi_classification_metrics(**data)
        else:
            raise ValueError("[ERROR] Do not support metric computation for "
                             f"task: {self.task} Currently supported tasks are:"
                             f" {self.support_tasks}")


    def post_process(self, output: torch.Tensor, **kwargs) -> np.ndarray:
        """ Re-format your model output to match the unified metrics.
        Input: 
            - output: any data type & shapes from your model.
            - Other necessary arguments.
        Return:
            - output (np.ndarray): re-formatted data to fit metrics. 
                                   It should be: (batch_size, class_number)
        """

        """ Rules of output. The rules is diff for vaious tasks:        
        - Classification (single/multi-class): 
            np.ndarry(float32) with shape (batch_size, class_number)
        """
        if self.task == "single_classification":
            # reshape to [batch_size, num_classes]
            pos, neg = output[:,0,:].detach().chunk(2, dim=-1)
            # swap the order of pos and neg
            output = torch.cat([neg, pos], dim=-1).cpu().numpy()
        elif args.multi_class:
            output = output[:,:,0].detach().cpu().numpy()
        else:
            output = output.detach().cpu().numpy()
        
        return output

    def prepare_eval_data(
            self, 
            output: torch.Tensor, 
            batch: Dict[str,Union[np.ndarray,torch.Tensor,float]],
        ) -> Dict[str,np.ndarray]:
        """Prepare the `data` required in self.compute_metrics()
        Input:
            - output: model output or prediction
            - batch: batch data from dataloader
        Return:
            - Dict[str,np.ndarray]
        """
        if self.task == "single_classification":
            data = dict(
                pred=output,
                label=batch["label"],
            )
        elif self.task == "multi_classification":
            data = dict(
                pred=output,
                label=batch["label"],
                categories=self.dataset.categories,
            )

        return data

    @torch.no_grad()
    def evaluate(self) -> None:
        ckpt_name = self.args.ckpt.split("/")[-1].split(".")[0]
        ckpt_dir = os.path.dirname(self.args.ckpt)
        buff_path = f"{ckpt_dir}/{ckpt_name}_{self.args.dataset}.pt"
        all_outputs, all_labels, latent_imgs = [], [], []

        if not os.path.exists(buff_path) or self.args.redo:
            for batch in tqdm(self.dataloader, ncols=100, total=len(self.dataloader)):
                input_img = batch["image"].to(self.device)
                output, latent_img, logits_pos, logits_neg = self.model(input_img)
                # post-process
                output = self.post_process(output)

                all_outputs.append(output)
                all_labels.append(batch["label"].cpu().numpy())
                latent_imgs.append(latent_img)
    
            latent_imgs = torch.cat(latent_imgs, dim=0)
            buff = dict(
                latent_imgs=latent_imgs,
                all_labels=all_labels,
            )
            torch.save(buff, buff_path)
        else:
            print("load from buff:", buff_path)
            buff = torch.load(buff_path)
            latent_imgs = buff["latent_imgs"]
            all_labels = buff["all_labels"]
            for batch in tqdm(range(len(self.dataloader)), ncols=100):
                input_img = None
                output, _, _, _ = self.model(input_img, latent_imgs[batch*self.args.batch_size:(batch+1)*self.args.batch_size])
                # post-process
                output = self.post_process(output)
                all_outputs.append(output)
                
        ckpt_dir = os.path.dirname(self.args.ckpt)
        save_dir = os.path.join(ckpt_dir, self.args.dataset, ckpt_name)
        metric_data = self.prepare_eval_data(
            np.concatenate(all_outputs), 
            dict(label=np.concatenate(all_labels))
        )
        softmax = True if "Path" in self.args.ckpt else False
        metrics = self.compute_metrics(metric_data, softmax=softmax)
        
        os.makedirs(save_dir, exist_ok=True)
        m = "ba" if softmax else "auc"
        metrics['cis'][m].to_csv(os.path.join(save_dir, f"{m}_ci.csv"))
        metrics['boot_stats'][m].to_csv(os.path.join(save_dir, f"{m}_bootstrap.csv"))
        print(f"metrics:{m}")
        print(metrics['cis'][m])

    def run(self):
        self.evaluate()


if __name__ == "__main__":
    args = BaseEvaluateEngine.get_arguments()
    engine = BaseEvaluateEngine(args)
    engine.run()

            