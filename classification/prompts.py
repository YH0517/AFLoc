from typing import List, Tuple


def _get_xray_text_promtps(category, args=None) -> Tuple[List, List]:
    dct = {
        "atelectasis": {
            "severity": ["", "mild", "minimal"],
            "subtype": [
                "subsegmental atelectasis",
                "linear atelectasis",
                "trace atelectasis",
                "bibasilar atelectasis",
                "retrocardiac atelectasis",
                "bandlike atelectasis",
                "residual atelectasis",
            ],
        "location": [
                "at the mid lung zone",
                "at the upper lung zone",
                "at the right lung zone",
                "at the left lung zone",
                "at the lung bases",
                "at the right lung base",
                "at the left lung base",
                "at the bilateral lung bases",
                "at the left lower lobe",
                "at the right lower lobe",
            ],
        },
        "cardiomegaly": {
            "severity": [""],
            "subtype": [
                "cardiac silhouette size is upper limits of normal",
                "cardiomegaly which is unchanged",
                "mildly prominent cardiac silhouette",
                "portable view of the chest demonstrates stable cardiomegaly",
                "portable view of the chest demonstrates mild cardiomegaly",
                "persistent severe cardiomegaly",
                "heart size is borderline enlarged",
                "cardiomegaly unchanged",
                "heart size is at the upper limits of normal",
                "redemonstration of cardiomegaly",
                "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
                "cardiac silhouette size is mildly enlarged",
                "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
                "heart size remains at mildly enlarged",
                "persistent cardiomegaly with prominent upper lobe vessels",
            ],
            "location": [""],
        },
        "consolidation": {
            "severity": ["", "increased", "improved", "apperance of"],
            "subtype": [
                "bilateral consolidation",
                "reticular consolidation",
                "retrocardiac consolidation",
                "patchy consolidation",
                "airspace consolidation",
                "partial consolidation",
            ],
            "location": [
                "at the lower lung zone",
                "at the upper lung zone",
                "at the left lower lobe",
                "at the right lower lobe",
                "at the left upper lobe",
                "at the right uppper lobe",
                "at the right lung base",
                "at the left lung base",
            ],
        },
        "edema": {
            "severity": [
                "mild",
                "presistent",
                "moderate",
            ],
            "subtype": [
                "pulmonary edema",
            ],
            "location": [                
                ""
                ],
        },
        "effusion": {
            "severity": [
                "small", 
                "stable", 
                "large", 
                "decreased", 
                "increased"
                         ],
            "location": [
                "left", 
                "right", 
                "tiny"
                ],
            "subtype": [
                "bilateral pleural effusion",
                "subpulmonic pleural effusion",
                "bilateral pleural effusion",
            ],
        },
        "pneumonia": {
            "severity": [
                "", 
                "mild", 
                "moderate", 
                "severe", 
                "improved", 
                "worsening"
            ],
            "subtype": [
                "pneumonia"
            ],
            "location": [
                "at the right lower lobe",
                "at the left lower lobe",
                "at the right upper lobe",
                "at the left upper lobe",
                "at the middle lobe",
                "at the lung bases",
                "at the bilateral lungs",
                "at the right lung",
                "at the left lung",
            ],
        },
        "pneumothorax": {
            "severity": [
                "", 
                "small", 
                "large", 
                "increased", 
                "stable", 
                "resolving"
            ],
            "subtype": [
                "pneumothorax"
            ],
            "location": [
                "right",
                "left",
                "bilateral",
                "at the apex",
                "at the upper lung zone",
                "at the mid lung zone",
                "at the lower lung zone",
            ],
        },
        "infiltration": {
            "severity": [
                ""
                         ],
            "subtype": [
                "diffuse infiltration",
                "patchy infiltration",
                "lobar infiltration",
                "interstitial infiltration",
            ],
            "location": [
                ""
            ],
        },
        "mass": {
            "severity": [
                "small", 
                "medium", 
                "large"
                ],
            "subtype": [
                "solitary mass",
                "multiple masses",
                "well-circumscribed mass",
                "spiculated mass",
            ],
            "location": [
                "at the right upper lobe",
                "at the left upper lobe",
                "at the right lower lobe",
                "at the left lower lobe",
                "mediastinal",
            ],
        },
        "nodule": {
            "severity": [
                "tiny", 
                "small", 
                "large"
                ],
            "subtype": [
                ""
            ],
            "location": [
                "at the right upper lobe",
                "at the left upper lobe",
                "at the right lower lobe",
                "at the left lower lobe",
                "peripheral",
                "central",
            ],
        },
        "emphysema": {
            "severity": [
                "mild", 
                "moderate", 
                "severe"
                ],
            "subtype": [
                "centrilobular emphysema",
                "panlobular emphysema",
                "paraseptal emphysema",
                "bullous emphysema",
            ],
            "location": [
                "upper zones", 
                "lower zones", 
                "diffuse"
                ],
        },
        "fibrosis": {
            "severity": ["mild", "moderate", "severe"],
            "subtype": [
                "reticular fibrosis",
                "honeycombing",
                "traction bronchiectasis",
            ],
            "location": ["upper zones", "lower zones", "diffuse"],
        },
        "pleural thickening": {
            "severity": ["mild", "moderate", "severe"],
            "subtype": ["diffuse pleural thickening", "localized pleural thickening"],
            "location": ["apical", "basal", "medial", "lateral", "right", "left"],
        },
        "hernia": {
            "severity": [
                "mild",
                "moderate",
                "severe"
                ], 
            "subtype": [
                "hiatal hernia",
                "diaphragmatic hernia",
                "parahiatal hernia",
            ],
            "location": [
                "right", 
                "left", 
                "central"
                ],
        },
        "fracture": {
            "severity": ["", "mild", "moderate", "severe"],
            "subtype": [
                "rib fracture",
                "vertebral fracture",
                "clavicle fracture",
                "scapula fracture",
            ],
            "location": ["right", "left", "central"],
        },
        "pleural effusion": {
            "severity": ["", "small", "moderate", "large"],
            "subtype": [
                "bilateral pleural effusion",
                "right pleural effusion",
                "left pleural effusion",
            ],
            "location": ["right", "left", "bilateral"],
        },
        "pleural other": {
            "severity": ["mild", "moderate", "severe"],
            "subtype": [
                "pleural calcification",
                "pleural plaques",
                "pleural cyst",
                "fibrothorax"
            ],
            "location": ["apical", "basal", "medial", "lateral", "right", "left", "diffuse"]
        },
        "enlarged cardiomediastinum": {
            "severity": ["mild", "moderate", "severe"],
            "subtype": [
                "aortic aneurysm",
                "mediastinal mass",
                "pericardial effusion",
                "cardiac enlargement"
            ],
            "location": ["anterior", "middle", "posterior", "superior", "inferior"]
        },
        "airspace opacity": {
            "severity": ["mild", "moderate", "severe"],
            "subtype": [
                "consolidation",
                "ground-glass opacity",
                "atelectasis",
                "alveolar hemorrhage",
                "pulmonary edema"
            ],
            "location": ["upper lobe", "middle lobe", "lower lobe", "perihilar", "peripheral", "right lung", "left lung", "bilateral"]
        },
        "calcification of the aorta": {
            "severity": ["mild", "moderate", "severe"],
            "subtype": [
                "intimal calcification",
                "medial calcification",
                "perivascular calcification",
                "diffuse calcification",
                "focal calcification"
            ],
            "location": ["ascending aorta", "descending aorta", "aortic arch", "abdominal aorta"],
        },
        "lung lesion": {
            "severity": ["small", "medium", "large"],
            "subtype": [
                "solitary lesion",
                "multiple lesions",
                "nodular lesion",
                "mass-like lesion",
                "cystic lesion"
            ],
            "location": [
                "right upper lobe",
                "left upper lobe",
                "right lower lobe",
                "left lower lobe",
                "right lung base",
                "left lung base",
                "bilateral lung bases"
            ],
        },
        "lung opacity": {
            "severity": ["mild", "moderate", "severe"],
            "subtype": [
                "consolidation",
                "ground-glass opacity",
                "infiltrate",
                "airspace opacity",
                "reticulonodular opacity"
            ],
            "location": [
                "right lung",
                "left lung",
                "bilateral lungs",
                "upper lobes",
                "lower lobes"
            ],
        },
        "normal": {
            "severity": [""],
            "subtype": [
                "no abnormalities detected",
                "normal findings",
                "unremarkable study"
            ],
            "location": [""],
        },
        "pneumomediastinum": {
            "severity": [""],
            "subtype": [
                "presence of free air in the mediastinum",
                "mild pneumomediastinum",
                "severe pneumomediastinum"
            ],
            "location": ["anterior mediastinum", "posterior mediastinum", "superior mediastinum"],
        },
        "pneumoperitoneum": {
            "severity": [""],
            "subtype": [
                "gas under diaphragm",
                "perforation of hollow viscus",
                "post-surgical pneumoperitoneum"
            ],
            "location": ["free air in the abdomen", "abdominal cavity"],
        },
        "subcutaneous emphysema": {
            "severity": ["mild", "moderate", "severe"],
            "subtype": [
                "localized subcutaneous emphysema",
                "extensive subcutaneous emphysema",
                "diffuse subcutaneous emphysema"
            ],
            "location": ["neck", "chest wall", "abdomen", "upper limbs", "lower limbs"],
        },
        "support devices": {
            "severity": [""],
            "subtype": [
                "endotracheal tube",
                "central venous catheter",
                "pacemaker",
                "internal defibrillator",
                "ventilator",
                "external fixator"
            ],
            "location": ["chest", "neck", "abdomen", "extremities"],
        },
        "tortuous aorta": {
            "severity": ["mild", "moderate", "severe"],
            "subtype": [
                "ascending aorta tortuosity",
                "descending aorta tortuosity",
                "aortic arch tortuosity"
            ],
            "location": ["aortic arch", "ascending aorta", "descending aorta"],
        },
        "bulla": {
            "severity": ["small", "medium", "large"],
            "subtype": [
                "apical bulla",
                "subpleural bulla",
                "bullous emphysema",
                "paraseptal bulla"
            ],
            "location": ["right lung", "left lung", "upper lobes", "lower lobes", "bilateral lungs"],
        },
        "cardiomyopathy": {
            "severity": ["mild", "moderate", "severe"],
            "subtype": [
                "dilated cardiomyopathy",
                "hypertrophic cardiomyopathy",
                "restrictive cardiomyopathy",
                "arrhythmogenic right ventricular cardiomyopathy"
            ],
            "location": ["left ventricle", "right ventricle", "bilateral ventricles"],
        },
        "hilum": {
            "severity": [""],
            "subtype": [
                "enlarged hilum",
                "abnormal hilum",
                "normal hilum",
            ],
            "location": [
                "left hilum", 
                "right hilum", 
                "bilateral hilar",
                ],
        },
        "osteopenia": {
            "severity": ["mild", "moderate", "severe"],
            "subtype": [
                "generalized osteopenia",
                "localized osteopenia"
            ],
            "location": ["spine", "hips", "wrists", "ankles"],
        },
        "scoliosis": {
            "severity": [
                ""
                ],
            "subtype": [
                "idiopathic scoliosis",
                "congenital scoliosis",
                "neuromuscular scoliosis",
                "degenerative scoliosis",
            ],
            "location": [
                ""
                ],
        },
    }
    pos_query = []

    if category.lower() in dct.keys():
        for sev in dct[category.lower()]["severity"]:
            for sub in dct[category.lower()]["subtype"]:
                for loc in dct[category.lower()]["location"]:
                    pos_query.append(
                        f"{category}:{sev} {sub} {loc}"
                    )
        neg_query = [
            f"no {category}. ",
        ]
    elif category.lower() == "no finding":
        pos_query.append("No finding")
        neg_query = [
            "finding",
        ]
    else:
        pos_query.append(f'Findings suggesting {category.lower()}')
        neg_query = [
            f"no {category}. ",
        ]

    return pos_query, neg_query


def get_all_text_prompts(categories, args):
    prompts_func_collection = {
        "Chest X-ray": _get_xray_text_promtps,
        "Pathology": _get_pathology_text_prompts,
    }

    prompts_func = prompts_func_collection[args.modality]

    prompts = {}
    for cat in categories:
        pos_query, neg_query = prompts_func(cat, args)
        prompts[cat] = {
            "pos": pos_query,
            "neg": neg_query,
        }
    return prompts

def label2cap(dataset):
    prompts_dct = {
        "SICAPV2": {
            "classnames": {
            "NC": "benign prostate glands",
            "G3": "prostate cancer, gleason pattern 3",
            "G4": "gleason grade 4",
            "G5": "prostate cancer, gleason grade 5",
            "Tumor": "prostate cancer"
            },
            "templates": "presence of CLASSNAME."
        },
        "Wsss4luad3": {
            "classnames": {
            "normal": "non-tumor",
            "stroma": "tumor-associated stromal tissue",
            "tumor": "invasive carcinoma"
            },
            "templates": "CLASSNAME is present."
        },
        "DHMCLUAD": {
            "classnames": {
            "papillary": "lung adenocarcinoma, papillary growth pattern",
            "solid": "lung adenocarcinoma with a predominantly solid growth pattern",
            "micropapillary": "micropapillary pattern adenocarcinoma of the lung",
            "acinar": "lung adenocarcinoma, acinar predominant histological subtype",
            "lepidic": "lepidic pattern adenocarcinoma of the lung"
            },
            "templates": "a histopathological photograph of CLASSNAME.",
            "neg_templates": "no CLASSNAME is found. "
        }
    }
    prompts = prompts_dct[dataset]
    label_captions = dict()
    types = list(prompts['classnames'].keys())
    for type_name in types:
        label_captions[type_name] = []
        cls_name = prompts['classnames'][type_name]
        template = prompts['templates']
        cap = template.replace('CLASSNAME',cls_name)
        label_captions[type_name].append(cap)

    return label_captions, len(prompts)

def _get_pathology_text_prompts(cat, args):
    all_prompts, num_prompts = label2cap(dataset=args.dataset)
    pos_query = [all_prompts[cat][0]]
    neg_query = [f"no {cat}. "]
    return pos_query, neg_query


