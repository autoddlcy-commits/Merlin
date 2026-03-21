import copy

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import wordpunct_tokenize
#nltk：自然语言处理库；tokenize模块：文本分词；wordpunct_tokenize函数：分词器，按照字母和标点分开（文本："Hello, world! NLP is fun." -> tokens：['Hello', ',', 'world', '!', 'NLP', 'is', 'fun', '.']）
import torchvision

from merlin.models import i3res

#处理 CT 影像，输出 contrastive_features（512维）和 ehr_features（1692维，即EHR预测）
class ImageEncoder(nn.Module):
    def __init__(
        self,
        ImageEmbedding: bool = False,
        PhenotypeCls: bool = False,
        FiveYearPred: bool = False,
    ):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.PhenotypeCls = PhenotypeCls
        self.FiveYearPred = FiveYearPred
        # pytorch官方的 2D 图像分类模型，ResNet-152；已经在ImageNet训练好了，会看普通2D图像
        resnet = torchvision.models.resnet152(pretrained=True)
        self.i3_resnet = i3res.I3ResNet(
            copy.deepcopy(resnet),
            class_nb=1692 if not self.FiveYearPred else 6,
            conv_class=True,
            ImageEmbedding=self.ImageEmbedding,
            PhenotypeCls=self.PhenotypeCls,
            FiveYearPred=self.FiveYearPred,
        )

    def forward(self, image):
        # 模式1：仅输出 512 维ImageEmbedding，图像特征
        if self.ImageEmbedding:
            contrastive_features = self.i3_resnet(image)
            return contrastive_features
        # 模式2: 仅输出 1692 维Phenotype Predictions，表型预测概率
        elif self.PhenotypeCls:
            return self.i3_resnet(image)
        # 模式3:仅输出五年疾病预测
        elif self.FiveYearPred:
            return self.i3_resnet(image)
        # 模式4:默认模式（多模态训练时使用），输出Image Embeddings,Text Embeddings and Phenotype Predictions
        else:
            contrastive_features, ehr_features = self.i3_resnet(image)
            return contrastive_features, ehr_features

#TextEncoder：处理报告文本，输出 512 维文本特征
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder.gradient_checkpointing_enable()
        self.linear_layer = nn.Linear(768, 512)

    def forward(self, text_labels):
        text_labels = [sanitize_report(text) for text in text_labels]
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        text_embeddings = self.linear_layer(text_embeddings)
        return text_embeddings


class MerlinArchitecture(nn.Module):
    def __init__(
        self,
        init_logit_scale: float = 1.0,
        ImageEmbedding: bool = False,
        PhenotypeCls: bool = False,
        FiveYearPred: bool = False,
    ):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.PhenotypeCls = PhenotypeCls
        self.FiveYearPred = FiveYearPred
        self.encode_image = ImageEncoder(
            ImageEmbedding=self.ImageEmbedding,
            PhenotypeCls=self.PhenotypeCls,
            FiveYearPred=self.FiveYearPred,
        )
        self.encode_text = TextEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(self, image, text=None):
        if self.ImageEmbedding and text is None:
            image_features = self.encode_image(image)
            return image_features
        elif self.PhenotypeCls and text is None:
            phenotype_features = self.encode_image(image)
            return phenotype_features
        elif self.FiveYearPred and text is None:
            five_year_features = self.encode_image(image)
            return five_year_features
        elif self.ImageEmbedding and text is not None:
            raise ValueError("Text input not required for image embedding")
        elif self.PhenotypeCls and text is not None:
            raise ValueError("Text input not required for phenotype classification")
        elif self.FiveYearPred and text is not None:
            raise ValueError("Text input not required for five year disease prediction")
        elif text is None:
            raise ValueError("Text input required for Image and Text embedding")
            
        #图像、EHR、文本编码
        image_features, ehr_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        #维度对齐（单样本填充batch位为第0位）
        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)
            
        #L2归一化：归一化后的向量模长全部变为 1
        ###图像编码器（ResNet/ViT）和文本编码器（BERT/Transformer）输出的数值量级通常不一样；这样对齐到同一个单位超球面上，让不同模态的特征在同一个“量级”下进行比较。
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #image_features.norm(dim=-1, keepdim=True)：计算每个特征向量的 L2 范数（模长）
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        #输出：(图像特征, EHR特征, 文本特征)
        return (
            image_features,
            ehr_features,
            text_features,
        )


def sanitize_report(report):
    report = report.lower()
    return " ".join(wordpunct_tokenize(report))
